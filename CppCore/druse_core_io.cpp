// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core_internal.h"

// Structure IO, PDB/molblock utilities, and Vina typing.

#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <ForceField/ForceField.h>
#include <ForceField/PositionConstraint.h>

DruseMoleculeResult* druse_add_hydrogens_pdb(const char *pdbContent) {
    if (!pdbContent || !pdbContent[0]) return make_error("Empty PDB content");
    try {
        std::unique_ptr<RWMol> mol(PDBBlockToMol(std::string(pdbContent), true, true, false));
        if (!mol) return make_error("Failed to parse PDB block");

        MolOps::addHs(*mol, false, true);
        return mol_to_result(*mol, "protein");
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error adding hydrogens to PDB");
    }
}

DruseMoleculeResult* druse_compute_charges_pdb(const char *pdbContent) {
    if (!pdbContent || !pdbContent[0]) return make_error("Empty PDB content");
    try {
        std::unique_ptr<RWMol> mol(PDBBlockToMol(std::string(pdbContent), true, true, false));
        if (!mol) return make_error("Failed to parse PDB block");

        try { computeGasteigerCharges(*mol); } catch (...) {}
        return mol_to_result(*mol, "protein");
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error computing PDB charges");
    }
}

DruseMoleculeResult* druse_compute_charges_molblock(const char *molBlock) {
    if (!molBlock || !molBlock[0]) return make_error("Empty mol block");
    try {
        std::unique_ptr<RWMol> mol(MolBlockToMol(std::string(molBlock), true, false, false));
        if (!mol) return make_error("Failed to parse mol block");

        RWMol chargedMol(*mol);
        try { MolOps::addHs(chargedMol); } catch (...) {}
        try { computeGasteigerCharges(chargedMol); } catch (...) {}

        const unsigned int atomCount = mol->getNumAtoms();
        for (unsigned int i = 0; i < atomCount && i < chargedMol.getNumAtoms(); ++i) {
            double gc = 0.0;
            try {
                chargedMol.getAtomWithIdx(i)->getProp("_GasteigerCharge", gc);
                mol->getAtomWithIdx(i)->setProp("_GasteigerCharge", gc);
            } catch (...) {}
        }
        return mol_to_result(*mol, "ligand");
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error computing mol block charges");
    }
}

namespace {

/// Heavy-atom graph of `mol` with elements and connectivity only (no charges, bond orders
/// or hydrogens), for matching structures whose bond orders or protonation differ.
/// `heavyIndices` receives the source index of each skeleton atom.
std::unique_ptr<RWMol> heavy_skeleton(const ROMol &mol, std::vector<unsigned> &heavyIndices) {
    auto skeleton = std::make_unique<RWMol>();
    std::vector<int> toSkeleton(mol.getNumAtoms(), -1);
    heavyIndices.clear();
    for (const auto *atom : mol.atoms()) {
        if (atom->getAtomicNum() <= 1) continue;
        Atom bare(atom->getAtomicNum());
        bare.setNoImplicit(true);
        toSkeleton[atom->getIdx()] = (int)skeleton->addAtom(&bare, true, false);
        heavyIndices.push_back(atom->getIdx());
    }
    for (const auto *bond : mol.bonds()) {
        int a = toSkeleton[bond->getBeginAtomIdx()], b = toSkeleton[bond->getEndAtomIdx()];
        if (a >= 0 && b >= 0 && !skeleton->getBondBetweenAtoms(a, b)) skeleton->addBond(a, b, Bond::SINGLE);
    }
    MolOps::fastFindRings(*skeleton);
    return skeleton;
}

} // namespace

DruseMoleculeResult* druse_prepare_on_template(
    const char *templateMolBlock, const char *targetMolBlock, const char *name,
    bool minimize, bool computeCharges, double *outEnergy
) {
    if (outEnergy) *outEnergy = NAN;
    if (!templateMolBlock || !templateMolBlock[0] || !targetMolBlock || !targetMolBlock[0])
        return make_error("Empty mol block");
    try {
        std::unique_ptr<RWMol> tmpl(MolBlockToMol(std::string(templateMolBlock), false, false, false));
        std::unique_ptr<RWMol> target(MolBlockToMol(std::string(targetMolBlock), true, false, false));
        if (!tmpl || !target || tmpl->getNumConformers() == 0 || target->getNumConformers() == 0)
            return make_error("Failed to parse mol block");

        std::vector<unsigned> tmplHeavy, targetHeavy;
        auto tmplSkeleton = heavy_skeleton(*tmpl, tmplHeavy);
        auto targetSkeleton = heavy_skeleton(*target, targetHeavy);
        if (tmplHeavy.size() != targetHeavy.size() ||
            tmplSkeleton->getNumBonds() != targetSkeleton->getNumBonds())
            return make_error("Heavy atoms differ from the experimental structure");
        SubstructMatchParameters params;
        params.maxMatches = 1;
        params.uniquify = false;
        auto matches = SubstructMatch(*tmplSkeleton, *targetSkeleton, params);
        if (matches.empty()) return make_error("Heavy atoms differ from the experimental structure");

        // Target heavy atoms on the experimental coordinates.
        auto &conf = target->getConformer();
        const auto &tmplConf = tmpl->getConformer();
        for (const auto &[targetSkelIdx, tmplSkelIdx] : matches.front()) {
            conf.setAtomPos(targetHeavy[targetSkelIdx], tmplConf.getAtomPos(tmplHeavy[tmplSkelIdx]));
        }

        // Hydrogens: RDKit places them on the heavy-atom geometry of a hydrogen-free copy
        // (heavy atoms keep their relative order there), and each heavy atom's new
        // hydrogens are copied onto its own hydrogens in the target (which may carry
        // fewer, e.g. polar hydrogens only).
        RWMol placed(*target);
        MolOps::removeAllHs(placed);
        MolOps::addHs(placed, false, true);
        const auto &placedConf = placed.getConformer();
        for (unsigned k = 0; k < targetHeavy.size(); ++k) {
            std::vector<unsigned> own, fresh;
            for (const auto *nbr : target->atomNeighbors(target->getAtomWithIdx(targetHeavy[k])))
                if (nbr->getAtomicNum() == 1) own.push_back(nbr->getIdx());
            for (const auto *nbr : placed.atomNeighbors(placed.getAtomWithIdx(k)))
                if (nbr->getAtomicNum() == 1) fresh.push_back(nbr->getIdx());
            if (own.size() > fresh.size()) return make_error("Hydrogen placement failed");
            for (size_t h = 0; h < own.size(); ++h) conf.setAtomPos(own[h], placedConf.getAtomPos(fresh[h]));
        }

        MMFF::MMFFMolProperties mmffProps(*target);
        if (mmffProps.isValid()) {
            if (minimize) {
                std::unique_ptr<ForceFields::ForceField> ff(MMFF::constructForceField(*target, &mmffProps));
                if (ff) {
                    for (unsigned idx : targetHeavy) {
                        ff->contribs().emplace_back(new ForceFields::PositionConstraintContrib(ff.get(), idx, 0.3, 100.0));
                    }
                    ff->initialize();
                    ff->minimize(2000);
                }
            }
            std::unique_ptr<ForceFields::ForceField> energyFF(MMFF::constructForceField(*target, &mmffProps));
            if (energyFF && outEnergy) {
                energyFF->initialize();
                *outEnergy = energyFF->calcEnergy();
            }
        }

        if (computeCharges) {
            try { computeGasteigerCharges(*target); } catch (...) {}
        }
        try { MolOps::assignStereochemistryFrom3D(*target); } catch (...) {}
        return mol_to_result(*target, name ? name : "ligand");
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error preparing ligand on template");
    }
}

const char* druse_molblock_to_smiles(const char *molBlock) {
    if (!molBlock || !molBlock[0]) return nullptr;
    try {
        std::unique_ptr<RWMol> mol(MolBlockToMol(std::string(molBlock), true, false, false));
        if (!mol) return nullptr;

        try {
            unsigned int failedOp = 0;
            MolOps::sanitizeMol(*mol, failedOp,
                MolOps::SANITIZE_ALL ^ MolOps::SANITIZE_PROPERTIES);
        } catch (...) {
            try { MolOps::findSSSR(*mol); } catch (...) {}
        }

        std::string smi = MolToSmiles(*mol);
        if (smi.empty()) return nullptr;

        char *result = new char[smi.size() + 1];
        std::strcpy(result, smi.c_str());
        return result;
    } catch (...) {
        return nullptr;
    }
}

void druse_free_string(const char *str) {
    delete[] str;
}

DruseMoleculeResult* druse_atoms_bonds_to_smiles(
    const DruseAtom *atoms,
    int32_t atomCount,
    const DruseBond *bonds,
    int32_t bondCount,
    const char *name
) {
    if (!atoms || atomCount <= 0) return make_error("No atoms provided");
    try {
        auto mol = std::make_unique<RWMol>();

        for (int32_t i = 0; i < atomCount; i++) {
            Atom atom(atoms[i].atomicNum);
            atom.setFormalCharge(atoms[i].formalCharge);
            mol->addAtom(&atom, true, false);
        }

        for (int32_t i = 0; i < bondCount && bonds; i++) {
            int a1 = bonds[i].atom1;
            int a2 = bonds[i].atom2;
            if (a1 < 0 || a1 >= atomCount || a2 < 0 || a2 >= atomCount) continue;
            // RDKit throws on self-bonds and duplicate bonds, which would
            // discard the whole molecule; skip them instead.
            if (a1 == a2 || mol->getBondBetweenAtoms(a1, a2)) continue;
            Bond::BondType bt;
            switch (bonds[i].order) {
                case 2:  bt = Bond::DOUBLE;   break;
                case 3:  bt = Bond::TRIPLE;   break;
                case 4:  bt = Bond::AROMATIC; break;
                default: bt = Bond::SINGLE;   break;
            }
            mol->addBond(a1, a2, bt);
        }

        auto *conf = new Conformer(atomCount);
        for (int32_t i = 0; i < atomCount; i++) {
            conf->setAtomPos(i, RDGeom::Point3D(atoms[i].x, atoms[i].y, atoms[i].z));
        }
        mol->addConformer(conf, true);

        try {
            unsigned int failedOp = 0;
            MolOps::sanitizeMol(*mol, failedOp,
                MolOps::SANITIZE_ALL ^ MolOps::SANITIZE_PROPERTIES);
        } catch (...) {
            try { MolOps::findSSSR(*mol); } catch (...) {}
        }

        bool is3D = false;
        for (int32_t i = 0; i < atomCount; i++) {
            if (std::abs(atoms[i].z) > 0.01f) { is3D = true; break; }
        }
        if (is3D) {
            try { MolOps::assignStereochemistryFrom3D(*mol); } catch (...) {}
        }

        return mol_to_result(*mol, name ? name : "ligand");
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error building molecule from atoms/bonds");
    }
}

int32_t druse_compute_vina_types_molblock(const char *molBlock, int32_t *outTypes, int32_t maxAtoms) {
    if (!molBlock || !molBlock[0] || !outTypes || maxAtoms <= 0) {
        return -1;
    }
    try {
        std::unique_ptr<RWMol> mol(MolBlockToMol(std::string(molBlock), true, false, false));
        if (!mol) return -1;

        const int atomCount = static_cast<int>(mol->getNumAtoms());
        if (atomCount > maxAtoms) return -1;

        RWMol typedMol(*mol);
        try { MolOps::addHs(typedMol); } catch (...) {}

        std::vector<bool> donorFlags;
        std::vector<bool> acceptorFlags;
        compute_donor_acceptor_flags(typedMol, donorFlags, acceptorFlags);

        for (int i = 0; i < atomCount; ++i) {
            outTypes[i] = vina_xs_type_for_atom(
                typedMol, *typedMol.getAtomWithIdx(i), donorFlags, acceptorFlags
            );
        }
        return atomCount;
    } catch (...) {
        return -1;
    }
}
