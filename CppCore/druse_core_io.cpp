// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core_internal.h"

// Structure IO, PDB/molblock utilities, and Vina typing.

#include <GraphMol/FileParsers/FileParsers.h>

DruseMoleculeResult* druse_add_hydrogens_pdb(const char *pdbContent) {
    if (!pdbContent || !pdbContent[0]) return make_error("Empty PDB content");
    try {
        std::unique_ptr<RWMol> mol(PDBBlockToMol(std::string(pdbContent), true, true, false));
        if (!mol) return make_error("Failed to parse PDB block");

        MolOps::addHs(*mol, false, true);
        return mol_to_result(*mol, "protein");
    } catch (const std::exception &e) {
        return make_error(e.what());
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
            mol->addAtom(&atom, true, true);
        }

        for (int32_t i = 0; i < bondCount && bonds; i++) {
            int a1 = bonds[i].atom1;
            int a2 = bonds[i].atom2;
            if (a1 < 0 || a1 >= atomCount || a2 < 0 || a2 >= atomCount) continue;
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
