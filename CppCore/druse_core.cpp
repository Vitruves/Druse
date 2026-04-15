// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core_internal.h"

#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace RDKit;

namespace {

constexpr int32_t VINA_XS_C_H = 0;
constexpr int32_t VINA_XS_C_P = 1;
constexpr int32_t VINA_XS_N_P = 2;
constexpr int32_t VINA_XS_N_D = 3;
constexpr int32_t VINA_XS_N_A = 4;
constexpr int32_t VINA_XS_N_DA = 5;
constexpr int32_t VINA_XS_O_P = 6;
constexpr int32_t VINA_XS_O_D = 7;
constexpr int32_t VINA_XS_O_A = 8;
constexpr int32_t VINA_XS_O_DA = 9;
constexpr int32_t VINA_XS_S_P = 10;
constexpr int32_t VINA_XS_P_P = 11;
constexpr int32_t VINA_XS_F_H = 12;
constexpr int32_t VINA_XS_Cl_H = 13;
constexpr int32_t VINA_XS_Br_H = 14;
constexpr int32_t VINA_XS_I_H = 15;
constexpr int32_t VINA_XS_Si = 16;
constexpr int32_t VINA_XS_At = 17;
constexpr int32_t VINA_XS_Met_D = 18;
constexpr int32_t VINA_XS_OTHER = 31;

bool is_metal_atomic_num(unsigned int atomicNum) {
    switch (atomicNum) {
        case 3: case 4: case 11: case 12: case 13: case 19: case 20:
        case 21: case 22: case 23: case 24: case 25: case 26: case 27:
        case 28: case 29: case 30: case 31:
            return true;
        default:
            return false;
    }
}

bool is_hetero_atomic_num(unsigned int atomicNum) {
    return atomicNum != 1 && atomicNum != 6;
}

bool bonded_to_heteroatom(const Atom &atom) {
    const auto &mol = atom.getOwningMol();
    for (const auto *nbr : mol.atomNeighbors(&atom)) {
        if (is_hetero_atomic_num(nbr->getAtomicNum())) {
            return true;
        }
    }
    return false;
}

std::unique_ptr<MolChemicalFeatureFactory> make_vina_feature_factory() {
    static constexpr const char *kBaseFeaturesPath = "/opt/homebrew/opt/rdkit/share/RDKit/Data/BaseFeatures.fdef";
    std::ifstream input(kBaseFeaturesPath);
    if (!input.good()) {
        return nullptr;
    }

    std::stringstream buffer;
    buffer << input.rdbuf();
    return std::unique_ptr<MolChemicalFeatureFactory>(buildFeatureFactory(buffer.str()));
}

} // namespace

const MolChemicalFeatureFactory *vina_feature_factory() {
    static const std::unique_ptr<MolChemicalFeatureFactory> factory = make_vina_feature_factory();
    return factory.get();
}

void compute_donor_acceptor_flags(
    const ROMol &mol,
    std::vector<bool> &donorFlags,
    std::vector<bool> &acceptorFlags
) {
    donorFlags.assign(mol.getNumAtoms(), false);
    acceptorFlags.assign(mol.getNumAtoms(), false);

    if (const auto *factory = vina_feature_factory()) {
        auto features = factory->getFeaturesForMol(mol);
        for (const auto &feature : features) {
            if (!feature) {
                continue;
            }

            const std::string &family = feature->getFamily();
            const bool isDonor = family == "Donor";
            const bool isAcceptor = family == "Acceptor";
            if (!isDonor && !isAcceptor) {
                continue;
            }

            for (const auto *atom : feature->getAtoms()) {
                if (!atom) {
                    continue;
                }
                const unsigned int idx = atom->getIdx();
                if (idx >= donorFlags.size()) {
                    continue;
                }
                if (isDonor) {
                    donorFlags[idx] = true;
                }
                if (isAcceptor) {
                    acceptorFlags[idx] = true;
                }
            }
        }
    }
}

int32_t vina_xs_type_for_atom(
    const ROMol &,
    const Atom &atom,
    const std::vector<bool> &donorFlags,
    const std::vector<bool> &acceptorFlags
) {
    const unsigned int idx = atom.getIdx();
    const bool donor = idx < donorFlags.size() ? donorFlags[idx] : false;
    const bool acceptor = idx < acceptorFlags.size() ? acceptorFlags[idx] : false;

    switch (atom.getAtomicNum()) {
        case 6:
            return bonded_to_heteroatom(atom) ? VINA_XS_C_P : VINA_XS_C_H;
        case 7:
            if (acceptor && donor) return VINA_XS_N_DA;
            if (acceptor) return VINA_XS_N_A;
            if (donor) return VINA_XS_N_D;
            return VINA_XS_N_P;
        case 8:
            if (acceptor && donor) return VINA_XS_O_DA;
            if (acceptor) return VINA_XS_O_A;
            if (donor) return VINA_XS_O_D;
            return VINA_XS_O_P;
        case 16:
            return VINA_XS_S_P;
        case 15:
            return VINA_XS_P_P;
        case 9:
            return VINA_XS_F_H;
        case 17:
            return VINA_XS_Cl_H;
        case 35:
            return VINA_XS_Br_H;
        case 53:
            return VINA_XS_I_H;
        case 14:
            return VINA_XS_Si;
        case 85:
            return VINA_XS_At;
        default:
            if (is_metal_atomic_num(atom.getAtomicNum())) {
                return VINA_XS_Met_D;
            }
            return VINA_XS_OTHER;
    }
}

// ============================================================================
// Helpers
// ============================================================================

DruseMoleculeResult* make_error(const char *msg) {
    auto *res = new DruseMoleculeResult();
    memset(res, 0, sizeof(DruseMoleculeResult));
    res->success = false;
    strncpy(res->errorMessage, msg, sizeof(res->errorMessage) - 1);
    return res;
}

void mmff_minimize_single(RWMol &mol, int confId) {
    MMFF::MMFFOptimizeMolecule(mol, 1000, "MMFF94", 10.0, confId);
}

void mmff_minimize_confs_pick_best(RWMol &mol) {
    std::vector<std::pair<int, double>> results;
    MMFF::MMFFOptimizeMoleculeConfs(mol, results);
    if (results.empty()) return;

    double bestE = 1e10;
    int bestC = 0;
    for (int i = 0; i < (int)results.size(); i++) {
        if (results[i].first >= 0 && results[i].second < bestE) {
            bestE = results[i].second;
            bestC = i;
        }
    }
    if (bestC != 0 && mol.getNumConformers() > (unsigned)bestC) {
        const auto &bp = mol.getConformer(bestC);
        auto &fp = mol.getConformer(0);
        for (unsigned int a = 0; a < mol.getNumAtoms(); a++)
            fp.setAtomPos(a, bp.getAtomPos(a));
    }
}

int embed_molecule(RWMol &mol) {
    DGeomHelpers::EmbedParameters params = DGeomHelpers::ETKDGv3;
    params.randomSeed = 42;
    return DGeomHelpers::EmbedMolecule(mol, params);
}

void populate_pdb_metadata(const RWMol &mol, const Atom &atom, DruseAtom &da) {
    auto populate_from_info = [&](const AtomPDBResidueInfo &info) {
        std::string atomName = info.getName();
        if (!atomName.empty()) strncpy(da.name, atomName.c_str(), sizeof(da.name) - 1);

        std::string residueName = info.getResidueName();
        if (!residueName.empty()) strncpy(da.residueName, residueName.c_str(), sizeof(da.residueName) - 1);

        std::string chainID = info.getChainId();
        if (!chainID.empty()) strncpy(da.chainID, chainID.c_str(), sizeof(da.chainID) - 1);

        std::string altLoc = info.getAltLoc();
        if (!altLoc.empty()) strncpy(da.altLoc, altLoc.c_str(), sizeof(da.altLoc) - 1);

        da.residueSeq = info.getResidueNumber();
        da.occupancy = (float)info.getOccupancy();
        da.tempFactor = (float)info.getTempFactor();
        da.isHetAtom = info.getIsHeteroAtom();
    };

    if (const auto *info = dynamic_cast<const AtomPDBResidueInfo*>(atom.getMonomerInfo())) {
        populate_from_info(*info);
        return;
    }

    if (atom.getAtomicNum() == 1) {
        for (const auto *nbr : mol.atomNeighbors(&atom)) {
            if (const auto *nbrInfo = dynamic_cast<const AtomPDBResidueInfo*>(nbr->getMonomerInfo())) {
                populate_from_info(*nbrInfo);
                if (!da.name[0]) strncpy(da.name, "H", sizeof(da.name) - 1);
                return;
            }
        }
    }
}

std::vector<int> embed_multiple(RWMol &mol, int numConfs) {
    DGeomHelpers::EmbedParameters params = DGeomHelpers::ETKDGv3;
    params.randomSeed = 42;
    return DGeomHelpers::EmbedMultipleConfs(mol, numConfs, params);
}

DruseMoleculeResult* mol_to_result(RWMol &mol, const char *name) {
    auto *res = new DruseMoleculeResult();
    memset(res, 0, sizeof(DruseMoleculeResult));
    res->success = true;

    if (name && name[0])
        strncpy(res->name, name, sizeof(res->name) - 1);

    try {
        std::string smi = MolToSmiles(mol);
        strncpy(res->smiles, smi.c_str(), sizeof(res->smiles) - 1);
    } catch (...) {}

    if (mol.getNumConformers() == 0) {
        // No 3D coords — just return basic info
        res->atomCount = 0;
        res->bondCount = 0;
        res->atoms = nullptr;
        res->bonds = nullptr;
        return res;
    }

    // Use the first available conformer (ID may not be 0 after conformer pruning)
    const auto &conf = *mol.beginConformers()->get();
    int natoms = (int)mol.getNumAtoms();
    res->atomCount = natoms;
    res->atoms = new DruseAtom[natoms];

    for (int i = 0; i < natoms; i++) {
        const auto &pos = conf.getAtomPos(i);
        const auto *atom = mol.getAtomWithIdx(i);
        DruseAtom &da = res->atoms[i];
        memset(&da, 0, sizeof(DruseAtom));
        da.x = (float)pos.x;
        da.y = (float)pos.y;
        da.z = (float)pos.z;
        da.atomicNum = atom->getAtomicNum();
        da.formalCharge = atom->getFormalCharge();

        double gc = 0.0;
        try { atom->getProp("_GasteigerCharge", gc); } catch (...) {}
        da.charge = (float)gc;

        std::string sym = atom->getSymbol();
        strncpy(da.symbol, sym.c_str(), sizeof(da.symbol) - 1);
        populate_pdb_metadata(mol, *atom, da);
        if (!da.name[0]) strncpy(da.name, sym.c_str(), sizeof(da.name) - 1);
    }

    int nbonds = (int)mol.getNumBonds();
    res->bondCount = nbonds;
    res->bonds = new DruseBond[nbonds];

    for (int i = 0; i < nbonds; i++) {
        const auto *bond = mol.getBondWithIdx(i);
        DruseBond &db = res->bonds[i];
        db.atom1 = bond->getBeginAtomIdx();
        db.atom2 = bond->getEndAtomIdx();
        switch (bond->getBondType()) {
            case Bond::SINGLE:   db.order = 1; break;
            case Bond::DOUBLE:   db.order = 2; break;
            case Bond::TRIPLE:   db.order = 3; break;
            case Bond::AROMATIC: db.order = 4; break;
            default:             db.order = 1; break;
        }
    }

    res->molecularWeight = (float)Descriptors::calcAMW(mol);
    try { res->logP = (float)Descriptors::calcClogP(mol); } catch (...) {}
    res->tpsa = (float)Descriptors::calcTPSA(mol);
    res->hbd = Descriptors::calcNumHBD(mol);
    res->hba = Descriptors::calcNumHBA(mol);
    res->rotatableBonds = Descriptors::calcNumRotatableBonds(mol);
    res->numConformers = mol.getNumConformers();

    return res;
}

// ============================================================================
// SMILES → 3D
// ============================================================================

DruseMoleculeResult* druse_smiles_to_3d(const char *smiles, const char *name) {
    return druse_smiles_to_3d_conformers(smiles, name, 1, true);
}

DruseMoleculeResult* druse_smiles_to_3d_conformers(
    const char *smiles, const char *name,
    int32_t numConformers, bool mmffMinimize
) {
    if (!smiles || !smiles[0]) return make_error("Empty SMILES string");

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return make_error("Failed to parse SMILES");

        MolOps::addHs(*mol);

        int confId;
        if (numConformers <= 1) {
            confId = embed_molecule(*mol);
        } else {
            auto confIds = embed_multiple(*mol, numConformers);
            if (confIds.empty()) return make_error("Conformer generation failed");
            confId = confIds[0];
        }

        if (confId < 0) return make_error("3D embedding failed");

        if (mmffMinimize) {
            try {
                if (numConformers > 1)
                    mmff_minimize_confs_pick_best(*mol);
                else
                    mmff_minimize_single(*mol);
            } catch (...) {}
        }

        return mol_to_result(*mol, name);
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error in SMILES to 3D");
    }
}

// ============================================================================
// Preparation
// ============================================================================

DruseMoleculeResult* druse_add_hydrogens(const char *smiles, const char *name) {
    if (!smiles || !smiles[0]) return make_error("Empty SMILES");
    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return make_error("Failed to parse SMILES");
        MolOps::addHs(*mol);
        embed_molecule(*mol);
        return mol_to_result(*mol, name);
    } catch (const std::exception &e) {
        return make_error(e.what());
    }
}

DruseMoleculeResult* druse_compute_gasteiger_charges(const char *smiles, const char *name) {
    if (!smiles || !smiles[0]) return make_error("Empty SMILES");
    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return make_error("Failed to parse SMILES");
        MolOps::addHs(*mol);
        embed_molecule(*mol);
        computeGasteigerCharges(*mol);
        return mol_to_result(*mol, name);
    } catch (const std::exception &e) {
        return make_error(e.what());
    }
}

/// Apply dominant protonation state (WASH) at given pH using the shared pKa table.
/// Modifies the molecule in-place: protonates strong bases, deprotonates strong acids.
/// Used internally by ensemble pipeline Phase A; NOT called by prepareLigand (which
/// preserves the as-drawn state, leaving protomer enumeration to prepareEnsemble).
__attribute__((unused))
static void washProtonation(RWMol &mol, double pH = 7.4, double threshold = 2.0) {
    auto sites = detectIonSitesInternal(mol);
    for (const auto &[atomIdx, groupIdx, isAcid, defaultPKa] : sites) {
        Atom *atom = mol.getAtomWithIdx(atomIdx);
        double deltaPH = pH - defaultPKa; // positive = pH above pKa

        if (isAcid && deltaPH > threshold) {
            // pH well above pKa: deprotonate acid
            int nH = atom->getTotalNumHs();
            if (nH > 0) {
                atom->setNumExplicitHs(nH - 1);
                atom->setFormalCharge(atom->getFormalCharge() - 1);
            }
        } else if (!isAcid && deltaPH < -threshold) {
            // pH well below pKa: protonate base
            int nH = atom->getTotalNumHs();
            atom->setNumExplicitHs(nH + 1);
            atom->setFormalCharge(atom->getFormalCharge() + 1);
        }
    }
    try { MolOps::sanitizeMol(mol); } catch (...) {}
}

DruseMoleculeResult* druse_prepare_ligand(
    const char *smiles, const char *name,
    int32_t numConformers, bool addHydrogens,
    bool minimize, bool computeCharges
) {
    if (!smiles || !smiles[0]) return make_error("Empty SMILES");
    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return make_error("Failed to parse SMILES");

        MolOps::sanitizeMol(*mol);

        // NOTE: No automatic WASH here. Protonation state enumeration is handled by
        // druse_prepare_ligand_ensemble() which generates all protomers with Boltzmann
        // populations. This allows the user to see and select minority species
        // (e.g. 10% neutral form of a guanidine) that may be biologically relevant.
        // The SMILES is used as-drawn; explicit charges like [NH2+] are preserved.

        if (addHydrogens) MolOps::addHs(*mol);

        if (numConformers <= 1) {
            if (embed_molecule(*mol) < 0) return make_error("Embedding failed");
        } else {
            auto cids = embed_multiple(*mol, numConformers);
            if (cids.empty()) return make_error("Conformer generation failed");
        }

        if (minimize) {
            try {
                if (numConformers > 1)
                    mmff_minimize_confs_pick_best(*mol);
                else
                    mmff_minimize_single(*mol);
            } catch (...) {}
        }

        if (computeCharges) {
            try { computeGasteigerCharges(*mol); } catch (...) {}
        }

        return mol_to_result(*mol, name);
    } catch (const std::exception &e) {
        return make_error(e.what());
    }
}

// ============================================================================
// Descriptors
// ============================================================================

DruseDescriptors druse_compute_descriptors(const char *smiles) {
    DruseDescriptors desc;
    memset(&desc, 0, sizeof(desc));
    if (!smiles || !smiles[0]) return desc;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return desc;

        desc.molecularWeight = (float)Descriptors::calcAMW(*mol);
        desc.exactMW = (float)Descriptors::calcExactMW(*mol);
        desc.logP = (float)Descriptors::calcClogP(*mol);
        desc.tpsa = (float)Descriptors::calcTPSA(*mol);
        desc.hbd = Descriptors::calcNumHBD(*mol);
        desc.hba = Descriptors::calcNumHBA(*mol);
        desc.rotatableBonds = Descriptors::calcNumRotatableBonds(*mol);
        desc.rings = Descriptors::calcNumRings(*mol);
        desc.aromaticRings = Descriptors::calcNumAromaticRings(*mol);
        desc.heavyAtomCount = mol->getNumHeavyAtoms();
        desc.fractionCSP3 = (float)Descriptors::calcFractionCSP3(*mol);

        desc.lipinski = (desc.molecularWeight <= 500 && desc.logP <= 5.0f &&
                         desc.hbd <= 5 && desc.hba <= 10);
        desc.veber = (desc.rotatableBonds <= 10 && desc.tpsa <= 140.0f);
    } catch (...) {}

    return desc;
}

// ============================================================================
// Batch
// ============================================================================

DruseMoleculeResult** druse_batch_process(
    const char **smiles_array, const char **name_array,
    int32_t count, bool addHydrogens, bool minimize, bool computeCharges
) {
    auto **results = new DruseMoleculeResult*[count];
    for (int32_t i = 0; i < count; i++) {
        const char *smi = smiles_array[i];
        const char *nm = (name_array && name_array[i]) ? name_array[i] : "";
        results[i] = druse_prepare_ligand(smi, nm, 1, addHydrogens, minimize, computeCharges);
    }
    return results;
}

// ============================================================================
// Conformer Generation (return all)
// ============================================================================

DruseMoleculeResult* mol_to_result_conf(RWMol &mol, const char *name, int confId) {
    // Like mol_to_result but uses a specific conformer
    auto *res = new DruseMoleculeResult();
    memset(res, 0, sizeof(DruseMoleculeResult));
    res->success = true;
    if (name && name[0]) strncpy(res->name, name, sizeof(res->name) - 1);

    try {
        std::string smi = MolToSmiles(mol);
        strncpy(res->smiles, smi.c_str(), sizeof(res->smiles) - 1);
    } catch (...) {}

    if ((unsigned)confId >= mol.getNumConformers()) {
        res->atomCount = 0;
        res->bondCount = 0;
        return res;
    }

    const auto &conf = mol.getConformer(confId);
    int natoms = (int)mol.getNumAtoms();
    res->atomCount = natoms;
    res->atoms = new DruseAtom[natoms];

    for (int i = 0; i < natoms; i++) {
        const auto &pos = conf.getAtomPos(i);
        const auto *atom = mol.getAtomWithIdx(i);
        DruseAtom &da = res->atoms[i];
        memset(&da, 0, sizeof(DruseAtom));
        da.x = (float)pos.x;
        da.y = (float)pos.y;
        da.z = (float)pos.z;
        da.atomicNum = atom->getAtomicNum();
        da.formalCharge = atom->getFormalCharge();
        double gc = 0.0;
        try { atom->getProp("_GasteigerCharge", gc); } catch (...) {}
        da.charge = (float)gc;
        std::string sym = atom->getSymbol();
        strncpy(da.symbol, sym.c_str(), sizeof(da.symbol) - 1);
        populate_pdb_metadata(mol, *atom, da);
        if (!da.name[0]) strncpy(da.name, sym.c_str(), sizeof(da.name) - 1);
    }

    int nbonds = (int)mol.getNumBonds();
    res->bondCount = nbonds;
    res->bonds = new DruseBond[nbonds];
    for (int i = 0; i < nbonds; i++) {
        const auto *bond = mol.getBondWithIdx(i);
        res->bonds[i].atom1 = bond->getBeginAtomIdx();
        res->bonds[i].atom2 = bond->getEndAtomIdx();
        switch (bond->getBondType()) {
            case Bond::SINGLE:   res->bonds[i].order = 1; break;
            case Bond::DOUBLE:   res->bonds[i].order = 2; break;
            case Bond::TRIPLE:   res->bonds[i].order = 3; break;
            case Bond::AROMATIC: res->bonds[i].order = 4; break;
            default:             res->bonds[i].order = 1; break;
        }
    }

    res->molecularWeight = (float)Descriptors::calcAMW(mol);
    res->numConformers = mol.getNumConformers();
    return res;
}

DruseConformerSet* druse_generate_conformers(
    const char *smiles, const char *name,
    int32_t numConformers, bool minimize
) {
    auto *set = new DruseConformerSet();
    memset(set, 0, sizeof(DruseConformerSet));

    if (!smiles || !smiles[0]) return set;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return set;

        MolOps::addHs(*mol);
        auto cids = embed_multiple(*mol, numConformers);
        if (cids.empty()) return set;

        // MMFF minimize all conformers
        std::vector<double> energies(cids.size(), 0.0);
        if (minimize) {
            std::vector<std::pair<int, double>> results;
            MMFF::MMFFOptimizeMoleculeConfs(*mol, results);
            for (size_t i = 0; i < results.size() && i < energies.size(); i++) {
                energies[i] = results[i].second;
            }
        }

        // Sort conformers by energy
        std::vector<int> sorted_indices(cids.size());
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&energies](int a, int b) { return energies[a] < energies[b]; });

        int count = (int)sorted_indices.size();
        set->count = count;
        set->conformers = new DruseMoleculeResult*[count];
        set->energies = new double[count];

        try { computeGasteigerCharges(*mol); } catch (...) {}

        for (int i = 0; i < count; i++) {
            int confIdx = sorted_indices[i];
            set->conformers[i] = mol_to_result_conf(*mol, name, confIdx);
            set->energies[i] = energies[confIdx];
        }
    } catch (...) {}

    return set;
}

void druse_free_conformer_set(DruseConformerSet *set) {
    if (!set) return;
    if (set->conformers) {
        for (int i = 0; i < set->count; i++)
            druse_free_molecule_result(set->conformers[i]);
        delete[] set->conformers;
    }
    delete[] set->energies;
    delete set;
}

// Memory
// ============================================================================

void druse_free_molecule_result(DruseMoleculeResult *result) {
    if (!result) return;
    delete[] result->atoms;
    delete[] result->bonds;
    delete result;
}

void druse_free_batch_results(DruseMoleculeResult **results, int32_t count) {
    if (!results) return;
    for (int32_t i = 0; i < count; i++)
        druse_free_molecule_result(results[i]);
    delete[] results;
}

const char* druse_rdkit_version(void) {
    static std::string ver = RDKit::rdkitVersion;
    return ver.c_str();
}
