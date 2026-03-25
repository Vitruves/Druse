#include "druse_core.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/PartialCharges/GasteigerCharges.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Depictor/RDDepictor.h>
#include <GraphMol/MolChemicalFeatures/MolChemicalFeature.h>
#include <GraphMol/MolChemicalFeatures/MolChemicalFeatureFactory.h>
#include <GraphMol/MonomerInfo.h>
#include <RDGeneral/versions.h>

#include <cstring>
#include <fstream>
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

} // namespace

// ============================================================================
// Helpers
// ============================================================================

static DruseMoleculeResult* make_error(const char *msg) {
    auto *res = new DruseMoleculeResult();
    memset(res, 0, sizeof(DruseMoleculeResult));
    res->success = false;
    strncpy(res->errorMessage, msg, sizeof(res->errorMessage) - 1);
    return res;
}

static void mmff_minimize_single(RWMol &mol, int confId = -1) {
    MMFF::MMFFOptimizeMolecule(mol, 1000, "MMFF94", 10.0, confId);
}

static void mmff_minimize_confs_pick_best(RWMol &mol) {
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

static int embed_molecule(RWMol &mol) {
    DGeomHelpers::EmbedParameters params = DGeomHelpers::ETKDGv3;
    params.randomSeed = 42;
    return DGeomHelpers::EmbedMolecule(mol, params);
}

static void populate_pdb_metadata(const RWMol &mol, const Atom &atom, DruseAtom &da) {
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

static std::vector<int> embed_multiple(RWMol &mol, int numConfs) {
    DGeomHelpers::EmbedParameters params = DGeomHelpers::ETKDGv3;
    params.randomSeed = 42;
    return DGeomHelpers::EmbedMultipleConfs(mol, numConfs, params);
}

static DruseMoleculeResult* mol_to_result(RWMol &mol, const char *name) {
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

static DruseMoleculeResult* mol_to_result_conf(RWMol &mol, const char *name, int confId) {
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

// ============================================================================
// Tautomer & Protomer Enumeration
// ============================================================================

#include <GraphMol/MolStandardize/Tautomer.h>
#include <GraphMol/Substruct/SubstructMatch.h>

/// Helper: generate 3D for a molecule and return MMFF energy, or NaN on failure.
static double embed_and_minimize(RWMol &mol) {
    MolOps::addHs(mol, false, true);
    int cid = embed_molecule(mol);
    if (cid < 0) {
        // Fallback: compute 2D coordinates so atoms aren't empty
        try { RDDepict::compute2DCoords(mol); } catch (...) {}
        return NAN;
    }
    std::vector<std::pair<int, double>> results;
    MMFF::MMFFOptimizeMoleculeConfs(mol, results);
    if (!results.empty() && results[0].first >= 0)
        return results[0].second;
    return NAN;
}

DruseVariantSet* druse_enumerate_tautomers(
    const char *smiles, const char *name,
    int32_t maxTautomers, double energyCutoff
) {
    auto *set = new DruseVariantSet();
    memset(set, 0, sizeof(DruseVariantSet));

    (void)energyCutoff; // ranking now uses tautomer score instead of MMFF energy

    if (!smiles || !smiles[0]) return set;

    try {
        std::unique_ptr<ROMol> mol(SmilesToMol(smiles));
        if (!mol) return set;

        // Collect unique tautomers — use Kekulized SMILES for dedup so aromatic
        // tautomers (e.g. guanine keto/enol) aren't collapsed to the same string.
        struct TautEntry {
            std::string smi;
            std::unique_ptr<RWMol> rwmol;
            double energy;
            int score;
        };
        std::vector<TautEntry> entries;
        std::set<std::string> seen;

        // Try enumeration (may throw for certain aromatic heterocycles)
        try {
            MolStandardize::TautomerEnumerator enumerator;
            enumerator.setMaxTautomers(std::max(maxTautomers, (int32_t)2));
            enumerator.setMaxTransforms(200);
            auto result = enumerator.enumerate(*mol);

            for (const auto &taut : result) {
                // Kekulize a copy to get explicit bond-order SMILES for dedup
                std::string dedupSmi;
                try {
                    RWMol kekulized(*taut);
                    MolOps::Kekulize(kekulized);
                    dedupSmi = MolToSmiles(kekulized);
                } catch (...) {
                    dedupSmi = MolToSmiles(*taut);
                }
                if (seen.count(dedupSmi)) continue;
                seen.insert(dedupSmi);

                auto rw = std::make_unique<RWMol>(*taut);
                int score = MolStandardize::TautomerScoringFunctions::scoreTautomer(*rw);

                // Use tautomer score for ranking instead of expensive 3D embedding.
                // Higher score = more stable tautomer, so negate for energy-like sorting.
                double energy = -score;

                std::string displaySmi = MolToSmiles(*taut);
                entries.push_back({displaySmi, std::move(rw), energy, score});
                if ((int)entries.size() >= maxTautomers) break;
            }
        } catch (...) {
            // Enumeration failed — fall through to fallback below
        }

        // Always include the input molecule if nothing was found
        if (entries.empty()) {
            auto rw = std::make_unique<RWMol>(*mol);
            int score = MolStandardize::TautomerScoringFunctions::scoreTautomer(*rw);
            entries.push_back({MolToSmiles(*mol), std::move(rw), (double)-score, score});
        }

        // Sort by energy (lowest = most stable tautomer first)
        std::sort(entries.begin(), entries.end(), [](const TautEntry &a, const TautEntry &b) {
            return a.energy < b.energy;
        });

        int count = (int)entries.size();
        if (count == 0) return set;

        set->count = count;
        set->variants = new DruseMoleculeResult*[count];
        set->scores = new double[count];
        set->infos = new DruseVariantInfo[count];

        for (int i = 0; i < count; i++) {
            set->variants[i] = mol_to_result(*entries[i].rwmol, name);
            set->scores[i] = entries[i].energy;
            set->infos[i].kind = 0; // tautomer
            snprintf(set->infos[i].label, sizeof(set->infos[i].label),
                     "Tautomer %d (score %d)", i + 1, entries[i].score);
        }
    } catch (...) {}

    return set;
}

DruseVariantSet* druse_enumerate_protomers(
    const char *smiles, const char *name,
    int32_t maxProtomers, double pH, double pkaThreshold
) {
    auto *set = new DruseVariantSet();
    memset(set, 0, sizeof(DruseVariantSet));

    if (!smiles || !smiles[0]) return set;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return set;

        // Ionizable group definitions: SMARTS, pKa, isAcid
        struct IonizableGroup {
            const char *name;
            const char *smarts;
            double pKa;
            bool isAcid; // true=acid (deprotonates at high pH), false=base (protonates at low pH)
        };
        static const IonizableGroup groups[] = {
            {"Carboxylic acid",  "[CX3](=O)[OX2H1]",           4.0,  true},
            {"Phosphoric acid",  "[OX2H1]P(=O)",               2.1,  true},
            {"Sulfonamide NH",   "[NX3H1]S(=O)(=O)",          10.0,  true},
            {"Phenol",           "[OX2H1]c",                  10.0,  true},
            {"Thiol",            "[SX2H1]",                    8.3,  true},
            {"Primary amine",    "[NX3H2;!$(NC=O);!$(NS=O)]", 10.5,  false},
            {"Secondary amine",  "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc)]", 11.0, false},
            {"Imidazole",        "[nH1]1cc[nH0]c1",            6.0,  false},
            {"Guanidine",        "[NX3H2]C(=[NX2H0])[NX3H2]", 12.5,  false},
        };

        // Find ionizable sites: each site = (group index, matched atom index, is ambiguous)
        struct IonizableSite {
            int groupIdx;
            int atomIdx;    // index of the ionizable atom (H-bearing for acids, N for bases)
            std::string label;
        };
        std::vector<IonizableSite> ambiguousSites;

        for (int g = 0; g < (int)(sizeof(groups) / sizeof(groups[0])); g++) {
            std::unique_ptr<ROMol> pattern(SmartsToMol(groups[g].smarts));
            if (!pattern) continue;

            std::vector<MatchVectType> matches;
            SubstructMatch(*mol, *pattern, matches);
            for (const auto &match : matches) {
                if (match.empty()) continue;
                // The ionizable atom is typically the first matched atom for acids,
                // or the nitrogen for bases
                int aidx = match[0].second;

                // Henderson-Hasselbalch: is this site ambiguous at the target pH?
                double deltaPKa = std::abs(pH - groups[g].pKa);
                if (deltaPKa < pkaThreshold) {
                    ambiguousSites.push_back({g, aidx, groups[g].name});
                }
            }
        }

        // Generate protomers: combinatorial expansion of ambiguous sites
        // Each site can be in state 0 (as-drawn) or state 1 (toggled)
        int nSites = std::min((int)ambiguousSites.size(), 10); // cap to avoid explosion
        int nCombinations = 1 << nSites;
        nCombinations = std::min(nCombinations, (int)maxProtomers);

        struct ProtomerEntry {
            std::string smi;
            std::unique_ptr<RWMol> rwmol;
            double energy;
            std::string label;
        };
        std::vector<ProtomerEntry> entries;
        std::set<std::string> seen;

        for (int combo = 0; combo < nCombinations; combo++) {
            auto rw = std::make_unique<RWMol>(*mol);

            std::string comboLabel;
            for (int s = 0; s < nSites; s++) {
                bool toggle = (combo >> s) & 1;
                if (!toggle) continue; // keep as-drawn

                const auto &site = ambiguousSites[s];
                const auto &grp = groups[site.groupIdx];
                Atom *atom = rw->getAtomWithIdx(site.atomIdx);

                if (grp.isAcid) {
                    // Deprotonate: remove one H, set formal charge -1
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                } else {
                    // Protonate: add one H, set formal charge +1
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                }

                if (!comboLabel.empty()) comboLabel += ", ";
                comboLabel += std::string(grp.isAcid ? "deprot " : "prot ") + grp.name;
            }

            // Sanitize; skip if invalid
            try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }

            std::string protSmi = MolToSmiles(*rw);
            if (seen.count(protSmi)) continue;
            seen.insert(protSmi);

            if (comboLabel.empty()) comboLabel = "Parent (as-drawn)";

            double energy = embed_and_minimize(*rw);
            entries.push_back({protSmi, std::move(rw), energy, comboLabel});
        }

        // Sort by energy
        std::sort(entries.begin(), entries.end(), [](const ProtomerEntry &a, const ProtomerEntry &b) {
            if (std::isnan(a.energy)) return false;
            if (std::isnan(b.energy)) return true;
            return a.energy < b.energy;
        });

        int count = std::min((int)entries.size(), (int)maxProtomers);
        if (count == 0) return set;

        set->count = count;
        set->variants = new DruseMoleculeResult*[count];
        set->scores = new double[count];
        set->infos = new DruseVariantInfo[count];

        for (int i = 0; i < count; i++) {
            set->variants[i] = mol_to_result(*entries[i].rwmol, name);
            set->scores[i] = entries[i].energy;
            set->infos[i].kind = 1; // protomer
            snprintf(set->infos[i].label, sizeof(set->infos[i].label),
                     "%s", entries[i].label.c_str());
        }
    } catch (...) {}

    return set;
}

void druse_free_variant_set(DruseVariantSet *set) {
    if (!set) return;
    if (set->variants) {
        for (int i = 0; i < set->count; i++)
            druse_free_molecule_result(set->variants[i]);
        delete[] set->variants;
    }
    delete[] set->scores;
    delete[] set->infos;
    delete set;
}

// ============================================================================
// Unified Ligand Ensemble Preparation
// ============================================================================
//
// Pipeline: SMILES → protomers(pH) → tautomers(each) → dedup → conformers(each)
//           → addH + MMFF minimize + Gasteiger charges → Boltzmann weights
//
// This produces a chemically realistic batch of molecular forms at the target pH,
// where tautomeric, protonation, and conformational diversity are all represented
// in realistic proportions based on MMFF94 energetics.

DruseEnsembleResult* druse_prepare_ligand_ensemble(
    const char *smiles, const char *name,
    double pH, double pkaThreshold,
    int32_t maxTautomers, int32_t maxProtomers,
    double energyCutoff, int32_t conformersPerForm,
    double temperature
) {
    auto *result = new DruseEnsembleResult();
    memset(result, 0, sizeof(DruseEnsembleResult));
    result->numConformersPerForm = conformersPerForm;

    if (!smiles || !smiles[0]) {
        result->success = false;
        snprintf(result->errorMessage, 512, "Empty SMILES");
        return result;
    }

    try {
        std::unique_ptr<RWMol> parentMol(SmilesToMol(smiles));
        if (!parentMol) {
            result->success = false;
            snprintf(result->errorMessage, 512, "Invalid SMILES: %s", smiles);
            return result;
        }

        // ===================================================================
        // Step 1: Enumerate protomers at target pH
        // ===================================================================
        // Each protomer is a distinct protonation state (different formal charges)

        struct ChemicalForm {
            std::string smi;               // canonical SMILES (dedup key)
            std::unique_ptr<RWMol> mol;    // sanitized molecule
            std::string label;             // human-readable description
            int kind;                      // 0=parent, 1=tautomer, 2=protomer, 3=taut+prot
        };
        std::vector<ChemicalForm> allForms;
        std::set<std::string> seenSMILES;

        // Ionizable groups (same as druse_enumerate_protomers)
        struct IonGrp { const char *name; const char *smarts; double pKa; bool isAcid; };
        static const IonGrp ionGroups[] = {
            {"Carboxylic acid",  "[CX3](=O)[OX2H1]",           4.0,  true},
            {"Phosphoric acid",  "[OX2H1]P(=O)",               2.1,  true},
            {"Sulfonamide NH",   "[NX3H1]S(=O)(=O)",          10.0,  true},
            {"Phenol",           "[OX2H1]c",                  10.0,  true},
            {"Thiol",            "[SX2H1]",                    8.3,  true},
            {"Primary amine",    "[NX3H2;!$(NC=O);!$(NS=O)]", 10.5,  false},
            {"Secondary amine",  "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc)]", 11.0, false},
            {"Imidazole",        "[nH1]1cc[nH0]c1",            6.0,  false},
            {"Guanidine",        "[NX3H2]C(=[NX2H0])[NX3H2]", 12.5,  false},
        };

        // Find ambiguous ionizable sites
        struct AmbSite { int groupIdx; int atomIdx; };
        std::vector<AmbSite> ambSites;
        for (int g = 0; g < (int)(sizeof(ionGroups)/sizeof(ionGroups[0])); g++) {
            std::unique_ptr<ROMol> pat(SmartsToMol(ionGroups[g].smarts));
            if (!pat) continue;
            std::vector<MatchVectType> matches;
            SubstructMatch(*parentMol, *pat, matches);
            for (const auto &m : matches) {
                if (m.empty()) continue;
                if (std::abs(pH - ionGroups[g].pKa) < pkaThreshold) {
                    ambSites.push_back({g, m[0].second});
                }
            }
        }

        // Generate protomer combinations
        int nSites = std::min((int)ambSites.size(), 10);
        int nCombos = std::min(1 << nSites, (int)maxProtomers);

        struct ProtomerForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
        };
        std::vector<ProtomerForm> protomers;

        for (int combo = 0; combo < nCombos; combo++) {
            auto rw = std::make_unique<RWMol>(*parentMol);
            std::string comboLabel;

            for (int s = 0; s < nSites; s++) {
                if (!((combo >> s) & 1)) continue;
                const auto &site = ambSites[s];
                const auto &grp = ionGroups[site.groupIdx];
                Atom *atom = rw->getAtomWithIdx(site.atomIdx);

                if (grp.isAcid) {
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                } else {
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                }
                if (!comboLabel.empty()) comboLabel += "+";
                comboLabel += std::string(grp.isAcid ? "deprot_" : "prot_") + grp.name;
            }

            try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
            std::string pSmi = MolToSmiles(*rw);
            if (seenSMILES.count(pSmi)) continue;
            seenSMILES.insert(pSmi);  // track for dedup across protomers

            if (comboLabel.empty()) comboLabel = "Parent";
            protomers.push_back({pSmi, std::move(rw), comboLabel});
        }

        // If no protomers generated (no ambiguous sites), use parent as-is
        if (protomers.empty()) {
            auto rw = std::make_unique<RWMol>(*parentMol);
            protomers.push_back({MolToSmiles(*rw), std::move(rw), "Parent"});
        }

        // ===================================================================
        // Step 2: For each protomer, enumerate tautomers
        // ===================================================================
        // Cap total forms to avoid combinatorial explosion
        int maxTotalForms = maxProtomers * maxTautomers;
        if (maxTotalForms > 200) maxTotalForms = 200;

        for (auto &prot : protomers) {
            if ((int)allForms.size() >= maxTotalForms) break;

            try {
                MolStandardize::TautomerEnumerator enumerator;
                enumerator.setMaxTautomers(std::max(maxTautomers, 2));
                enumerator.setMaxTransforms(200); // capped to avoid hanging on complex molecules
                auto tautResult = enumerator.enumerate(*prot.mol);

                int tautCount = 0;
                for (const auto &taut : tautResult) {
                    if (tautCount >= maxTautomers || (int)allForms.size() >= maxTotalForms) break;
                    auto rw = std::make_unique<RWMol>(*taut);
                    try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
                    std::string tSmi = MolToSmiles(*rw);

                    if (seenSMILES.count(tSmi)) continue;
                    seenSMILES.insert(tSmi);

                    // Determine kind: is this from a non-parent protomer AND a tautomer?
                    bool isFromProtomer = (prot.label != "Parent");
                    bool isFromTautomer = (tSmi != prot.smi);
                    int kind = 0; // parent
                    if (isFromProtomer && isFromTautomer) kind = 3;      // protomer+tautomer
                    else if (isFromProtomer) kind = 2;                    // protomer only
                    else if (isFromTautomer) kind = 1;                    // tautomer only

                    std::string label = prot.label;
                    if (isFromTautomer) {
                        if (label != "Parent") label += "_";
                        else label = "";
                        label += "Taut" + std::to_string(tautCount + 1);
                    }

                    allForms.push_back({tSmi, std::move(rw), label, kind});
                    tautCount++;
                }

                // If enumeration produced nothing new, add the protomer itself
                if (tautCount == 0 && !seenSMILES.count(prot.smi)) {
                    seenSMILES.insert(prot.smi);
                    auto rw = std::make_unique<RWMol>(*prot.mol);
                    int kind = (prot.label != "Parent") ? 2 : 0;
                    allForms.push_back({prot.smi, std::move(rw), prot.label, kind});
                }
            } catch (...) {
                // Tautomer enumeration failed; add protomer directly
                if (!seenSMILES.count(prot.smi)) {
                    seenSMILES.insert(prot.smi);
                    auto rw = std::make_unique<RWMol>(*prot.mol);
                    int kind = (prot.label != "Parent") ? 2 : 0;
                    allForms.push_back({prot.smi, std::move(rw), prot.label, kind});
                }
            }
        }

        if (allForms.empty()) {
            result->success = false;
            snprintf(result->errorMessage, 512, "No valid chemical forms generated");
            return result;
        }

        result->numForms = (int)allForms.size();

        // ===================================================================
        // Step 3: For each form, generate conformers + full preparation
        // ===================================================================
        // Each form: addH → embed N conformers → MMFF minimize → Gasteiger charges

        struct PreparedConformer {
            std::unique_ptr<RWMol> mol;
            double energy;
            int formIdx;
            int confIdx;
            std::string label;
            std::string smi;
            int kind;
        };
        std::vector<PreparedConformer> allConformers;

        for (int fi = 0; fi < (int)allForms.size(); fi++) {
            auto &form = allForms[fi];

            // Clone and add hydrogens
            auto mol = std::make_unique<RWMol>(*form.mol);
            MolOps::addHs(*mol, false, true); // addCoords=true

            // Generate multiple conformers
            auto cids = embed_multiple(*mol, conformersPerForm);
            if (cids.empty()) {
                // Fallback: single conformer via basic embedding
                int cid = embed_molecule(*mol);
                if (cid >= 0) cids.push_back(cid);
                else {
                    // Last resort: 2D coords
                    try { RDDepict::compute2DCoords(*mol); } catch (...) {}
                    cids.push_back(0);
                }
            }

            // MMFF94 minimize all conformers
            std::vector<std::pair<int, double>> mmffResults;
            try { MMFF::MMFFOptimizeMoleculeConfs(*mol, mmffResults); } catch (...) {}

            // Gasteiger charges
            try { computeGasteigerCharges(*mol); } catch (...) {}

            // Extract each conformer as a separate entry
            std::vector<std::pair<int, double>> indexed;
            for (int ci = 0; ci < (int)cids.size(); ci++) {
                double e = (ci < (int)mmffResults.size() && mmffResults[ci].first >= 0)
                           ? mmffResults[ci].second : 0.0;
                indexed.push_back({ci, e});
            }

            // Sort by energy within this form
            std::sort(indexed.begin(), indexed.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });

            for (int ri = 0; ri < (int)indexed.size(); ri++) {
                int confId = cids[indexed[ri].first];
                double energy = indexed[ri].second;

                // Create a single-conformer copy for this entry
                auto confMol = std::make_unique<RWMol>(*mol);

                // Keep only the target conformer
                if (confMol->getNumConformers() > 1) {
                    std::vector<unsigned int> toRemove;
                    for (auto it = confMol->beginConformers(); it != confMol->endConformers(); ++it) {
                        if ((int)(*it)->getId() != confId) {
                            toRemove.push_back((*it)->getId());
                        }
                    }
                    for (auto cid : toRemove) {
                        confMol->removeConformer(cid);
                    }
                }

                std::string confLabel = form.label;
                if (indexed.size() > 1) {
                    confLabel += "_Conf" + std::to_string(ri + 1);
                }

                allConformers.push_back({
                    std::move(confMol), energy, fi, ri,
                    confLabel, form.smi, form.kind
                });
            }
        }

        // ===================================================================
        // Step 4: Energy cutoff filter
        // ===================================================================
        if (energyCutoff > 0 && !allConformers.empty()) {
            double bestE = 1e30;
            for (const auto &c : allConformers) {
                if (!std::isnan(c.energy) && c.energy < bestE) bestE = c.energy;
            }
            allConformers.erase(
                std::remove_if(allConformers.begin(), allConformers.end(),
                    [&](const PreparedConformer &c) {
                        return !std::isnan(c.energy) && c.energy > bestE + energyCutoff;
                    }),
                allConformers.end()
            );
        }

        // ===================================================================
        // Step 5: Compute Boltzmann weights
        // ===================================================================
        // w_i = exp(-E_i / (kB * T)) / Z, where Z = sum of all weights
        // kB in kcal/(mol·K) = 0.001987204
        double kBT = 0.001987204 * temperature;
        if (kBT < 1e-10) kBT = 0.593; // room temperature fallback

        // Find minimum energy for numerical stability
        double minE = 1e30;
        for (const auto &c : allConformers) {
            if (!std::isnan(c.energy) && c.energy < minE) minE = c.energy;
        }

        std::vector<double> boltzWeights(allConformers.size(), 0.0);
        double partitionZ = 0.0;
        for (size_t i = 0; i < allConformers.size(); i++) {
            double dE = std::isnan(allConformers[i].energy) ? 50.0 : (allConformers[i].energy - minE);
            boltzWeights[i] = std::exp(-dE / kBT);
            partitionZ += boltzWeights[i];
        }
        if (partitionZ > 0) {
            for (auto &w : boltzWeights) w /= partitionZ;
        }

        // ===================================================================
        // Step 6: Build result
        // ===================================================================
        int count = (int)allConformers.size();
        result->count = count;
        result->members = new DruseEnsembleMember[count];
        result->success = true;

        for (int i = 0; i < count; i++) {
            auto &c = allConformers[i];
            auto &m = result->members[i];

            m.molecule = mol_to_result(*c.mol, name);
            m.mmffEnergy = c.energy;
            m.boltzmannWeight = boltzWeights[i];
            m.kind = c.kind;
            m.conformerIndex = c.confIdx;
            m.formIndex = c.formIdx;
            snprintf(m.label, sizeof(m.label), "%s", c.label.c_str());
            snprintf(m.smiles, sizeof(m.smiles), "%s", c.smi.c_str());
        }

    } catch (const std::exception &e) {
        result->success = false;
        snprintf(result->errorMessage, 512, "Ensemble preparation failed: %s", e.what());
    } catch (...) {
        result->success = false;
        snprintf(result->errorMessage, 512, "Ensemble preparation failed (unknown error)");
    }

    return result;
}

void druse_free_ensemble_result(DruseEnsembleResult *result) {
    if (!result) return;
    if (result->members) {
        for (int i = 0; i < result->count; i++) {
            druse_free_molecule_result(result->members[i].molecule);
        }
        delete[] result->members;
    }
    delete result;
}

// ============================================================================
// Protein Preparation (PDB-based)
// ============================================================================

#include <GraphMol/FileParsers/FileParsers.h>

DruseMoleculeResult* druse_add_hydrogens_pdb(const char *pdbContent) {
    if (!pdbContent || !pdbContent[0]) return make_error("Empty PDB content");
    try {
        std::unique_ptr<RWMol> mol(PDBBlockToMol(std::string(pdbContent), true, true, false));
        if (!mol) return make_error("Failed to parse PDB block");

        MolOps::addHs(*mol, false, true); // addCoords=true
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

        // Add atoms
        for (int32_t i = 0; i < atomCount; i++) {
            Atom atom(atoms[i].atomicNum);
            atom.setFormalCharge(atoms[i].formalCharge);
            mol->addAtom(&atom, true, true);
        }

        // Add bonds
        for (int32_t i = 0; i < bondCount && bonds; i++) {
            int a1 = bonds[i].atom1;
            int a2 = bonds[i].atom2;
            if (a1 < 0 || a1 >= atomCount || a2 < 0 || a2 >= atomCount) continue;
            Bond::BondType bt;
            switch (bonds[i].order) {
                case 2:  bt = Bond::DOUBLE;   break;
                case 3:  bt = Bond::TRIPLE;   break;
                case 4:  bt = Bond::AROMATIC;  break;
                default: bt = Bond::SINGLE;    break;
            }
            mol->addBond(a1, a2, bt);
        }

        // Add 3D conformer
        auto *conf = new Conformer(atomCount);
        for (int32_t i = 0; i < atomCount; i++) {
            conf->setAtomPos(i, RDGeom::Point3D(atoms[i].x, atoms[i].y, atoms[i].z));
        }
        mol->addConformer(conf, true);

        // Sanitize (compute aromaticity, ring info, etc.)
        try {
            unsigned int failedOp = 0;
            MolOps::sanitizeMol(*mol, failedOp,
                MolOps::SANITIZE_ALL ^ MolOps::SANITIZE_PROPERTIES);
        } catch (...) {
            // If full sanitize fails, try a lighter cleanup
            try { MolOps::findSSSR(*mol); } catch (...) {}
        }

        // Infer chiral centers and E/Z bonds from 3D geometry
        // Skip for 2D coordinates (all z ≈ 0) — can crash with degenerate geometry
        bool is3D = false;
        for (int32_t i = 0; i < atomCount; i++) {
            if (std::abs(atoms[i].z) > 0.01f) { is3D = true; break; }
        }
        if (is3D) {
            try {
                MolOps::assignStereochemistryFrom3D(*mol);
            } catch (...) {}
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
            outTypes[i] = vina_xs_type_for_atom(typedMol, *typedMol.getAtomWithIdx(i), donorFlags, acceptorFlags);
        }
        return atomCount;
    } catch (...) {
        return -1;
    }
}

// ============================================================================
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

// ============================================================================
// Torsion Tree
// ============================================================================

#include <GraphMol/RingInfo.h>
#include <queue>
#include <unordered_set>

static std::unordered_set<int> bfs_from_excluding(
    const ROMol &mol, int startAtom, int excludeAtom
) {
    std::unordered_set<int> visited;
    std::queue<int> q;
    q.push(startAtom);
    visited.insert(startAtom);
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        for (const auto *bond : mol.atomBonds(mol.getAtomWithIdx(cur))) {
            int nbr = bond->getOtherAtomIdx(cur);
            if (nbr == excludeAtom) continue;
            if (visited.count(nbr)) continue;
            visited.insert(nbr);
            q.push(nbr);
        }
    }
    return visited;
}

static DruseTorsionTree* build_torsion_tree_for_mol(ROMol &mol) {
    struct RotBond {
        int atom1, atom2;
        std::vector<int> movingAtoms;
    };
    std::vector<RotBond> rotBonds;

    for (unsigned int bi = 0; bi < mol.getNumBonds(); bi++) {
        const auto *bond = mol.getBondWithIdx(bi);
        if (bond->getBondType() != Bond::SINGLE) continue;
        if (mol.getRingInfo()->numBondRings(bi) != 0) continue;

        int a1 = bond->getBeginAtomIdx();
        int a2 = bond->getEndAtomIdx();
        const auto *atom1 = mol.getAtomWithIdx(a1);
        const auto *atom2 = mol.getAtomWithIdx(a2);

        if (atom1->getAtomicNum() == 1 || atom2->getAtomicNum() == 1) continue;
        if (atom1->getDegree() <= 1 || atom2->getDegree() <= 1) continue;

        auto fwd = bfs_from_excluding(mol, a2, a1);
        auto bwd = bfs_from_excluding(mol, a1, a2);

        RotBond rb;
        if (fwd.size() <= bwd.size()) {
            rb.atom1 = a1;
            rb.atom2 = a2;
            rb.movingAtoms.assign(fwd.begin(), fwd.end());
        } else {
            rb.atom1 = a2;
            rb.atom2 = a1;
            rb.movingAtoms.assign(bwd.begin(), bwd.end());
        }
        std::sort(rb.movingAtoms.begin(), rb.movingAtoms.end());
        rotBonds.push_back(std::move(rb));
    }

    // The docking engine uploads heavy atoms only, so remap the torsion tree
    // into heavy-atom space while preserving the source molecule's atom order.
    int numAllAtoms = (int)mol.getNumAtoms();
    std::vector<int> fullToHeavy(numAllAtoms, -1);
    int heavyIdx = 0;
    for (int i = 0; i < numAllAtoms; i++) {
        if (mol.getAtomWithIdx(i)->getAtomicNum() != 1) {
            fullToHeavy[i] = heavyIdx++;
        }
    }

    for (auto &rb : rotBonds) {
        rb.atom1 = fullToHeavy[rb.atom1];
        rb.atom2 = fullToHeavy[rb.atom2];
        std::vector<int> heavyMoving;
        heavyMoving.reserve(rb.movingAtoms.size());
        for (int idx : rb.movingAtoms) {
            int hi = fullToHeavy[idx];
            if (hi >= 0) heavyMoving.push_back(hi);
        }
        rb.movingAtoms = std::move(heavyMoving);
    }

    rotBonds.erase(
        std::remove_if(rotBonds.begin(), rotBonds.end(),
            [](const RotBond &rb) { return rb.atom1 < 0 || rb.atom2 < 0 || rb.movingAtoms.empty(); }),
        rotBonds.end());

    int numHeavy = heavyIdx;
    std::vector<int> atomOrder(numHeavy, -1);
    if (numHeavy > 0) {
        std::vector<std::vector<int>> heavyAdj(numHeavy);
        for (unsigned int bi = 0; bi < mol.getNumBonds(); bi++) {
            const auto *bond = mol.getBondWithIdx(bi);
            int ha = fullToHeavy[bond->getBeginAtomIdx()];
            int hb = fullToHeavy[bond->getEndAtomIdx()];
            if (ha >= 0 && hb >= 0) {
                heavyAdj[ha].push_back(hb);
                heavyAdj[hb].push_back(ha);
            }
        }
        std::queue<int> q;
        q.push(0);
        atomOrder[0] = 0;
        int ord = 1;
        while (!q.empty()) {
            int cur = q.front(); q.pop();
            for (int nbr : heavyAdj[cur]) {
                if (atomOrder[nbr] < 0) {
                    atomOrder[nbr] = ord++;
                    q.push(nbr);
                }
            }
        }
    }

    std::sort(rotBonds.begin(), rotBonds.end(),
        [&atomOrder](const RotBond &a, const RotBond &b) {
            return atomOrder[a.atom1] < atomOrder[b.atom1];
        });

    auto *tree = new DruseTorsionTree();
    memset(tree, 0, sizeof(DruseTorsionTree));

    tree->edgeCount = (int32_t)rotBonds.size();
    if (rotBonds.empty()) {
        return tree;
    }

    tree->edges = new DruseTorsionEdge[tree->edgeCount];

    int totalMoving = 0;
    for (const auto &rb : rotBonds) totalMoving += (int)rb.movingAtoms.size();
    tree->totalMovingAtoms = totalMoving;
    tree->movingAtomIndices = new int32_t[totalMoving];

    int offset = 0;
    for (int i = 0; i < tree->edgeCount; i++) {
        tree->edges[i].atom1 = rotBonds[i].atom1;
        tree->edges[i].atom2 = rotBonds[i].atom2;
        tree->edges[i].movingStart = offset;
        tree->edges[i].movingCount = (int32_t)rotBonds[i].movingAtoms.size();
        for (int idx : rotBonds[i].movingAtoms) {
            tree->movingAtomIndices[offset++] = idx;
        }
    }

    return tree;
}

DruseTorsionTree* druse_build_torsion_tree(const char *smiles) {
    if (!smiles || !smiles[0]) return nullptr;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return nullptr;

        MolOps::addHs(*mol);
        MolOps::findSSSR(*mol);
        return build_torsion_tree_for_mol(*mol);
    } catch (...) {
        return nullptr;
    }
}

DruseTorsionTree* druse_build_torsion_tree_molblock(const char *molBlock) {
    if (!molBlock || !molBlock[0]) return nullptr;

    try {
        std::unique_ptr<RWMol> mol(MolBlockToMol(std::string(molBlock), true, false, false));
        if (!mol) return nullptr;

        MolOps::addHs(*mol);
        MolOps::findSSSR(*mol);
        return build_torsion_tree_for_mol(*mol);
    } catch (...) {
        return nullptr;
    }
}

void druse_free_torsion_tree(DruseTorsionTree *tree) {
    if (!tree) return;
    delete[] tree->edges;
    delete[] tree->movingAtomIndices;
    delete tree;
}

// ============================================================================
// Spatial Queries (nanoflann KD-tree)
// ============================================================================

#include <nanoflann.hpp>

namespace {

struct PointCloud3f {
    const float *pts;
    int32_t count;

    inline size_t kdtree_get_point_count() const { return (size_t)count; }

    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx * 3 + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
};

using KDTree3f = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud3f>,
    PointCloud3f, 3
>;

struct KDTreeHandle {
    PointCloud3f cloud;
    std::unique_ptr<KDTree3f> tree;
};

} // anonymous namespace

DruseKDTree druse_build_kdtree(const float *positions, int32_t numPoints) {
    if (!positions || numPoints <= 0) return nullptr;

    auto *handle = new KDTreeHandle();
    handle->cloud.pts = positions;
    handle->cloud.count = numPoints;

    handle->tree = std::make_unique<KDTree3f>(
        3, handle->cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)
    );
    handle->tree->buildIndex();

    return static_cast<DruseKDTree>(handle);
}

int32_t druse_kdtree_radius_search(
    DruseKDTree tree,
    const float *queryPoint,
    float radius,
    int32_t *outIndices,
    int32_t maxResults
) {
    if (!tree || !queryPoint || !outIndices || maxResults <= 0) return 0;

    auto *handle = static_cast<KDTreeHandle*>(tree);

    nanoflann::SearchParameters params;
    params.sorted = true;

    std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
    size_t found = handle->tree->radiusSearch(
        queryPoint, radius * radius, matches, params
    );

    int32_t count = std::min((int32_t)found, maxResults);
    for (int32_t i = 0; i < count; i++) {
        outIndices[i] = (int32_t)matches[i].first;
    }
    return count;
}

void druse_free_kdtree(DruseKDTree tree) {
    if (!tree) return;
    auto *handle = static_cast<KDTreeHandle*>(tree);
    delete handle;
}

// ============================================================================
// Linear Algebra (Eigen - Kabsch)
// ============================================================================

#include <Eigen/Dense>

float druse_kabsch_superpose(
    const float *mobile, const float *reference, int32_t n,
    float *rotation_out, float *translation_out
) {
    if (!mobile || !reference || n <= 0) return -1.0f;

    // Map input arrays to Eigen matrices (Nx3, row-major)
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> mob(mobile, n, 3);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> ref(reference, n, 3);

    // Compute centroids
    Eigen::Vector3f centroid_mob = mob.colwise().mean();
    Eigen::Vector3f centroid_ref = ref.colwise().mean();

    // Center both point sets
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> mob_c = mob.rowwise() - centroid_mob.transpose();
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> ref_c = ref.rowwise() - centroid_ref.transpose();

    // Compute cross-covariance matrix H = mobile^T * reference
    Eigen::Matrix3f H = mob_c.transpose() * ref_c;

    // SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    // Ensure right-handed coordinate system
    float d = (V * U.transpose()).determinant();
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    if (d < 0.0f) S(2, 2) = -1.0f;

    // Rotation matrix
    Eigen::Matrix3f R = V * S * U.transpose();

    // Translation
    Eigen::Vector3f t = centroid_ref - R * centroid_mob;

    // Write outputs (row-major 3x3)
    if (rotation_out) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                rotation_out[i * 3 + j] = R(i, j);
    }
    if (translation_out) {
        translation_out[0] = t(0);
        translation_out[1] = t(1);
        translation_out[2] = t(2);
    }

    // Compute RMSD of transformed mobile vs reference
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> transformed =
        (mob * R.transpose()).rowwise() + t.transpose();
    float sum_sq = (transformed - ref).squaredNorm();
    return std::sqrt(sum_sq / (float)n);
}

float druse_compute_rmsd(const float *a, const float *b, int32_t n) {
    if (!a || !b || n <= 0) return -1.0f;

    float sum_sq = 0.0f;
    for (int32_t i = 0; i < n * 3; i++) {
        float d = a[i] - b[i];
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / (float)n);
}

// ============================================================================
// Energy Minimization (compatibility wrapper, currently RDKit MMFF94)
// ============================================================================

DruseMoleculeResult* druse_minimize_lbfgs(const char *smiles, const char *name, int32_t maxIters) {
    if (!smiles || !smiles[0]) return make_error("Empty SMILES");
    if (maxIters <= 0) maxIters = 2000;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return make_error("Failed to parse SMILES");

        MolOps::addHs(*mol);
        if (embed_molecule(*mol) < 0)
            return make_error("3D embedding failed");

        // MMFF94 with extended iterations for thorough minimization
        auto res = MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 1e-6);
        if (res.first < 0) {
            // MMFF failed, try MMFF94s fallback
            MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94s", 1e-6);
        }

        try { computeGasteigerCharges(*mol); } catch (...) {}

        return mol_to_result(*mol, name);
    } catch (const std::exception &e) {
        return make_error(e.what());
    } catch (...) {
        return make_error("Unknown error in L-BFGS minimization");
    }
}

// ============================================================================
// MMFF94 Strain Energy
// ============================================================================

double druse_mmff_strain_energy(const char *smiles, const float *heavyPositions, int32_t numHeavy) {
    if (!smiles || !heavyPositions || numHeavy <= 0) return NAN;
    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return NAN;

        MolOps::addHs(*mol);
        if (embed_molecule(*mol) < 0) return NAN;

        // Map heavy atom positions onto the molecule's conformer
        auto &conf = mol->getConformer();
        int heavyIdx = 0;
        for (unsigned i = 0; i < mol->getNumAtoms() && heavyIdx < numHeavy; i++) {
            if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) {
                conf.setAtomPos(i, RDGeom::Point3D(
                    heavyPositions[heavyIdx * 3],
                    heavyPositions[heavyIdx * 3 + 1],
                    heavyPositions[heavyIdx * 3 + 2]
                ));
                heavyIdx++;
            }
        }

        // Set up MMFF and fix heavy atoms, optimize H positions only
        MMFF::MMFFMolProperties mmffProps(*mol);
        if (!mmffProps.isValid()) return NAN;

        auto *ff = MMFF::constructForceField(*mol, &mmffProps);
        if (!ff) return NAN;

        for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
            if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) {
                ff->fixedPoints().push_back(i);
            }
        }
        ff->minimize(200);  // Quick H optimization

        double energy = ff->calcEnergy();
        delete ff;
        return energy;
    } catch (...) {
        return NAN;
    }
}

double druse_mmff_reference_energy(const char *smiles) {
    if (!smiles || !smiles[0]) return NAN;
    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return NAN;

        MolOps::addHs(*mol);
        if (embed_molecule(*mol) < 0) return NAN;
        mmff_minimize_single(*mol);

        MMFF::MMFFMolProperties mmffProps(*mol);
        if (!mmffProps.isValid()) return NAN;

        auto *ff = MMFF::constructForceField(*mol, &mmffProps);
        if (!ff) return NAN;

        double energy = ff->calcEnergy();
        delete ff;
        return energy;
    } catch (...) {
        return NAN;
    }
}

// ============================================================================
// mmCIF Parser
// ============================================================================

#include <sstream>
#include <map>
#include <cmath>

DruseMoleculeResult* druse_parse_mmcif(const char *content) {
    return druse_parse_structure(content);
}

// ============================================================================
// Electrostatic Potential
// ============================================================================

void druse_compute_esp(
    const float *atomPositions, const float *charges, int32_t nAtoms,
    const float *surfacePoints, int32_t nSurface,
    float *outESP
) {
    if (!atomPositions || !charges || !surfacePoints || !outESP) return;
    if (nAtoms <= 0 || nSurface <= 0) return;

    // Coulomb constant in kcal/(mol*e^2*A): 332.06
    const float KE = 332.06f;

    for (int32_t s = 0; s < nSurface; s++) {
        float sx = surfacePoints[s * 3 + 0];
        float sy = surfacePoints[s * 3 + 1];
        float sz = surfacePoints[s * 3 + 2];
        float esp = 0.0f;

        for (int32_t a = 0; a < nAtoms; a++) {
            float dx = sx - atomPositions[a * 3 + 0];
            float dy = sy - atomPositions[a * 3 + 1];
            float dz = sz - atomPositions[a * 3 + 2];
            float r2 = dx*dx + dy*dy + dz*dz;
            float r = std::sqrt(r2);
            if (r < 0.1f) r = 0.1f; // clamp to avoid singularity

            // Distance-dependent dielectric: eps = 4*r
            esp += KE * charges[a] / (4.0f * r * r);
        }

        outESP[s] = esp;
    }
}

// ============================================================================
// Parallel Batch Processing (TBB)
// ============================================================================

#include <tbb/parallel_for.h>

DruseMoleculeResult** druse_batch_process_parallel(
    const char **smiles_array,
    const char **name_array,
    int32_t count,
    bool addHydrogens,
    bool minimize,
    bool computeCharges,
    const volatile int32_t *cancel_flag
) {
    if (!smiles_array || count <= 0) return nullptr;

    auto **results = new DruseMoleculeResult*[count];
    // Zero-initialize so cancelled slots are nullptr (caller can detect skipped entries)
    for (int i = 0; i < count; ++i) results[i] = nullptr;

    tbb::parallel_for(0, (int)count, [&](int i) {
        // Early exit: if cancellation requested, skip remaining molecules
        if (cancel_flag && __atomic_load_n(cancel_flag, __ATOMIC_ACQUIRE)) return;

        const char *smi = smiles_array[i];
        const char *nm = (name_array && name_array[i]) ? name_array[i] : "";
        results[i] = druse_prepare_ligand(smi, nm, 1, addHydrogens, minimize, computeCharges);
    });

    return results;
}

void druse_atomic_cancel_store(volatile int32_t *flag, int32_t value) {
    __atomic_store_n(flag, value, __ATOMIC_RELEASE);
}

int32_t druse_atomic_cancel_load(const volatile int32_t *flag) {
    return __atomic_load_n(flag, __ATOMIC_ACQUIRE);
}

// ============================================================================
// Morgan Fingerprint
// ============================================================================

#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/ExplicitBitVect.h>

DruseFingerprint* druse_morgan_fingerprint(const char *smiles, int32_t radius, int32_t nBits) {
    if (!smiles || !smiles[0]) return nullptr;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return nullptr;

        std::unique_ptr<ExplicitBitVect> fp(
            MorganFingerprints::getFingerprintAsBitVect(*mol, radius, nBits)
        );

        auto *result = new DruseFingerprint();
        result->numBits = nBits;
        result->bits = new float[nBits];

        for (int32_t i = 0; i < nBits; i++) {
            result->bits[i] = fp->getBit(i) ? 1.0f : 0.0f;
        }

        return result;
    } catch (...) {
        return nullptr;
    }
}

void druse_free_fingerprint(DruseFingerprint *fp) {
    if (!fp) return;
    delete[] fp->bits;
    delete fp;
}
