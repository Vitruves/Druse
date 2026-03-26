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

// Shared ionizable group table (used by protomer enumeration, site detection, ensemble)
struct IonizableGroupDef {
    const char *name;
    const char *smarts;
    double pKa;
    bool isAcid;
};

static const IonizableGroupDef kIonizableGroups[] = {
    // =====================================================================
    // ACIDS — ordered most specific first (first atom-match wins)
    // Sources: Williams pKa tables, Evans/Ripin DMSO→H2O, Jencks/Westheimer,
    //          FLogD acidbase.csv (2445 entries), cchem acidbase.c
    // =====================================================================

    // ---- Sulfonic acids (pKa < 0, always deprotonated at physiological pH) ----
    {"Trifluoromethane-sulfonic", "[OX2H1]S(=O)(=O)C(F)(F)F", -14.0, true},
    {"Sulfonic acid",    "[OX2H1]S(=O)(=O)",          -2.6,  true},

    // ---- Phosphoric / Phosphonic acids ----
    {"Phosphoric acid",  "[OX2H1]P(=O)([OX2])[OX2]",  2.1,  true},  // H3PO4 first pKa
    {"Phosphonate",      "[OX2H1]P(=O)([CX4])",        2.4,  true},  // alkylphosphonates
    {"Phosphoric ester", "[OX2H1]P(=O)",               1.5,  true},

    // ---- Carboxylic acids — specific subtypes first ----
    // Polyhaloacetic acids
    {"Trifluoroacetic",  "[CX3](=O)([OX2H1])C(F)(F)F", 0.5,  true},
    {"Trichloroacetic",  "[CX3](=O)([OX2H1])C(Cl)(Cl)Cl", 0.65, true},
    {"Difluoroacetic",   "[CX3](=O)([OX2H1])C([F,Cl])F", 1.35, true},
    {"Dichloroacetic",   "[CX3](=O)([OX2H1])C(Cl)Cl",  1.29, true},
    {"Chloroacetic",     "[CX3](=O)([OX2H1])CCl",      2.86, true},
    {"Fluoroacetic",     "[CX3](=O)([OX2H1])CF",       2.66, true},
    // Alpha-keto acids (pyruvic, glyoxylic)
    {"Alpha-keto acid",  "[CX3](=O)([OX2H1])C(=O)",    2.5,  true},
    // Alpha-cyano acids
    {"Alpha-cyano acid", "[CX3](=O)([OX2H1])CC#N",     2.5,  true},
    // Oxalic acid
    {"Oxalic acid",      "[CX3](=O)([OX2H1])C(=O)[OX2H1]", 1.25, true},
    // Maleic/fumaric (dicarboxylic, unsaturated)
    {"Maleic acid",      "[CX3](=O)([OX2H1])/C=C\\C(=O)[OX2H1]", 1.92, true},
    // Aromatic carboxylic — ortho-substituted
    {"o-Nitrobenzoic",   "[CX3](=O)([OX2H1])c1ccccc1[NX3+](=O)[O-]", 2.17, true},
    // Aromatic carboxylic — with strong EWG
    {"ArCOOH p-NO2",     "[CX3](=O)([OX2H1])c1ccc([NX3+](=O)[O-])cc1", 3.44, true},
    {"ArCOOH m-NO2",     "[CX3](=O)([OX2H1])c1cccc([NX3+](=O)[O-])c1", 3.45, true},
    {"ArCOOH p-Cl",      "[CX3](=O)([OX2H1])c1ccc(Cl)cc1", 3.99, true},
    {"ArCOOH p-F",       "[CX3](=O)([OX2H1])c1ccc(F)cc1",  4.14, true},
    {"ArCOOH p-Br",      "[CX3](=O)([OX2H1])c1ccc(Br)cc1", 4.00, true},
    {"ArCOOH p-CN",      "[CX3](=O)([OX2H1])c1ccc(C#N)cc1", 3.55, true},
    {"ArCOOH p-CF3",     "[CX3](=O)([OX2H1])c1ccc(C(F)(F)F)cc1", 3.79, true},
    // Aromatic carboxylic — with EDG
    {"ArCOOH p-OMe",     "[CX3](=O)([OX2H1])c1ccc(OC)cc1", 4.47, true},
    {"ArCOOH p-OH",      "[CX3](=O)([OX2H1])c1ccc(O)cc1",  4.58, true},
    {"ArCOOH p-NH2",     "[CX3](=O)([OX2H1])c1ccc(N)cc1",  4.92, true},
    {"ArCOOH p-NMe2",    "[CX3](=O)([OX2H1])c1ccc(N(C)C)cc1", 5.03, true},
    {"ArCOOH p-Me",      "[CX3](=O)([OX2H1])c1ccc(C)cc1",  4.34, true},
    // Benzoic acid (generic aromatic)
    {"Benzoic acid",     "[CX3](=O)([OX2H1])c",       4.20,  true},
    // Aliphatic dicarboxylic second pKa (succinic, glutaric)
    {"Succinic acid",    "[CX3](=O)([OX2H1])CCC(=O)[OX2H1]", 4.19, true},
    {"Glutaric acid",    "[CX3](=O)([OX2H1])CCCC(=O)[OX2H1]", 4.34, true},
    // Simple aliphatic
    {"Acetic acid",      "[CX3](=O)([OX2H1])[CX4H3]", 4.76, true},
    {"Propionic acid",   "[CX3](=O)([OX2H1])[CX4H2][CX4]", 4.88, true},
    {"Formic acid",      "[CX3H1](=O)[OX2H1]",        3.77, true},
    // Generic carboxylic (fallback)
    {"Carboxylic acid",  "[CX3](=O)[OX2H1]",           4.0,  true},

    // ---- Tetrazoles (bioisostere of COOH) ----
    {"Tetrazole",        "[nH1]1nnn[nH0]1",            4.9,  true},
    {"Tetrazole 2",      "[nH1]1nn[nH0]n1",            4.9,  true},

    // ---- Sulfonamides (N-H) — acidity depends on substituent ----
    {"Saccharin NH",     "[nH1]1c2ccccc2S(=O)(=O)1",   1.6,  true},  // cyclic sulfonamide
    {"ArSO2NHAr",        "[NX3H1](S(=O)(=O)c)c",       6.3,  true},  // PhSO2NHPh
    {"CF3-sulfonamide",  "[NX3H1]S(=O)(=O)C(F)(F)F",   5.8,  true},
    {"ArSO2NH2",         "[NX3H2]S(=O)(=O)c",          10.0, true},  // PhSO2NH2
    {"Sulfonamide NH",   "[NX3H1]S(=O)(=O)",           10.0, true},

    // ---- Phenols — specific subtypes first ----
    // Polynitrophenols
    {"2,4,6-Trinitro-phenol", "[OX2H1]c1c([NX3+](=O)[O-])cc([NX3+](=O)[O-])cc1[NX3+](=O)[O-]", 0.3, true},
    {"2,4-Dinitrophenol","[OX2H1]c1ccc([NX3+](=O)[O-])cc1[NX3+](=O)[O-]", 4.1, true},
    // Nitrophenols
    {"p-Nitrophenol",    "[OX2H1]c1ccc([NX3+](=O)[O-])cc1", 7.14, true},
    {"m-Nitrophenol",    "[OX2H1]c1cccc([NX3+](=O)[O-])c1", 8.35, true},
    {"o-Nitrophenol",    "[OX2H1]c1ccccc1[NX3+](=O)[O-]",   7.23, true},
    // Halophenols
    {"p-Fluorophenol",   "[OX2H1]c1ccc(F)cc1",         9.95, true},
    {"p-Chlorophenol",   "[OX2H1]c1ccc(Cl)cc1",        9.38, true},
    {"p-Bromophenol",    "[OX2H1]c1ccc(Br)cc1",        9.34, true},
    // Cyanophenol
    {"p-Cyanophenol",    "[OX2H1]c1ccc(C#N)cc1",       7.95, true},
    // Trifluoromethyl phenol
    {"p-CF3-phenol",     "[OX2H1]c1ccc(C(F)(F)F)cc1",  8.68, true},
    // Methoxyphenol (EDG → higher pKa)
    {"p-Methoxyphenol",  "[OX2H1]c1ccc(OC)cc1",       10.20, true},
    // Aminophenol
    {"p-Aminophenol",    "[OX2H1]c1ccc(N)cc1",        10.30, true},
    // Naphthol
    {"1-Naphthol",       "[OX2H1]c1cccc2ccccc12",      9.34, true},
    {"2-Naphthol",       "[OX2H1]c1ccc2ccccc2c1",      9.51, true},
    // Catechol, resorcinol, hydroquinone
    {"Catechol",         "[OX2H1]c1ccccc1[OX2H1]",     9.45, true},
    {"Resorcinol",       "[OX2H1]c1cccc([OX2H1])c1",   9.15, true},
    {"Hydroquinone",     "[OX2H1]c1ccc([OX2H1])cc1",   9.85, true},
    // Simple phenol (generic aromatic OH fallback)
    {"Phenol",           "[OX2H1]c",                   10.0,  true},

    // ---- Hydroxamic acids ----
    {"Hydroxamic acid",  "[OX2H1]NC(=O)",               8.0,  true},
    {"Hydroxamic acid 2","[NX3H1]C(=O)[OX2H1]",         8.0,  true},

    // ---- Thiols ----
    {"Thiophenol p-NO2", "[SX2H1]c1ccc([NX3+](=O)[O-])cc1", 4.7, true},
    {"Thiophenol",       "[SX2H1]c",                    6.5,  true},
    {"Cysteine thiol",   "[SX2H1]CC(N)C(=O)",           8.3,  true},
    {"Benzyl thiol",     "[SX2H1]Cc",                    9.4,  true},
    {"Aliphatic thiol",  "[SX2H1][CX4]",               10.0,  true},
    {"Thiol",            "[SX2H1]",                     8.3,  true},

    // ---- Imides (cyclic NH between two C=O) ----
    {"Barbiturate NH",   "[NH1]1C(=O)[NH1]C(=O)CC1=O",  4.0,  true},
    {"Hydantoin NH",     "[NH1]1C(=O)[NH1]C(=O)C1",     8.7,  true},
    {"Succinimide",      "[NH1]1C(=O)CCC1=O",           9.6,  true},
    {"Phthalimide",      "[NH1]1C(=O)c2ccccc2C1=O",     8.3,  true},
    {"Maleimide",        "[NH1]1C(=O)C=CC1=O",          7.0,  true},

    // ---- Oximes ----
    {"Oxime OH",         "[OX2H1]/N=C",                10.0,  true},
    {"Amidoxime NH",     "[OX2H1]N=C(N)",               9.0,  true},

    // ---- N-H heterocycle acids ----
    {"Triazole NH 123",  "[nH1]1nncc1",                 9.4,  true},  // 1,2,3-triazole
    {"Triazole NH 124",  "[nH1]1ncnc1",                10.0,  true},  // 1,2,4-triazole
    {"Benzimidazole NH", "[nH1]1cnc2ccccc12",          12.0,  true},  // benzimidazole NH
    {"Indole NH",        "[nH1]1ccc2ccccc12",          17.0,  true},  // very weak acid
    {"Pyrrole NH",       "[nH1]1cccc1",                17.5,  true},  // very weak acid

    // ---- Boronic acids ----
    {"Boronic acid",     "[OX2H1]B([OX2H1])",           8.8,  true},

    // ---- Sulfinic acids ----
    {"Sulfinic acid",    "[OX2H1]S(=O)c",               1.8,  true},
    {"Sulfinic acid ali","[OX2H1]S(=O)[CX4]",           2.0,  true},

    // ---- N-Acyl sulfonamides (very acidic NH, drug-relevant) ----
    {"N-acyl sulfonamide","[NH1](C(=O))S(=O)(=O)",      2.5,  true},
    {"Sulfonylcarboxamide","[NH1](S(=O)(=O))C(=O)c",    3.0,  true},

    // ---- Carbon acids (C-H acidity, drug-relevant enolizable systems) ----
    // β-Diketones
    {"Acetylacetone CH", "[CH2](C(=O)C)C(=O)C",         8.95, true},  // 2,4-pentanedione
    {"ArCO-CH2-COAr",   "[CH2](C(=O)c)C(=O)c",         8.5,  true},
    {"ArCO-CH2-COCH3",  "[CH2](C(=O)c)C(=O)C",         9.0,  true},
    // Cyanoacetates / malononitriles
    {"Malononitrile CH", "[CH2](C#N)C#N",              11.2,  true},  // (NC)2CH2
    {"Cyanoacetate CH",  "[CH2](C#N)C(=O)O",           10.7,  true},
    {"Cyanoacetamide CH","[CH2](C#N)C(=O)N",           11.5,  true},
    // Malonates
    {"Malonate diester", "[CH2](C(=O)OC)C(=O)OC",     12.9,  true},  // diethyl malonate
    {"Malonic acid CH",  "[CH2](C(=O)[OH])C(=O)[OH]",  2.83, true},
    // Nitroalkanes
    {"Nitromethane",     "[CH3][NX3+](=O)[O-]",        10.2,  true},
    {"Nitroethane",      "[CH2]([CX4])[NX3+](=O)[O-]", 8.6,  true},
    {"Dinitromethane",   "[CH1]([NX3+](=O)[O-])[NX3+](=O)[O-]", 3.6, true},
    {"Phenylnitromethane","[CH2](c)[NX3+](=O)[O-]",    7.1,  true},
    // Sulfonylmethane
    {"Bis-sulfonylmethane","[CH2](S(=O)(=O))S(=O)(=O)", 12.3, true},

    // ---- Enols (drug-relevant) ----
    {"Ascorbic acid",    "OC1OC(=O)C(O)=C1O",          4.1,  true},
    {"Squaric acid",     "[OH]C1=C([OH])C(=O)C1=O",    1.5,  true},
    {"Tropolone",        "[OH]c1cccccc1=O",             6.95, true},

    // ---- Hydroxypyridines / Pyridinones (tautomeric, act as acids) ----
    {"2-Hydroxypyridine","[OH]c1ccccn1",                0.75, true},   // → 2-pyridinone, special
    {"3-Hydroxypyridine","[OH]c1cccnc1",               8.72, true},
    {"4-Hydroxypyridine","[OH]c1ccncc1",               11.09, true},  // → 4-pyridinone
    {"8-Hydroxyquinoline","[OH]c1ccc2ncccc2c1",         9.81, true},

    // ---- Nucleobase acids (NH) ----
    {"Uracil NH",        "[NH1]1C(=O)[NH1]C(=O)C=C1",  9.5,  true},  // uracil
    {"Thymine NH",       "[NH1]1C(=O)[NH1]C(=O)C(C)=C1", 9.9, true},
    {"Xanthine NH",      "[NH1]1C(=O)[NH1]C(=O)c2[nH]cnc12", 7.44, true},

    // ---- Thioamide NH ----
    {"Thioamide NH",     "[NX3H1]C(=S)",               13.0,  true},  // weak acid, rarely relevant
    {"Thiourea NH",      "[NX3H1]C(=S)N",              21.0, true},   // very weak acid, rarely relevant in aq

    // ---- Additional phenols (ortho/meta substituents from Williams tables) ----
    {"o-Chlorophenol",   "[OX2H1]c1ccccc1Cl",           8.48, true},
    {"o-Bromophenol",    "[OX2H1]c1ccccc1Br",           8.42, true},
    {"m-Chlorophenol",   "[OX2H1]c1cccc(Cl)c1",         9.02, true},
    {"m-Bromophenol",    "[OX2H1]c1cccc(Br)c1",         9.03, true},
    {"m-Fluorophenol",   "[OX2H1]c1cccc(F)c1",          9.28, true},
    {"o-Fluorophenol",   "[OX2H1]c1ccccc1F",            8.81, true},
    {"m-Cyanophenol",    "[OX2H1]c1cccc(C#N)c1",        8.61, true},
    {"p-SO2Me-phenol",   "[OX2H1]c1ccc(S(=O)(=O)C)cc1", 8.47, true},
    {"p-Acetylphenol",   "[OX2H1]c1ccc(C(=O)C)cc1",     8.05, true},
    {"2,4-diCl-phenol",  "[OX2H1]c1ccc(Cl)cc1Cl",       7.85, true},
    {"2,4,6-triCl-phenol","[OX2H1]c1c(Cl)cc(Cl)cc1Cl",  6.15, true},
    {"Pentachlorophenol","[OX2H1]c1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", 4.7, true},
    {"p-Hydroxyphenol",  "[OX2H1]c1ccc([OX2H1])cc1",    9.85, true},  // hydroquinone
    {"Salicylaldehyde OH","[OX2H1]c1ccccc1C=O",         8.34, true},
    {"Salicylic acid OH","[OX2H1]c1ccccc1C(=O)[OH]",   13.0, true},  // intramolecular H-bond, high pKa for OH
    {"p-Hydroxybenz OH", "[OX2H1]c1ccc(C(=O))cc1",      8.0,  true},
    {"m-Methoxyphenol",  "[OX2H1]c1cccc(OC)c1",         9.65, true},
    {"p-tBu-phenol",     "[OX2H1]c1ccc(C(C)(C)C)cc1",  10.23, true},
    {"2,6-di-tBu-phenol","[OX2H1]c1c(C(C)(C)C)cccc1C(C)(C)C", 11.70, true},

    // ---- Salicylic acid (COOH) ----
    {"Salicylic acid",   "[CX3](=O)([OX2H1])c1ccccc1O",  2.97, true},

    // ---- Additional carboxylic acids from Williams ----
    // Amino acids (α-COOH)
    {"Glycine COOH",     "[CX3](=O)([OX2H1])[CX4]([NX3H2,NX4H3+])", 2.35, true},
    {"Proline COOH",     "[CX3](=O)([OX2H1])C1CCCN1",   1.99, true},
    // Cinnamic/crotonic
    {"Cinnamic acid",    "[CX3](=O)([OX2H1])/C=C/c",    4.44, true},
    {"Crotonic acid",    "[CX3](=O)([OX2H1])/C=C/C",    4.69, true},
    // Pyruvic acid
    {"Pyruvic acid",     "[CX3](=O)([OX2H1])C(=O)C",    2.50, true},
    // Tartaric, citric, lactic
    {"Lactic acid",      "[CX3](=O)([OX2H1])C(O)C",     3.86, true},
    {"Mandelic acid",    "[CX3](=O)([OX2H1])C(O)c",     3.41, true},
    {"Glycolic acid",    "[CX3](=O)([OX2H1])CO",        3.82, true},
    // Nicotinic acid (aromatic N + COOH)
    {"Picolinic acid",   "[CX3](=O)([OX2H1])c1ccccn1",  5.25, true},  // 2-pyridine COOH (zwitterionic)
    // Dicarboxylic aliphatic
    {"Oxalacetic acid",  "[CX3](=O)([OX2H1])CC(=O)C(=O)[OX2H1]", 2.56, true},
    {"Malonic acid",     "[CX3](=O)([OX2H1])CC(=O)[OX2H1]", 2.83, true},
    {"Adipic acid",      "[CX3](=O)([OX2H1])CCCCC(=O)[OX2H1]", 4.42, true},

    // ---- Phosphonamides ----
    {"Phosphonamidate",  "[OX2H1]P(=O)(N)",             3.0,  true},

    // ---- Selenol ----
    {"Selenol",          "[SeX2H1]",                     5.2,  true},

    // =====================================================================
    // BASES — conjugate acid pKa (pH where 50% protonated)
    // Ordered most specific first
    // =====================================================================

    // ---- Guanidines / Amidines (very strong bases) ----
    // Guanidine: any N-C(=N)-N connectivity — multiple SMARTS to catch all tautomers
    {"Guanidine",        "[NX3]C(=[NX2])[NX3]",        12.5,  false},
    {"Guanidine alt1",   "[NX3]C([NX3])=[NX2]",        12.5,  false},  // RDKit tautomer
    {"Guanidine alt2",   "[NH2]C(=N)N",                12.5,  false},  // explicit H form
    {"Guanidine alt3",   "NC(N)=[NH]",                 12.5,  false},  // another tautomer
    {"Guanidine charged","[NH2]C(=[NH2+])N",           12.5,  false},  // already protonated check
    {"Guanidine charged2","NC(=[NH2+])[NH2]",          12.5,  false},  // reverse
    // Amidines: N-C(=N)-C (not N)
    {"Amidine",          "[NX3]C(=[NX2])[!N]",         11.6,  false},
    {"Amidine alt",      "[NH2]C(=[NH])c",             11.6,  false},  // benzamidine form
    {"Amidine alt2",     "[NH2]/C(=N\\H)c",            11.6,  false},  // E/Z
    {"Amidine alt3",     "[NH2]C(=[NH])[CX4]",         12.4,  false},  // acetamidine
    {"Amidine charged",  "[NH2]C(=[NH2+])c",           11.6,  false},  // already protonated
    {"DBU",              "C1=NCCCN1CCC",               12.0,  false},  // diazabicycloundecene
    {"Acetamidine",      "[NX3]C(=[NX2])C",            12.4,  false},

    // ---- Saturated N-heterocycles — most specific ring patterns first ----

    // Piperazine — critical for drug molecules, 2 distinct pKa values
    // N-Aryl piperazine: aryl-N is much less basic (~3.5), other N is basic (~8.5)
    {"Piperaz N-aryl arN","[NX3H0;R1](c)1CC[NX3;R1]CC1", 3.9, false}, // Ar-N of N-arylpiperazine
    {"Piperaz N-aryl alkN","[NX3;R1]1CC[NX3H0;R1](c)CC1", 8.5, false}, // alk-N of N-arylpiperazine
    // N-Acyl piperazine (amide N, not basic): handled by !$(NC=O) exclusion below
    // N-Alkyl piperazine: N-alkyl pKa ~9.0, NH pKa ~5.3 (diamine depression)
    {"Piperaz NH di-sub","[NX3H1;R1]1CC[NX3H0;R1]([CX4])CC1", 5.3, false}, // NH when other N is alkylated
    {"Piperaz NR di-sub","[NX3H0;R1]([CX4])1CC[NX3H1;R1]CC1", 9.0, false}, // NR when other is NH
    // Unsubstituted piperazine: pKa1=9.83, pKa2=5.33
    {"Piperazine NH 1st","[NX3H1;R1]1CC[NX3H1;R1]CC1",  9.8, false},  // first (more basic) N
    // Piperazine di-NR: both N alkylated
    {"Piperaz NR,NR 1st","[NX3H0;R1]([CX4])1CC[NX3H0;R1]([CX4])CC1", 9.0, false},
    // N-Tosylpiperazine
    {"1-Tosylpiperazine","[NX3H1;R1]1CC[NX3;R1](S(=O)(=O))CC1", 7.4, false},
    // N-Benzoylpiperazine
    {"N-Bz piperazine",  "[NX3H1;R1]1CC[NX3;R1](C(=O)c)CC1", 7.8, false},
    // Fallback piperazine N (NH in ring with another N, not amide/sulfonamide)
    {"Piperazine N",     "[NX3H1;R1;!$(NC=O);!$(NS(=O)=O)]1CC[NX3;R1]CC1", 9.0, false},

    // Morpholine
    {"4-Aryl morpholine","[NX3H0;R1](c)1CCOCC1",       7.4,  false},
    {"N-Me morpholine",  "[NX3H0;R1](C)1CCOCC1",       7.4,  false},
    {"Morpholine NH",    "[NX3H1;R1]1CCOCC1",           8.33, false},
    {"Morpholine N",     "[NX3;R1;!$(NC=O)]1CCOCC1",    8.33, false},

    // Thiomorpholine
    {"Thiomorpholine NH","[NX3H1;R1]1CCSCC1",           8.70, false},
    {"Thiomorpholine N", "[NX3;R1;!$(NC=O)]1CCSCC1",    8.70, false},

    // Pyrrolidine
    {"Pyrrolidine NH",   "[NX3H1;R1;!$(NC=O);!$(NS=O)]1CCCC1", 11.27, false},
    {"N-Me pyrrolidine", "[NX3H0;R1;!$(NC=O)](C)1CCCC1", 10.46, false},

    // Piperidine
    {"4-Aryl piperidine","[NX3H1;R1;!$(NC=O)]1CCC(c)CC1", 10.1, false},
    {"N-Me piperidine",  "[NX3H0;R1;!$(NC=O)](C)1CCCCC1", 10.08, false},
    {"N-Bz piperidine",  "[NX3H0;R1](Cc)1CCCCC1",      9.6,  false},
    {"Piperidine NH",    "[NX3H1;R1;!$(NC=O);!$(NS=O)]1CCCCC1", 11.22, false},
    {"Piperidine N",     "[NX3;R1;!$(NC=O);!$(NS=O)]1CCCCC1", 10.5, false},

    // Azetidine
    {"Azetidine NH",     "[NX3H1;R1;!$(NC=O)]1CCC1",    11.3, false},

    // Azepane (7-membered ring)
    {"Azepane NH",       "[NX3H1;R1;!$(NC=O)]1CCCCCC1",  10.5, false},

    // DABCO (1,4-diazabicyclo[2.2.2]octane)
    {"DABCO N",          "[NX3;R2]1CC[NX3;R2]CC1",      8.82, false},

    // Quinuclidine
    {"Quinuclidine N",   "[NX3;R2;!$(Nc)]1CC2CCC(C1)C2", 11.0, false},

    // ---- Aromatic heterocyclic bases ----

    // Imidazoles (protonated at =N, not NH)
    {"4-Me-imidazole",   "[nH0;X2]1c(C)[nH1]cc1",       7.45, false},
    {"2-Me-imidazole",   "[nH0;X2]1cc[nH1]c1C",          7.75, false},
    {"Benzimidazole =N", "[nH0;X2]1c[nH1]c2ccccc12",     5.53, false},
    {"Imidazole =N",     "[nH0;X2]1cc[nH1]c1",            6.95, false},
    {"Imidazole =N alt", "[nH0;X2]1c[nH1]cc1",            6.95, false},

    // Pyridines — substituted (most specific first)
    {"4-DMAP",           "[nH0;X2]1cc(N(C)C)ccc1",        9.70, false},  // 4-dimethylaminopyridine
    {"4-Aminopyridine",  "[nH0;X2]1cc(N)ccc1",            9.17, false},
    {"2-Aminopyridine",  "[nH0;X2]1cccc(N)c1",            6.86, false},
    {"4-Me-pyridine",    "[nH0;X2]1cc(C)ccc1",            6.02, false},
    {"3-Me-pyridine",    "[nH0;X2]1ccc(C)cc1",            5.68, false},
    {"2-Me-pyridine",    "[nH0;X2]1cccc(C)c1",            5.97, false},
    {"2,6-diMe-pyridine","[nH0;X2]1c(C)ccc(C)c1",         6.77, false},  // 2,6-lutidine
    {"2,4,6-triMe-pyridine","[nH0;X2]1c(C)cc(C)c(C)c1",   7.48, false},  // collidine
    {"4-OMe-pyridine",   "[nH0;X2]1cc(OC)ccc1",           6.62, false},
    {"2-OMe-pyridine",   "[nH0;X2]1cccc(OC)c1",           3.28, false},
    {"3-OH-pyridine",    "[nH0;X2]1ccc(O)cc1",            4.86, false},
    {"3-CN-pyridine",    "[nH0;X2]1ccc(C#N)cc1",          1.45, false},  // nicotinonitrile
    {"3-NO2-pyridine",   "[nH0;X2]1ccc([NX3+](=O)[O-])cc1", 0.81, false},
    {"2-Cl-pyridine",    "[nH0;X2]1cccc(Cl)c1",           0.72, false},
    {"3-Cl-pyridine",    "[nH0;X2]1ccc(Cl)cc1",           2.84, false},
    {"4-Cl-pyridine",    "[nH0;X2]1cc(Cl)ccc1",           3.83, false},
    {"3-F-pyridine",     "[nH0;X2]1ccc(F)cc1",            2.97, false},
    {"2-F-pyridine",     "[nH0;X2]1cccc(F)c1",           -0.44, false},
    {"3-Br-pyridine",    "[nH0;X2]1ccc(Br)cc1",           2.84, false},
    {"3-COOH-pyridine",  "[nH0;X2]1ccc(C(=O)O)cc1",       3.13, false},  // nicotinic acid
    {"3-CO2Et-pyridine", "[nH0;X2]1ccc(C(=O)OCC)cc1",     3.35, false},

    // Quinoline / Isoquinoline
    {"Isoquinoline",     "[nH0;X2]1ccc2ccccc2c1",         5.14, false},
    {"Quinoline",        "[nH0;X2]1cccc2ccccc12",          4.85, false},
    {"Acridine N",       "[nH0;X2]1cccc2cc3ccccc3cc12",    5.60, false},

    // Diazines
    {"Pyridazine N",     "[nH0;X2]1[nH0;X2]cccc1",        2.33, false},
    {"Pyrimidine N",     "[nH0;X2]1cc[nH0;X2]cc1",        1.10, false},
    {"Pyrazine N",       "[nH0;X2]1cc[nH0;X2]cc1",        0.60, false},

    // Benzodiazines
    {"Quinazoline N",    "[nH0;X2]1c[nH0;X2]c2ccccc2c1",  3.31, false},
    {"Quinoxaline N",    "[nH0;X2]1[nH0;X2]cc2ccccc2c1",  0.60, false},
    {"Phthalazine N",    "[nH0;X2]1[nH0;X2]cc2ccccc12",   3.47, false},
    {"Cinnoline N",      "[nH0;X2]1[nH0;X2]c2ccccc2cc1",  2.64, false},

    // Pyrazole, triazole (as bases — protonation of =N)
    {"Pyrazole =N",      "[nH0;X2]1cc[nH1]c1",            2.5,  false},  // weak base
    {"1,2,4-Triazole =N","[nH0;X2]1c[nH1]nc1",            2.2,  false},
    {"Benzotriazole =N", "[nH0;X2]1[nH0;X2][nH1]c2ccccc12", 1.6, false},

    // Generic pyridine (fallback — any 6-ring aromatic N without H)
    {"Pyridine N",       "[nH0;X2;R1]1ccccc1",            5.14, false},

    // ---- 5-membered heterocyclic bases (Williams p22-24, Evans p2) ----
    // Thiazole
    {"2-Aminothiazole",  "[nH0;X2]1csc(N)c1",             5.36, false},  // key drug fragment!
    {"4-Me-thiazole",    "[nH0;X2]1csc(C)c1",             3.5,  false},
    {"Thiazole =N",      "[nH0;X2]1cscc1",                2.5,  false},
    {"Benzothiazole =N", "[nH0;X2]1c2ccccc2sc1",          1.2,  false},
    {"2-Aminobenzothiazole","[nH0;X2]1c2ccccc2sc1N",      4.51, false},
    // Oxazole
    {"Benzoxazole =N",   "[nH0;X2]1c2ccccc2oc1",         -0.2,  false},
    {"2-Aminobenzoxazole","[nH0;X2]1c2ccccc2oc1N",        3.73, false},
    {"Oxazole =N",       "[nH0;X2]1cocc1",               0.8,  false},
    {"Oxazoline =N",     "N=1CCOC1",                       4.8,  false},  // 2-oxazoline
    // Isoxazole, isothiazole
    {"Isoxazole =N",     "[nH0;X2]1oncc1",               -2.0,  false},  // very weak base
    {"Isothiazole =N",   "[nH0;X2]1sncc1",                0.5,  false},
    // Thiadiazoles
    {"1,2,4-Thiadiazole","[nH0;X2]1ncs[nH0]1",           -1.0,  false},
    {"1,3,4-Thiadiazole","[nH0;X2]1[nH0]csc1",            1.0,  false},
    {"2-Amino-1,3,4-thiadiazole","[nH0;X2]1[nH0]c(N)sc1", 3.5, false},
    // Oxadiazoles
    {"1,2,4-Oxadiazole", "[nH0;X2]1nco[nH0]1",           -2.0,  false},
    {"1,3,4-Oxadiazole", "[nH0;X2]1[nH0]coc1",           -1.5,  false},
    {"2-Amino-1,3,4-oxadiazole","[nH0;X2]1[nH0]c(N)oc1",  2.0, false},
    // Indazole (as base)
    {"Indazole =N",      "[nH0;X2]1[nH1]c2ccccc12",       1.4,  false},
    // Imidazo[1,2-a]pyridine (fused, common scaffold)
    {"Imidazo[1,2-a]pyr","[nH0;X2]1ccn2ccccc12",          6.0,  false},

    // ---- Purine / nucleobase bases (protonation at N) ----
    {"Adenine N1",       "[nH0;X2]1c2[nH]cnc2nc(N)c1",    4.15, false},
    {"Guanine N7",       "[nH0;X2]1cnc2C(=O)[NH]C(N)=Nc12", 3.3, false},
    {"Purine N",         "[nH0;X2]1c2[nH]cnc2ncc1",       2.52, false},
    {"Caffeine N",       "[nH0;X2]1c2n(C)c(=O)n(C)c(=O)c2n(C)c1", 0.6, false},

    // ---- Aminopyrimidines / Aminopyrazines (key drug scaffolds) ----
    {"4-Aminopyrimidine","[nH0;X2]1cc(N)ncn1",            5.71, false},
    {"2-Aminopyrimidine","[nH0;X2]1ccnc(N)n1",            3.54, false},
    {"4,6-Diamino-pyrimidine","[nH0;X2]1c(N)cc(N)nc1",    7.26, false},
    {"2-Aminopyrazine",  "[nH0;X2]1cnc(N)cn1",            3.14, false},
    {"5-Aminopyrimidine","[nH0;X2]1cc(N)cnc1",            2.83, false},
    {"Aminotriazine",    "[nH0;X2]1nc(N)ncn1",            5.0,  false},  // melamine-like

    // ---- Aminoquinolines / Aminoisoquinolines ----
    {"2-Aminoquinoline", "[nH0;X2]1cccc2ccc(N)cc12",      7.34, false},
    {"4-Aminoquinoline", "[nH0;X2]1cccc2c(N)cccc12",      9.17, false},
    {"6-Aminoquinoline", "[nH0;X2]1cccc2ccc(N)cc12",      5.63, false},
    {"1-Aminoisoquinoline","[nH0;X2]1cc(N)c2ccccc2c1",    7.62, false},
    {"3-Aminoisoquinoline","[nH0;X2]1cnc(N)c2ccccc12",    5.05, false},
    {"Benzoquinoline N", "[nH0;X2]1cccc2cccc3ccccc123",   5.05, false},
    {"Phenanthroline N", "[nH0;X2]1cccc2c1ccc1cccnc12",   4.27, false},

    // Generic aromatic =N in ring (e.g., misc heterocycles, must be after specific ones)
    {"Het aromatic =N",  "[nH0;X2;R1]",                   3.5,  false},

    // ---- Aliphatic amines ----
    // Primary amines — specific subtypes
    {"alpha-Amino acid", "[NX3H2][CX4H1](C(=O)[O,N])",    9.0,  false},  // glycine-like
    {"Benzylamine",      "[NX3H2]Cc",                      9.34, false},
    {"CF3-ethylamine",   "[NX3H2]CC(F)(F)F",               5.7,  false},
    {"2-Fluoroethylamine","[NX3H2]CCF",                    8.5,  false},
    {"2-Methoxyethylamine","[NX3H2]CCOC",                  9.2,  false},
    {"Ethanolamine",     "[NX3H2]CCO",                     9.50, false},
    {"2-Cyanoethylamine","[NX3H2]CCC#N",                   7.9,  false},
    {"Allylamine",       "[NX3H2]CC=C",                    9.49, false},
    {"Methyl amine",     "[NX3H2;!$(NC=O);!$(NS=O);!$(Nc)]C", 10.6, false},
    {"Primary amine",    "[NX3H2;!$(NC=O);!$(NS=O);!$(Nc)]", 10.5, false},

    // Secondary amines — specific subtypes
    {"N,O-dimethylhydroxylamine","[NX3H0](C)(OC)",         4.75, false},
    {"N-methylhydroxylamine","[NX3H1](C)O",                5.96, false},
    {"Diethylamine",     "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R]([CX4][CX4])[CX4][CX4]", 10.98, false},
    {"Dimethylamine",    "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R](C)C", 10.64, false},
    {"N-Me benzylamine", "[NX3H0;!$(NC=O);!$(NS=O);!R](C)Cc", 9.6, false},
    {"Dibenzylamine",    "[NX3H1;!$(NC=O);!$(NS=O);!R](Cc)Cc", 8.52, false},
    {"Secondary amine",  "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R]", 10.5, false},

    // Tertiary amines
    {"Trimethylamine",   "[NX3H0;!$(NC=O);!$(NS=O);!$(Nc);!R](C)(C)C", 9.76, false},
    {"Triethylamine",    "[NX3H0;!$(NC=O);!$(NS=O);!$(Nc);!R]([CX4][CX4])([CX4][CX4])[CX4][CX4]", 10.75, false},
    {"Tertiary amine",   "[NX3H0;!$(NC=O);!$(NS=O);!$(Nc);!R]([CX4])([CX4])[CX4]", 9.8, false},

    // ---- Anilines (aromatic amines — weak bases) ----
    {"p-NO2-aniline",    "[NX3H2]c1ccc([NX3+](=O)[O-])cc1", 1.0, false},
    {"m-NO2-aniline",    "[NX3H2]c1cccc([NX3+](=O)[O-])c1", 2.47, false},
    {"p-CN-aniline",     "[NX3H2]c1ccc(C#N)cc1",          1.74, false},
    {"p-Cl-aniline",     "[NX3H2]c1ccc(Cl)cc1",           3.98, false},
    {"p-Br-aniline",     "[NX3H2]c1ccc(Br)cc1",           3.91, false},
    {"p-F-aniline",      "[NX3H2]c1ccc(F)cc1",            4.52, false},
    {"p-CF3-aniline",    "[NX3H2]c1ccc(C(F)(F)F)cc1",     2.57, false},
    {"p-OMe-aniline",    "[NX3H2]c1ccc(OC)cc1",           5.29, false},
    {"p-Me-aniline",     "[NX3H2]c1ccc(C)cc1",            5.07, false},
    {"p-NH2-aniline",    "[NX3H2]c1ccc(N)cc1",            6.08, false},  // p-phenylenediamine
    {"2-Naphthylamine",  "[NX3H2]c1ccc2ccccc2c1",         4.16, false},
    {"1-Naphthylamine",  "[NX3H2]c1cccc2ccccc12",         3.92, false},
    {"Aniline",          "[NX3H2]c",                       4.58, false},
    {"N-Me aniline",     "[NX3H1;!$(NC=O)](C)c",           4.85, false},
    {"N,N-diMe aniline", "[NX3H0;!$(NC=O)](C)(C)c",        5.07, false},

    // ---- Additional saturated heterocycles / bicyclic ----
    {"Thiazolidine NH",  "[NX3H1;R1]1CCSC1",              6.31, false},  // from Williams
    {"Tetrahydroisoquinoline","[NX3H1;R1]1CCc2ccccc2C1",   9.5, false},  // common drug scaffold
    {"Tropane N",        "[NX3;R2]1CC2CCC(C1)CC2",        10.0, false},  // cocaine/atropine scaffold
    {"Proton sponge",    "[NX3](C)(C)c1cccc2c1cccc2[NX3](C)C", 12.1, false}, // 1,8-bis(NMe2)naphthalene
    {"Decahydroquinoline","[NX3H1;R1]1CCCCC1C1CCCCC1",     11.0, false},
    {"2-Methylazetidine","[NX3H1;R1]1CC(C)C1",            11.3, false},

    // ---- Diamines (important: pKa depression for second N) ----
    {"Ethylenediamine N1","[NX3H2]CC[NX3H2]",             10.0, false},  // first pKa
    // note: second pKa (7.0) handled by Henderson-Hasselbalch + charge effect
    {"1,3-Diaminopropane","[NX3H2]CCC[NX3H2]",           10.6, false},
    {"1,4-Diaminobutane","[NX3H2]CCCC[NX3H2]",           10.8, false},  // putrescine
    {"1,5-Diaminopentane","[NX3H2]CCCCC[NX3H2]",         10.9, false}, // cadaverine
    {"1,2-Diaminopropane","[NX3H2]CC([NX3H2])C",          9.97, false},

    // ---- Amino acids (amine side) ----
    {"Lysine epsilon-NH2","[NX3H2]CCCCC(N)C(=O)",          10.5, false},
    {"Histidine imidazole","[nH0;X2]1cc([CH2]C(N)C(=O))[nH1]c1", 6.04, false},
    {"Arginine guanidine","[NX3H2]C(=[NX2H0])[NX3H1]CCCC(N)C(=O)", 12.48, false},

    // ---- Ephedrine / phenethylamine class ----
    {"Ephedrine",        "[NX3H1](C)C(O)c",               9.6,  false},
    {"Amphetamine",      "[NX3H2]CC(C)c",                  9.9,  false},
    {"Phenethylamine",   "[NX3H2]CCc",                     9.83, false},

    // ---- Hydroxylamine ----
    {"Hydroxylamine",    "[NX3H2]O",                       5.97, false},
    {"N,N-dimethylhydroxylamine","[NX3H0](C)(C)O",         4.75, false},

    // ---- Hydrazines / Hydrazides ----
    {"Phenylhydrazine",  "[NX3H1]([NX3])c",                5.21, false},
    {"Hydrazine",        "[NX3H2][NX3H2]",                 8.07, false},
    {"Methylhydrazine",  "[NX3H1]([NX3])C",                7.87, false},
    {"N,N-Dimethylhydrazine","[NX3H0]([NX3])(C)C",         7.21, false},
    {"Acetohydrazide",   "[NX3H1]([NX3H2])C(=O)",          3.24, false},
    {"Semicarbazide",    "[NX3H2][NX3H1]C(=O)[NX3H2]",    3.66, false},
    {"Isoniazid hydrazide","[NX3H2][NX3H1]C(=O)c1ccncc1",  3.5, false},  // INH

    // ---- Guanidine / Amidine variants ----
    {"Biguanide",        "[NX3]C(=N)NC(=N)N",             11.5, false},
    {"Phenylguanidine",  "[NX3H2]C(=[NX2])[NX3H1]c",     10.9, false},
    {"Acetylguanidine",  "[NX3H2]C(=[NX2])NC(=O)C",       8.3, false},
    {"Cyanoguanidine",   "[NX3H2]C(=[NX2])NC#N",          0.4, false},   // very weak base

    // ---- Urea / Thiourea (very weak bases) ----
    {"Urea N",           "[NX3H2]C(=O)[NX3H2]",           0.18, false}, // from Williams
    {"Thiourea N",       "[NX3H2]C(=S)[NX3H2]",          -0.96, false}, // from Williams

    // ---- Nitrogen mustard class / Aziridine ----
    {"Aziridine NH",     "[NX3H1;R1]1CC1",                8.0,  false},

    // ---- Pyridine N-oxide (as base, less basic than pyridine) ----
    // N-oxides have the oxygen as a base, very weak
    {"Pyridine N-oxide", "[nX3;R1]([O-])1ccccc1",         0.8,  false},

    // =====================================================================
    // GENERIC FALLBACK PATTERNS — catch anything the specific patterns above miss
    // These are the broadest possible SMARTS, placed LAST so specific patterns win
    // =====================================================================

    // ---- Generic acid fallbacks ----
    // Any OH on heteroatom (not alcohol, not water) — rare but covers exotic acids
    {"Generic S-OH acid","[OX2H1]S",                       2.0,  true},  // any S-OH (sulfenic, sulfinic, sulfonic)
    {"Generic P-OH acid","[OX2H1]P",                       2.0,  true},  // any P-OH
    {"Generic ArOH",     "[OX2H1]a",                      10.0,  true},  // any aromatic OH (phenol-like)
    {"Generic COOH",     "[CX3](=O)[OX2H1]",              4.0,  true},  // any carboxylic acid
    {"Generic SH",       "[SX2H1]",                        8.3,  true},  // any thiol
    {"Generic NH-SO2",   "[NH]S(=O)(=O)",                 10.0,  true},  // any sulfonamide NH
    {"Generic NH-CO-NH-CO","[NH1](C=O)C=O",               9.0,  true},  // any imide NH

    // ---- Generic base fallbacks ----
    // Any C(=N)N connectivity (guanidine/amidine family) — broadest possible
    {"Generic C=N-N",    "[NX3;!$(NC=O);!$(NS=O)]C=[NX2]",  11.0, false},  // amidine/guanidine generic
    {"Generic N=C-N",    "[NX2]=[CX3][NX3;!$(NC=O)]",       11.0, false},  // reverse match

    // Any saturated ring N not amide/sulfonamide (covers all N-heterocycles)
    {"Generic ring NH sat","[NX3H1;R;!$(NC=O);!$(NS(=O)=O);!a]", 9.5, false},  // any sat ring NH
    {"Generic ring NR sat","[NX3H0;R;!$(NC=O);!$(NS(=O)=O);!a]([CX4])", 9.0, false}, // any sat ring NR

    // Any aromatic ring =N (pyridine-like, all heterocyclic bases)
    {"Generic arom =N",  "[nH0;X2]",                        4.0, false},  // any aromatic =N

    // Any aliphatic NH2 not amide
    {"Generic prim amine","[NX3H2;!$(NC=O);!$(NS=O);!$(NC=S)]", 10.5, false},
    // Any aliphatic NH not amide, not aromatic, not ring
    {"Generic sec amine", "[NX3H1;!$(NC=O);!$(NS=O);!$(NC=S);!$(Nc);!R]", 10.5, false},
    // Any aliphatic tertiary N not amide
    {"Generic tert amine","[NX3H0;!$(NC=O);!$(NS=O);!$(NC=S);!$(Nc);!R]([CX4])([CX4])[CX4]", 9.8, false},

    // Any ring NH (aromatic, e.g. imidazole NH, pyrazole NH — protonation as base at =N)
    // This handles protonation at the OTHER nitrogen in the ring
    {"Generic arom ring NH","[nH1]",                        5.0, false},  // very broad fallback

    // ---- Amides (extremely weak bases — not usually relevant at pH 7.4) ----
    // Amide N is essentially non-basic (pKa ~-0.5) so we omit it.
    // The exclusion patterns !$(NC=O) above prevent amide N from matching amine patterns.
};
static constexpr int kNumIonizableGroups = sizeof(kIonizableGroups) / sizeof(kIonizableGroups[0]);

/// Lazily compiled SMARTS cache — compiled once on first use, reused forever.
/// This avoids recompiling 345 SMARTS patterns on every call to detectIonSitesInternal.
/// C++11 static local initialization is thread-safe (ISO C++11 §6.7): no explicit mutex needed.
static const std::vector<std::unique_ptr<ROMol>>& getCompiledPatterns() {
    static const std::vector<std::unique_ptr<ROMol>> patterns = []() {
        std::vector<std::unique_ptr<ROMol>> p(kNumIonizableGroups);
        for (int g = 0; g < kNumIonizableGroups; g++) {
            p[g].reset(SmartsToMol(kIonizableGroups[g].smarts));
        }
        return p;
    }();
    return patterns;
}

/// Detect all ionizable sites in a molecule (dedup by atom index, first match wins).
static std::vector<std::tuple<int, int, bool, double>> detectIonSitesInternal(const ROMol &mol) {
    const auto &patterns = getCompiledPatterns();
    std::vector<std::tuple<int, int, bool, double>> sites; // (atomIdx, groupIdx, isAcid, defaultPKa)
    std::set<int> seen;
    for (int g = 0; g < kNumIonizableGroups; g++) {
        if (!patterns[g]) continue;
        std::vector<MatchVectType> matches;
        SubstructMatch(mol, *patterns[g], matches);
        for (const auto &m : matches) {
            if (m.empty()) continue;
            int aIdx = m[0].second;
            if (seen.count(aIdx)) continue;
            seen.insert(aIdx);
            sites.emplace_back(aIdx, g, kIonizableGroups[g].isAcid, kIonizableGroups[g].pKa);
        }
    }
    return sites;
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

        // Reuse the shared ionizable group table (kIonizableGroups)
        // Find ionizable sites: each site = (group index, matched atom index, is ambiguous)
        struct IonizableSite {
            int groupIdx;
            int atomIdx;    // index of the ionizable atom (H-bearing for acids, N for bases)
            std::string label;
        };
        std::vector<IonizableSite> ambiguousSites;
        std::set<int> seenAtomIdx;

        const auto &patterns = getCompiledPatterns();
        for (int g = 0; g < kNumIonizableGroups; g++) {
            if (!patterns[g]) continue;

            std::vector<MatchVectType> matches;
            SubstructMatch(*mol, *patterns[g], matches);
            for (const auto &match : matches) {
                if (match.empty()) continue;
                int aidx = match[0].second;
                if (seenAtomIdx.count(aidx)) continue;

                // Henderson-Hasselbalch: is this site ambiguous at the target pH?
                double deltaPKa = std::abs(pH - kIonizableGroups[g].pKa);
                if (deltaPKa < pkaThreshold) {
                    ambiguousSites.push_back({g, aidx, kIonizableGroups[g].name});
                    seenAtomIdx.insert(aidx);
                }
            }
        }

        // Generate protomers: combinatorial expansion of ambiguous sites
        // Each site can be in state 0 (as-drawn) or state 1 (toggled)
        int nSites = std::min((int)ambiguousSites.size(), 10); // cap to avoid explosion
        int nCombinations = 1 << nSites;

        // Precompute Henderson-Hasselbalch populations for ALL combos (cheap math only),
        // then sort by population so we try the most probable combos first.
        // This avoids the bug where capping nCombinations to maxProtomers would only
        // explore combos 0..maxProtomers-1, ignoring higher-indexed ionizable sites.
        struct ComboPopulation { int combo; double population; };
        std::vector<ComboPopulation> combosByPop;
        combosByPop.reserve(nCombinations);
        for (int combo = 0; combo < nCombinations; combo++) {
            double pop = 1.0;
            for (int s = 0; s < nSites; s++) {
                bool toggle = (combo >> s) & 1;
                const auto &grp = kIonizableGroups[ambiguousSites[s].groupIdx];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - grp.pKa));
                double fracDeprot = 1.0 - fracProt;
                if (toggle) {
                    // Toggled: acid → deprotonated, base → protonated
                    pop *= grp.isAcid ? fracDeprot : fracProt;
                } else {
                    // As-drawn: acid → protonated, base → deprotonated
                    pop *= grp.isAcid ? fracProt : fracDeprot;
                }
            }
            combosByPop.push_back({combo, pop});
        }
        std::sort(combosByPop.begin(), combosByPop.end(),
            [](const ComboPopulation &a, const ComboPopulation &b) {
                return a.population > b.population;
            });
        int nToTry = std::min((int)combosByPop.size(), (int)maxProtomers);

        struct ProtomerEntry {
            std::string smi;
            std::unique_ptr<RWMol> rwmol;
            double energy;
            std::string label;
        };
        std::vector<ProtomerEntry> entries;
        std::set<std::string> seen;

        for (int ci = 0; ci < nToTry; ci++) {
            int combo = combosByPop[ci].combo;
            auto rw = std::make_unique<RWMol>(*mol);

            std::string comboLabel;
            for (int s = 0; s < nSites; s++) {
                bool toggle = (combo >> s) & 1;
                if (!toggle) continue; // keep as-drawn

                const auto &site = ambiguousSites[s];
                const auto &grp = kIonizableGroups[site.groupIdx];
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
        // Step 1: WASH — apply dominant protonation at target pH
        // ===================================================================
        // First set the dominant ionization state, then enumerate alternatives
        // for groups near the pH-pKa boundary.
        //
        // - Acids with pKa << pH → deprotonate (dominant form is conjugate base)
        // - Bases with pKa >> pH → protonate (dominant form is conjugate acid)
        // - Groups near boundary (|pH - pKa| < threshold) → enumerate both states

        struct ChemicalForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            int kind;  // 0=parent, 1=tautomer, 2=protomer, 3=taut+prot
            double hhPopulation = 1.0;  // Henderson-Hasselbalch population fraction [0,1]
        };
        std::vector<ChemicalForm> allForms;
        std::set<std::string> seenSMILES;

        // Reuse the shared ionizable group table (kIonizableGroups)
        // Detect all ionizable sites
        struct IonSite { int groupIdx; int atomIdx; bool isAcid; double groupPKa; };
        std::vector<IonSite> allSites;
        std::set<int> seenAtomIdx;
        const auto &cachedPats = getCompiledPatterns();
        for (int g = 0; g < kNumIonizableGroups; g++) {
            if (!cachedPats[g]) continue;
            std::vector<MatchVectType> matches;
            SubstructMatch(*parentMol, *cachedPats[g], matches);
            for (const auto &m : matches) {
                if (m.empty()) continue;
                int aIdx = m[0].second;
                if (seenAtomIdx.count(aIdx)) continue;
                seenAtomIdx.insert(aIdx);
                allSites.push_back({g, aIdx, kIonizableGroups[g].isAcid, kIonizableGroups[g].pKa});
            }
        }

        // Phase A: Apply dominant protonation state (WASH)
        // For acids: deprotonate if pH > pKa + threshold (clearly ionized)
        // For bases: protonate if pH < pKa - threshold (clearly protonated)
        auto washedMol = std::make_unique<RWMol>(*parentMol);
        std::string washLog;
        for (const auto &site : allSites) {
            Atom *atom = washedMol->getAtomWithIdx(site.atomIdx);
            double deltaPH = pH - site.groupPKa;  // positive = pH above pKa

            if (site.isAcid && deltaPH > pkaThreshold) {
                // pH well above pKa: acid is deprotonated (dominant)
                int nH = atom->getTotalNumHs();
                if (nH > 0) {
                    atom->setNumExplicitHs(nH - 1);
                    atom->setFormalCharge(atom->getFormalCharge() - 1);
                    if (!washLog.empty()) washLog += ", ";
                    washLog += std::string("deprot ") + kIonizableGroups[site.groupIdx].name;
                }
            } else if (!site.isAcid && deltaPH < -pkaThreshold) {
                // pH well below pKa: base is protonated (dominant)
                int nH = atom->getTotalNumHs();
                atom->setNumExplicitHs(nH + 1);
                atom->setFormalCharge(atom->getFormalCharge() + 1);
                if (!washLog.empty()) washLog += ", ";
                washLog += std::string("prot ") + kIonizableGroups[site.groupIdx].name;
            }
            // else: site is near boundary → will be enumerated in Phase B
        }
        try { MolOps::sanitizeMol(*washedMol); } catch (...) {
            // Wash failed — fall back to original
            washedMol = std::make_unique<RWMol>(*parentMol);
        }

        // Phase B: Enumerate protomers for ambiguous sites (near pH-pKa boundary)
        std::vector<IonSite> ambSites;
        for (const auto &site : allSites) {
            double deltaPH = std::abs(pH - site.groupPKa);
            if (deltaPH <= pkaThreshold) {
                ambSites.push_back(site);
            }
        }

        int nAmbSites = std::min((int)ambSites.size(), 10);
        int nTotalCombos = 1 << nAmbSites;

        // Precompute Henderson-Hasselbalch populations for ALL combos (cheap math only),
        // then sort by population so we try the most probable combos first.
        // This avoids the bug where capping nCombos to maxProtomers would only
        // explore combos 0..maxProtomers-1, ignoring higher-indexed ionizable sites.
        struct ComboPopulation { int combo; double population; };
        std::vector<ComboPopulation> combosByPop;
        combosByPop.reserve(nTotalCombos);
        for (int combo = 0; combo < nTotalCombos; combo++) {
            double pop = 1.0;
            for (int s = 0; s < nAmbSites; s++) {
                const auto &site = ambSites[s];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.groupPKa));
                double fracDeprot = 1.0 - fracProt;
                bool toggle = (combo >> s) & 1;
                if (!toggle) {
                    pop *= site.isAcid ? fracProt : fracDeprot;
                } else {
                    pop *= site.isAcid ? fracDeprot : fracProt;
                }
            }
            combosByPop.push_back({combo, pop});
        }
        std::sort(combosByPop.begin(), combosByPop.end(),
            [](const ComboPopulation &a, const ComboPopulation &b) {
                return a.population > b.population;
            });
        int nCombos = std::min((int)combosByPop.size(), (int)maxProtomers);

        struct ProtomerForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            double hhPopulation;  // Henderson-Hasselbalch population fraction
        };
        std::vector<ProtomerForm> protomers;

        // Iterate combos in order of decreasing HH population
        for (int ci = 0; ci < nCombos; ci++) {
            int combo = combosByPop[ci].combo;
            auto rw = std::make_unique<RWMol>(*washedMol);
            std::string comboLabel;
            double hhPop = 1.0;

            for (int s = 0; s < nAmbSites; s++) {
                const auto &site = ambSites[s];
                // Henderson-Hasselbalch: fraction protonated
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.groupPKa));
                double fracDeprot = 1.0 - fracProt;

                bool toggle = (combo >> s) & 1;
                if (!toggle) {
                    // Keep as-drawn (washed state)
                    // Washed state for acids with pH > pKa+threshold = deprotonated (dominant)
                    // Washed state for bases with pH < pKa-threshold = protonated (dominant)
                    // For ambiguous sites (within threshold), washed = as-drawn from parent SMILES
                    // The dominant fraction:
                    if (site.isAcid) {
                        // Washed left this acid as-drawn (near boundary), i.e. protonated
                        hhPop *= fracProt;
                    } else {
                        // Washed left this base as-drawn (near boundary), i.e. deprotonated
                        hhPop *= fracDeprot;
                    }
                    continue;
                }

                // Toggle this site
                Atom *atom = rw->getAtomWithIdx(site.atomIdx);
                if (site.isAcid) {
                    // Toggle: deprotonate the acid
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                    hhPop *= fracDeprot;  // deprotonated fraction
                } else {
                    // Toggle: protonate the base
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                    hhPop *= fracProt;    // protonated fraction
                }
                if (!comboLabel.empty()) comboLabel += "+";
                comboLabel += std::string(site.isAcid ? "deprot_" : "prot_") +
                    kIonizableGroups[site.groupIdx].name;
            }

            try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
            std::string pSmi = MolToSmiles(*rw);
            if (seenSMILES.count(pSmi)) continue;
            seenSMILES.insert(pSmi);

            if (comboLabel.empty()) comboLabel = "Parent";
            protomers.push_back({pSmi, std::move(rw), comboLabel, hhPop});
        }

        // If no protomers generated, use washed molecule
        if (protomers.empty()) {
            auto rw = std::make_unique<RWMol>(*washedMol);
            std::string wSmi = MolToSmiles(*rw);
            seenSMILES.insert(wSmi);
            protomers.push_back({wSmi, std::move(rw), "Parent", 1.0});
        }

        // ===================================================================
        // Step 2: For each protomer, enumerate tautomers
        // ===================================================================
        int maxTotalForms = maxProtomers * maxTautomers;
        if (maxTotalForms > 200) maxTotalForms = 200;

        for (auto &prot : protomers) {
            if ((int)allForms.size() >= maxTotalForms) break;

            // Always add the protomer itself first (before tautomer enumeration)
            if (!seenSMILES.count(prot.smi)) {
                seenSMILES.insert(prot.smi);
            }
            {
                auto rw = std::make_unique<RWMol>(*prot.mol);
                int kind = (prot.label != "Parent") ? 2 : 0;
                allForms.push_back({prot.smi, std::move(rw), prot.label, kind, prot.hhPopulation});
            }

            // Then enumerate tautomers of this protomer
            try {
                MolStandardize::TautomerEnumerator enumerator;
                enumerator.setMaxTautomers(std::max(maxTautomers, 2));
                enumerator.setMaxTransforms(200);
                auto tautResult = enumerator.enumerate(*prot.mol);

                int tautCount = 0;
                for (const auto &taut : tautResult) {
                    if (tautCount >= maxTautomers || (int)allForms.size() >= maxTotalForms) break;
                    auto rw = std::make_unique<RWMol>(*taut);
                    try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
                    std::string tSmi = MolToSmiles(*rw);

                    if (seenSMILES.count(tSmi)) continue;
                    seenSMILES.insert(tSmi);

                    bool isFromProtomer = (prot.label != "Parent");
                    int kind = isFromProtomer ? 3 : 1;  // taut+prot or taut only

                    std::string label = prot.label;
                    if (label != "Parent") label += "_";
                    else label = "";
                    label += "Taut" + std::to_string(tautCount + 1);

                    allForms.push_back({tSmi, std::move(rw), label, kind, prot.hhPopulation});
                    tautCount++;
                }
            } catch (...) {
                // Tautomer enumeration failed; protomer already added above
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
            double hhPopulation;  // Henderson-Hasselbalch population from protomer
        };
        std::vector<PreparedConformer> allConformers;

        for (int fi = 0; fi < (int)allForms.size(); fi++) {
            auto &form = allForms[fi];

            // Clone and add hydrogens
            auto mol = std::make_unique<RWMol>(*form.mol);
            MolOps::addHs(*mol, false, true); // addCoords=true

            // Generate multiple conformers with RMSD-based pruning during embedding
            DGeomHelpers::EmbedParameters embedParams = DGeomHelpers::ETKDGv3;
            embedParams.randomSeed = 42;
            embedParams.pruneRmsThresh = 0.25;  // RMSD dedup threshold (Å)
            embedParams.numThreads = 0;         // use all cores
            auto cids = DGeomHelpers::EmbedMultipleConfs(*mol, conformersPerForm, embedParams);
            if (cids.empty()) {
                // Fallback: single conformer
                int cid = DGeomHelpers::EmbedMolecule(*mol, embedParams);
                if (cid >= 0) cids.push_back(cid);
                else {
                    // Last resort: basic ETKDG without pruning
                    cid = embed_molecule(*mol);
                    if (cid >= 0) cids.push_back(cid);
                }
            }

            if (cids.empty()) continue;  // skip this form if embedding completely failed

            // MMFF94 minimize all conformers (maxIters=1000)
            std::vector<std::pair<int, double>> mmffResults;
            try { MMFF::MMFFOptimizeMoleculeConfs(*mol, mmffResults, /*numThreads=*/0,
                    /*maxIters=*/1000, /*mmffVariant=*/"MMFF94"); } catch (...) {}

            // Gasteiger charges
            try { computeGasteigerCharges(*mol); } catch (...) {}

            // Extract each conformer, filtering out failures, clashes, and bad energies
            std::vector<std::pair<int, double>> indexed;
            for (int ci = 0; ci < (int)cids.size(); ci++) {
                double e = std::numeric_limits<double>::quiet_NaN();
                if (ci < (int)mmffResults.size()) {
                    // mmffResults.first: 0=converged, 1=not converged, -1=setup failed
                    if (mmffResults[ci].first == 0) {
                        e = mmffResults[ci].second;
                    } else {
                        continue;  // skip non-converged and failed conformers
                    }
                }

                if (std::isnan(e) || e > 1e6) continue;

                // Steric clash check: non-bonded heavy atoms < 1.0 Å apart
                bool hasClash = false;
                const Conformer &conf = mol->getConformer(cids[ci]);
                int nAtoms = mol->getNumAtoms();
                for (int a = 0; a < nAtoms && !hasClash; a++) {
                    if (mol->getAtomWithIdx(a)->getAtomicNum() == 1) continue;
                    const auto &pa = conf.getAtomPos(a);
                    // Check for degenerate coords (atom at origin or identical to another)
                    if (pa.x == 0.0 && pa.y == 0.0 && pa.z == 0.0) { hasClash = true; break; }
                    for (int b = a + 1; b < nAtoms && !hasClash; b++) {
                        if (mol->getAtomWithIdx(b)->getAtomicNum() == 1) continue;
                        if (mol->getBondBetweenAtoms(a, b)) continue;  // skip bonded pairs
                        const auto &pb = conf.getAtomPos(b);
                        double d2 = (pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y) + (pa.z-pb.z)*(pa.z-pb.z);
                        if (d2 < 1.0) hasClash = true;  // 1.0 Å² = 1.0 Å threshold
                    }
                }
                if (hasClash) continue;

                indexed.push_back({ci, e});
            }

            // If all conformers rejected, try keeping best MMFF result as fallback
            if (indexed.empty() && !cids.empty()) {
                // Find first converged conformer
                for (int ci = 0; ci < (int)cids.size(); ci++) {
                    if (ci < (int)mmffResults.size() && mmffResults[ci].first == 0) {
                        indexed.push_back({ci, mmffResults[ci].second});
                        break;
                    }
                }
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
                    confLabel, form.smi, form.kind, form.hhPopulation
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
        // Step 5: Compute population weights = HH fraction × Boltzmann
        // ===================================================================
        // Henderson-Hasselbalch gives the population fraction for each protomer:
        //   fraction = product of per-site 1/(1+10^(pH-pKa)) or its complement
        // Boltzmann gives the conformer distribution within each form:
        //   w_i = exp(-E_i / (kB * T))
        // Combined: finalWeight_i = hhPopulation_i × boltzmann_i / Z
        //
        // This ensures that e.g. a 10% minority protonation state is visible
        // in the UI even if its conformers are higher energy.

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
            double boltzFactor = std::exp(-dE / kBT);
            double hhFactor = allConformers[i].hhPopulation;
            boltzWeights[i] = hhFactor * boltzFactor;
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

// ============================================================================
// MARK: - Fragment Decomposition
// ============================================================================

#include <GraphMol/Substruct/SubstructMatch.h>
#include <DataStructs/BitOps.h>
#include <queue>
#include <set>

DruseFragmentResult* druse_decompose_fragments(
    const char *smiles,
    const char *scaffoldSmarts
) {
    auto *result = new DruseFragmentResult();
    std::memset(result, 0, sizeof(DruseFragmentResult));
    result->success = false;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) {
            std::strncpy(result->errorMessage, "Failed to parse SMILES", 511);
            return result;
        }
        MolOps::addHs(*mol);
        MolOps::findSSSR(*mol);
        const auto &ringInfo = *mol->getRingInfo();

        // Collect heavy atom indices
        std::vector<int> heavyIdx;
        for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
            if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) {
                heavyIdx.push_back(i);
            }
        }
        int nHeavy = (int)heavyIdx.size();
        if (nHeavy == 0) {
            std::strncpy(result->errorMessage, "No heavy atoms", 511);
            return result;
        }

        // Map original atom idx → heavy atom index
        std::vector<int> origToHeavy(mol->getNumAtoms(), -1);
        for (int i = 0; i < nHeavy; i++) {
            origToHeavy[heavyIdx[i]] = i;
        }

        // Identify rotatable bonds between heavy atoms (non-ring single bonds, both ends have >= 2 heavy neighbors)
        std::set<int> rotatableBondSet;
        for (auto bondIt = mol->beginBonds(); bondIt != mol->endBonds(); ++bondIt) {
            const Bond *bond = *bondIt;
            if (bond->getBondTypeAsDouble() != 1.0) continue; // only single bonds
            if (ringInfo.numBondRings(bond->getIdx()) > 0) continue; // not in ring

            int a1 = bond->getBeginAtomIdx();
            int a2 = bond->getEndAtomIdx();
            if (mol->getAtomWithIdx(a1)->getAtomicNum() <= 1) continue;
            if (mol->getAtomWithIdx(a2)->getAtomicNum() <= 1) continue;

            // Both ends need at least 2 heavy neighbors (otherwise terminal — keep in same fragment)
            int heavyNeighA = 0, heavyNeighB = 0;
            for (auto ni = mol->getAtomNeighbors(mol->getAtomWithIdx(a1)); ni.first != ni.second; ++ni.first) {
                if (mol->getAtomWithIdx(*ni.first)->getAtomicNum() > 1) heavyNeighA++;
            }
            for (auto ni = mol->getAtomNeighbors(mol->getAtomWithIdx(a2)); ni.first != ni.second; ++ni.first) {
                if (mol->getAtomWithIdx(*ni.first)->getAtomicNum() > 1) heavyNeighB++;
            }
            if (heavyNeighA >= 2 && heavyNeighB >= 2) {
                rotatableBondSet.insert(bond->getIdx());
            }
        }

        // Build adjacency among heavy atoms, excluding rotatable bonds
        std::vector<std::vector<int>> adj(nHeavy);
        // Also track which bonds are rotatable for connectivity info
        struct RotBondInfo { int heavyA; int heavyB; int bondIdx; };
        std::vector<RotBondInfo> rotBonds;

        for (auto bondIt = mol->beginBonds(); bondIt != mol->endBonds(); ++bondIt) {
            const Bond *bond = *bondIt;
            int a1 = origToHeavy[bond->getBeginAtomIdx()];
            int a2 = origToHeavy[bond->getEndAtomIdx()];
            if (a1 < 0 || a2 < 0) continue;

            if (rotatableBondSet.count(bond->getIdx())) {
                rotBonds.push_back({a1, a2, (int)bond->getIdx()});
                continue; // don't add to adjacency — these cut fragments
            }
            adj[a1].push_back(a2);
            adj[a2].push_back(a1);
        }

        // Flood-fill to assign fragment membership
        std::vector<int> fragMembership(nHeavy, -1);
        int numFrags = 0;
        for (int i = 0; i < nHeavy; i++) {
            if (fragMembership[i] >= 0) continue;
            int fid = numFrags++;
            std::queue<int> q;
            q.push(i);
            fragMembership[i] = fid;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : adj[u]) {
                    if (fragMembership[v] < 0) {
                        fragMembership[v] = fid;
                        q.push(v);
                    }
                }
            }
        }

        // Fragment sizes
        std::vector<int> fragSizes(numFrags, 0);
        for (int i = 0; i < nHeavy; i++) {
            fragSizes[fragMembership[i]]++;
        }

        // Determine anchor: largest fragment (or scaffold-containing if specified)
        int anchorIdx = 0;
        int maxSize = 0;
        for (int f = 0; f < numFrags; f++) {
            if (fragSizes[f] > maxSize) { maxSize = fragSizes[f]; anchorIdx = f; }
        }

        // If scaffold SMARTS provided, find which fragment best matches it
        if (scaffoldSmarts && std::strlen(scaffoldSmarts) > 0) {
            std::unique_ptr<ROMol> pattern(SmartsToMol(scaffoldSmarts));
            if (pattern) {
                std::vector<MatchVectType> matches;
                SubstructMatch(*mol, *pattern, matches);
                if (!matches.empty()) {
                    // Count matched atoms per fragment
                    std::vector<int> fragMatchCount(numFrags, 0);
                    for (const auto &pair : matches[0]) {
                        int origIdx = pair.second;
                        int hIdx = origToHeavy[origIdx];
                        if (hIdx >= 0 && hIdx < nHeavy) {
                            fragMatchCount[fragMembership[hIdx]]++;
                        }
                    }
                    int bestFrag = anchorIdx;
                    int bestCount = 0;
                    for (int f = 0; f < numFrags; f++) {
                        if (fragMatchCount[f] > bestCount) {
                            bestCount = fragMatchCount[f];
                            bestFrag = f;
                        }
                    }
                    anchorIdx = bestFrag;
                }
            }
        }

        // Build fragment connectivity from rotatable bonds
        struct Connection { int parentFrag; int childFrag; int atomA; int atomB; };
        std::vector<Connection> connections;
        for (const auto &rb : rotBonds) {
            int fA = fragMembership[rb.heavyA];
            int fB = fragMembership[rb.heavyB];
            if (fA != fB) {
                connections.push_back({fA, fB, rb.heavyA, rb.heavyB});
            }
        }

        // BFS from anchor to order fragments (parent → child direction)
        std::vector<bool> visited(numFrags, false);
        std::vector<Connection> orderedConn;
        std::queue<int> bfsQ;
        bfsQ.push(anchorIdx);
        visited[anchorIdx] = true;
        while (!bfsQ.empty()) {
            int curFrag = bfsQ.front(); bfsQ.pop();
            for (const auto &c : connections) {
                int other = -1;
                Connection oriented = c;
                if (c.parentFrag == curFrag && !visited[c.childFrag]) {
                    other = c.childFrag;
                    oriented = {curFrag, other, c.atomA, c.atomB};
                } else if (c.childFrag == curFrag && !visited[c.parentFrag]) {
                    other = c.parentFrag;
                    oriented = {curFrag, other, c.atomB, c.atomA};
                }
                if (other >= 0) {
                    visited[other] = true;
                    orderedConn.push_back(oriented);
                    bfsQ.push(other);
                }
            }
        }

        // Compute fragment centroids (using heavy atom coordinates from the original mol)
        // Need 3D coords — generate if not present
        if (mol->getNumConformers() == 0) {
            MolOps::removeHs(*mol);
            MolOps::addHs(*mol);
            auto embedParams = DGeomHelpers::srETKDGv3;
            DGeomHelpers::EmbedMolecule(*mol, embedParams);
        }
        std::vector<float> centroids(numFrags * 3, 0.0f);
        if (mol->getNumConformers() > 0) {
            const auto &conf = mol->getConformer(0);
            for (int i = 0; i < nHeavy; i++) {
                int origAtom = heavyIdx[i];
                int frag = fragMembership[i];
                auto pos = conf.getAtomPos(origAtom);
                centroids[frag * 3 + 0] += (float)pos.x;
                centroids[frag * 3 + 1] += (float)pos.y;
                centroids[frag * 3 + 2] += (float)pos.z;
            }
            for (int f = 0; f < numFrags; f++) {
                if (fragSizes[f] > 0) {
                    centroids[f * 3 + 0] /= fragSizes[f];
                    centroids[f * 3 + 1] /= fragSizes[f];
                    centroids[f * 3 + 2] /= fragSizes[f];
                }
            }
        }

        // Fill result
        result->numHeavyAtoms = nHeavy;
        result->numFragments = numFrags;
        result->anchorFragmentIdx = anchorIdx;
        result->fragmentMembership = new int32_t[nHeavy];
        for (int i = 0; i < nHeavy; i++) result->fragmentMembership[i] = fragMembership[i];
        result->fragmentSizes = new int32_t[numFrags];
        for (int f = 0; f < numFrags; f++) result->fragmentSizes[f] = fragSizes[f];

        int nConn = (int)orderedConn.size();
        result->numConnections = nConn;
        result->connections = new int32_t[nConn * 4];
        for (int i = 0; i < nConn; i++) {
            result->connections[i * 4 + 0] = orderedConn[i].parentFrag;
            result->connections[i * 4 + 1] = orderedConn[i].childFrag;
            result->connections[i * 4 + 2] = orderedConn[i].atomA;
            result->connections[i * 4 + 3] = orderedConn[i].atomB;
        }

        result->centroids = new float[numFrags * 3];
        std::memcpy(result->centroids, centroids.data(), numFrags * 3 * sizeof(float));

        result->success = true;
        return result;
    } catch (const std::exception &e) {
        std::strncpy(result->errorMessage, e.what(), 511);
        return result;
    } catch (...) {
        std::strncpy(result->errorMessage, "Unknown error in fragment decomposition", 511);
        return result;
    }
}

void druse_free_fragment_result(DruseFragmentResult *result) {
    if (!result) return;
    delete[] result->fragmentMembership;
    delete[] result->fragmentSizes;
    delete[] result->connections;
    delete[] result->centroids;
    delete result;
}

// ============================================================================
// MARK: - Scaffold Matching & Tanimoto Similarity
// ============================================================================

DruseScaffoldMatch* druse_match_scaffold(const char *smiles, const char *scaffoldSmarts) {
    auto *result = new DruseScaffoldMatch();
    std::memset(result, 0, sizeof(DruseScaffoldMatch));
    result->hasMatch = false;
    result->tanimotoSimilarity = 0.0f;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return result;

        // Substructure match
        std::unique_ptr<ROMol> pattern(SmartsToMol(scaffoldSmarts));
        if (!pattern) return result;

        std::vector<MatchVectType> matches;
        SubstructMatch(*mol, *pattern, matches);

        if (!matches.empty()) {
            result->hasMatch = true;
            // Collect matched heavy atom indices
            std::vector<int32_t> matchedHeavy;
            // Build heavy-atom index mapping
            std::vector<int> heavyMap; // orig idx of each heavy atom
            for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
                if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) {
                    heavyMap.push_back(i);
                }
            }
            std::map<int, int> origToHeavy;
            for (int i = 0; i < (int)heavyMap.size(); i++) {
                origToHeavy[heavyMap[i]] = i;
            }
            for (const auto &pair : matches[0]) {
                auto it = origToHeavy.find(pair.second);
                if (it != origToHeavy.end()) {
                    matchedHeavy.push_back(it->second);
                }
            }
            result->matchCount = (int32_t)matchedHeavy.size();
            result->matchedAtomIndices = new int32_t[matchedHeavy.size()];
            std::memcpy(result->matchedAtomIndices, matchedHeavy.data(), matchedHeavy.size() * sizeof(int32_t));
        }

        // Tanimoto similarity (Morgan radius=2, 2048 bits)
        std::unique_ptr<RWMol> scaffoldMol(SmilesToMol(scaffoldSmarts));
        if (scaffoldMol) {
            std::unique_ptr<ExplicitBitVect> fp1(
                MorganFingerprints::getFingerprintAsBitVect(*mol, 2, 2048));
            std::unique_ptr<ExplicitBitVect> fp2(
                MorganFingerprints::getFingerprintAsBitVect(*scaffoldMol, 2, 2048));
            if (fp1 && fp2) {
                result->tanimotoSimilarity = (float)TanimotoSimilarity(*fp1, *fp2);
            }
        }

        return result;
    } catch (...) {
        return result;
    }
}

void druse_free_scaffold_match(DruseScaffoldMatch *result) {
    if (!result) return;
    delete[] result->matchedAtomIndices;
    delete result;
}

float druse_tanimoto_similarity(const char *smiles1, const char *smiles2) {
    try {
        std::unique_ptr<RWMol> mol1(SmilesToMol(smiles1));
        std::unique_ptr<RWMol> mol2(SmilesToMol(smiles2));
        if (!mol1 || !mol2) return 0.0f;

        std::unique_ptr<ExplicitBitVect> fp1(
            MorganFingerprints::getFingerprintAsBitVect(*mol1, 2, 2048));
        std::unique_ptr<ExplicitBitVect> fp2(
            MorganFingerprints::getFingerprintAsBitVect(*mol2, 2, 2048));
        if (!fp1 || !fp2) return 0.0f;

        return (float)TanimotoSimilarity(*fp1, *fp2);
    } catch (...) {
        return 0.0f;
    }
}

// ============================================================================
// MARK: - Ionizable Site Detection
// ============================================================================

DruseIonSiteResult* druse_detect_ionizable_sites(const char *smiles) {
    auto *result = new DruseIonSiteResult();
    result->sites = nullptr;
    result->count = 0;

    if (!smiles || !smiles[0]) return result;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return result;

        auto sites = detectIonSitesInternal(*mol);
        if (sites.empty()) return result;

        result->count = (int32_t)sites.size();
        result->sites = new DruseIonSite[sites.size()];

        for (size_t i = 0; i < sites.size(); i++) {
            auto &[atomIdx, groupIdx, isAcid, defaultPKa] = sites[i];
            auto &out = result->sites[i];
            out.atomIdx = atomIdx;
            out.isAcid = isAcid;
            out.defaultPKa = defaultPKa;
            snprintf(out.groupName, sizeof(out.groupName), "%s", kIonizableGroups[groupIdx].name);
        }
    } catch (...) {}

    return result;
}

void druse_free_ion_sites(DruseIonSiteResult *result) {
    if (!result) return;
    delete[] result->sites;
    delete result;
}

// ============================================================================
// MARK: - Per-Site Protomer Pair Generation
// ============================================================================

DruseSiteProtomerPair* druse_generate_site_protomers(
    const char *smiles, int32_t atomIdx, bool isAcid
) {
    auto *result = new DruseSiteProtomerPair();
    memset(result, 0, sizeof(DruseSiteProtomerPair));

    if (!smiles || !smiles[0]) {
        result->success = false;
        snprintf(result->errorMessage, sizeof(result->errorMessage), "Empty SMILES");
        return result;
    }

    try {
        std::unique_ptr<RWMol> parentMol(SmilesToMol(smiles));
        if (!parentMol) {
            result->success = false;
            snprintf(result->errorMessage, sizeof(result->errorMessage), "Invalid SMILES");
            return result;
        }

        // Compute total formal charge of parent
        int parentCharge = 0;
        for (auto atom : parentMol->atoms()) parentCharge += atom->getFormalCharge();

        // --- Protonated form ---
        auto protMol = std::make_unique<RWMol>(*parentMol);
        // --- Deprotonated form ---
        auto deprotMol = std::make_unique<RWMol>(*parentMol);

        if (isAcid) {
            // Acid: parent is protonated (has H). Deprotonated = remove H, charge -1
            Atom *depAtom = deprotMol->getAtomWithIdx(atomIdx);
            int nH = depAtom->getTotalNumHs();
            if (nH > 0) {
                depAtom->setNumExplicitHs(nH - 1);
                depAtom->setFormalCharge(depAtom->getFormalCharge() - 1);
            }
            result->protonatedCharge = parentCharge;
            result->deprotonatedCharge = parentCharge - 1;
        } else {
            // Base: parent is deprotonated (neutral). Protonated = add H, charge +1
            Atom *protAtom = protMol->getAtomWithIdx(atomIdx);
            int nH = protAtom->getTotalNumHs();
            protAtom->setNumExplicitHs(nH + 1);
            protAtom->setFormalCharge(protAtom->getFormalCharge() + 1);
            result->protonatedCharge = parentCharge + 1;
            result->deprotonatedCharge = parentCharge;
        }

        // Sanitize both
        try { MolOps::sanitizeMol(*protMol); } catch (...) {}
        try { MolOps::sanitizeMol(*deprotMol); } catch (...) {}

        // Add hydrogens and generate 3D for both
        auto prepare3D = [](RWMol &mol) {
            MolOps::addHs(mol, false, true);
            int cid = embed_molecule(mol);
            if (cid < 0) {
                try { RDDepict::compute2DCoords(mol); } catch (...) {}
            }
            // MMFF minimize
            std::vector<std::pair<int,double>> mmffRes;
            try { MMFF::MMFFOptimizeMoleculeConfs(mol, mmffRes, 0, 500); } catch (...) {}
            // Gasteiger charges
            try { computeGasteigerCharges(mol); } catch (...) {}
        };

        prepare3D(*protMol);
        prepare3D(*deprotMol);

        result->protonated = mol_to_result(*protMol, "protonated");
        result->deprotonated = mol_to_result(*deprotMol, "deprotonated");
        result->success = true;

    } catch (const std::exception &e) {
        result->success = false;
        snprintf(result->errorMessage, sizeof(result->errorMessage), "%s", e.what());
    } catch (...) {
        result->success = false;
        snprintf(result->errorMessage, sizeof(result->errorMessage), "Unknown error");
    }

    return result;
}

void druse_free_site_protomer_pair(DruseSiteProtomerPair *result) {
    if (!result) return;
    druse_free_molecule_result(result->protonated);
    druse_free_molecule_result(result->deprotonated);
    delete result;
}

// ============================================================================
// MARK: - Ensemble with pKa Overrides
// ============================================================================

DruseEnsembleResult* druse_prepare_ligand_ensemble_v2(
    const char *smiles, const char *name,
    double pH, double pkaThreshold,
    int32_t maxTautomers, int32_t maxProtomers,
    double energyCutoff, int32_t conformersPerForm,
    double temperature,
    const double *sitePKa, int32_t nSitePKa
) {
    // If no pKa overrides, delegate to the original function
    if (!sitePKa || nSitePKa <= 0) {
        return druse_prepare_ligand_ensemble(smiles, name, pH, pkaThreshold,
            maxTautomers, maxProtomers, energyCutoff, conformersPerForm, temperature);
    }

    // This version replaces the hardcoded pKa detection with caller-provided values.
    // The algorithm is identical to druse_prepare_ligand_ensemble except for step 1.
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

        // Step 1: Use CALLER-PROVIDED pKa values for wash + protomer enumeration
        auto allSites = detectIonSitesInternal(*parentMol);

        // Phase A: WASH — apply dominant protonation using computed pKa
        auto washedMol = std::make_unique<RWMol>(*parentMol);
        struct AmbSite { int groupIdx; int atomIdx; bool isAcid; double pKa; };
        std::vector<AmbSite> ambSites;

        for (size_t i = 0; i < allSites.size() && (int)i < nSitePKa; i++) {
            auto &[atomIdx, groupIdx, isAcid, _] = allSites[i];
            double computedPKa = sitePKa[i];
            double deltaPH = pH - computedPKa;

            if (isAcid && deltaPH > pkaThreshold) {
                // pH well above pKa: deprotonate (dominant)
                Atom *atom = washedMol->getAtomWithIdx(atomIdx);
                int nH = atom->getTotalNumHs();
                if (nH > 0) {
                    atom->setNumExplicitHs(nH - 1);
                    atom->setFormalCharge(atom->getFormalCharge() - 1);
                }
            } else if (!isAcid && deltaPH < -pkaThreshold) {
                // pH well below pKa: protonate (dominant)
                Atom *atom = washedMol->getAtomWithIdx(atomIdx);
                int nH = atom->getTotalNumHs();
                atom->setNumExplicitHs(nH + 1);
                atom->setFormalCharge(atom->getFormalCharge() + 1);
            } else if (std::abs(deltaPH) <= pkaThreshold) {
                // Near boundary: enumerate both states
                ambSites.push_back({groupIdx, atomIdx, isAcid, computedPKa});
            }
        }
        try { MolOps::sanitizeMol(*washedMol); } catch (...) {
            washedMol = std::make_unique<RWMol>(*parentMol);
        }

        struct ChemicalForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            int kind;
            double hhPopulation = 1.0;
        };
        std::vector<ChemicalForm> allForms;
        std::set<std::string> seenSMILES;

        // Phase B: Generate protomer combinations from ambiguous sites
        int nSites = std::min((int)ambSites.size(), 10);
        int nCombos = std::min(1 << nSites, (int)maxProtomers);

        struct ProtomerForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            double hhPopulation;
        };
        std::vector<ProtomerForm> protomers;

        // combo=0 = washed molecule, higher combos toggle ambiguous sites
        for (int combo = 0; combo < nCombos; combo++) {
            auto rw = std::make_unique<RWMol>(*washedMol);
            std::string comboLabel;
            double hhPop = 1.0;

            for (int s = 0; s < nSites; s++) {
                const auto &site = ambSites[s];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.pKa));
                double fracDeprot = 1.0 - fracProt;

                bool toggle = (combo >> s) & 1;
                if (!toggle) {
                    hhPop *= site.isAcid ? fracProt : fracDeprot;
                    continue;
                }

                Atom *atom = rw->getAtomWithIdx(site.atomIdx);
                if (site.isAcid) {
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                    hhPop *= fracDeprot;
                } else {
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                    hhPop *= fracProt;
                }
                if (!comboLabel.empty()) comboLabel += "+";
                comboLabel += std::string(site.isAcid ? "deprot_" : "prot_") +
                    kIonizableGroups[site.groupIdx].name;
            }

            try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
            std::string pSmi = MolToSmiles(*rw);
            if (seenSMILES.count(pSmi)) continue;
            seenSMILES.insert(pSmi);

            if (comboLabel.empty()) comboLabel = "Parent";
            protomers.push_back({pSmi, std::move(rw), comboLabel, hhPop});
        }

        if (protomers.empty()) {
            auto rw = std::make_unique<RWMol>(*washedMol);
            std::string wSmi = MolToSmiles(*rw);
            seenSMILES.insert(wSmi);
            protomers.push_back({wSmi, std::move(rw), "Parent", 1.0});
        }

        // Step 2: Tautomer enumeration (always add protomer first, then its tautomers)
        int maxTotalForms = maxProtomers * maxTautomers;
        if (maxTotalForms > 200) maxTotalForms = 200;

        for (auto &prot : protomers) {
            if ((int)allForms.size() >= maxTotalForms) break;

            // Always add the protomer itself first
            if (!seenSMILES.count(prot.smi)) seenSMILES.insert(prot.smi);
            {
                auto rw = std::make_unique<RWMol>(*prot.mol);
                int kind = (prot.label != "Parent") ? 2 : 0;
                allForms.push_back({prot.smi, std::move(rw), prot.label, kind, prot.hhPopulation});
            }

            // Then enumerate tautomers
            try {
                MolStandardize::TautomerEnumerator enumerator;
                enumerator.setMaxTautomers(std::max(maxTautomers, 2));
                enumerator.setMaxTransforms(200);
                auto tautResult = enumerator.enumerate(*prot.mol);

                int tautCount = 0;
                for (const auto &taut : tautResult) {
                    if (tautCount >= maxTautomers || (int)allForms.size() >= maxTotalForms) break;
                    auto rw = std::make_unique<RWMol>(*taut);
                    try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
                    std::string tSmi = MolToSmiles(*rw);
                    if (seenSMILES.count(tSmi)) continue;
                    seenSMILES.insert(tSmi);

                    bool isFromProtomer = (prot.label != "Parent");
                    int kind = isFromProtomer ? 3 : 1;

                    std::string label = prot.label;
                    if (label != "Parent") label += "_";
                    else label = "";
                    label += "Taut" + std::to_string(tautCount + 1);

                    allForms.push_back({tSmi, std::move(rw), label, kind, prot.hhPopulation});
                    tautCount++;
                }
            } catch (...) {
                // Tautomer enumeration failed; protomer already added above
            }
        }

        if (allForms.empty()) {
            result->success = false;
            snprintf(result->errorMessage, 512, "No valid chemical forms generated");
            return result;
        }
        result->numForms = (int)allForms.size();

        // Steps 3-6: Conformer generation, energy filter, Boltzmann (reuse from original)
        struct PreparedConformer {
            std::unique_ptr<RWMol> mol;
            double energy;
            int formIdx;
            int confIdx;
            std::string label;
            std::string smi;
            int kind;
            double hhPopulation;  // Henderson-Hasselbalch population from protomer
        };
        std::vector<PreparedConformer> allConformers;

        for (int fi = 0; fi < (int)allForms.size(); fi++) {
            auto &form = allForms[fi];
            auto mol = std::make_unique<RWMol>(*form.mol);
            MolOps::addHs(*mol, false, true);

            // Conformers with RMSD pruning
            DGeomHelpers::EmbedParameters ep = DGeomHelpers::ETKDGv3;
            ep.randomSeed = 42;
            ep.pruneRmsThresh = 0.25;
            ep.numThreads = 0;
            auto cids = DGeomHelpers::EmbedMultipleConfs(*mol, conformersPerForm, ep);
            if (cids.empty()) {
                int cid = DGeomHelpers::EmbedMolecule(*mol, ep);
                if (cid >= 0) cids.push_back(cid);
                else {
                    cid = embed_molecule(*mol);
                    if (cid >= 0) cids.push_back(cid);
                }
            }
            if (cids.empty()) continue;

            std::vector<std::pair<int, double>> mmffResults;
            try { MMFF::MMFFOptimizeMoleculeConfs(*mol, mmffResults, 0, 1000, "MMFF94"); } catch (...) {}
            try { computeGasteigerCharges(*mol); } catch (...) {}

            std::vector<std::pair<int, double>> indexed;
            for (int ci = 0; ci < (int)cids.size(); ci++) {
                double e = std::numeric_limits<double>::quiet_NaN();
                if (ci < (int)mmffResults.size()) {
                    if (mmffResults[ci].first == 0) e = mmffResults[ci].second;
                    else continue;  // skip non-converged
                }
                if (std::isnan(e) || e > 1e6) continue;

                bool hasClash = false;
                const Conformer &conf = mol->getConformer(cids[ci]);
                int nAtoms = mol->getNumAtoms();
                for (int a = 0; a < nAtoms && !hasClash; a++) {
                    if (mol->getAtomWithIdx(a)->getAtomicNum() == 1) continue;
                    const auto &pa = conf.getAtomPos(a);
                    if (pa.x == 0.0 && pa.y == 0.0 && pa.z == 0.0) { hasClash = true; break; }
                    for (int b = a + 1; b < nAtoms && !hasClash; b++) {
                        if (mol->getAtomWithIdx(b)->getAtomicNum() == 1) continue;
                        if (mol->getBondBetweenAtoms(a, b)) continue;
                        const auto &pb = conf.getAtomPos(b);
                        double d2 = (pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y) + (pa.z-pb.z)*(pa.z-pb.z);
                        if (d2 < 1.0) hasClash = true;
                    }
                }
                if (hasClash) continue;
                indexed.push_back({ci, e});
            }

            if (indexed.empty() && !cids.empty()) {
                for (int ci = 0; ci < (int)cids.size(); ci++) {
                    if (ci < (int)mmffResults.size() && mmffResults[ci].first == 0) {
                        indexed.push_back({ci, mmffResults[ci].second});
                        break;
                    }
                }
            }

            std::sort(indexed.begin(), indexed.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });

            for (int ri = 0; ri < (int)indexed.size(); ri++) {
                int confId = cids[indexed[ri].first];
                double energy = indexed[ri].second;

                auto confMol = std::make_unique<RWMol>(*mol);
                if (confMol->getNumConformers() > 1) {
                    std::vector<unsigned int> toRemove;
                    for (auto it = confMol->beginConformers(); it != confMol->endConformers(); ++it) {
                        if ((int)(*it)->getId() != confId) toRemove.push_back((*it)->getId());
                    }
                    for (auto cid : toRemove) confMol->removeConformer(cid);
                }

                std::string confLabel = form.label;
                if (indexed.size() > 1) confLabel += "_Conf" + std::to_string(ri + 1);

                allConformers.push_back({std::move(confMol), energy, fi, ri, confLabel, form.smi, form.kind, form.hhPopulation});
            }
        }

        // Energy cutoff
        if (energyCutoff > 0 && !allConformers.empty()) {
            double bestE = 1e30;
            for (const auto &c : allConformers)
                if (!std::isnan(c.energy) && c.energy < bestE) bestE = c.energy;
            allConformers.erase(
                std::remove_if(allConformers.begin(), allConformers.end(),
                    [&](const PreparedConformer &c) { return !std::isnan(c.energy) && c.energy > bestE + energyCutoff; }),
                allConformers.end());
        }

        // Population weights = HH fraction × Boltzmann (same as v1)
        double kBT = 0.001987204 * temperature;
        if (kBT < 1e-10) kBT = 0.593;
        double minE = 1e30;
        for (const auto &c : allConformers)
            if (!std::isnan(c.energy) && c.energy < minE) minE = c.energy;

        std::vector<double> boltzWeights(allConformers.size(), 0.0);
        double partitionZ = 0.0;
        for (size_t i = 0; i < allConformers.size(); i++) {
            double dE = std::isnan(allConformers[i].energy) ? 50.0 : (allConformers[i].energy - minE);
            double boltzFactor = std::exp(-dE / kBT);
            double hhFactor = allConformers[i].hhPopulation;
            boltzWeights[i] = hhFactor * boltzFactor;
            partitionZ += boltzWeights[i];
        }
        if (partitionZ > 0) for (auto &w : boltzWeights) w /= partitionZ;

        // Build result
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
