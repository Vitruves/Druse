#include "druse_core_internal.h"

// Shared ionization state lookup and detection helpers.

#include <GraphMol/Substruct/SubstructMatch.h>
#include <set>
#include <tuple>

// Shared ionizable group table (used by protomer enumeration, site detection, ensemble)
const IonizableGroupDef kIonizableGroups[] = {
    // =====================================================================
    // ACIDS — ordered most specific first (first atom-match wins)
    // Sources: Williams pKa tables, Evans/Ripin DMSO→H2O, Jencks/Westheimer,
    //          FLogD acidbase.csv (2445 entries), cchem acidbase.c
    // =====================================================================

    // ---- Sulfonic acids (pKa < 0, always deprotonated at physiological pH) ----
    {"Trifluoromethane-sulfonic", "[OX2H1]S(=O)(=O)C(F)(F)F", -14.0, true},
    {"Sulfonic acid",    "[OX2H1]S(=O)(=O)",          -2.6,  true},

    // ---- Phosphoric / Phosphonic acids ----
    {"Phosphoric acid",  "[OX2H1]P(=O)([OX2])[OX2]",  2.1,  true},
    {"Phosphonate",      "[OX2H1]P(=O)([CX4])",        2.4,  true},
    {"Phosphoric ester", "[OX2H1]P(=O)",               1.5,  true},

    // ---- Carboxylic acids — specific subtypes first ----
    {"Trifluoroacetic",  "[CX3](=O)([OX2H1])C(F)(F)F", 0.5,  true},
    {"Trichloroacetic",  "[CX3](=O)([OX2H1])C(Cl)(Cl)Cl", 0.65, true},
    {"Difluoroacetic",   "[CX3](=O)([OX2H1])C([F,Cl])F", 1.35, true},
    {"Dichloroacetic",   "[CX3](=O)([OX2H1])C(Cl)Cl",  1.29, true},
    {"Chloroacetic",     "[CX3](=O)([OX2H1])CCl",      2.86, true},
    {"Fluoroacetic",     "[CX3](=O)([OX2H1])CF",       2.66, true},
    {"Alpha-keto acid",  "[CX3](=O)([OX2H1])C(=O)",    2.5,  true},
    {"Alpha-cyano acid", "[CX3](=O)([OX2H1])CC#N",     2.5,  true},
    {"Oxalic acid",      "[CX3](=O)([OX2H1])C(=O)[OX2H1]", 1.25, true},
    {"Maleic acid",      "[CX3](=O)([OX2H1])/C=C\\C(=O)[OX2H1]", 1.92, true},
    {"o-Nitrobenzoic",   "[CX3](=O)([OX2H1])c1ccccc1[NX3+](=O)[O-]", 2.17, true},
    {"ArCOOH p-NO2",     "[CX3](=O)([OX2H1])c1ccc([NX3+](=O)[O-])cc1", 3.44, true},
    {"ArCOOH m-NO2",     "[CX3](=O)([OX2H1])c1cccc([NX3+](=O)[O-])c1", 3.45, true},
    {"ArCOOH p-Cl",      "[CX3](=O)([OX2H1])c1ccc(Cl)cc1", 3.99, true},
    {"ArCOOH p-F",       "[CX3](=O)([OX2H1])c1ccc(F)cc1",  4.14, true},
    {"ArCOOH p-Br",      "[CX3](=O)([OX2H1])c1ccc(Br)cc1", 4.00, true},
    {"ArCOOH p-CN",      "[CX3](=O)([OX2H1])c1ccc(C#N)cc1", 3.55, true},
    {"ArCOOH p-CF3",     "[CX3](=O)([OX2H1])c1ccc(C(F)(F)F)cc1", 3.79, true},
    {"ArCOOH p-OMe",     "[CX3](=O)([OX2H1])c1ccc(OC)cc1", 4.47, true},
    {"ArCOOH p-OH",      "[CX3](=O)([OX2H1])c1ccc(O)cc1",  4.58, true},
    {"ArCOOH p-NH2",     "[CX3](=O)([OX2H1])c1ccc(N)cc1",  4.92, true},
    {"ArCOOH p-NMe2",    "[CX3](=O)([OX2H1])c1ccc(N(C)C)cc1", 5.03, true},
    {"ArCOOH p-Me",      "[CX3](=O)([OX2H1])c1ccc(C)cc1",  4.34, true},
    {"Benzoic acid",     "[CX3](=O)([OX2H1])c",       4.20,  true},
    {"Succinic acid",    "[CX3](=O)([OX2H1])CCC(=O)[OX2H1]", 4.19, true},
    {"Glutaric acid",    "[CX3](=O)([OX2H1])CCCC(=O)[OX2H1]", 4.34, true},
    {"Acetic acid",      "[CX3](=O)([OX2H1])[CX4H3]", 4.76, true},
    {"Propionic acid",   "[CX3](=O)([OX2H1])[CX4H2][CX4]", 4.88, true},
    {"Formic acid",      "[CX3H1](=O)[OX2H1]",        3.77, true},
    {"Carboxylic acid",  "[CX3](=O)[OX2H1]",           4.0,  true},

    // ---- Tetrazoles (bioisostere of COOH) ----
    {"Tetrazole",        "[nH1]1nnn[nH0]1",            4.9,  true},
    {"Tetrazole 2",      "[nH1]1nn[nH0]n1",            4.9,  true},

    // ---- Sulfonamides (N-H) — acidity depends on substituent ----
    {"Saccharin NH",     "[nH1]1c2ccccc2S(=O)(=O)1",   1.6,  true},
    {"ArSO2NHAr",        "[NX3H1](S(=O)(=O)c)c",       6.3,  true},
    {"CF3-sulfonamide",  "[NX3H1]S(=O)(=O)C(F)(F)F",   5.8,  true},
    {"ArSO2NH2",         "[NX3H2]S(=O)(=O)c",          10.0, true},
    {"Sulfonamide NH",   "[NX3H1]S(=O)(=O)",           10.0, true},

    // ---- Phenols — specific subtypes first ----
    {"2,4,6-Trinitro-phenol", "[OX2H1]c1c([NX3+](=O)[O-])cc([NX3+](=O)[O-])cc1[NX3+](=O)[O-]", 0.3, true},
    {"2,4-Dinitrophenol","[OX2H1]c1ccc([NX3+](=O)[O-])cc1[NX3+](=O)[O-]", 4.1, true},
    {"p-Nitrophenol",    "[OX2H1]c1ccc([NX3+](=O)[O-])cc1", 7.14, true},
    {"m-Nitrophenol",    "[OX2H1]c1cccc([NX3+](=O)[O-])c1", 8.35, true},
    {"o-Nitrophenol",    "[OX2H1]c1ccccc1[NX3+](=O)[O-]",   7.23, true},
    {"p-Fluorophenol",   "[OX2H1]c1ccc(F)cc1",         9.95, true},
    {"p-Chlorophenol",   "[OX2H1]c1ccc(Cl)cc1",        9.38, true},
    {"p-Bromophenol",    "[OX2H1]c1ccc(Br)cc1",        9.34, true},
    {"p-Cyanophenol",    "[OX2H1]c1ccc(C#N)cc1",       7.95, true},
    {"p-CF3-phenol",     "[OX2H1]c1ccc(C(F)(F)F)cc1",  8.68, true},
    {"p-Methoxyphenol",  "[OX2H1]c1ccc(OC)cc1",       10.20, true},
    {"p-Aminophenol",    "[OX2H1]c1ccc(N)cc1",        10.30, true},
    {"1-Naphthol",       "[OX2H1]c1cccc2ccccc12",      9.34, true},
    {"2-Naphthol",       "[OX2H1]c1ccc2ccccc2c1",      9.51, true},
    {"Catechol",         "[OX2H1]c1ccccc1[OX2H1]",     9.45, true},
    {"Resorcinol",       "[OX2H1]c1cccc([OX2H1])c1",   9.15, true},
    {"Hydroquinone",     "[OX2H1]c1ccc([OX2H1])cc1",   9.85, true},
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
    {"Triazole NH 123",  "[nH1]1nncc1",                 9.4,  true},
    {"Triazole NH 124",  "[nH1]1ncnc1",                10.0,  true},
    {"Benzimidazole NH", "[nH1]1cnc2ccccc12",          12.0,  true},
    {"Indole NH",        "[nH1]1ccc2ccccc12",          17.0,  true},
    {"Pyrrole NH",       "[nH1]1cccc1",                17.5,  true},

    // ---- Boronic acids ----
    {"Boronic acid",     "[OX2H1]B([OX2H1])",           8.8,  true},

    // ---- Sulfinic acids ----
    {"Sulfinic acid",    "[OX2H1]S(=O)c",               1.8,  true},
    {"Sulfinic acid ali","[OX2H1]S(=O)[CX4]",           2.0,  true},

    // ---- N-Acyl sulfonamides (very acidic NH, drug-relevant) ----
    {"N-acyl sulfonamide","[NH1](C(=O))S(=O)(=O)",      2.5,  true},
    {"Sulfonylcarboxamide","[NH1](S(=O)(=O))C(=O)c",    3.0,  true},

    // ---- Carbon acids (C-H acidity, drug-relevant enolizable systems) ----
    {"Acetylacetone CH", "[CH2](C(=O)C)C(=O)C",         8.95, true},
    {"ArCO-CH2-COAr",   "[CH2](C(=O)c)C(=O)c",         8.5,  true},
    {"ArCO-CH2-COCH3",  "[CH2](C(=O)c)C(=O)C",         9.0,  true},
    {"Malononitrile CH", "[CH2](C#N)C#N",              11.2,  true},
    {"Cyanoacetate CH",  "[CH2](C#N)C(=O)O",           10.7,  true},
    {"Cyanoacetamide CH","[CH2](C#N)C(=O)N",           11.5,  true},
    {"Malonate diester", "[CH2](C(=O)OC)C(=O)OC",     12.9,  true},
    {"Malonic acid CH",  "[CH2](C(=O)[OH])C(=O)[OH]",  2.83, true},
    {"Nitromethane",     "[CH3][NX3+](=O)[O-]",        10.2,  true},
    {"Nitroethane",      "[CH2]([CX4])[NX3+](=O)[O-]", 8.6,  true},
    {"Dinitromethane",   "[CH1]([NX3+](=O)[O-])[NX3+](=O)[O-]", 3.6, true},
    {"Phenylnitromethane","[CH2](c)[NX3+](=O)[O-]",    7.1,  true},
    {"Bis-sulfonylmethane","[CH2](S(=O)(=O))S(=O)(=O)", 12.3, true},

    // ---- Enols (drug-relevant) ----
    {"Ascorbic acid",    "OC1OC(=O)C(O)=C1O",          4.1,  true},
    {"Squaric acid",     "[OH]C1=C([OH])C(=O)C1=O",    1.5,  true},
    {"Tropolone",        "[OH]c1cccccc1=O",             6.95, true},

    // ---- Hydroxypyridines / Pyridinones (tautomeric, act as acids) ----
    {"2-Hydroxypyridine","[OH]c1ccccn1",                0.75, true},
    {"3-Hydroxypyridine","[OH]c1cccnc1",               8.72, true},
    {"4-Hydroxypyridine","[OH]c1ccncc1",               11.09, true},
    {"8-Hydroxyquinoline","[OH]c1ccc2ncccc2c1",         9.81, true},

    // ---- Nucleobase acids (NH) ----
    {"Uracil NH",        "[NH1]1C(=O)[NH1]C(=O)C=C1",  9.5,  true},
    {"Thymine NH",       "[NH1]1C(=O)[NH1]C(=O)C(C)=C1", 9.9, true},
    {"Xanthine NH",      "[NH1]1C(=O)[NH1]C(=O)c2[nH]cnc12", 7.44, true},

    // ---- Thioamide NH ----
    {"Thioamide NH",     "[NX3H1]C(=S)",               13.0,  true},
    {"Thiourea NH",      "[NX3H1]C(=S)N",              21.0, true},

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
    {"p-Hydroxyphenol",  "[OX2H1]c1ccc([OX2H1])cc1",    9.85, true},
    {"Salicylaldehyde OH","[OX2H1]c1ccccc1C=O",         8.34, true},
    {"Salicylic acid OH","[OX2H1]c1ccccc1C(=O)[OH]",   13.0, true},
    {"p-Hydroxybenz OH", "[OX2H1]c1ccc(C(=O))cc1",      8.0,  true},
    {"m-Methoxyphenol",  "[OX2H1]c1cccc(OC)c1",         9.65, true},
    {"p-tBu-phenol",     "[OX2H1]c1ccc(C(C)(C)C)cc1",  10.23, true},
    {"2,6-di-tBu-phenol","[OX2H1]c1c(C(C)(C)C)cccc1C(C)(C)C", 11.70, true},

    // ---- Salicylic acid (COOH) ----
    {"Salicylic acid",   "[CX3](=O)([OX2H1])c1ccccc1O",  2.97, true},

    // ---- Additional carboxylic acids from Williams ----
    {"Glycine COOH",     "[CX3](=O)([OX2H1])[CX4]([NX3H2,NX4H3+])", 2.35, true},
    {"Proline COOH",     "[CX3](=O)([OX2H1])C1CCCN1",   1.99, true},
    {"Cinnamic acid",    "[CX3](=O)([OX2H1])/C=C/c",    4.44, true},
    {"Crotonic acid",    "[CX3](=O)([OX2H1])/C=C/C",    4.69, true},
    {"Pyruvic acid",     "[CX3](=O)([OX2H1])C(=O)C",    2.50, true},
    {"Lactic acid",      "[CX3](=O)([OX2H1])C(O)C",     3.86, true},
    {"Mandelic acid",    "[CX3](=O)([OX2H1])C(O)c",     3.41, true},
    {"Glycolic acid",    "[CX3](=O)([OX2H1])CO",        3.82, true},
    {"Picolinic acid",   "[CX3](=O)([OX2H1])c1ccccn1",  5.25, true},
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
    {"Guanidine",        "[NX3]C(=[NX2])[NX3]",        12.5,  false},
    {"Guanidine alt1",   "[NX3]C([NX3])=[NX2]",        12.5,  false},
    {"Guanidine alt2",   "[NH2]C(=N)N",                12.5,  false},
    {"Guanidine alt3",   "NC(N)=[NH]",                 12.5,  false},
    {"Guanidine charged","[NH2]C(=[NH2+])N",           12.5,  false},
    {"Guanidine charged2","NC(=[NH2+])[NH2]",          12.5,  false},
    {"Amidine",          "[NX3]C(=[NX2])[!N]",         11.6,  false},
    {"Amidine alt",      "[NH2]C(=[NH])c",             11.6,  false},
    {"Amidine alt2",     "[NH2]/C(=N\\H)c",            11.6,  false},
    {"Amidine alt3",     "[NH2]C(=[NH])[CX4]",         12.4,  false},
    {"Amidine charged",  "[NH2]C(=[NH2+])c",           11.6,  false},
    {"DBU",              "C1=NCCCN1CCC",               12.0,  false},
    {"Acetamidine",      "[NX3]C(=[NX2])C",            12.4,  false},

    // ---- Saturated N-heterocycles — most specific ring patterns first ----
    {"Piperaz N-aryl arN","[NX3H0;R1](c)1CC[NX3;R1]CC1", 3.9, false},
    {"Piperaz N-aryl alkN","[NX3;R1]1CC[NX3H0;R1](c)CC1", 8.5, false},
    {"Piperaz NH di-sub","[NX3H1;R1]1CC[NX3H0;R1]([CX4])CC1", 5.3, false},
    {"Piperaz NR di-sub","[NX3H0;R1]([CX4])1CC[NX3H1;R1]CC1", 9.0, false},
    {"Piperazine NH 1st","[NX3H1;R1]1CC[NX3H1;R1]CC1",  9.8, false},
    {"Piperaz NR,NR 1st","[NX3H0;R1]([CX4])1CC[NX3H0;R1]([CX4])CC1", 9.0, false},
    {"1-Tosylpiperazine","[NX3H1;R1]1CC[NX3;R1](S(=O)(=O))CC1", 7.4, false},
    {"N-Bz piperazine",  "[NX3H1;R1]1CC[NX3;R1](C(=O)c)CC1", 7.8, false},
    {"Piperazine N",     "[NX3H1;R1;!$(NC=O);!$(NS(=O)=O)]1CC[NX3;R1]CC1", 9.0, false},
    {"4-Aryl morpholine","[NX3H0;R1](c)1CCOCC1",       7.4,  false},
    {"N-Me morpholine",  "[NX3H0;R1](C)1CCOCC1",       7.4,  false},
    {"Morpholine NH",    "[NX3H1;R1]1CCOCC1",           8.33, false},
    {"Morpholine N",     "[NX3;R1;!$(NC=O)]1CCOCC1",    8.33, false},
    {"Thiomorpholine NH","[NX3H1;R1]1CCSCC1",           8.70, false},
    {"Thiomorpholine N", "[NX3;R1;!$(NC=O)]1CCSCC1",    8.70, false},
    {"Pyrrolidine NH",   "[NX3H1;R1;!$(NC=O);!$(NS=O)]1CCCC1", 11.27, false},
    {"N-Me pyrrolidine", "[NX3H0;R1;!$(NC=O)](C)1CCCC1", 10.46, false},
    {"4-Aryl piperidine","[NX3H1;R1;!$(NC=O)]1CCC(c)CC1", 10.1, false},
    {"N-Me piperidine",  "[NX3H0;R1;!$(NC=O)](C)1CCCCC1", 10.08, false},
    {"N-Bz piperidine",  "[NX3H0;R1](Cc)1CCCCC1",      9.6,  false},
    {"Piperidine NH",    "[NX3H1;R1;!$(NC=O);!$(NS=O)]1CCCCC1", 11.22, false},
    {"Piperidine N",     "[NX3;R1;!$(NC=O);!$(NS=O)]1CCCCC1", 10.5, false},
    {"Azetidine NH",     "[NX3H1;R1;!$(NC=O)]1CCC1",    11.3, false},
    {"Azepane NH",       "[NX3H1;R1;!$(NC=O)]1CCCCCC1",  10.5, false},
    {"DABCO N",          "[NX3;R2]1CC[NX3;R2]CC1",      8.82, false},
    {"Quinuclidine N",   "[NX3;R2;!$(Nc)]1CC2CCC(C1)C2", 11.0, false},

    // ---- Aromatic heterocyclic bases ----
    {"4-Me-imidazole",   "[nH0;X2]1c(C)[nH1]cc1",       7.45, false},
    {"2-Me-imidazole",   "[nH0;X2]1cc[nH1]c1C",          7.75, false},
    {"Benzimidazole =N", "[nH0;X2]1c[nH1]c2ccccc12",     5.53, false},
    {"Imidazole =N",     "[nH0;X2]1cc[nH1]c1",            6.95, false},
    {"Imidazole =N alt", "[nH0;X2]1c[nH1]cc1",            6.95, false},
    {"4-DMAP",           "[nH0;X2]1cc(N(C)C)ccc1",        9.70, false},
    {"4-Aminopyridine",  "[nH0;X2]1cc(N)ccc1",            9.17, false},
    {"2-Aminopyridine",  "[nH0;X2]1cccc(N)c1",            6.86, false},
    {"4-Me-pyridine",    "[nH0;X2]1cc(C)ccc1",            6.02, false},
    {"3-Me-pyridine",    "[nH0;X2]1ccc(C)cc1",            5.68, false},
    {"2-Me-pyridine",    "[nH0;X2]1cccc(C)c1",            5.97, false},
    {"2,6-diMe-pyridine","[nH0;X2]1c(C)ccc(C)c1",         6.77, false},
    {"2,4,6-triMe-pyridine","[nH0;X2]1c(C)cc(C)c(C)c1",   7.48, false},
    {"4-OMe-pyridine",   "[nH0;X2]1cc(OC)ccc1",           6.62, false},
    {"2-OMe-pyridine",   "[nH0;X2]1cccc(OC)c1",           3.28, false},
    {"3-OH-pyridine",    "[nH0;X2]1ccc(O)cc1",            4.86, false},
    {"3-CN-pyridine",    "[nH0;X2]1ccc(C#N)cc1",          1.45, false},
    {"3-NO2-pyridine",   "[nH0;X2]1ccc([NX3+](=O)[O-])cc1", 0.81, false},
    {"2-Cl-pyridine",    "[nH0;X2]1cccc(Cl)c1",           0.72, false},
    {"3-Cl-pyridine",    "[nH0;X2]1ccc(Cl)cc1",           2.84, false},
    {"4-Cl-pyridine",    "[nH0;X2]1cc(Cl)ccc1",           3.83, false},
    {"3-F-pyridine",     "[nH0;X2]1ccc(F)cc1",            2.97, false},
    {"2-F-pyridine",     "[nH0;X2]1cccc(F)c1",           -0.44, false},
    {"3-Br-pyridine",    "[nH0;X2]1ccc(Br)cc1",           2.84, false},
    {"3-COOH-pyridine",  "[nH0;X2]1ccc(C(=O)O)cc1",       3.13, false},
    {"3-CO2Et-pyridine", "[nH0;X2]1ccc(C(=O)OCC)cc1",     3.35, false},
    {"Isoquinoline",     "[nH0;X2]1ccc2ccccc2c1",         5.14, false},
    {"Quinoline",        "[nH0;X2]1cccc2ccccc12",          4.85, false},
    {"Acridine N",       "[nH0;X2]1cccc2cc3ccccc3cc12",    5.60, false},
    {"Pyridazine N",     "[nH0;X2]1[nH0;X2]cccc1",        2.33, false},
    {"Pyrimidine N",     "[nH0;X2]1c[nH0;X2]ccc1",        1.10, false},
    {"Pyrazine N",       "[nH0;X2]1cc[nH0;X2]cc1",        0.60, false},
    {"Quinazoline N",    "[nH0;X2]1c[nH0;X2]c2ccccc2c1",  3.31, false},
    {"Quinoxaline N",    "[nH0;X2]1[nH0;X2]cc2ccccc2c1",  0.60, false},
    {"Phthalazine N",    "[nH0;X2]1[nH0;X2]cc2ccccc12",   3.47, false},
    {"Cinnoline N",      "[nH0;X2]1[nH0;X2]c2ccccc2cc1",  2.64, false},
    {"Pyrazole =N",      "[nH0;X2]1cc[nH1]c1",            2.5,  false},
    {"1,2,4-Triazole =N","[nH0;X2]1c[nH1]nc1",            2.2,  false},
    {"Benzotriazole =N", "[nH0;X2]1[nH0;X2][nH1]c2ccccc12", 1.6, false},
    {"Pyridine N",       "[nH0;X2;R1]1ccccc1",            5.14, false},

    // ---- 5-membered heterocyclic bases ----
    {"2-Aminothiazole",  "[nH0;X2]1csc(N)c1",             5.36, false},
    {"4-Me-thiazole",    "[nH0;X2]1csc(C)c1",             3.5,  false},
    {"Thiazole =N",      "[nH0;X2]1cscc1",                2.5,  false},
    {"Benzothiazole =N", "[nH0;X2]1c2ccccc2sc1",          1.2,  false},
    {"2-Aminobenzothiazole","[nH0;X2]1c2ccccc2sc1N",      4.51, false},
    {"Benzoxazole =N",   "[nH0;X2]1c2ccccc2oc1",         -0.2,  false},
    {"2-Aminobenzoxazole","[nH0;X2]1c2ccccc2oc1N",        3.73, false},
    {"Oxazole =N",       "[nH0;X2]1cocc1",               0.8,  false},
    {"Oxazoline =N",     "N=1CCOC1",                       4.8,  false},
    {"Isoxazole =N",     "[nH0;X2]1oncc1",               -2.0,  false},
    {"Isothiazole =N",   "[nH0;X2]1sncc1",                0.5,  false},
    {"1,2,4-Thiadiazole","[nH0;X2]1ncs[nH0]1",           -1.0,  false},
    {"1,3,4-Thiadiazole","[nH0;X2]1[nH0]csc1",            1.0,  false},
    {"2-Amino-1,3,4-thiadiazole","[nH0;X2]1[nH0]c(N)sc1", 3.5, false},
    {"1,2,4-Oxadiazole", "[nH0;X2]1nco[nH0]1",           -2.0,  false},
    {"1,3,4-Oxadiazole", "[nH0;X2]1[nH0]coc1",           -1.5,  false},
    {"2-Amino-1,3,4-oxadiazole","[nH0;X2]1[nH0]c(N)oc1",  2.0, false},
    {"Indazole =N",      "[nH0;X2]1[nH1]c2ccccc12",       1.4,  false},
    {"Imidazo[1,2-a]pyr","[nH0;X2]1ccn2ccccc12",          6.0,  false},

    // ---- Purine / nucleobase bases ----
    {"Adenine N1",       "[nH0;X2]1c2[nH]cnc2nc(N)c1",    4.15, false},
    {"Guanine N7",       "[nH0;X2]1cnc2C(=O)[NH]C(N)=Nc12", 3.3, false},
    {"Purine N",         "[nH0;X2]1c2[nH]cnc2ncc1",       2.52, false},
    {"Caffeine N",       "[nH0;X2]1c2n(C)c(=O)n(C)c(=O)c2n(C)c1", 0.6, false},

    // ---- Aminopyrimidines / Aminopyrazines ----
    {"4-Aminopyrimidine","[nH0;X2]1c[nH0;X2]c(N)cc1",    5.71, false},
    {"2-Aminopyrimidine","[nH0;X2]1c(N)[nH0;X2]ccc1",    3.54, false},
    {"4,6-Diamino-pyrimidine","[nH0;X2]1c[nH0;X2]c(N)cc(N)1", 7.26, false},
    {"2-Aminopyrazine",  "[nH0;X2]1c(N)c[nH0;X2]cc1",    3.14, false},
    {"5-Aminopyrimidine","[nH0;X2]1c[nH0;X2]cc(N)c1",    2.83, false},
    {"Aminotriazine",    "[nH0;X2]1c(N)[nH0;X2]c[nH0;X2]c1", 5.0, false},

    // ---- Aminoquinolines / Aminoisoquinolines ----
    {"2-Aminoquinoline", "[nH0;X2]1cccc2ccc(N)cc12",      7.34, false},
    {"4-Aminoquinoline", "[nH0;X2]1cccc2c(N)cccc12",      9.17, false},
    {"6-Aminoquinoline", "[nH0;X2]1cccc2ccc(N)cc12",      5.63, false},
    {"1-Aminoisoquinoline","[nH0;X2]1cc(N)c2ccccc2c1",    7.62, false},
    {"3-Aminoisoquinoline","[nH0;X2]1cnc(N)c2ccccc12",    5.05, false},
    {"Benzoquinoline N", "[nH0;X2]1cccc2cccc3ccccc123",   5.05, false},
    {"Phenanthroline N", "[nH0;X2]1cccc2c1ccc1cccnc12",   4.27, false},
    {"Het aromatic =N",  "[nH0;X2;R1]",                   3.5,  false},

    // ---- Aliphatic amines ----
    {"alpha-Amino acid", "[NX3H2][CX4H1](C(=O)[O,N])",    9.0,  false},
    {"Benzylamine",      "[NX3H2]Cc",                      9.34, false},
    {"CF3-ethylamine",   "[NX3H2]CC(F)(F)F",               5.7,  false},
    {"2-Fluoroethylamine","[NX3H2]CCF",                    8.5,  false},
    {"2-Methoxyethylamine","[NX3H2]CCOC",                  9.2,  false},
    {"Ethanolamine",     "[NX3H2]CCO",                     9.50, false},
    {"2-Cyanoethylamine","[NX3H2]CCC#N",                   7.9,  false},
    {"Allylamine",       "[NX3H2]CC=C",                    9.49, false},
    {"Methyl amine",     "[NX3H2;!$(NC=O);!$(NS=O);!$(Nc)]C", 10.6, false},
    {"Primary amine",    "[NX3H2;!$(NC=O);!$(NS=O);!$(Nc)]", 10.5, false},
    {"N,O-dimethylhydroxylamine","[NX3H0](C)(OC)",         4.75, false},
    {"N-methylhydroxylamine","[NX3H1](C)O",                5.96, false},
    {"Diethylamine",     "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R]([CX4][CX4])[CX4][CX4]", 10.98, false},
    {"Dimethylamine",    "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R](C)C", 10.64, false},
    {"N-Me benzylamine", "[NX3H0;!$(NC=O);!$(NS=O);!R](C)Cc", 9.6, false},
    {"Dibenzylamine",    "[NX3H1;!$(NC=O);!$(NS=O);!R](Cc)Cc", 8.52, false},
    {"Secondary amine",  "[NX3H1;!$(NC=O);!$(NS=O);!$(Nc);!R]", 10.5, false},
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
    {"p-NH2-aniline",    "[NX3H2]c1ccc(N)cc1",            6.08, false},
    {"2-Naphthylamine",  "[NX3H2]c1ccc2ccccc2c1",         4.16, false},
    {"1-Naphthylamine",  "[NX3H2]c1cccc2ccccc12",         3.92, false},
    {"Aniline",          "[NX3H2]c",                       4.58, false},
    {"N-Me aniline",     "[NX3H1;!$(NC=O)](C)c",           4.85, false},
    {"N,N-diMe aniline", "[NX3H0;!$(NC=O)](C)(C)c",        5.07, false},

    // ---- Additional saturated heterocycles / bicyclic ----
    {"Thiazolidine NH",  "[NX3H1;R1]1CCSC1",              6.31, false},
    {"Tetrahydroisoquinoline","[NX3H1;R1]1CCc2ccccc2C1",   9.5, false},
    {"Tropane N",        "[NX3;R2]1CC2CCC(C1)CC2",        10.0, false},
    {"Proton sponge",    "[NX3](C)(C)c1cccc2c1cccc2[NX3](C)C", 12.1, false},
    {"Decahydroquinoline","[NX3H1;R1]1CCCCC1C1CCCCC1",     11.0, false},
    {"2-Methylazetidine","[NX3H1;R1]1CC(C)C1",            11.3, false},

    // ---- Diamines (important: pKa depression for second N) ----
    {"Ethylenediamine N1","[NX3H2]CC[NX3H2]",             10.0, false},
    {"1,3-Diaminopropane","[NX3H2]CCC[NX3H2]",           10.6, false},
    {"1,4-Diaminobutane","[NX3H2]CCCC[NX3H2]",           10.8, false},
    {"1,5-Diaminopentane","[NX3H2]CCCCC[NX3H2]",         10.9, false},
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
    {"Isoniazid hydrazide","[NX3H2][NX3H1]C(=O)c1ccncc1",  3.5, false},

    // ---- Guanidine / Amidine variants ----
    {"Biguanide",        "[NX3]C(=N)NC(=N)N",             11.5, false},
    {"Phenylguanidine",  "[NX3H2]C(=[NX2])[NX3H1]c",     10.9, false},
    {"Acetylguanidine",  "[NX3H2]C(=[NX2])NC(=O)C",       8.3, false},
    {"Cyanoguanidine",   "[NX3H2]C(=[NX2])NC#N",          0.4, false},

    // ---- Urea / Thiourea (very weak bases) ----
    {"Urea N",           "[NX3H2]C(=O)[NX3H2]",           0.18, false},
    {"Thiourea N",       "[NX3H2]C(=S)[NX3H2]",          -0.96, false},

    // ---- Nitrogen mustard class / Aziridine ----
    {"Aziridine NH",     "[NX3H1;R1]1CC1",                8.0,  false},

    // ---- Pyridine N-oxide (as base, less basic than pyridine) ----
    {"Pyridine N-oxide", "[nX3;R1]([O-])1ccccc1",         0.8,  false},

    // =====================================================================
    // GENERIC FALLBACK PATTERNS — catch anything the specific patterns above miss
    // =====================================================================
    {"Generic S-OH acid","[OX2H1]S",                       2.0,  true},
    {"Generic P-OH acid","[OX2H1]P",                       2.0,  true},
    {"Generic ArOH",     "[OX2H1]a",                      10.0,  true},
    {"Generic COOH",     "[CX3](=O)[OX2H1]",              4.0,  true},
    {"Generic SH",       "[SX2H1]",                        8.3,  true},
    {"Generic NH-SO2",   "[NH]S(=O)(=O)",                 10.0,  true},
    {"Generic NH-CO-NH-CO","[NH1](C=O)C=O",               9.0,  true},
    {"Generic C=N-N",    "[NX3;!$(NC=O);!$(NS=O)]C=[NX2]",  11.0, false},
    {"Generic N=C-N",    "[NX2]=[CX3][NX3;!$(NC=O)]",       11.0, false},
    {"Generic ring NH sat","[NX3H1;R;!$(NC=O);!$(NS(=O)=O);!a]", 9.5, false},
    {"Generic ring NR sat","[NX3H0;R;!$(NC=O);!$(NS(=O)=O);!a]([CX4])", 9.0, false},
    {"Generic arom =N",  "[nH0;X2]",                        4.0, false},
    {"Generic prim amine","[NX3H2;!$(NC=O);!$(NS=O);!$(NC=S)]", 10.5, false},
    {"Generic sec amine", "[NX3H1;!$(NC=O);!$(NS=O);!$(NC=S);!$(Nc);!R]", 10.5, false},
    {"Generic tert amine","[NX3H0;!$(NC=O);!$(NS=O);!$(NC=S);!$(Nc);!R]([CX4])([CX4])[CX4]", 9.8, false},
    {"Generic arom ring NH","[nH1]",                        5.0, false},
};

const int kNumIonizableGroups = sizeof(kIonizableGroups) / sizeof(kIonizableGroups[0]);

/// Lazily compiled SMARTS cache — compiled once on first use, reused forever.
const std::vector<std::unique_ptr<ROMol>>& getCompiledPatterns() {
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
std::vector<std::tuple<int, int, bool, double>> detectIonSitesInternal(const ROMol &mol) {
    const auto &patterns = getCompiledPatterns();
    std::vector<std::tuple<int, int, bool, double>> sites;
    std::set<int> seen;
    const auto *ri = mol.getRingInfo();
    for (int g = 0; g < kNumIonizableGroups; g++) {
        if (!patterns[g]) continue;
        std::vector<MatchVectType> matches;
        SubstructMatch(mol, *patterns[g], matches);
        for (const auto &m : matches) {
            if (m.empty()) continue;
            int aIdx = m[0].second;
            if (!seen.count(aIdx)) {
                seen.insert(aIdx);
                sites.emplace_back(aIdx, g, kIonizableGroups[g].isAcid, kIonizableGroups[g].pKa);
            }

            if (m.size() > 2 && ri && ri->numAtomRings(aIdx) > 0) {
                const Atom *primary = mol.getAtomWithIdx(aIdx);
                int pAN = primary->getAtomicNum();
                bool pArom = primary->getIsAromatic();
                unsigned pDeg = primary->getDegree();
                unsigned pH = primary->getTotalNumHs();
                for (size_t mi = 1; mi < m.size(); mi++) {
                    int otherIdx = m[mi].second;
                    if (seen.count(otherIdx)) continue;
                    const Atom *other = mol.getAtomWithIdx(otherIdx);
                    if (other->getAtomicNum() == pAN &&
                        ri->numAtomRings(otherIdx) > 0 &&
                        other->getIsAromatic() == pArom &&
                        other->getDegree() == pDeg &&
                        other->getTotalNumHs() == pH) {
                        seen.insert(otherIdx);
                        sites.emplace_back(otherIdx, g, kIonizableGroups[g].isAcid,
                                           kIonizableGroups[g].pKa);
                    }
                }
            }

            if (kIonizableGroups[g].isAcid && mol.getAtomWithIdx(aIdx)->getTotalNumHs() == 0) {
                for (size_t mi = 1; mi < m.size(); mi++) {
                    int otherIdx = m[mi].second;
                    if (seen.count(otherIdx)) continue;
                    const Atom *other = mol.getAtomWithIdx(otherIdx);
                    int an = other->getAtomicNum();
                    if (other->getTotalNumHs() > 0 && (an == 8 || an == 16 || an == 7)) {
                        seen.insert(otherIdx);
                        sites.emplace_back(otherIdx, g, true, kIonizableGroups[g].pKa);
                        break;
                    }
                }
            }
        }
    }
    return sites;
}
