// ============================================================================
// druse_xtb.cpp — GFN2-xTB Mulliken charge calculator
//
// Self-contained implementation of the GFN2-xTB tight-binding method.
// Computes Mulliken partial charges via self-consistent charge (SCC) iteration.
//
// Theory overview:
//   GFN2-xTB is a semi-empirical tight-binding DFT method. The total energy is:
//     E_total = E_rep + E_elec + E_disp + E_XB
//   where E_elec comes from solving the SCC-DFTB equations iteratively:
//     H(q) * C = S * C * ε
//   The Hamiltonian H depends on Mulliken charges q, which depend on eigenvectors
//   C, creating a self-consistent loop. Convergence is aided by charge mixing.
//
// Uses Apple Accelerate framework for LAPACK (dsygv_) and BLAS (dgemm_).
// No Fortran or external xTB library dependencies.
//
// Reference: Bannwarth, Ehlert, Grimme, JCTC 2019, 15, 1652-1671
// Parameters: tblite GFN2-xTB parametrization (gfn2-molmom.toml)
// ============================================================================

#include "druse_xtb.h"

// Use the new Accelerate LAPACK headers (avoids CLAPACK deprecation warnings).
// ACCELERATE_NEW_LAPACK is defined via CMake target_compile_definitions.
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdio>

// ============================================================================
// MARK: - Physical Constants
// ============================================================================

static constexpr double BOHR_TO_ANG = 0.529177210903;   // 1 Bohr in Angstrom
static constexpr double ANG_TO_BOHR = 1.0 / BOHR_TO_ANG;
static constexpr double EV_TO_HARTREE = 1.0 / 27.211386245988;
static constexpr double PI = 3.14159265358979323846;

// ============================================================================
// MARK: - GFN2 Element Parameters
// ============================================================================

/// GFN2-xTB parameters for a single element.
/// Extracted from the tblite GFN2 parametrization TOML.
struct GFN2Element {
    int atomicNumber;
    int nShells;                // number of valence shells (1-3)
    int shellAngMom[3];         // angular momentum per shell: 0=s, 1=p, 2=d
    double slaterExp[3];        // Slater orbital exponents (ζ)
    double selfEnergy[3];       // shell self-energy levels (eV)
    double refOcc[3];           // reference occupations
    double gam;                 // chemical hardness / Hubbard U (a.u.)
    double lgam[3];             // shell-resolved hardness multipliers
    double gam3;                // third-order Γ correction
    double zeff;                // effective nuclear charge (for repulsion)
    double arep;                // repulsion exponent prefactor
    double en;                  // electronegativity (Pauling)
    double shpoly[3];           // shell polynomial coefficients (raw, NOT pre-scaled)
    double kcn[3];              // coordination number dependence of levels
    int ngauss[3];              // number of Gaussians for STO expansion (3 or 4)
    double atomicRad;           // covalent/atomic radius in Bohr (for shellPoly distance scaling)
};

// GFN2-xTB Hamiltonian global parameters
static constexpr double GFN2_WEXP = 0.5;       // distance weighting exponent
static constexpr double GFN2_ENSCALE = 0.02;   // electronegativity scaling
static constexpr double GFN2_KPOL = 2.0;       // polarization scaling

// Shell-pair scaling constants (Hamiltonian)
static constexpr double GFN2_KAB_SS = 1.85;
static constexpr double GFN2_KAB_PP = 2.23;
static constexpr double GFN2_KAB_DD = 2.23;
static constexpr double GFN2_KAB_SD = 2.00;
static constexpr double GFN2_KAB_PD = 2.00;

// Third-order shell scaling
static constexpr double GFN2_THIRDORDER_S = 1.0;
static constexpr double GFN2_THIRDORDER_P = 0.5;
static constexpr double GFN2_THIRDORDER_D = 0.25;

// Repulsion parameters
static constexpr double GFN2_KEXP = 1.5;
static constexpr double GFN2_KLIGHT = 1.0;

// Charge interaction parameters
static constexpr double GFN2_GEXP = 2.0;

// Maximum supported atomic number
static constexpr int MAX_ATOMIC_NUM = 86;

// Parameter table — only drug-relevant elements are populated.
// Shell angular momenta encode the shell type:
//   For main-group rows 1-2: s=0, p=1
//   For row 3+: s=0, p=1, d=2
//   For transition metals: d=2 is listed first (matching tblite convention)
static GFN2Element gfn2Params[MAX_ATOMIC_NUM + 1];
static bool gfn2ParamsInitialized = false;

/// Helper to set up one element's parameters.
/// shpoly: raw values from param file (NOT pre-divided by 100)
/// kcn: pre-scaled by 0.1 (matching read_gfn_param.f90)
/// atomicRad_angstrom: covalent radius in Angstroms (converted to Bohr internally)
static void setElement(int Z, int nShells,
                       const int angMom[], const double slater[],
                       const double levels[], const double refOcc[],
                       double gam, const double lgam[], double gam3,
                       double zeff, double arep, double en,
                       const double shpoly[], const double kcn[],
                       const int ngauss[], double atomicRad_angstrom) {
    auto &e = gfn2Params[Z];
    e.atomicNumber = Z;
    e.nShells = nShells;
    for (int i = 0; i < nShells; i++) {
        e.shellAngMom[i] = angMom[i];
        e.slaterExp[i] = slater[i];
        e.selfEnergy[i] = levels[i];
        e.refOcc[i] = refOcc[i];
        e.lgam[i] = lgam[i];
        e.shpoly[i] = shpoly[i];
        e.kcn[i] = kcn[i];
        e.ngauss[i] = ngauss[i];
    }
    for (int i = nShells; i < 3; i++) {
        e.shellAngMom[i] = -1;
        e.slaterExp[i] = 0;
        e.selfEnergy[i] = 0;
        e.refOcc[i] = 0;
        e.lgam[i] = 0;
        e.shpoly[i] = 0;
        e.kcn[i] = 0;
        e.ngauss[i] = 0;
    }
    e.gam = gam;
    e.gam3 = gam3;
    e.zeff = zeff;
    e.arep = arep;
    e.en = en;
    e.atomicRad = atomicRad_angstrom * ANG_TO_BOHR;
}

/// Initialize the GFN2 parameter table for all supported elements.
/// Values taken directly from tblite gfn2-molmom.toml parametrization.
static void initGFN2Params() {
    if (gfn2ParamsInitialized) return;
    std::memset(gfn2Params, 0, sizeof(gfn2Params));

    // H (Z=1): 1s  — atomicRad=0.32 Å
    {
        int am[] = {0}; double sl[] = {1.23}; double lv[] = {-10.707211};
        double ro[] = {1.0}; double lg[] = {1.0}; double sp[] = {-0.00953618};
        double kc[] = {-0.05}; int ng[] = {3};
        setElement(1, 1, am, sl, lv, ro, 0.405771, lg, 0.08, 1.105388, 2.213717, 2.20, sp, kc, ng, 0.32);
    }
    // He (Z=2): 1s, 2p  — atomicRad=0.37 Å
    {
        int am[] = {0, 1}; double sl[] = {1.669667, 1.5}; double lv[] = {-23.716445, -1.822307};
        double ro[] = {2.0, 0.0}; double lg[] = {1.0, 1.0}; double sp[] = {-0.04386816, 0.00710647};
        double kc[] = {0.2074275, 0.0}; int ng[] = {3, 4};
        setElement(2, 2, am, sl, lv, ro, 0.642029, lg, 0.20, 1.094283, 3.604670, 3.0, sp, kc, ng, 0.37);
    }
    // Li (Z=3): 2s, 2p  — atomicRad=1.30 Å
    {
        int am[] = {0, 1}; double sl[] = {0.75006, 0.557848}; double lv[] = {-4.9, -2.217789};
        double ro[] = {1.0, 0.0}; double lg[] = {1.0, 1.197261}; double sp[] = {-0.04750398, 0.20424920};
        double kc[] = {0.1620836, -0.0623876}; int ng[] = {4, 4};
        setElement(3, 2, am, sl, lv, ro, 0.245006, lg, 0.1303821, 1.289367, 0.475307, 0.98, sp, kc, ng, 1.30);
    }
    // Be (Z=4): 2s, 2p  — atomicRad=0.99 Å
    {
        int am[] = {0, 1}; double sl[] = {1.03472, 0.949332}; double lv[] = {-7.743081, -3.133433};
        double ro[] = {2.0, 0.0}; double lg[] = {1.0, 1.9658467}; double sp[] = {-0.07910394, -0.00476438};
        double kc[] = {0.1187759, 0.0550528}; int ng[] = {4, 4};
        setElement(4, 2, am, sl, lv, ro, 0.684789, lg, 0.0574239, 4.221216, 0.939696, 1.57, sp, kc, ng, 0.99);
    }
    // B (Z=5): 2s, 2p  — atomicRad=0.84 Å
    {
        int am[] = {0, 1}; double sl[] = {1.479444, 1.479805}; double lv[] = {-9.224376, -7.419002};
        double ro[] = {2.0, 1.0}; double lg[] = {1.0, 1.399408}; double sp[] = {-0.0518315, -0.02453322};
        double kc[] = {0.0120462, -0.0141086}; int ng[] = {4, 4};
        setElement(5, 2, am, sl, lv, ro, 0.513556, lg, 0.0946104, 7.192431, 1.373856, 2.04, sp, kc, ng, 0.84);
    }
    // C (Z=6): 2s, 2p  — atomicRad=0.75 Å
    {
        int am[] = {0, 1}; double sl[] = {2.096432, 1.8}; double lv[] = {-13.970922, -10.063292};
        double ro[] = {1.0, 3.0}; double lg[] = {1.0, 1.1056358}; double sp[] = {-0.02294321, -0.00271102};
        double kc[] = {-0.0102144, 0.0161657}; int ng[] = {4, 4};
        setElement(6, 2, am, sl, lv, ro, 0.538015, lg, 0.15, 4.231078, 1.247655, 2.55, sp, kc, ng, 0.75);
    }
    // N (Z=7): 2s, 2p  — atomicRad=0.71 Å
    {
        int am[] = {0, 1}; double sl[] = {2.339881, 2.014332}; double lv[] = {-16.686243, -12.523956};
        double ro[] = {1.5, 3.5}; double lg[] = {1.0, 1.1164892}; double sp[] = {-0.08506003, -0.02504201};
        double kc[] = {-0.1955336, 0.0561076}; int ng[] = {4, 4};
        setElement(7, 2, am, sl, lv, ro, 0.461493, lg, -0.063978, 5.242592, 1.682689, 3.04, sp, kc, ng, 0.71);
    }
    // O (Z=8): 2s, 2p  — atomicRad=0.64 Å
    {
        int am[] = {0, 1}; double sl[] = {2.439742, 2.137023}; double lv[] = {-20.229985, -15.503117};
        double ro[] = {2.0, 4.0}; double lg[] = {1.0, 1.149702}; double sp[] = {-0.14955291, -0.03350819};
        double kc[] = {0.0117826, -0.0145102}; int ng[] = {4, 4};
        setElement(8, 2, am, sl, lv, ro, 0.451896, lg, -0.0517134, 5.784415, 2.165712, 3.44, sp, kc, ng, 0.64);
    }
    // F (Z=9): 2s, 2p  — atomicRad=0.60 Å
    {
        int am[] = {0, 1}; double sl[] = {2.416361, 2.308399}; double lv[] = {-23.458179, -15.746583};
        double ro[] = {2.0, 5.0}; double lg[] = {1.0, 1.1677376}; double sp[] = {-0.13011924, -0.12300828};
        double kc[] = {0.0394362, -0.0538373}; int ng[] = {4, 4};
        setElement(9, 2, am, sl, lv, ro, 0.531518, lg, 0.1426212, 7.021486, 2.421394, 3.98, sp, kc, ng, 0.60);
    }
    // Na (Z=11): 3s, 3p  — atomicRad=1.60 Å
    {
        int am[] = {0, 1}; double sl[] = {0.763787, 0.573553}; double lv[] = {-4.546934, -1.332719};
        double ro[] = {1.0, 0.0}; double lg[] = {1.0, 1.1018894}; double sp[] = {-0.04033495, 0.20873908};
        double kc[] = {-0.0042211, -0.0144323}; int ng[] = {4, 4};
        setElement(11, 2, am, sl, lv, ro, 0.271056, lg, 0.1798727, 5.244917, 0.572728, 0.93, sp, kc, ng, 1.60);
    }
    // Mg (Z=12): 3s, 3p, 3d  — atomicRad=1.40 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {1.184203, 0.717769, 1.3}; double lv[] = {-6.339908, -0.697688, -1.458197};
        double ro[] = {2.0, 0.0, 0.0}; double lg[] = {1.0, 2.4, 0.95}; double sp[] = {-0.11167374, 0.39076962, 0.12691061};
        double kc[] = {0.1164444, -0.0079924, 0.1192409}; int ng[] = {4, 4, 3};
        setElement(12, 3, am, sl, lv, ro, 0.344822, lg, 0.2349164, 18.083164, 0.917975, 1.31, sp, kc, ng, 1.40);
    }
    // P (Z=15): 3s, 3p, 3d  — atomicRad=1.09 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {1.816945, 1.903247, 1.167533}; double lv[] = {-17.518756, -9.842286, -0.444893};
        double ro[] = {1.5, 3.5, 0.0}; double lg[] = {1.0, 0.844194, 0.65}; double sp[] = {-0.19831771, -0.05515577, 0.26397535};
        double kc[] = {0.054761, -0.048993, 0.2429507}; int ng[] = {4, 4, 3};
        setElement(15, 3, am, sl, lv, ro, 0.297739, lg, 0.0711291, 19.683502, 1.143343, 2.19, sp, kc, ng, 1.09);
    }
    // S (Z=16): 3s, 3p, 3d  — atomicRad=1.04 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {1.981333, 2.025643, 1.702555}; double lv[] = {-20.029654, -11.377694, -0.420282};
        double ro[] = {2.0, 4.0, 0.0}; double lg[] = {1.0, 0.8914134, 0.75}; double sp[] = {-0.2585552, -0.08048064, 0.25993857};
        double kc[] = {-0.0256951, -0.0098465, 0.200769}; int ng[] = {4, 4, 3};
        setElement(16, 3, am, sl, lv, ro, 0.339971, lg, -0.0501722, 14.995090, 1.214553, 2.58, sp, kc, ng, 1.04);
    }
    // Cl (Z=17): 3s, 3p, 3d  — atomicRad=1.00 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {2.485265, 2.199650, 2.476089}; double lv[] = {-29.278781, -12.673758, -0.240338};
        double ro[] = {2.0, 5.0, 0.0}; double lg[] = {1.0, 1.49894, 1.5}; double sp[] = {-0.16562004, -0.0698643, 0.38045622};
        double kc[] = {0.0617972, -0.0181618, 0.1672768}; int ng[] = {4, 4, 3};
        setElement(17, 3, am, sl, lv, ro, 0.248514, lg, 0.1495483, 17.353134, 1.577144, 3.16, sp, kc, ng, 1.00);
    }
    // K (Z=19): 4s, 4p  — atomicRad=2.00 Å
    {
        int am[] = {0, 1}; double sl[] = {0.875961, 0.631694}; double lv[] = {-4.510348, -0.934377};
        double ro[] = {1.0, 0.0}; double lg[] = {1.0, 1.3483655}; double sp[] = {-0.0607606, 0.21187329};
        double kc[] = {-0.0339245, 0.0174542}; int ng[] = {4, 4};
        setElement(19, 2, am, sl, lv, ro, 0.247602, lg, 0.2033085, 10.439482, 0.482206, 0.82, sp, kc, ng, 2.00);
    }
    // Ca (Z=20): 4s, 4p, 3d  — atomicRad=1.74 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {1.26713, 0.786247, 1.38}; double lv[] = {-5.056506, -1.150304, -0.776883};
        double ro[] = {2.0, 0.0, 0.0}; double lg[] = {1.0, 2.5, 0.75}; double sp[] = {-0.09718719, 0.31973372, 0.09528649};
        double kc[] = {0.057093, -0.0074926, 0.1013752}; int ng[] = {4, 4, 3};
        setElement(20, 3, am, sl, lv, ro, 0.320378, lg, 0.2006898, 14.786701, 0.683051, 1.0, sp, kc, ng, 1.74);
    }
    // Mn (Z=25): 3d, 4s, 4p  — atomicRad=1.29 Å
    {
        int am[] = {2, 0, 1}; double sl[] = {1.83925, 1.22219, 1.240215}; double lv[] = {-10.120933, -5.617346, -4.198724};
        double ro[] = {5.0, 1.0, 1.0}; double lg[] = {1.0545811, 1.0, 0.4}; double sp[] = {-0.31255885, 0.28519691, 0.26346555};
        double kc[] = {-0.0195827, -0.0275, -0.0015839}; int ng[] = {3, 4, 4};
        setElement(25, 3, am, sl, lv, ro, 0.346651, lg, 0.06, 18.760605, 1.0711, 1.55, sp, kc, ng, 1.29);
    }
    // Fe (Z=26): 3d, 4s, 4p  — atomicRad=1.24 Å
    {
        int am[] = {2, 0, 1}; double sl[] = {1.911049, 1.022393, 1.294467}; double lv[] = {-10.035473, -5.402911, -3.308988};
        double ro[] = {6.0, 1.0, 1.0}; double lg[] = {1.4046615, 1.0, 0.35}; double sp[] = {-0.28614961, 0.11527794, 0.3945989};
        double kc[] = {-0.0274654, -0.4049876, -0.075648}; int ng[] = {3, 4, 4};
        setElement(26, 3, am, sl, lv, ro, 0.271594, lg, -0.05, 20.360089, 1.113422, 1.83, sp, kc, ng, 1.24);
    }
    // Co (Z=27): 3d, 4s, 4p  — atomicRad=1.18 Å
    {
        int am[] = {2, 0, 1}; double sl[] = {2.326507, 1.464221, 1.298678}; double lv[] = {-10.58043, -8.596723, -2.585753};
        double ro[] = {7.0, 1.0, 1.0}; double lg[] = {0.7581507, 1.0, 0.35}; double sp[] = {-0.22355636, 0.0916846, 0.25424719};
        double kc[] = {0.012198, -0.0227872, 0.0076513}; int ng[] = {3, 4, 4};
        setElement(27, 3, am, sl, lv, ro, 0.47776, lg, 0.03, 27.127744, 1.241717, 1.88, sp, kc, ng, 1.18);
    }
    // Ni (Z=28): 3d, 4s, 4p  — atomicRad=1.17 Å
    {
        int am[] = {2, 0, 1}; double sl[] = {2.430756, 1.469945, 1.317046}; double lv[] = {-12.712236, -8.524281, -2.878873};
        double ro[] = {8.0, 1.0, 1.0}; double lg[] = {0.9388812, 1.0, 0.4}; double sp[] = {-0.2538564, 0.2083955, 0.30886445};
        double kc[] = {-0.0066417, 0.0310301, 0.0226796}; int ng[] = {3, 4, 4};
        setElement(28, 3, am, sl, lv, ro, 0.34497, lg, -0.02, 10.533269, 1.077516, 1.91, sp, kc, ng, 1.17);
    }
    // Cu (Z=29): 3d, 4s, 4p  — atomicRad=1.22 Å
    {
        int am[] = {2, 0, 1}; double sl[] = {2.375425, 1.550837, 1.984703}; double lv[] = {-9.506548, -6.922958, -2.267723};
        double ro[] = {10.0, 1.0, 0.0}; double lg[] = {2.3333066, 1.0, 1.07}; double sp[] = {-0.26508943, 0.17798264, 0.14977818};
        double kc[] = {-0.0173684, 0.3349047, -0.2619446}; int ng[] = {3, 4, 4};
        setElement(29, 3, am, sl, lv, ro, 0.202969, lg, 0.05, 9.913846, 0.998768, 1.90, sp, kc, ng, 1.22);
    }
    // Zn (Z=30): 4s, 4p  — atomicRad=1.20 Å
    {
        int am[] = {0, 1}; double sl[] = {1.664847, 1.176434}; double lv[] = {-7.177294, -0.991895};
        double ro[] = {2.0, 0.0}; double lg[] = {1.0, 1.0684343}; double sp[] = {-0.09240315, 0.22271839};
        double kc[] = {0.201191, -0.0055135}; int ng[] = {4, 4};
        setElement(30, 2, am, sl, lv, ro, 0.564152, lg, 0.2312896, 22.099503, 1.160262, 1.65, sp, kc, ng, 1.20);
    }
    // Br (Z=35): 4s, 4p, 4d  — atomicRad=1.17 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {2.077587, 2.26312, 1.845038}; double lv[] = {-23.583718, -12.588824, 0.04798};
        double ro[] = {2.0, 5.0, 0.0}; double lg[] = {1.0, 1.5203002, 1.4}; double sp[] = {-0.25005079, -0.14520078, 0.36614038};
        double kc[] = {0.000615, -0.0058347, 0.225018}; int ng[] = {4, 4, 3};
        setElement(35, 3, am, sl, lv, ro, 0.261253, lg, 0.13, 32.845361, 1.296174, 2.96, sp, kc, ng, 1.17);
    }
    // I (Z=53): 5s, 5p, 5d  — atomicRad=1.36 Å
    {
        int am[] = {0, 1, 2}; double sl[] = {2.1595, 2.308379, 1.691185}; double lv[] = {-20.949407, -12.180159, -0.266596};
        double ro[] = {2.0, 5.0, 0.0}; double lg[] = {1.0, 0.9661265, 1.3}; double sp[] = {-0.26957547, -0.14183312, 0.28211905};
        double kc[] = {-0.050615, 0.0084766, 0.3077127}; int ng[] = {4, 4, 3};
        setElement(53, 3, am, sl, lv, ro, 0.383124, lg, 0.12, 63.319176, 1.017946, 2.66, sp, kc, ng, 1.36);
    }

    gfn2ParamsInitialized = true;
}

/// Number of basis functions for a given angular momentum.
static inline int basisCount(int l) {
    // s=1, p=3, d=5
    return 2 * l + 1;
}

// ============================================================================
// MARK: - Covalent Radii (Bohr)
// ============================================================================

/// D3-type covalent radii used for coordination number calculation.
/// Values in Bohr. Index by atomic number.
static const double covRadBohr[] = {
    0.0,    // dummy
    0.586,  // H
    0.529,  // He
    2.419,  // Li
    1.814,  // Be
    1.559,  // B
    1.436,  // C
    1.351,  // N
    1.247,  // O
    1.163,  // F
    1.098,  // Ne
    2.852,  // Na
    2.608,  // Mg
    2.457,  // Al
    2.213,  // Si
    2.100,  // P
    1.948,  // S
    1.890,  // Cl
    1.814,  // Ar
    3.663,  // K
    3.286,  // Ca
    2.852,  // Sc
    2.665,  // Ti
    2.571,  // V
    2.476,  // Cr
    2.457,  // Mn
    2.438,  // Fe
    2.381,  // Co
    2.343,  // Ni
    2.476,  // Cu
    2.381,  // Zn
    2.343,  // Ga
    2.213,  // Ge
    2.157,  // As
    2.100,  // Se
    2.062,  // Br
    1.986,  // Kr
    3.891,  // Rb
    3.514,  // Sr
    3.192,  // Y
    2.946,  // Zr
    2.814,  // Nb
    2.703,  // Mo
    2.627,  // Tc
    2.571,  // Ru
    2.552,  // Rh
    2.571,  // Pd
    2.703,  // Ag
    2.627,  // Cd
    2.703,  // In
    2.627,  // Sn
    2.571,  // Sb
    2.514,  // Te
    2.476,  // I
    2.438,  // Xe
};
static const int covRadSize = sizeof(covRadBohr) / sizeof(covRadBohr[0]);

static double getCovRad(int Z) {
    if (Z > 0 && Z < covRadSize) return covRadBohr[Z];
    return 2.5; // fallback
}

// ============================================================================
// MARK: - STO-nG Expansion Coefficients
// ============================================================================

// Slater-type orbitals are expanded as a sum of Gaussian primitives.
// We use STO-3G for d orbitals and STO-4G or STO-3G for s/p as specified
// by the ngauss parameter per shell.
//
// The coefficients below are for a unit-exponent STO (ζ=1).
// For STO with exponent ζ, scale Gaussian exponents by ζ².
//
// References:
//   Stewart, JCP 1970, 52, 431
//   Hehre, Stewart, Pople, JCP 1969, 51, 2657

struct STOnGData {
    int n;             // number of Gaussians
    double alpha[6];   // Gaussian exponents (for ζ=1 STO)
    double coeff[6];   // contraction coefficients (normalized)
};

// STO-3G for 1s orbital
static const STOnGData sto3g_1s = {3,
    {2.227660, 0.405771, 0.109818},
    {0.154329, 0.535328, 0.444635}
};

// STO-4G for 1s orbital
static const STOnGData sto4g_1s = {4,
    {5.216844, 0.938523, 0.295862, 0.106538},
    {0.069884, 0.229589, 0.432738, 0.386587}
};

// STO-3G for 2s orbital
static const STOnGData sto3g_2s = {3,
    {2.581579, 0.532101, 0.159737},
    {-0.089967, 0.402425, 0.687075}
};

// STO-4G for 2s orbital
static const STOnGData sto4g_2s = {4,
    {6.164129, 1.097730, 0.359337, 0.133084},
    {-0.038907, 0.148795, 0.463026, 0.501207}
};

// STO-3G for 2p orbital
static const STOnGData sto3g_2p = {3,
    {1.151398, 0.286400, 0.096215},
    {0.155916, 0.607684, 0.391957}
};

// STO-4G for 2p orbital
static const STOnGData sto4g_2p = {4,
    {2.355468, 0.590513, 0.204666, 0.078683},
    {0.065024, 0.261779, 0.469811, 0.339982}
};

// STO-3G for 3s orbital
static const STOnGData sto3g_3s = {3,
    {3.275717, 0.751978, 0.267487},
    {0.027628, -0.184000, 0.793418}
};

// STO-4G for 3s orbital
static const STOnGData sto4g_3s = {4,
    {7.858272, 1.508030, 0.536228, 0.214618},
    {0.013570, -0.072591, 0.311540, 0.696087}
};

// STO-3G for 3p orbital
static const STOnGData sto3g_3p = {3,
    {1.627559, 0.439985, 0.163314},
    {0.061368, 0.383688, 0.659699}
};

// STO-4G for 3p orbital
static const STOnGData sto4g_3p = {4,
    {3.393880, 0.905151, 0.333086, 0.137147},
    {0.024893, 0.145399, 0.437962, 0.501474}
};

// STO-3G for 3d orbital
static const STOnGData sto3g_3d = {3,
    {1.108350, 0.358801, 0.145585},
    {0.136082, 0.517084, 0.467531}
};

// STO-4G for 4s orbital
static const STOnGData sto4g_4s = {4,
    {10.07836, 2.040939, 0.767600, 0.321494},
    {-0.006573, 0.034458, -0.147524, 0.824936}
};

// STO-4G for 4p orbital
static const STOnGData sto4g_4p = {4,
    {4.763754, 1.311499, 0.499453, 0.213043},
    {0.010168, 0.080974, 0.373816, 0.601725}
};

// STO-3G for 4d orbital
static const STOnGData sto3g_4d = {3,
    {1.300990, 0.442205, 0.186977},
    {0.103889, 0.467799, 0.517195}
};

// STO-4G for 5s orbital
static const STOnGData sto4g_5s = {4,
    {12.69627, 2.679300, 1.040640, 0.447637},
    {0.003477, -0.018754, 0.079660, -0.429900}
};

// STO-4G for 5p orbital
static const STOnGData sto4g_5p = {4,
    {6.300430, 1.784596, 0.696580, 0.303178},
    {-0.004169, 0.041169, 0.307389, 0.667401}
};

// STO-3G for 5d orbital
static const STOnGData sto3g_5d = {3,
    {1.488756, 0.523099, 0.228321},
    {0.083588, 0.425447, 0.557553}
};

/// Get the STO-nG expansion for a given shell type.
/// principalQN: principal quantum number (1-5)
/// angMom: angular momentum (0=s, 1=p, 2=d)
/// ngauss: requested number of Gaussians (3 or 4)
static const STOnGData* getSTOnG(int principalQN, int angMom, int ngauss) {
    if (angMom == 0) {
        // s orbitals
        switch (principalQN) {
            case 1: return (ngauss >= 4) ? &sto4g_1s : &sto3g_1s;
            case 2: return (ngauss >= 4) ? &sto4g_2s : &sto3g_2s;
            case 3: return (ngauss >= 4) ? &sto4g_3s : &sto3g_3s;
            case 4: return &sto4g_4s;
            case 5: return &sto4g_5s;
            default: return &sto4g_4s;
        }
    } else if (angMom == 1) {
        // p orbitals
        switch (principalQN) {
            case 2: return (ngauss >= 4) ? &sto4g_2p : &sto3g_2p;
            case 3: return (ngauss >= 4) ? &sto4g_3p : &sto3g_3p;
            case 4: return &sto4g_4p;
            case 5: return &sto4g_5p;
            default: return &sto4g_4p;
        }
    } else {
        // d orbitals
        switch (principalQN) {
            case 3: return &sto3g_3d;
            case 4: return &sto3g_4d;
            case 5: return &sto3g_5d;
            default: return &sto3g_3d;
        }
    }
}

/// Determine principal quantum number from shell label convention.
/// For main-group: row 1 → n=1, row 2 → n=2, row 3 → n=3, row 4 → n=4, row 5 → n=5
/// For transition metals (d shells): n is one less than the row.
static int getPrincipalQN(int Z, int shellIdx, int angMom) {
    // Determine the period (row) of the element
    int period;
    if (Z <= 2) period = 1;
    else if (Z <= 10) period = 2;
    else if (Z <= 18) period = 3;
    else if (Z <= 36) period = 4;
    else if (Z <= 54) period = 5;
    else if (Z <= 86) period = 6;
    else period = 7;

    if (angMom == 2) {
        // d orbitals: n = period - 1 for transition metals
        // But for main-group with polarization d: n = period
        if (Z >= 21 && Z <= 30) return 3;   // 3d TM
        if (Z >= 39 && Z <= 48) return 4;   // 4d TM
        if (Z >= 57 && Z <= 80) return 5;   // 5d TM/lanthanides
        // Main group with d polarization
        return period;
    }

    return period;
}

// ============================================================================
// MARK: - Gaussian Overlap Integrals
// ============================================================================

/// Compute overlap integral between two primitive Gaussians centered at
/// positions A and B with exponents alpha and beta, and angular momenta lA and lB.
///
/// For s-s overlap:  S = (pi/(a+b))^(3/2) * exp(-a*b/(a+b) * R^2)
/// For s-p overlap:  additional (P-B) factor
/// For p-p overlap:  Obara-Saika recurrence relation
/// For d overlaps:   similarly extended
///
/// We compute this via Obara-Saika recursion for general (l,m) quantum numbers.

/// Boys-like auxiliary: compute Hermite expansion coefficients E_{ij}^t
/// using the McMurchie-Davidson scheme.
/// This gives the overlap distribution coefficient for Cartesian Gaussian
/// product between two 1D Gaussians with centers at PA and PB distances,
/// exponents p = a + b.

/// Compute 1D overlap integral using Obara-Saika recursion.
/// a, b: Gaussian exponents
/// la, lb: angular momentum in this Cartesian direction
/// PA, PB: distances from Gaussian product center P to centers A and B
static double overlap1D(double a, double b, int la, int lb, double PA, double PB) {
    double p = a + b;
    double mu = a * b / p;

    // Recursion base cases:
    // E(0,0) = 1
    // E(i+1, j) = PA * E(i,j) + 1/(2p) * (i * E(i-1,j) + j * E(i,j-1))  [simplified]
    // Actually, for overlap: S(i+1,j) = PA * S(i,j) + 1/(2p) * (i*S(i-1,j) + j*S(i,j-1))

    // Use a small 2D table for recursion (max l = 2 for d orbitals)
    double S[4][4];
    std::memset(S, 0, sizeof(S));

    // S(0,0) = 1 (before prefactor)
    S[0][0] = 1.0;

    // Build up la direction
    for (int i = 0; i < la; i++) {
        S[i + 1][0] = PA * S[i][0];
        if (i > 0) S[i + 1][0] += i * S[i - 1][0] / (2.0 * p);
    }

    // Build up lb direction
    for (int j = 0; j < lb; j++) {
        S[0][j + 1] = PB * S[0][j];
        if (j > 0) S[0][j + 1] += j * S[0][j - 1] / (2.0 * p);

        for (int i = 1; i <= la; i++) {
            S[i][j + 1] = PB * S[i][j] + i * S[i - 1][j] / (2.0 * p);
            if (j > 0) S[i][j + 1] += j * S[i][j - 1] / (2.0 * p);
        }
    }

    return S[la][lb];
}

/// Compute the overlap integral between two Cartesian Gaussian primitives.
/// Returns the integral ⟨G_A(a, lx_a, ly_a, lz_a) | G_B(b, lx_b, ly_b, lz_b)⟩
///
/// a, b: Gaussian exponents
/// Ax, Ay, Az, Bx, By, Bz: center coordinates (Bohr)
/// lxa..lzb: Cartesian angular momentum components
static double gaussianOverlap(double a, double b,
                              double Ax, double Ay, double Az,
                              double Bx, double By, double Bz,
                              int lxa, int lya, int lza,
                              int lxb, int lyb, int lzb) {
    double p = a + b;
    double mu = a * b / p;

    // Gaussian product center
    double Px = (a * Ax + b * Bx) / p;
    double Py = (a * Ay + b * By) / p;
    double Pz = (a * Az + b * Bz) / p;

    double PA_x = Px - Ax, PA_y = Py - Ay, PA_z = Pz - Az;
    double PB_x = Px - Bx, PB_y = Py - By, PB_z = Pz - Bz;

    double R2 = (Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) + (Az - Bz) * (Az - Bz);

    double prefactor = std::exp(-mu * R2) * std::pow(PI / p, 1.5);

    double Sx = overlap1D(a, b, lxa, lxb, PA_x, PB_x);
    double Sy = overlap1D(a, b, lya, lyb, PA_y, PB_y);
    double Sz = overlap1D(a, b, lza, lzb, PA_z, PB_z);

    return prefactor * Sx * Sy * Sz;
}

// Cartesian components for each angular momentum shell.
// s: (0,0,0)
// p: (1,0,0), (0,1,0), (0,0,1)
// d: (2,0,0), (1,1,0), (1,0,1), (0,2,0), (0,1,1), (0,0,2)
//    but we use the 5 spherical harmonics (real solid harmonics)

// For simplicity and correctness we use Cartesian Gaussians then transform
// to spherical for d orbitals. For s and p, Cartesian = spherical.

// Cartesian d has 6 components; spherical d has 5.
// The transformation from 6 Cartesian to 5 spherical d:
//   d_z2     = (2*zz - xx - yy) / (2*sqrt(3))  → normalize
//   d_xz     = xz
//   d_yz     = yz
//   d_x2-y2  = (xx - yy) / 2
//   d_xy     = xy
//
// We handle this by directly computing spherical d overlap via
// Cartesian components with proper transformation coefficients.

struct CartComponent {
    int lx, ly, lz;
    double coeff;  // coefficient in the spherical harmonic expansion
};

// Spherical harmonics for d orbitals (m = -2, -1, 0, 1, 2)
// expressed as linear combinations of Cartesian d functions
// d_{z²} (m=0):     (2zz - xx - yy) * N  where N = 1/(2√3) for norm
// d_{xz} (m=1):     xz
// d_{yz} (m=-1):    yz
// d_{x²-y²} (m=2):  (xx - yy)/2
// d_{xy} (m=-2):    xy

static const int nCartD = 6;
// Indices: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
static const int cartD_lx[] = {2, 1, 1, 0, 0, 0};
static const int cartD_ly[] = {0, 1, 0, 2, 1, 0};
static const int cartD_lz[] = {0, 0, 1, 0, 1, 2};

// For each of the 5 spherical d harmonics, give the Cartesian decomposition
struct SphericalD {
    int nTerms;
    int cartIdx[3];    // indices into the 6 Cartesian d functions
    double cartCoeff[3];
};

// Pre-normalized transformation (assumes Cartesian Gaussians are individually normalized):
// d(0) = d_{z²}: -0.5 * d_{xx} - 0.5 * d_{yy} + 1.0 * d_{zz}  (then renormalize)
// d(+1) = d_{xz}: 1.0 * d_{xz}
// d(-1) = d_{yz}: 1.0 * d_{yz}
// d(+2) = d_{x²-y²}: sqrt(3)/2 * (d_{xx} - d_{yy})
// d(-2) = d_{xy}: 1.0 * d_{xy}
//
// However, for overlap integrals it's simpler to directly compute using
// unnormalized Cartesian Gaussians and build the spherical overlap from those.
// We'll take the approach of computing the full 6x6 Cartesian overlap block
// and transforming to 5x5 spherical.

// Cartesian-to-spherical transformation matrix for d orbitals (5 x 6)
// Rows: spherical d functions (z², xz, yz, x²-y², xy)
// Cols: Cartesian d functions (xx, xy, xz, yy, yz, zz)
// sqrt(3)/2 ≈ 0.866025403784
static constexpr double SQRT3_HALF = 0.86602540378443864676;

static const double cart2sph_d[5][6] = {
    // Cartesian-to-spherical transformation for d orbitals.
    // Rows: spherical d functions (z², xz, yz, x²-y², xy)
    // Cols: Cartesian d functions (xx, xy, xz, yy, yz, zz)
    //
    // For computing overlap S(sph) = T * S(cart) * T^T:
    {-0.5, 0.0, 0.0, -0.5, 0.0, 1.0},         // d_{z²}
    { 0.0, 0.0, 1.0,  0.0, 0.0, 0.0},          // d_{xz}
    { 0.0, 0.0, 0.0,  0.0, 1.0, 0.0},          // d_{yz}
    { SQRT3_HALF, 0.0, 0.0, -SQRT3_HALF, 0.0, 0.0},  // d_{x²-y²}
    { 0.0, 1.0, 0.0,  0.0, 0.0, 0.0},          // d_{xy}
};

// ============================================================================
// MARK: - Coordination Number
// ============================================================================

/// Compute D3-type coordination number for each atom.
/// Uses an exponential counting function:
///   CN_i = Σ_{j≠i} 1 / (1 + exp(-16*(4/3 * (R_cov_i + R_cov_j)/R_ij - 1)))
static void computeCN(const double *pos_bohr, const int *Z, int natom,
                       std::vector<double> &cn) {
    cn.assign(natom, 0.0);
    for (int i = 0; i < natom; i++) {
        for (int j = 0; j < i; j++) {
            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double rij = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (rij < 1e-6) continue;

            double rcov = getCovRad(Z[i]) + getCovRad(Z[j]);
            double arg = -16.0 * (4.0/3.0 * rcov / rij - 1.0);
            double count = 1.0 / (1.0 + std::exp(arg));
            cn[i] += count;
            cn[j] += count;
        }
    }
}

// ============================================================================
// MARK: - Exponential Gamma Function (Coulomb Interaction)
// ============================================================================

/// Compute the exponential gamma function for DFTB Coulomb interaction.
///
/// For two atoms i,j with Hubbard parameters U_i and U_j at distance R:
///   γ(R, U_i, U_j) describes the Coulomb interaction between charge
///   distributions centered on atoms i and j.
///
/// At short range, it interpolates between the on-site Hubbard U and
/// the long-range 1/R Coulomb interaction. The exponential form (Klopman)
/// gives smoother potentials than the original DFTB Mataga-Nishimoto formula.
///
/// Parameters:
///   r_bohr: interatomic distance in Bohr
///   ui, uj: Hubbard parameters (chemical hardness) in Hartree

static double gamma_asymmetric(double r, double taui, double tauj) {
    // One term of the asymmetric gamma function
    // Used when taui ≠ tauj
    double ti2 = taui * taui;
    double tj2 = tauj * tauj;
    double diff2 = ti2 - tj2;
    if (std::fabs(diff2) < 1e-14) return 0.0;

    double prefac = 0.5 * ti2 * ti2 / (diff2 * diff2);
    double expr = std::exp(-taui * r);
    // Include the polynomial correction
    double poly = 1.0 - (tj2 / ti2 - tj2) * r;
    // Full asymmetric expression
    return prefac * expr * (taui - (2.0 * taui / (ti2 - tj2)));
}

/// Ohno-Klopman gamma function for GFN2-xTB Coulomb interaction.
/// gamma(R, gi, gj) = 1 / (R^gExp + gij^(-gExp))^(1/gExp)
/// where gij = 0.5*(gi + gj) (arithmetic average of shell hardnesses)
/// and gExp = 2.0 for GFN2 (alphaj parameter).
/// On-site (R=0): gamma = gij.
static double ohno_gamma(double r_bohr, double gi, double gj) {
    double gij = 0.5 * (gi + gj);
    if (r_bohr < 1e-8) {
        return gij;  // On-site limit
    }
    // gExp = 2.0: gamma = 1/sqrt(R^2 + 1/gij^2)
    double gij_inv2 = 1.0 / (gij * gij);
    return 1.0 / std::sqrt(r_bohr * r_bohr + gij_inv2);
}

// ============================================================================
// MARK: - Shell-Resolved Gamma with GFN2 Averaging
// ============================================================================

/// Shell-resolved Ohno-Klopman gamma using arithmetic averaging.
/// Shell hardness: gi = gam * lgam[shell]
static double shell_gamma(double r_bohr, double gam_i, double lgam_i,
                          double gam_j, double lgam_j) {
    double gi = gam_i * lgam_i;
    double gj = gam_j * lgam_j;
    return ohno_gamma(r_bohr, gi, gj);
}

// ============================================================================
// MARK: - Repulsion Energy
// ============================================================================

/// Compute the GFN2-xTB repulsion energy.
/// E_rep = Σ_{A>B} Z_eff_A * Z_eff_B / R_AB * exp(-sqrt(α_A * α_B) * R_AB^k_exp)
///
/// This is a simple repulsive potential that keeps atoms apart at short range.
/// Z_eff is the effective nuclear charge, and arep controls the steepness.
static double computeRepulsion(const double *pos_bohr, const int *Z, int natom) {
    double Erep = 0.0;
    for (int i = 0; i < natom; i++) {
        const auto &pi = gfn2Params[Z[i]];
        if (pi.atomicNumber == 0) continue;
        for (int j = 0; j < i; j++) {
            const auto &pj = gfn2Params[Z[j]];
            if (pj.atomicNumber == 0) continue;

            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < 1e-6) continue;

            double alpha = std::sqrt(pi.arep * pj.arep);

            // k_exp for light atoms (H-He) can differ
            double kexp = GFN2_KEXP;
            // For H-H or H-X interactions, potentially use klight
            // (In practice klight=1.0 so this is a no-op for GFN2 but included for completeness)

            double rk = std::pow(r, kexp);
            double rep = pi.zeff * pj.zeff / r * std::exp(-alpha * rk);
            Erep += rep;
        }
    }
    return Erep;
}

// ============================================================================
// MARK: - Overlap & Hamiltonian Matrix Construction
// ============================================================================

/// Information about a single shell in the basis set.
struct ShellInfo {
    int atomIdx;        // atom index
    int shellIdx;       // shell index on that atom (0, 1, 2)
    int angMom;         // angular momentum (0=s, 1=p, 2=d)
    int basisStart;     // starting index in the full basis
    int basisCount;     // number of basis functions (1, 3, or 5)
    double slaterExp;   // Slater exponent
    double selfEnergy;  // self-energy level (eV)
    double refOcc;      // reference occupation
    double hubbard;     // effective Hubbard U for this shell
    int principalQN;    // principal quantum number
    int ngauss;         // number of Gaussians in STO expansion
};

/// Build the overlap matrix S between all basis functions.
/// Uses STO-nG expansion: each STO is approximated by n Gaussians,
/// and Gaussian overlap integrals are computed analytically.
static void buildOverlapMatrix(const std::vector<ShellInfo> &shells,
                               const double *pos_bohr,
                               int nbasis,
                               std::vector<double> &S) {
    S.assign(nbasis * nbasis, 0.0);

    // For each pair of shells, compute the overlap block
    for (size_t si = 0; si < shells.size(); si++) {
        const auto &shA = shells[si];
        int atomA = shA.atomIdx;
        double Ax = pos_bohr[3*atomA], Ay = pos_bohr[3*atomA+1], Az = pos_bohr[3*atomA+2];
        int lA = shA.angMom;
        double zetaA = shA.slaterExp;
        const STOnGData *stoA = getSTOnG(shA.principalQN, lA, shA.ngauss);

        for (size_t sj = si; sj < shells.size(); sj++) {
            const auto &shB = shells[sj];
            int atomB = shB.atomIdx;
            double Bx = pos_bohr[3*atomB], By = pos_bohr[3*atomB+1], Bz = pos_bohr[3*atomB+2];
            int lB = shB.angMom;
            double zetaB = shB.slaterExp;
            const STOnGData *stoB = getSTOnG(shB.principalQN, lB, shB.ngauss);

            // Number of Cartesian components
            int nCartA = (lA + 1) * (lA + 2) / 2;  // s:1, p:3, d:6
            int nCartB = (lB + 1) * (lB + 2) / 2;

            // Compute Cartesian overlap block using STO-nG primitives
            // S_cart[ia][ib] = Σ_k Σ_l c_k * c_l * ⟨G_k|G_l⟩
            std::vector<double> S_cart(nCartA * nCartB, 0.0);

            // Generate Cartesian angular momentum components
            auto makeCart = [](int l, std::vector<int> &lx, std::vector<int> &ly, std::vector<int> &lz) {
                lx.clear(); ly.clear(); lz.clear();
                for (int ix = l; ix >= 0; ix--) {
                    for (int iy = l - ix; iy >= 0; iy--) {
                        int iz = l - ix - iy;
                        lx.push_back(ix); ly.push_back(iy); lz.push_back(iz);
                    }
                }
            };

            std::vector<int> lxA, lyA, lzA, lxB, lyB, lzB;
            makeCart(lA, lxA, lyA, lzA);
            makeCart(lB, lxB, lyB, lzB);

            // Loop over STO-nG primitive pairs
            for (int k = 0; k < stoA->n; k++) {
                double alphaK = stoA->alpha[k] * zetaA * zetaA;  // scale by ζ²
                double cK = stoA->coeff[k];
                for (int l = 0; l < stoB->n; l++) {
                    double alphaL = stoB->alpha[l] * zetaB * zetaB;
                    double cL = stoB->coeff[l];
                    double cc = cK * cL;

                    // Compute overlap for each Cartesian component pair
                    for (int ia = 0; ia < nCartA; ia++) {
                        for (int ib = 0; ib < nCartB; ib++) {
                            double ov = gaussianOverlap(alphaK, alphaL,
                                                        Ax, Ay, Az, Bx, By, Bz,
                                                        lxA[ia], lyA[ia], lzA[ia],
                                                        lxB[ib], lyB[ib], lzB[ib]);
                            S_cart[ia * nCartB + ib] += cc * ov;
                        }
                    }
                }
            }

            // Transform Cartesian → spherical if needed
            // s and p: Cartesian = spherical (nCart == nBasis)
            // d: need to transform 6 Cartesian → 5 spherical

            int nA = shA.basisCount;  // spherical count
            int nB = shB.basisCount;

            if (lA <= 1 && lB <= 1) {
                // Both s or p: direct copy
                for (int ia = 0; ia < nA; ia++) {
                    for (int ib = 0; ib < nB; ib++) {
                        double val = S_cart[ia * nCartB + ib];
                        int ii = shA.basisStart + ia;
                        int jj = shB.basisStart + ib;
                        S[ii * nbasis + jj] = val;
                        if (si != sj) S[jj * nbasis + ii] = val;
                    }
                }
            } else if (lA == 2 && lB <= 1) {
                // A is d (6 cart → 5 sph), B is s or p
                for (int ma = 0; ma < 5; ma++) {
                    for (int ib = 0; ib < nB; ib++) {
                        double val = 0.0;
                        for (int ca = 0; ca < 6; ca++) {
                            val += cart2sph_d[ma][ca] * S_cart[ca * nCartB + ib];
                        }
                        int ii = shA.basisStart + ma;
                        int jj = shB.basisStart + ib;
                        S[ii * nbasis + jj] = val;
                        if (si != sj) S[jj * nbasis + ii] = val;
                    }
                }
            } else if (lA <= 1 && lB == 2) {
                // A is s or p, B is d
                for (int ia = 0; ia < nA; ia++) {
                    for (int mb = 0; mb < 5; mb++) {
                        double val = 0.0;
                        for (int cb = 0; cb < 6; cb++) {
                            val += cart2sph_d[mb][cb] * S_cart[ia * nCartB + cb];
                        }
                        int ii = shA.basisStart + ia;
                        int jj = shB.basisStart + mb;
                        S[ii * nbasis + jj] = val;
                        if (si != sj) S[jj * nbasis + ii] = val;
                    }
                }
            } else {
                // Both d: transform both indices
                for (int ma = 0; ma < 5; ma++) {
                    for (int mb = 0; mb < 5; mb++) {
                        double val = 0.0;
                        for (int ca = 0; ca < 6; ca++) {
                            for (int cb = 0; cb < 6; cb++) {
                                val += cart2sph_d[ma][ca] * cart2sph_d[mb][cb]
                                       * S_cart[ca * nCartB + cb];
                            }
                        }
                        int ii = shA.basisStart + ma;
                        int jj = shB.basisStart + mb;
                        S[ii * nbasis + jj] = val;
                        if (si != sj) S[jj * nbasis + ii] = val;
                    }
                }
            }
        }
    }
}

/// Get the shell-pair scaling factor K_{l_i, l_j} from the Hamiltonian parameters.
static double getShellPairK(int li, int lj) {
    int lmin = std::min(li, lj);
    int lmax = std::max(li, lj);
    if (lmin == 0 && lmax == 0) return GFN2_KAB_SS;
    if (lmin == 0 && lmax == 1) return 0.5 * (GFN2_KAB_SS + GFN2_KAB_PP);
    if (lmin == 1 && lmax == 1) return GFN2_KAB_PP;
    if (lmin == 0 && lmax == 2) return GFN2_KAB_SD;
    if (lmin == 1 && lmax == 2) return GFN2_KAB_PD;
    if (lmin == 2 && lmax == 2) return GFN2_KAB_DD;
    return GFN2_KAB_PP; // fallback
}

/// Build the zero-th order Hamiltonian H0.
/// In GFN2-xTB, the off-diagonal elements are:
///   H0_{μν} = K_{l_μ, l_ν} * S_{μν} * (h_μ + h_ν) / 2
/// where h_μ is the CN-dependent self-energy of shell μ, and
/// K is a shell-pair scaling factor.
///
/// The diagonal elements are the self-energies themselves:
///   H0_{μμ} = h_μ
///
/// The self-energies are modified by:
///   1. Coordination number dependence: h(CN) = h0 + kcn * CN
///   2. Shell polynomial correction: h_poly = h * (1 + shpoly * (CN - CN_ref))
///   3. Electronegativity scaling between different atoms
static void buildH0(const std::vector<ShellInfo> &shells,
                    const double *pos_bohr, const int *Z, int natom,
                    int nbasis,
                    const std::vector<double> &S,
                    const std::vector<double> &cn,
                    std::vector<double> &H0) {
    H0.assign(nbasis * nbasis, 0.0);

    // Compute effective shell levels including CN dependence
    // Reference: hamiltonian.F90 getSelfEnergyFlat
    //   selfEnergy = base - kCN * CN  (sign is MINUS)
    std::vector<double> shellLevels(shells.size());
    for (size_t si = 0; si < shells.size(); si++) {
        const auto &sh = shells[si];
        int ai = sh.atomIdx;
        const auto &par = gfn2Params[Z[ai]];

        // Base self-energy in eV → convert to Hartree
        double h0 = sh.selfEnergy * EV_TO_HARTREE;

        // CN-dependent shift: h0 = h0 - kCN * CN (reference sign convention)
        double kcn_val = par.kcn[sh.shellIdx] * EV_TO_HARTREE;
        h0 -= kcn_val * cn[ai];

        shellLevels[si] = h0;
    }

    // Fill H0 matrix
    for (size_t si = 0; si < shells.size(); si++) {
        const auto &shA = shells[si];
        int atomA = shA.atomIdx;

        // Diagonal blocks: self-energy (no shellPoly on diagonal)
        for (int ia = 0; ia < shA.basisCount; ia++) {
            int ii = shA.basisStart + ia;
            H0[ii * nbasis + ii] = shellLevels[si];
        }

        // Off-diagonal blocks
        for (size_t sj = si + 1; sj < shells.size(); sj++) {
            const auto &shB = shells[sj];
            int atomB = shB.atomIdx;

            if (atomA == atomB) continue;  // on-site off-diag is zero in DFTB

            // Interatomic distance
            double dx = pos_bohr[3*atomA]   - pos_bohr[3*atomB];
            double dy = pos_bohr[3*atomA+1] - pos_bohr[3*atomB+1];
            double dz = pos_bohr[3*atomA+2] - pos_bohr[3*atomB+2];
            double rab = std::sqrt(dx*dx + dy*dy + dz*dz);

            // Shell-pair scaling constant K (ss=1.85, pp=2.23, etc.)
            double K = getShellPairK(shA.angMom, shB.angMom);

            // Electronegativity scaling
            double enA = gfn2Params[Z[atomA]].en;
            double enB = gfn2Params[Z[atomB]].en;
            double enDiff = enA - enB;
            double enScale = 1.0 + GFN2_ENSCALE * enDiff * enDiff;

            // Distance-dependent shellPoly correction (reference: scc_core.f90 lines 728-754)
            // r = rab / (Ri + Rj), then rf = 1.0 + shpoly * sqrt(r)
            // shpoly values are pre-scaled by 0.01 in our parameter table
            double radSum = gfn2Params[Z[atomA]].atomicRad + gfn2Params[Z[atomB]].atomicRad;
            double rNorm = (radSum > 1e-8) ? rab / radSum : 0.0;
            double sqrtR = std::sqrt(rNorm);
            double rf1 = 1.0 + gfn2Params[Z[atomA]].shpoly[shA.shellIdx] * sqrtR;
            double rf2 = 1.0 + gfn2Params[Z[atomB]].shpoly[shB.shellIdx] * sqrtR;
            double shPoly = rf1 * rf2;

            // Slater exponent ratio scaling (reference: hamiltonian.F90 lines 245-251)
            // zetaij = (2*sqrt(zi*zj)/(zi+zj))^wExp, wExp=0.5
            double zi = gfn2Params[Z[atomA]].slaterExp[shA.shellIdx];
            double zj = gfn2Params[Z[atomB]].slaterExp[shB.shellIdx];
            double zetaRatio = 2.0 * std::sqrt(zi * zj) / (zi + zj);
            double zetaij = std::sqrt(zetaRatio);  // wExp = 0.5

            // hav = 0.5 * K * (hi + hj) * zetaij * enScale * shellPoly
            double hAvg = 0.5 * K * (shellLevels[si] + shellLevels[sj]) * zetaij * enScale * shPoly;

            for (int ia = 0; ia < shA.basisCount; ia++) {
                for (int ib = 0; ib < shB.basisCount; ib++) {
                    int ii = shA.basisStart + ia;
                    int jj = shB.basisStart + ib;
                    double val = hAvg * S[ii * nbasis + jj];
                    H0[ii * nbasis + jj] = val;
                    H0[jj * nbasis + ii] = val;
                }
            }
        }
    }
}

// ============================================================================
// MARK: - Generalized Eigenvalue Problem Solver
// ============================================================================

/// Solve the generalized symmetric eigenvalue problem: H*C = S*C*ε
/// Uses LAPACK dsygv from Apple's Accelerate framework.
///
/// On input:
///   H: n×n symmetric matrix (Hamiltonian), column-major
///   S: n×n symmetric positive-definite matrix (overlap), column-major
///   n: matrix dimension
///
/// On output:
///   eigenvalues: n eigenvalues in ascending order
///   eigenvectors: n×n matrix of eigenvectors (columns), column-major
///
/// Returns true on success.
static bool solveGenEig(const std::vector<double> &H_in,
                        const std::vector<double> &S_in,
                        int n,
                        std::vector<double> &eigenvalues,
                        std::vector<double> &eigenvectors) {
    if (n == 0) return false;

    // dsygv modifies both matrices in place, so copy them
    // Our matrices are stored row-major, but LAPACK expects column-major.
    // Since H and S are symmetric, row-major = column-major. Good.
    std::vector<double> A(H_in);   // will contain eigenvectors on output
    std::vector<double> B(S_in);   // will contain Cholesky factor on output
    eigenvalues.resize(n);

    int itype = 1;                  // A*x = lambda*B*x
    char jobz = 'V';               // compute eigenvalues and eigenvectors
    char uplo = 'U';               // upper triangle stored
    int N = n;
    int lda = n;
    int ldb = n;
    int lwork = std::max(1, 3 * N - 1);
    std::vector<double> work(lwork);
    int info = 0;

    dsygv_(&itype, &jobz, &uplo, &N, A.data(), &lda,
           B.data(), &ldb, eigenvalues.data(),
           work.data(), &lwork, &info);

    if (info != 0) {
        return false;
    }

    eigenvectors = std::move(A);
    return true;
}

// ============================================================================
// MARK: - Mulliken Population Analysis
// ============================================================================

/// Compute Mulliken charges from density matrix P and overlap matrix S.
///
/// The Mulliken population on atom A is:
///   q_A = Z_A - Σ_{μ∈A} Σ_ν (P*S)_{μν}
/// where the sum over μ runs over basis functions centered on atom A.
///
/// Shell population excess (positive = gained electrons, for SCC convergence):
///   Δpop_shell = Σ_{μ∈shell} (PS)_{μμ} - n_ref_shell
static void mullikenCharges(const std::vector<double> &P,
                            const std::vector<double> &S,
                            const std::vector<ShellInfo> &shells,
                            int nbasis, int natom,
                            std::vector<double> &shellCharges,
                            std::vector<double> &atomCharges) {
    // Compute PS product diagonal: (P*S)_{μμ} = Σ_ν P_{μν} * S_{νμ}
    // Actually for Mulliken we need: gross population = Σ_ν P_{μν} * S_{μν}
    // (element-wise product summed over one index)

    shellCharges.resize(shells.size(), 0.0);
    atomCharges.assign(natom, 0.0);

    for (size_t si = 0; si < shells.size(); si++) {
        const auto &sh = shells[si];
        double pop = 0.0;
        // Use BLAS ddot for each basis function row: pop += dot(P_row, S_row)
        for (int ia = 0; ia < sh.basisCount; ia++) {
            int mu = sh.basisStart + ia;
            pop += cblas_ddot(nbasis, &P[mu * nbasis], 1, &S[mu * nbasis], 1);
        }
        shellCharges[si] = pop - sh.refOcc;
        atomCharges[sh.atomIdx] += shellCharges[si];
    }
}

// ============================================================================
// MARK: - SCC Potential (Charge-Dependent Hamiltonian Shift)
// ============================================================================

/// Build the charge-dependent potential shift for the Hamiltonian.
/// In SCC-DFTB, the Hamiltonian is modified as:
///   H_{μν} = H0_{μν} + 1/2 * S_{μν} * Σ_C (γ_{AC} + γ_{BC}) * Δq_C
/// where A, B are the atoms of basis functions μ, ν.
///
/// This includes both second-order (γ * Δq) and third-order (Γ * Δq²) terms.
static void buildChargeShift(const std::vector<ShellInfo> &shells,
                             const double *pos_bohr, const int *Z, int natom,
                             int nbasis,
                             const std::vector<double> &shellCharges,
                             const std::vector<double> &atomCharges,
                             const std::vector<double> &S,
                             std::vector<double> &Vshift) {
    Vshift.assign(nbasis * nbasis, 0.0);

    // Precompute shell-resolved gamma matrix
    // γ_{shell_i, shell_j} between all shell pairs
    int nshells = (int)shells.size();

    // Compute the potential at each shell due to all charge fluctuations
    // V_i = Σ_j γ_{ij} * Δq_j  (second order)
    //      + Γ_i * Δq_i²        (third order, on-site only)
    std::vector<double> shellPotential(nshells, 0.0);

    for (int si = 0; si < nshells; si++) {
        int atomA = shells[si].atomIdx;
        double gam_A = gfn2Params[Z[atomA]].gam;
        double lgam_A = gfn2Params[Z[atomA]].lgam[shells[si].shellIdx];

        double V = 0.0;

        for (int sj = 0; sj < nshells; sj++) {
            int atomB = shells[sj].atomIdx;
            double gam_B = gfn2Params[Z[atomB]].gam;
            double lgam_B = gfn2Params[Z[atomB]].lgam[shells[sj].shellIdx];

            double r;
            if (atomA == atomB) {
                r = 0.0;
            } else {
                double dx = pos_bohr[3*atomA]   - pos_bohr[3*atomB];
                double dy = pos_bohr[3*atomA+1] - pos_bohr[3*atomB+1];
                double dz = pos_bohr[3*atomA+2] - pos_bohr[3*atomB+2];
                r = std::sqrt(dx*dx + dy*dy + dz*dz);
            }

            double gam = shell_gamma(r, gam_A, lgam_A, gam_B, lgam_B);
            V += gam * shellCharges[sj];
        }

        // Third-order correction (on-site):
        // V_3rd = Γ_A * Δq_A  (derivative of 1/3 * Γ * Δq³)
        int l = shells[si].angMom;
        double thirdScale = (l == 0) ? GFN2_THIRDORDER_S :
                            (l == 1) ? GFN2_THIRDORDER_P : GFN2_THIRDORDER_D;
        double gam3 = gfn2Params[Z[atomA]].gam3 * thirdScale;
        V += gam3 * atomCharges[atomA] * atomCharges[atomA];

        shellPotential[si] = V;
    }

    // Build the potential matrix:
    // Vshift_{μν} = 1/2 * S_{μν} * (V_{shell(μ)} + V_{shell(ν)})
    for (int si = 0; si < nshells; si++) {
        for (int sj = si; sj < nshells; sj++) {
            double Vavg = 0.5 * (shellPotential[si] + shellPotential[sj]);
            const auto &shA = shells[si];
            const auto &shB = shells[sj];

            for (int ia = 0; ia < shA.basisCount; ia++) {
                for (int ib = 0; ib < shB.basisCount; ib++) {
                    int ii = shA.basisStart + ia;
                    int jj = shB.basisStart + ib;
                    double val = Vavg * S[ii * nbasis + jj];
                    Vshift[ii * nbasis + jj] = val;
                    if (ii != jj) Vshift[jj * nbasis + ii] = val;
                }
            }
        }
    }
}

/// Compute the Coulomb energy from shell charges.
/// E_coul = 1/2 * Σ_{ij} Δq_i * γ_{ij} * Δq_j
///        + 1/3 * Σ_A Γ_A * Δq_A³  (third order)
static double computeCoulombEnergy(const std::vector<ShellInfo> &shells,
                                   const double *pos_bohr, const int *Z, int natom,
                                   const std::vector<double> &shellCharges,
                                   const std::vector<double> &atomCharges) {
    int nshells = (int)shells.size();
    double E2 = 0.0;

    for (int si = 0; si < nshells; si++) {
        int atomA = shells[si].atomIdx;
        double gam_A = gfn2Params[Z[atomA]].gam;
        double lgam_A = gfn2Params[Z[atomA]].lgam[shells[si].shellIdx];

        for (int sj = si; sj < nshells; sj++) {
            int atomB = shells[sj].atomIdx;
            double gam_B = gfn2Params[Z[atomB]].gam;
            double lgam_B = gfn2Params[Z[atomB]].lgam[shells[sj].shellIdx];

            double r;
            if (atomA == atomB) {
                r = 0.0;
            } else {
                double dx = pos_bohr[3*atomA]   - pos_bohr[3*atomB];
                double dy = pos_bohr[3*atomA+1] - pos_bohr[3*atomB+1];
                double dz = pos_bohr[3*atomA+2] - pos_bohr[3*atomB+2];
                r = std::sqrt(dx*dx + dy*dy + dz*dz);
            }

            double gam = shell_gamma(r, gam_A, lgam_A, gam_B, lgam_B);
            double contrib = shellCharges[si] * gam * shellCharges[sj];
            if (si == sj) {
                E2 += 0.5 * contrib;
            } else {
                E2 += contrib;
            }
        }
    }

    // Third-order energy: 1/3 * Σ_A Γ_A * Δq_A³
    double E3 = 0.0;
    for (int a = 0; a < natom; a++) {
        // Use the average third-order parameter weighted by shell charges
        double dq = atomCharges[a];
        double gam3 = gfn2Params[Z[a]].gam3;
        // Shell-resolved third-order contributions
        E3 += (1.0 / 3.0) * gam3 * dq * dq * dq;
    }

    return E2 + E3;
}

// ============================================================================
// MARK: - Main SCC-DFTB Driver
// ============================================================================

extern "C" {

DruseXTBChargeResult* druse_xtb_compute_charges(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    int32_t maxIterations)
{
    // Initialize parameters if needed
    initGFN2Params();

    // Allocate result
    auto *result = new DruseXTBChargeResult();
    std::memset(result, 0, sizeof(DruseXTBChargeResult));
    result->atomCount = atomCount;

    if (atomCount <= 0) {
        result->success = false;
        std::snprintf(result->errorMessage, 512, "Invalid atom count: %d", atomCount);
        return result;
    }

    // Validate elements
    for (int i = 0; i < atomCount; i++) {
        int Z = atomicNumbers[i];
        if (Z <= 0 || Z > MAX_ATOMIC_NUM || gfn2Params[Z].atomicNumber == 0) {
            result->success = false;
            std::snprintf(result->errorMessage, 512,
                         "Unsupported element Z=%d at atom %d", Z, i);
            return result;
        }
    }

    // Convert positions from Angstrom to Bohr
    std::vector<double> pos_bohr(3 * atomCount);
    for (int i = 0; i < 3 * atomCount; i++) {
        pos_bohr[i] = positions[i] * ANG_TO_BOHR;
    }

    // Store atomic numbers as int array
    std::vector<int> Z(atomCount);
    for (int i = 0; i < atomCount; i++) Z[i] = atomicNumbers[i];

    // =========================================================================
    // Step 1: Set up the basis set — determine shells and total basis size
    // =========================================================================

    std::vector<ShellInfo> shells;
    int nbasis = 0;
    int totalElectrons = -totalCharge;  // electrons = nuclear charge - total charge

    for (int a = 0; a < atomCount; a++) {
        const auto &par = gfn2Params[Z[a]];
        totalElectrons += Z[a];  // we'll subtract core electrons below

        for (int s = 0; s < par.nShells; s++) {
            ShellInfo sh;
            sh.atomIdx = a;
            sh.shellIdx = s;
            sh.angMom = par.shellAngMom[s];
            sh.basisStart = nbasis;
            sh.basisCount = basisCount(sh.angMom);
            sh.slaterExp = par.slaterExp[s];
            sh.selfEnergy = par.selfEnergy[s];
            sh.refOcc = par.refOcc[s];
            sh.hubbard = par.gam * par.lgam[s];
            sh.principalQN = getPrincipalQN(Z[a], s, sh.angMom);
            sh.ngauss = par.ngauss[s];
            nbasis += sh.basisCount;
            shells.push_back(sh);
        }
    }

    // For GFN2-xTB, the valence electron count is the sum of reference occupations
    // (This automatically excludes core electrons.)
    int nValElectrons = 0;
    for (const auto &sh : shells) {
        nValElectrons += (int)std::round(sh.refOcc);
    }
    // Adjust for total charge
    nValElectrons -= totalCharge;

    if (nValElectrons < 0 || nValElectrons > 2 * nbasis) {
        result->success = false;
        std::snprintf(result->errorMessage, 512,
                     "Invalid electron count: %d valence electrons for %d basis functions",
                     nValElectrons, nbasis);
        return result;
    }

    int nOccupied = nValElectrons / 2;  // closed-shell: each orbital holds 2 electrons
    bool openShell = (nValElectrons % 2 != 0);
    // For simplicity, handle only closed-shell or HOMO singly occupied
    // (appropriate for most drug molecules)

    // =========================================================================
    // Step 2: Compute coordination numbers
    // =========================================================================

    std::vector<double> cn;
    computeCN(pos_bohr.data(), Z.data(), atomCount, cn);

    // =========================================================================
    // Step 3: Build overlap matrix S
    // =========================================================================

    std::vector<double> S;
    buildOverlapMatrix(shells, pos_bohr.data(), nbasis, S);

    // Verify overlap matrix: diagonal should be positive, off-diagonal bounded
    for (int i = 0; i < nbasis; i++) {
        if (S[i * nbasis + i] < 0.01) {
            // Try to fix near-zero diagonal by adding small value
            S[i * nbasis + i] = std::max(S[i * nbasis + i], 0.1);
        }
    }

    // Normalize the overlap matrix: ensure S_{ii} = 1 for orthonormal-like basis
    // This compensates for STO-nG normalization differences
    std::vector<double> normFactor(nbasis);
    for (int i = 0; i < nbasis; i++) {
        normFactor[i] = 1.0 / std::sqrt(std::fabs(S[i * nbasis + i]));
    }
    for (int i = 0; i < nbasis; i++) {
        for (int j = 0; j < nbasis; j++) {
            S[i * nbasis + j] *= normFactor[i] * normFactor[j];
        }
    }

    // =========================================================================
    // Step 4: Build core Hamiltonian H0
    // =========================================================================

    std::vector<double> H0;
    buildH0(shells, pos_bohr.data(), Z.data(), atomCount, nbasis, S, cn, H0);

    // Also normalize H0 consistently with the overlap normalization
    // (The self-energy diagonal is already correct; off-diagonal scaled with S)

    // =========================================================================
    // Step 5: Compute repulsion energy
    // =========================================================================

    double Erep = computeRepulsion(pos_bohr.data(), Z.data(), atomCount);

    // =========================================================================
    // Step 6: Initial guess — charges from electronegativity equalization
    // =========================================================================

    std::vector<double> shellCharges(shells.size(), 0.0);
    std::vector<double> atomCharges(atomCount, 0.0);

    // Start with zero charge fluctuations (reference occupation)
    // This is the standard starting point for SCC-DFTB

    // =========================================================================
    // Step 7: Self-Consistent Charge (SCC) Iteration
    // =========================================================================
    //
    // The SCC loop:
    // 1. Build H = H0 + V(q) where V depends on current charges
    // 2. Solve HC = SCε (generalized eigenvalue problem)
    // 3. Occupy lowest orbitals to get density matrix P
    // 4. Compute new Mulliken charges from P and S
    // 5. Mix new and old charges for stability
    // 6. Check convergence
    //
    // Convergence criterion: max |Δq| < 1e-6 e
    // Mixing: simple linear mixing with damping factor α
    // =========================================================================

    double convergenceThreshold = 1e-6;
    double mixingFactor = 0.3;  // charge mixing damping (0.3 = 30% new charges)
    bool converged = false;
    int iter = 0;
    double Eelec = 0.0;

    // For Anderson/Broyden mixing, store history
    std::vector<double> prevShellCharges(shells.size(), 0.0);
    std::vector<double> prevResidual(shells.size(), 0.0);
    bool hasHistory = false;

    std::vector<double> H(nbasis * nbasis);
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;

    for (iter = 0; iter < maxIterations; iter++) {
        // 7a. Build charge-dependent potential
        std::vector<double> Vshift;
        buildChargeShift(shells, pos_bohr.data(), Z.data(), atomCount,
                         nbasis, shellCharges, atomCharges, S, Vshift);

        // 7b. Construct full Hamiltonian: H = H0 + Vshift
        for (int i = 0; i < nbasis * nbasis; i++) {
            H[i] = H0[i] + Vshift[i];
        }

        // 7c. Solve generalized eigenvalue problem H*C = S*C*ε
        if (!solveGenEig(H, S, nbasis, eigenvalues, eigenvectors)) {
            // If eigenvalue solve fails, try increasing overlap regularization
            for (int i = 0; i < nbasis; i++) {
                S[i * nbasis + i] += 1e-6;
            }
            if (!solveGenEig(H, S, nbasis, eigenvalues, eigenvectors)) {
                result->success = false;
                std::snprintf(result->errorMessage, 512,
                             "Eigenvalue solve failed at SCC iteration %d", iter);
                return result;
            }
        }

        // 7d. Build density matrix P = C_occ * diag(f) * C_occ^T
        // Use BLAS dgemm for O(nbasis² × nOcc) instead of manual triple loop.
        // P = 2.0 * C_occ * C_occ^T (closed-shell occupied block)
        std::vector<double> P(nbasis * nbasis, 0.0);
        {
            int nOcc = openShell ? nOccupied + 1 : nOccupied;
            nOcc = std::min(nOcc, nbasis);

            // Build scaled occupied eigenvector block: C_scaled[k,mu] = sqrt(f_k) * C[k,mu]
            // eigenvectors are column-major from LAPACK: C[k*nbasis + mu]
            std::vector<double> C_scaled(nOcc * nbasis);
            for (int k = 0; k < nOccupied && k < nbasis; k++) {
                double scale = std::sqrt(2.0);
                for (int mu = 0; mu < nbasis; mu++) {
                    C_scaled[k * nbasis + mu] = scale * eigenvectors[k * nbasis + mu];
                }
            }
            if (openShell && nOccupied < nbasis) {
                // Singly-occupied HOMO: occupation = 1.0
                int k = nOccupied;
                for (int mu = 0; mu < nbasis; mu++) {
                    C_scaled[k * nbasis + mu] = eigenvectors[k * nbasis + mu];
                }
            }

            // P = C_scaled^T * C_scaled using BLAS dgemm
            // C_scaled is nOcc × nbasis (row-major), we want P = C_scaled^T * C_scaled
            // In column-major (BLAS convention): P(nbasis,nbasis) = C^T(nbasis,nOcc) * C(nOcc,nbasis)
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        nbasis, nbasis, nOcc,
                        1.0, C_scaled.data(), nbasis,
                             C_scaled.data(), nbasis,
                        0.0, P.data(), nbasis);
        }

        // 7e. Compute new Mulliken charges
        std::vector<double> newShellCharges, newAtomCharges;
        mullikenCharges(P, S, shells, nbasis, atomCount,
                        newShellCharges, newAtomCharges);

        // 7f. Check convergence
        double maxDelta = 0.0;
        for (size_t si = 0; si < shells.size(); si++) {
            double delta = std::fabs(newShellCharges[si] - shellCharges[si]);
            maxDelta = std::max(maxDelta, delta);
        }

        if (maxDelta < convergenceThreshold && iter > 0) {
            shellCharges = newShellCharges;
            atomCharges = newAtomCharges;
            converged = true;

            // Compute electronic energy:
            // E_elec = Σ_{μν} P_{μν} * H0_{μν} + E_coulomb
            Eelec = 0.0;
            for (int mu = 0; mu < nbasis; mu++) {
                for (int nu = 0; nu < nbasis; nu++) {
                    Eelec += P[mu * nbasis + nu] * H0[mu * nbasis + nu];
                }
            }
            Eelec += computeCoulombEnergy(shells, pos_bohr.data(), Z.data(),
                                          atomCount, shellCharges, atomCharges);
            break;
        }

        // 7g. Mix charges for stability
        // Use Anderson mixing if we have history, otherwise simple damping
        std::vector<double> residual(shells.size());
        for (size_t si = 0; si < shells.size(); si++) {
            residual[si] = newShellCharges[si] - shellCharges[si];
        }

        if (hasHistory && iter > 1) {
            // Anderson mixing: use current and previous residuals to
            // extrapolate better charges
            double dotRR = 0, dotRdR = 0;
            for (size_t si = 0; si < shells.size(); si++) {
                double dr = residual[si] - prevResidual[si];
                dotRR += residual[si] * residual[si];
                dotRdR += residual[si] * dr;
            }
            double beta = (std::fabs(dotRdR) > 1e-16) ? dotRR / dotRdR : 0.0;
            beta = std::max(-1.0, std::min(1.0, beta));

            for (size_t si = 0; si < shells.size(); si++) {
                double mixed = (1.0 - beta) * (shellCharges[si] + mixingFactor * residual[si])
                             + beta * (prevShellCharges[si] + mixingFactor * prevResidual[si]);
                prevShellCharges[si] = shellCharges[si];
                prevResidual[si] = residual[si];
                shellCharges[si] = mixed;
            }
        } else {
            // Simple damping
            for (size_t si = 0; si < shells.size(); si++) {
                prevShellCharges[si] = shellCharges[si];
                prevResidual[si] = residual[si];
                shellCharges[si] += mixingFactor * residual[si];
            }
            hasHistory = true;
        }

        // Recompute atom charges from shell charges
        std::fill(atomCharges.begin(), atomCharges.end(), 0.0);
        for (size_t si = 0; si < shells.size(); si++) {
            atomCharges[shells[si].atomIdx] += shellCharges[si];
        }

        // Enforce charge neutrality: shift charges so they sum to totalCharge
        double qsum = std::accumulate(atomCharges.begin(), atomCharges.end(), 0.0);
        double shift = (totalCharge - qsum) / atomCount;
        for (int a = 0; a < atomCount; a++) {
            atomCharges[a] += shift;
        }
        // Distribute shift back to shell charges proportionally
        for (size_t si = 0; si < shells.size(); si++) {
            shellCharges[si] += shift * shells[si].refOcc /
                std::max(1.0, (double)gfn2Params[Z[shells[si].atomIdx]].nShells);
        }
    }

    if (!converged) {
        // Even if not converged, return the best charges we have
        // This often gives reasonable results for charge estimation
        // Recompute electronic energy with final charges
        std::vector<double> Vshift;
        buildChargeShift(shells, pos_bohr.data(), Z.data(), atomCount,
                         nbasis, shellCharges, atomCharges, S, Vshift);
        for (int i = 0; i < nbasis * nbasis; i++) {
            H[i] = H0[i] + Vshift[i];
        }
        if (solveGenEig(H, S, nbasis, eigenvalues, eigenvectors)) {
            std::vector<double> P(nbasis * nbasis, 0.0);
            for (int k = 0; k < nOccupied; k++) {
                for (int mu = 0; mu < nbasis; mu++) {
                    for (int nu = 0; nu < nbasis; nu++) {
                        P[mu * nbasis + nu] += 2.0 * eigenvectors[k * nbasis + mu]
                                                    * eigenvectors[k * nbasis + nu];
                    }
                }
            }
            Eelec = 0.0;
            for (int mu = 0; mu < nbasis; mu++) {
                for (int nu = 0; nu < nbasis; nu++) {
                    Eelec += P[mu * nbasis + nu] * H0[mu * nbasis + nu];
                }
            }
            Eelec += computeCoulombEnergy(shells, pos_bohr.data(), Z.data(),
                                          atomCount, shellCharges, atomCharges);
        }
    }

    // =========================================================================
    // Step 8: Package results
    // =========================================================================

    result->charges = new float[atomCount];
    for (int i = 0; i < atomCount; i++) {
        // Internal SCC uses pop-refOcc (positive = gained electrons).
        // Output standard Mulliken charges: q = refOcc - pop (positive = lost electrons).
        result->charges[i] = -(float)atomCharges[i];
    }
    result->totalEnergy = (float)(Eelec + Erep);
    result->electronicEnergy = (float)Eelec;
    result->repulsionEnergy = (float)Erep;
    result->scfIterations = iter + 1;
    result->converged = converged;
    result->success = true;

    if (!converged) {
        std::snprintf(result->errorMessage, 512,
                     "SCC did not converge within %d iterations (last max|dq|>%.1e)",
                     maxIterations, convergenceThreshold);
    }

    return result;
}

void druse_xtb_free_result(DruseXTBChargeResult *result) {
    if (result) {
        delete[] result->charges;
        delete result;
    }
}

bool druse_xtb_available(void) {
    return true;
}

} // extern "C"
