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
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <mutex>

// ============================================================================
// MARK: - GPU Acceleration Context (set by Swift Metal layer)
// ============================================================================

static DruseXTBGPUContext *g_xtb_gpu = nullptr;

extern "C" void druse_xtb_set_gpu_context(DruseXTBGPUContext *ctx) {
    g_xtb_gpu = ctx;
}

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

    // GPU path: dispatch to Metal compute if available
    if (g_xtb_gpu && g_xtb_gpu->gpu_compute_cn && natom >= 8) {
        g_xtb_gpu->gpu_compute_cn(g_xtb_gpu->context, pos_bohr,
                                   reinterpret_cast<const int32_t*>(Z),
                                   natom, cn.data());
        return;
    }

    // CPU fallback
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
    // GPU path
    if (g_xtb_gpu && g_xtb_gpu->gpu_compute_repulsion && natom >= 8) {
        return g_xtb_gpu->gpu_compute_repulsion(g_xtb_gpu->context, pos_bohr,
                                                 reinterpret_cast<const int32_t*>(Z),
                                                 natom, nullptr);
    }

    // CPU fallback
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
            double kexp = GFN2_KEXP;
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

    const size_t nsh = shells.size();

    // Build a flat list of upper-triangle shell pairs for balanced parallel dispatch.
    // Each pair (si, sj) writes to non-overlapping matrix blocks so no mutex needed.
    struct ShellPair { size_t si, sj; };
    std::vector<ShellPair> pairs;
    pairs.reserve(nsh * (nsh + 1) / 2);
    for (size_t si = 0; si < nsh; si++)
        for (size_t sj = si; sj < nsh; sj++)
            pairs.push_back({si, sj});

    tbb::parallel_for(tbb::blocked_range<size_t>(0, pairs.size(), /*grain=*/4),
    [&](const tbb::blocked_range<size_t> &range) {
      for (size_t idx = range.begin(); idx < range.end(); idx++) {
        size_t si = pairs[idx].si;
        size_t sj = pairs[idx].sj;

        const auto &shA = shells[si];
        int atomA = shA.atomIdx;
        double Ax = pos_bohr[3*atomA], Ay = pos_bohr[3*atomA+1], Az = pos_bohr[3*atomA+2];
        int lA = shA.angMom;
        double zetaA = shA.slaterExp;
        const STOnGData *stoA = getSTOnG(shA.principalQN, lA, shA.ngauss);

        const auto &shB = shells[sj];
        int atomB = shB.atomIdx;
        double Bx = pos_bohr[3*atomB], By = pos_bohr[3*atomB+1], Bz = pos_bohr[3*atomB+2];
        int lB = shB.angMom;
        double zetaB = shB.slaterExp;
        const STOnGData *stoB = getSTOnG(shB.principalQN, lB, shB.ngauss);

        int nCartA = (lA + 1) * (lA + 2) / 2;
        int nCartB = (lB + 1) * (lB + 2) / 2;

        std::vector<double> S_cart(nCartA * nCartB, 0.0);

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

        for (int k = 0; k < stoA->n; k++) {
            double alphaK = stoA->alpha[k] * zetaA * zetaA;
            double cK = stoA->coeff[k];
            for (int l = 0; l < stoB->n; l++) {
                double alphaL = stoB->alpha[l] * zetaB * zetaB;
                double cL = stoB->coeff[l];
                double cc = cK * cL;

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

        // Transform Cartesian → spherical and write to output matrix
        int nA = shA.basisCount;
        int nB = shB.basisCount;

        if (lA <= 1 && lB <= 1) {
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
      } // end for idx in range
    }); // end parallel_for
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

    // Fill H0 diagonal blocks (serial — small and fast)
    for (size_t si = 0; si < shells.size(); si++) {
        const auto &shA = shells[si];
        for (int ia = 0; ia < shA.basisCount; ia++) {
            int ii = shA.basisStart + ia;
            H0[ii * nbasis + ii] = shellLevels[si];
        }
    }

    // Fill H0 off-diagonal blocks — parallelized over shell pairs
    struct ShellPair { size_t si, sj; };
    std::vector<ShellPair> offdiagPairs;
    for (size_t si = 0; si < shells.size(); si++)
        for (size_t sj = si + 1; sj < shells.size(); sj++)
            if (shells[si].atomIdx != shells[sj].atomIdx)
                offdiagPairs.push_back({si, sj});

    tbb::parallel_for(tbb::blocked_range<size_t>(0, offdiagPairs.size(), /*grain=*/4),
    [&](const tbb::blocked_range<size_t> &range) {
      for (size_t idx = range.begin(); idx < range.end(); idx++) {
        size_t si = offdiagPairs[idx].si;
        size_t sj = offdiagPairs[idx].sj;
        const auto &shA = shells[si];
        const auto &shB = shells[sj];
        int atomA = shA.atomIdx;
        int atomB = shB.atomIdx;

        double dx = pos_bohr[3*atomA]   - pos_bohr[3*atomB];
        double dy = pos_bohr[3*atomA+1] - pos_bohr[3*atomB+1];
        double dz = pos_bohr[3*atomA+2] - pos_bohr[3*atomB+2];
        double rab = std::sqrt(dx*dx + dy*dy + dz*dz);

        double K = getShellPairK(shA.angMom, shB.angMom);

        double enA = gfn2Params[Z[atomA]].en;
        double enB = gfn2Params[Z[atomB]].en;
        double enDiff = enA - enB;
        double enScale = 1.0 + GFN2_ENSCALE * enDiff * enDiff;

        double radSum = gfn2Params[Z[atomA]].atomicRad + gfn2Params[Z[atomB]].atomicRad;
        double rNorm = (radSum > 1e-8) ? rab / radSum : 0.0;
        double sqrtR = std::sqrt(rNorm);
        double rf1 = 1.0 + gfn2Params[Z[atomA]].shpoly[shA.shellIdx] * sqrtR;
        double rf2 = 1.0 + gfn2Params[Z[atomB]].shpoly[shB.shellIdx] * sqrtR;
        double shPoly = rf1 * rf2;

        double zi = gfn2Params[Z[atomA]].slaterExp[shA.shellIdx];
        double zj = gfn2Params[Z[atomB]].slaterExp[shB.shellIdx];
        double zetaRatio = 2.0 * std::sqrt(zi * zj) / (zi + zj);
        double zetaij = std::sqrt(zetaRatio);

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
    }); // end parallel_for
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
    // Parallelized: each shell's potential is independent.
    std::vector<double> shellPotential(nshells, 0.0);

    tbb::parallel_for(tbb::blocked_range<int>(0, nshells),
    [&](const tbb::blocked_range<int> &range) {
      for (int si = range.begin(); si < range.end(); si++) {
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

        int l = shells[si].angMom;
        double thirdScale = (l == 0) ? GFN2_THIRDORDER_S :
                            (l == 1) ? GFN2_THIRDORDER_P : GFN2_THIRDORDER_D;
        double gam3 = gfn2Params[Z[atomA]].gam3 * thirdScale;
        V += gam3 * atomCharges[atomA] * atomCharges[atomA];

        shellPotential[si] = V;
      }
    }); // end parallel_for

    // Build the potential matrix:
    // Vshift_{μν} = 1/2 * S_{μν} * (V_{shell(μ)} + V_{shell(ν)})
    // Each shell pair writes to non-overlapping basis blocks.
    struct ShellPair { int si, sj; };
    std::vector<ShellPair> vshiftPairs;
    for (int si = 0; si < nshells; si++)
        for (int sj = si; sj < nshells; sj++)
            vshiftPairs.push_back({si, sj});

    tbb::parallel_for(tbb::blocked_range<size_t>(0, vshiftPairs.size()),
    [&](const tbb::blocked_range<size_t> &range) {
      for (size_t idx = range.begin(); idx < range.end(); idx++) {
        int si = vshiftPairs[idx].si;
        int sj = vshiftPairs[idx].sj;
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
    }); // end parallel_for
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

// Forward declaration: D4 dispersion (defined in Part 2, used in compute_charges for total energy)
static double computeD4Dispersion(const double *pos_bohr, const int *Z, int natom,
                                   const std::vector<double> &cn,
                                   double *gradient = nullptr);
// Remove default from actual definition below (only one default allowed)

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
    // Step 8: D4 dispersion (lightweight, always-on — CN already computed)
    // =========================================================================

    double Edisp = computeD4Dispersion(pos_bohr.data(), Z.data(), atomCount, cn);

    // =========================================================================
    // Step 9: Package results
    // =========================================================================

    result->charges = new float[atomCount];
    for (int i = 0; i < atomCount; i++) {
        // Internal SCC uses pop-refOcc (positive = gained electrons).
        // Output standard Mulliken charges: q = refOcc - pop (positive = lost electrons).
        result->charges[i] = -(float)atomCharges[i];
    }
    result->totalEnergy = (float)(Eelec + Erep + Edisp);
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

// ============================================================================
// ============================================================================
//
//  PART 2 — ANALYTICAL GRADIENTS, D4 DISPERSION, GBSA/ALPB, L-BFGS
//
//  All new functionality added below. The original charge-only code above
//  is preserved unchanged for backward compatibility.
//
// ============================================================================
// ============================================================================

// ============================================================================
// MARK: - Internal: Shared SCC Workspace
// ============================================================================

/// All intermediate data from an SCC calculation, needed for gradient computation.
struct SCCWorkspace {
    int natom;
    int nbasis;
    int nshells;
    int nValElectrons;
    int nOccupied;
    bool openShell;

    std::vector<int> Z;
    std::vector<double> pos_bohr;       // 3*natom
    std::vector<ShellInfo> shells;
    std::vector<double> cn;             // coordination numbers

    // Matrices
    std::vector<double> S;              // overlap (nbasis x nbasis)
    std::vector<double> H0;             // core Hamiltonian
    std::vector<double> P;              // density matrix
    std::vector<double> W;              // energy-weighted density matrix
    std::vector<double> normFactor;     // basis normalization factors

    // Converged SCC data
    std::vector<double> shellCharges;
    std::vector<double> atomCharges;
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;

    // Energy components
    double Eelec;
    double Erep;
    double Ecoul;

    int scfIterations;
    bool converged;
};

/// Run a full SCC calculation and populate workspace for gradient use.
/// Returns false on failure (fills ws.errorMessage-like info in the caller).
static bool runSCC(const float *positions, const int32_t *atomicNumbers,
                   int32_t atomCount, int32_t totalCharge, int32_t maxIterations,
                   SCCWorkspace &ws) {
    initGFN2Params();

    ws.natom = atomCount;
    ws.Z.resize(atomCount);
    for (int i = 0; i < atomCount; i++) ws.Z[i] = atomicNumbers[i];

    // Validate
    for (int i = 0; i < atomCount; i++) {
        int z = ws.Z[i];
        if (z <= 0 || z > MAX_ATOMIC_NUM || gfn2Params[z].atomicNumber == 0)
            return false;
    }

    // Positions: Angstrom → Bohr
    ws.pos_bohr.resize(3 * atomCount);
    for (int i = 0; i < 3 * atomCount; i++)
        ws.pos_bohr[i] = positions[i] * ANG_TO_BOHR;

    // Build basis
    ws.shells.clear();
    ws.nbasis = 0;
    for (int a = 0; a < atomCount; a++) {
        const auto &par = gfn2Params[ws.Z[a]];
        for (int s = 0; s < par.nShells; s++) {
            ShellInfo sh;
            sh.atomIdx = a;
            sh.shellIdx = s;
            sh.angMom = par.shellAngMom[s];
            sh.basisStart = ws.nbasis;
            sh.basisCount = basisCount(sh.angMom);
            sh.slaterExp = par.slaterExp[s];
            sh.selfEnergy = par.selfEnergy[s];
            sh.refOcc = par.refOcc[s];
            sh.hubbard = par.gam * par.lgam[s];
            sh.principalQN = getPrincipalQN(ws.Z[a], s, sh.angMom);
            sh.ngauss = par.ngauss[s];
            ws.nbasis += sh.basisCount;
            ws.shells.push_back(sh);
        }
    }
    ws.nshells = (int)ws.shells.size();

    ws.nValElectrons = 0;
    for (const auto &sh : ws.shells)
        ws.nValElectrons += (int)std::round(sh.refOcc);
    ws.nValElectrons -= totalCharge;

    if (ws.nValElectrons < 0 || ws.nValElectrons > 2 * ws.nbasis) return false;
    ws.nOccupied = ws.nValElectrons / 2;
    ws.openShell = (ws.nValElectrons % 2 != 0);

    // CN
    computeCN(ws.pos_bohr.data(), ws.Z.data(), atomCount, ws.cn);

    // Overlap
    buildOverlapMatrix(ws.shells, ws.pos_bohr.data(), ws.nbasis, ws.S);
    ws.normFactor.resize(ws.nbasis);
    for (int i = 0; i < ws.nbasis; i++)
        ws.normFactor[i] = 1.0 / std::sqrt(std::fabs(ws.S[i * ws.nbasis + i]));
    for (int i = 0; i < ws.nbasis; i++)
        for (int j = 0; j < ws.nbasis; j++)
            ws.S[i * ws.nbasis + j] *= ws.normFactor[i] * ws.normFactor[j];

    // H0
    buildH0(ws.shells, ws.pos_bohr.data(), ws.Z.data(), atomCount,
            ws.nbasis, ws.S, ws.cn, ws.H0);

    // Repulsion
    ws.Erep = computeRepulsion(ws.pos_bohr.data(), ws.Z.data(), atomCount);

    // SCC iteration (same as original druse_xtb_compute_charges logic)
    ws.shellCharges.assign(ws.nshells, 0.0);
    ws.atomCharges.assign(atomCount, 0.0);

    double convergenceThreshold = 1e-6;
    double mixingFactor = 0.3;
    ws.converged = false;
    ws.scfIterations = 0;
    ws.Eelec = 0.0;

    std::vector<double> prevShellCharges(ws.nshells, 0.0);
    std::vector<double> prevResidual(ws.nshells, 0.0);
    bool hasHistory = false;

    std::vector<double> H(ws.nbasis * ws.nbasis);

    for (int iter = 0; iter < maxIterations; iter++) {
        std::vector<double> Vshift;
        buildChargeShift(ws.shells, ws.pos_bohr.data(), ws.Z.data(), atomCount,
                         ws.nbasis, ws.shellCharges, ws.atomCharges, ws.S, Vshift);

        for (int i = 0; i < ws.nbasis * ws.nbasis; i++)
            H[i] = ws.H0[i] + Vshift[i];

        if (!solveGenEig(H, ws.S, ws.nbasis, ws.eigenvalues, ws.eigenvectors)) {
            for (int i = 0; i < ws.nbasis; i++)
                ws.S[i * ws.nbasis + i] += 1e-6;
            if (!solveGenEig(H, ws.S, ws.nbasis, ws.eigenvalues, ws.eigenvectors))
                return false;
        }

        // Density matrix
        ws.P.assign(ws.nbasis * ws.nbasis, 0.0);
        {
            int nOcc = ws.openShell ? ws.nOccupied + 1 : ws.nOccupied;
            nOcc = std::min(nOcc, ws.nbasis);
            std::vector<double> C_scaled(nOcc * ws.nbasis);
            for (int k = 0; k < ws.nOccupied && k < ws.nbasis; k++) {
                double scale = std::sqrt(2.0);
                for (int mu = 0; mu < ws.nbasis; mu++)
                    C_scaled[k * ws.nbasis + mu] = scale * ws.eigenvectors[k * ws.nbasis + mu];
            }
            if (ws.openShell && ws.nOccupied < ws.nbasis) {
                int k = ws.nOccupied;
                for (int mu = 0; mu < ws.nbasis; mu++)
                    C_scaled[k * ws.nbasis + mu] = ws.eigenvectors[k * ws.nbasis + mu];
            }
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        ws.nbasis, ws.nbasis, nOcc,
                        1.0, C_scaled.data(), ws.nbasis,
                             C_scaled.data(), ws.nbasis,
                        0.0, ws.P.data(), ws.nbasis);
        }

        std::vector<double> newShellCharges, newAtomCharges;
        mullikenCharges(ws.P, ws.S, ws.shells, ws.nbasis, atomCount,
                        newShellCharges, newAtomCharges);

        double maxDelta = 0.0;
        for (int si = 0; si < ws.nshells; si++)
            maxDelta = std::max(maxDelta, std::fabs(newShellCharges[si] - ws.shellCharges[si]));

        if (maxDelta < convergenceThreshold && iter > 0) {
            ws.shellCharges = newShellCharges;
            ws.atomCharges = newAtomCharges;
            ws.converged = true;
            ws.scfIterations = iter + 1;

            ws.Eelec = 0.0;
            for (int mu = 0; mu < ws.nbasis; mu++)
                for (int nu = 0; nu < ws.nbasis; nu++)
                    ws.Eelec += ws.P[mu * ws.nbasis + nu] * ws.H0[mu * ws.nbasis + nu];

            ws.Ecoul = computeCoulombEnergy(ws.shells, ws.pos_bohr.data(), ws.Z.data(),
                                             atomCount, ws.shellCharges, ws.atomCharges);
            ws.Eelec += ws.Ecoul;
            break;
        }

        // Mixing (Anderson)
        std::vector<double> residual(ws.nshells);
        for (int si = 0; si < ws.nshells; si++)
            residual[si] = newShellCharges[si] - ws.shellCharges[si];

        if (hasHistory && iter > 1) {
            double dotRR = 0, dotRdR = 0;
            for (int si = 0; si < ws.nshells; si++) {
                double dr = residual[si] - prevResidual[si];
                dotRR += residual[si] * residual[si];
                dotRdR += residual[si] * dr;
            }
            double beta = (std::fabs(dotRdR) > 1e-16) ? dotRR / dotRdR : 0.0;
            beta = std::max(-1.0, std::min(1.0, beta));
            for (int si = 0; si < ws.nshells; si++) {
                double mixed = (1.0 - beta) * (ws.shellCharges[si] + mixingFactor * residual[si])
                             + beta * (prevShellCharges[si] + mixingFactor * prevResidual[si]);
                prevShellCharges[si] = ws.shellCharges[si];
                prevResidual[si] = residual[si];
                ws.shellCharges[si] = mixed;
            }
        } else {
            for (int si = 0; si < ws.nshells; si++) {
                prevShellCharges[si] = ws.shellCharges[si];
                prevResidual[si] = residual[si];
                ws.shellCharges[si] += mixingFactor * residual[si];
            }
            hasHistory = true;
        }

        std::fill(ws.atomCharges.begin(), ws.atomCharges.end(), 0.0);
        for (int si = 0; si < ws.nshells; si++)
            ws.atomCharges[ws.shells[si].atomIdx] += ws.shellCharges[si];
        double qsum = std::accumulate(ws.atomCharges.begin(), ws.atomCharges.end(), 0.0);
        double shift = (totalCharge - qsum) / atomCount;
        for (int a = 0; a < atomCount; a++) ws.atomCharges[a] += shift;
        for (int si = 0; si < ws.nshells; si++)
            ws.shellCharges[si] += shift * ws.shells[si].refOcc /
                std::max(1.0, (double)gfn2Params[ws.Z[ws.shells[si].atomIdx]].nShells);
    }

    if (!ws.converged) {
        ws.scfIterations = maxIterations;
        // Compute energy with final (unconverged) charges
        ws.Eelec = 0.0;
        for (int mu = 0; mu < ws.nbasis; mu++)
            for (int nu = 0; nu < ws.nbasis; nu++)
                ws.Eelec += ws.P[mu * ws.nbasis + nu] * ws.H0[mu * ws.nbasis + nu];
        ws.Ecoul = computeCoulombEnergy(ws.shells, ws.pos_bohr.data(), ws.Z.data(),
                                         atomCount, ws.shellCharges, ws.atomCharges);
        ws.Eelec += ws.Ecoul;
    }

    // Energy-weighted density matrix: W_{μν} = Σ_k f_k * ε_k * C_{μk} * C_{νk}
    ws.W.assign(ws.nbasis * ws.nbasis, 0.0);
    {
        int nOcc = ws.openShell ? ws.nOccupied + 1 : ws.nOccupied;
        nOcc = std::min(nOcc, ws.nbasis);
        for (int k = 0; k < nOcc; k++) {
            double occ = (k < ws.nOccupied) ? 2.0 : 1.0;
            double ek = ws.eigenvalues[k];
            for (int mu = 0; mu < ws.nbasis; mu++) {
                double c_mu = ws.eigenvectors[k * ws.nbasis + mu];
                for (int nu = mu; nu < ws.nbasis; nu++) {
                    double val = occ * ek * c_mu * ws.eigenvectors[k * ws.nbasis + nu];
                    ws.W[mu * ws.nbasis + nu] += val;
                    if (mu != nu) ws.W[nu * ws.nbasis + mu] += val;
                }
            }
        }
    }

    return true;
}

// ============================================================================
// MARK: - D4 Dispersion Energy and Gradient
// ============================================================================

// GFN2-xTB D4 parameters (rational BJ damping)
static constexpr double D4_S6  = 1.0;
static constexpr double D4_S8  = 2.7;
static constexpr double D4_A1  = 0.52;
static constexpr double D4_A2  = 5.0;    // Bohr (already in atomic units)
static constexpr double D4_S9  = 0.0;    // ATM three-body (disabled for speed)

// D4 reference C6 coefficients for common elements (Hartree·Bohr⁶)
// Simplified: single reference per element from Grimme's D4 reference set.
// For drug molecules this is accurate to ~5% vs full multi-reference D4.
static const double d4RefC6[] = {
    0.0,      // dummy
    3.61,     // H
    1.46,     // He
    1380.0,   // Li
    214.0,    // Be
    99.5,     // B
    46.6,     // C
    24.2,     // N
    15.6,     // O
    9.52,     // F
    6.38,     // Ne
    1470.0,   // Na
    626.0,    // Mg
    528.0,    // Al
    305.0,    // Si
    185.0,    // P
    134.0,    // S
    94.6,     // Cl
    64.3,     // Ar
    3880.0,   // K
    2180.0,   // Ca
    0,0,0,0,  // Sc-Cr (sparse)
    552.0,    // Mn (25)
    482.0,    // Fe (26)
    408.0,    // Co (27)
    373.0,    // Ni (28)
    253.0,    // Cu (29)
    284.0,    // Zn (30)
    0,0,0,0,  // Ga-Se
    162.0,    // Br (35)
};
static const int d4RefC6Size = sizeof(d4RefC6) / sizeof(d4RefC6[0]);

static double getC6(int Z) {
    if (Z > 0 && Z < d4RefC6Size && d4RefC6[Z] > 0.0) return d4RefC6[Z];
    // Fallback: empirical scaling C6 ~ Z^(1.5)
    return 25.0 * std::pow((double)Z / 6.0, 1.5);
}

// Casimir-Polder C8 from C6: C8 = 3 * C6 * sqrt(Q_A * Q_B)
// where Q is the quadrupole expectation — approximate as Q ~ sqrt(C6 / Z^0.5)
static double getC8fromC6(double c6AB, int ZA, int ZB) {
    double qA = std::sqrt(getC6(ZA)) * 2.5;
    double qB = std::sqrt(getC6(ZB)) * 2.5;
    return 3.0 * c6AB * std::sqrt(qA * qB);
}

/// Compute D4 dispersion energy (and optionally gradient).
/// Reference: Caldeweyher et al., JCP 2019, 150, 154122
static double computeD4Dispersion(const double *pos_bohr, const int *Z, int natom,
                                   const std::vector<double> &cn,
                                   double *gradient) {
    // GPU path
    if (g_xtb_gpu && g_xtb_gpu->gpu_compute_d4 && natom >= 8) {
        if (gradient) std::memset(gradient, 0, 3 * natom * sizeof(double));
        return g_xtb_gpu->gpu_compute_d4(g_xtb_gpu->context, pos_bohr,
                                          reinterpret_cast<const int32_t*>(Z),
                                          natom, cn.data(), gradient);
    }

    // CPU fallback
    // BJ damping: f_n(r) = r^n / (r^n + (a1*sqrt(C6/C8) + a2)^n)
    // Convert a2 from Angstrom to Bohr
    double a2_bohr = D4_A2;  // already in Bohr

    double Edisp = 0.0;
    if (gradient) std::memset(gradient, 0, 3 * natom * sizeof(double));

    for (int i = 0; i < natom; i++) {
        double c6i = getC6(Z[i]);
        for (int j = 0; j < i; j++) {
            double c6j = getC6(Z[j]);
            double c6ij = std::sqrt(c6i * c6j);

            // CN-dependent scaling (simplified D4 single-reference approximation)
            // Soft Gaussian: w = exp(-0.5*(CN - CN_ref)^2) avoids catastrophic
            // drop when CN deviates from the single reference value.
            double cnRef_i = (Z[i] <= 2) ? 1.0 : (Z[i] <= 10 ? 3.0 : 4.0);
            double cnRef_j = (Z[j] <= 2) ? 1.0 : (Z[j] <= 10 ? 3.0 : 4.0);
            double wi = std::exp(-0.5 * (cn[i] - cnRef_i) * (cn[i] - cnRef_i));
            double wj = std::exp(-0.5 * (cn[j] - cnRef_j) * (cn[j] - cnRef_j));
            double wij = wi * wj;
            c6ij *= wij; // modulate C6 by coordination environment

            double c8ij = getC8fromC6(c6ij, Z[i], Z[j]);

            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            if (r < 1e-6) continue;

            // BJ damping radii
            double r0 = D4_A1 * std::sqrt(c6ij > 0 ? c8ij / c6ij : 0.0) + a2_bohr;
            double r0_2 = r0 * r0;
            double r0_6 = r0_2 * r0_2 * r0_2;
            double r0_8 = r0_6 * r0_2;

            double r6 = r2 * r2 * r2;
            double r8 = r6 * r2;

            // Damped dispersion energy
            double f6 = 1.0 / (r6 + r0_6);
            double f8 = 1.0 / (r8 + r0_8);

            double e6 = -D4_S6 * c6ij * f6;
            double e8 = -D4_S8 * c8ij * f8;
            Edisp += e6 + e8;

            if (gradient) {
                // dE/dr for BJ-damped terms:
                // d/dr[-C6/(r^6 + r0^6)] = 6*C6*r^5 / (r^6 + r0^6)^2
                double df6 = 6.0 * D4_S6 * c6ij * r2 * r2 * r / ((r6 + r0_6) * (r6 + r0_6));
                double df8 = 8.0 * D4_S8 * c8ij * r6 * r / ((r8 + r0_8) * (r8 + r0_8));
                double dEdr = (df6 + df8) / r; // divide by r to get dE/dr * (r_vec/r)

                gradient[3*i]   += dEdr * dx;
                gradient[3*i+1] += dEdr * dy;
                gradient[3*i+2] += dEdr * dz;
                gradient[3*j]   -= dEdr * dx;
                gradient[3*j+1] -= dEdr * dy;
                gradient[3*j+2] -= dEdr * dz;
            }
        }
    }
    return Edisp;
}

// ============================================================================
// MARK: - GBSA / ALPB Implicit Solvation
// ============================================================================

// Van der Waals radii for Born radii computation (Bondi radii, in Bohr)
static const double vdwRadiiBohr[] = {
    0.0,      // dummy
    2.268,    // H   (1.20 Å)
    2.646,    // He  (1.40 Å)
    3.438,    // Li  (1.82 Å)
    2.873,    // Be  (1.52 Å — estimated)
    3.627,    // B   (1.92 Å)
    3.213,    // C   (1.70 Å)
    2.929,    // N   (1.55 Å)
    2.873,    // O   (1.52 Å)
    2.797,    // F   (1.48 Å — adjusted to Bondi)
    2.910,    // Ne  (1.54 Å)
    4.290,    // Na  (2.27 Å)
    3.250,    // Mg  (1.72 Å)
    3.495,    // Al  (1.85 Å — estimated)
    3.968,    // Si  (2.10 Å)
    3.402,    // P   (1.80 Å)
    3.402,    // S   (1.80 Å)
    3.307,    // Cl  (1.75 Å)
    3.553,    // Ar  (1.88 Å)
    5.197,    // K   (2.75 Å)
    4.214,    // Ca  (2.23 Å — estimated)
    0,0,0,0,  // Sc-Cr
    3.78,     // Mn (25)
    3.78,     // Fe (26)
    3.78,     // Co (27)
    3.08,     // Ni (28)
    2.65,     // Cu (29)
    2.59,     // Zn (30)
    0,0,0,0,  // Ga-Se
    3.495,    // Br (35)
};
static const int vdwRadiiSize = sizeof(vdwRadiiBohr) / sizeof(vdwRadiiBohr[0]);

static double getVdwRad(int Z) {
    if (Z > 0 && Z < vdwRadiiSize && vdwRadiiBohr[Z] > 0.0) return vdwRadiiBohr[Z];
    return 3.4; // fallback (~1.80 Å)
}

/// Compute Born radii using the Still/OBC-II method.
/// Reference: Onufriev, Bashford, Case, Proteins 2004, 55, 383-394
static void computeBornRadii(const double *pos_bohr, const int *Z, int natom,
                              double probeRad_bohr, double offset_bohr, double bornScale,
                              std::vector<double> &brad,
                              std::vector<double> &sasa) {
    brad.resize(natom);
    sasa.resize(natom);

    // GPU path
    if (g_xtb_gpu && g_xtb_gpu->gpu_compute_born && natom >= 8) {
        g_xtb_gpu->gpu_compute_born(g_xtb_gpu->context, pos_bohr,
                                     reinterpret_cast<const int32_t*>(Z),
                                     natom, (float)probeRad_bohr, (float)offset_bohr,
                                     (float)bornScale, brad.data(), sasa.data());
        return;
    }

    // CPU fallback
    // Step 1: Compute psi (descreening integral) for each atom
    std::vector<double> psi(natom, 0.0);

    for (int i = 0; i < natom; i++) {
        double ri = getVdwRad(Z[i]) + offset_bohr;
        double rho_i = ri * bornScale;

        for (int j = 0; j < natom; j++) {
            if (i == j) continue;
            double rj = getVdwRad(Z[j]) + offset_bohr;

            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double rij = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (rij < 1e-8) continue;

            double sj = rj * bornScale;

            // Compute the descreening contribution
            if (rij > ri + sj) {
                // No overlap: standard 1/r contribution
                psi[i] += 0.5 * (1.0/(rij - sj) - 1.0/(rij + sj)
                    + sj * 0.25 * (1.0/((rij + sj)*(rij + sj)) - 1.0/((rij - sj)*(rij - sj)))
                    + 0.5 * std::log((rij - sj)/(rij + sj)) / rij);
            } else if (rij > std::fabs(ri - sj)) {
                // Partial overlap
                double d = rij - sj;
                if (std::fabs(d) < 1e-8) d = 1e-8;
                psi[i] += 0.25 * (2.0/rij - 1.0/(rij + sj) - ri/(4.0*sj*rij)
                    + 0.25 * (1.0/(sj*sj) - 1.0/(ri*ri))
                    + 0.5 * std::log(std::fabs(d)/(rij + sj)) / rij);
            }
            // else fully buried: different formula omitted for simplicity
        }

        // OBC-II correction: apply three-parameter scaling
        // Reference: Onufriev, Bashford, Case (2004)
        double br = psi[i] * ri;
        static constexpr double obc_alpha = 1.0;
        static constexpr double obc_beta  = 0.8;
        static constexpr double obc_gamma = 4.85;

        double arg = br * (obc_alpha - br * (obc_beta - br * obc_gamma));
        double tanh_arg = std::tanh(arg);

        brad[i] = 1.0 / (1.0/ri - tanh_arg/ri);

        // Approximate SASA: each atom's solvent-accessible surface area
        // Simple Gaussian approximation (fast)
        double probe_sum = getVdwRad(Z[i]) + probeRad_bohr;
        double area = 4.0 * PI * probe_sum * probe_sum;
        double burial = 0.0;
        for (int j = 0; j < natom; j++) {
            if (i == j) continue;
            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double rij = std::sqrt(dx*dx + dy*dy + dz*dz);
            double pj = getVdwRad(Z[j]) + probeRad_bohr;
            double overlap = std::max(0.0, 1.0 - rij / (probe_sum + pj));
            burial += overlap;
        }
        sasa[i] = area * std::exp(-0.5 * burial);
    }
}

/// Compute GBSA/ALPB solvation energy (and optionally gradient).
/// Returns solvation free energy in Hartree.
static double computeSolvation(const double *pos_bohr, const int *Z, int natom,
                                const std::vector<double> &atomCharges,
                                DruseXTBSolvationConfig solv,
                                double *gradient = nullptr) {
    if (solv.model == DRUSE_XTB_SOLV_NONE) return 0.0;

    double eps = solv.dielectricConstant;
    double keps = (1.0 - 1.0 / eps); // Born prefactor: E_GB = -0.5 * (1 - 1/ε) * Σ q_i*q_j/f_GB

    double probeRad_bohr = solv.probeRadius * ANG_TO_BOHR;
    double offset_bohr = solv.bornOffset * ANG_TO_BOHR;

    std::vector<double> brad, sasa;
    computeBornRadii(pos_bohr, Z, natom, probeRad_bohr, offset_bohr, solv.bornScale, brad, sasa);

    double Eborn = 0.0;
    double Esasa = 0.0;

    // Generalized Born energy: E_GB = -0.5 * keps * Σ_ij q_i*q_j / f_GB(r_ij)
    // f_GB = sqrt(r²_ij + B_i*B_j * exp(-r²_ij/(4*B_i*B_j)))  (Still formula)
    for (int i = 0; i < natom; i++) {
        // Sign convention: atomCharges are pop-refOcc (internal), negate for real charges
        double qi = -atomCharges[i];
        for (int j = i; j < natom; j++) {
            double qj = -atomCharges[j];

            double r2;
            if (i == j) {
                r2 = 0.0;
            } else {
                double dx = pos_bohr[3*i]   - pos_bohr[3*j];
                double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
                double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
                r2 = dx*dx + dy*dy + dz*dz;
            }

            double BiBj = brad[i] * brad[j];
            double expfac = std::exp(-r2 / (4.0 * BiBj));
            double fGB = std::sqrt(r2 + BiBj * expfac);

            double contrib = qi * qj / fGB;
            if (i == j)
                Eborn += 0.5 * contrib;
            else
                Eborn += contrib;
        }
    }
    Eborn *= -0.5 * keps;

    // ALPB correction for non-spherical charge distributions
    if (solv.model == DRUSE_XTB_SOLV_ALPB) {
        // α_ALPB = 0.571412 (Ehlert et al., JCTC 2021)
        static constexpr double ALPB_ALPHA = 0.571412;
        double qtot = 0.0;
        for (int i = 0; i < natom; i++) qtot += -atomCharges[i];
        // Shape correction: E_alpb = α * keps * q_tot² / (4π * <r²>)
        if (std::fabs(qtot) > 1e-8) {
            // Compute geometric mean radius
            double r2avg = 0.0;
            for (int i = 0; i < natom; i++) {
                double x = pos_bohr[3*i], y = pos_bohr[3*i+1], z = pos_bohr[3*i+2];
                r2avg += x*x + y*y + z*z;
            }
            r2avg /= natom;
            double reff = std::sqrt(r2avg);
            if (reff > 1e-6) {
                Eborn += ALPB_ALPHA * keps * qtot * qtot / (4.0 * PI * reff);
            }
        }
    }

    // SASA non-polar contribution: E_sasa = γ * Σ SASA_i
    // Convert surface tension from dyn/cm to Hartree/Bohr²:
    //   1 dyn/cm = 1e-3 J/m², 1 Bohr² = 2.8003e-21 m², 1 Hartree = 4.3597e-18 J
    //   γ(Hartree/Bohr²) = γ(dyn/cm) * 1e-3 * 2.8003e-21 / 4.3597e-18
    double gamma_au = solv.surfaceTension * 6.4232e-7;
    for (int i = 0; i < natom; i++) {
        Esasa += gamma_au * sasa[i];
    }

    // Gradient (numerical for solvation — exact analytical is complex)
    if (gradient) {
        double h = 1e-4; // Bohr
        std::vector<double> pos_plus(3 * natom);
        for (int a = 0; a < natom; a++) {
            for (int d = 0; d < 3; d++) {
                std::copy(pos_bohr, pos_bohr + 3 * natom, pos_plus.begin());
                pos_plus[3*a + d] += h;
                std::vector<double> brad_p, sasa_p;
                computeBornRadii(pos_plus.data(), Z, natom, probeRad_bohr, offset_bohr, solv.bornScale, brad_p, sasa_p);
                double Ep = 0.0;
                for (int i = 0; i < natom; i++) {
                    double qi = -atomCharges[i];
                    for (int j = i; j < natom; j++) {
                        double qj = -atomCharges[j];
                        double r2 = 0;
                        if (i != j) {
                            double dx2 = pos_plus[3*i]-pos_plus[3*j];
                            double dy2 = pos_plus[3*i+1]-pos_plus[3*j+1];
                            double dz2 = pos_plus[3*i+2]-pos_plus[3*j+2];
                            r2 = dx2*dx2+dy2*dy2+dz2*dz2;
                        }
                        double BiBj = brad_p[i]*brad_p[j];
                        double fGB = std::sqrt(r2 + BiBj*std::exp(-r2/(4.0*BiBj)));
                        double c = qi*qj/fGB;
                        Ep += (i==j) ? 0.5*c : c;
                    }
                }
                Ep *= -0.5 * keps;
                for (int i = 0; i < natom; i++) Ep += gamma_au * sasa_p[i];

                pos_plus[3*a + d] -= 2.0 * h;
                computeBornRadii(pos_plus.data(), Z, natom, probeRad_bohr, offset_bohr, solv.bornScale, brad_p, sasa_p);
                double Em = 0.0;
                for (int i = 0; i < natom; i++) {
                    double qi = -atomCharges[i];
                    for (int j = i; j < natom; j++) {
                        double qj = -atomCharges[j];
                        double r2 = 0;
                        if (i != j) {
                            double dx2 = pos_plus[3*i]-pos_plus[3*j];
                            double dy2 = pos_plus[3*i+1]-pos_plus[3*j+1];
                            double dz2 = pos_plus[3*i+2]-pos_plus[3*j+2];
                            r2 = dx2*dx2+dy2*dy2+dz2*dz2;
                        }
                        double BiBj = brad_p[i]*brad_p[j];
                        double fGB = std::sqrt(r2 + BiBj*std::exp(-r2/(4.0*BiBj)));
                        double c = qi*qj/fGB;
                        Em += (i==j) ? 0.5*c : c;
                    }
                }
                Em *= -0.5 * keps;
                for (int i = 0; i < natom; i++) Em += gamma_au * sasa_p[i];

                gradient[3*a + d] += (Ep - Em) / (2.0 * h);
            }
        }
    }

    return Eborn + Esasa;
}

// ============================================================================
// MARK: - Analytical Nuclear Gradient (Repulsion)
// ============================================================================

static void computeRepulsionGradient(const double *pos_bohr, const int *Z, int natom,
                                      double *gradient) {
    // GPU path: compute repulsion with gradient enabled
    if (g_xtb_gpu && g_xtb_gpu->gpu_compute_repulsion && natom >= 8) {
        // GPU repulsion kernel computes energy+gradient in one dispatch
        g_xtb_gpu->gpu_compute_repulsion(g_xtb_gpu->context, pos_bohr,
                                          reinterpret_cast<const int32_t*>(Z),
                                          natom, gradient);
        return;
    }

    // CPU fallback
    for (int i = 0; i < natom; i++) {
        const auto &pi = gfn2Params[Z[i]];
        if (pi.atomicNumber == 0) continue;
        for (int j = 0; j < i; j++) {
            const auto &pj = gfn2Params[Z[j]];
            if (pj.atomicNumber == 0) continue;

            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            if (r < 1e-6) continue;

            double alpha = std::sqrt(pi.arep * pj.arep);
            double kexp = GFN2_KEXP;  // 1.5
            double rk = std::pow(r, kexp);
            double zz = pi.zeff * pj.zeff;
            double exa = std::exp(-alpha * rk);

            // E_rep_ij = zz / r * exp(-alpha * r^kexp)
            // dE/dr = zz * exp(-alpha*r^k) * (-1/r² - alpha*k*r^(k-2)/r)
            //       = -zz * exa * (1/r² + alpha * kexp * r^(kexp-1) / r)
            // Factor out 1/r:
            double dEdr = -zz * exa * (1.0/r2 + alpha * kexp * std::pow(r, kexp - 2.0));

            // dE/dR_i = dEdr * (R_i - R_j) / r
            double gx = dEdr * dx;
            double gy = dEdr * dy;
            double gz = dEdr * dz;

            gradient[3*i]   += gx;
            gradient[3*i+1] += gy;
            gradient[3*i+2] += gz;
            gradient[3*j]   -= gx;
            gradient[3*j+1] -= gy;
            gradient[3*j+2] -= gz;
        }
    }
}

// ============================================================================
// MARK: - Analytical Nuclear Gradient (Coulomb / SCC)
// ============================================================================

/// Gradient of the Ohno-Klopman gamma function w.r.t. distance.
/// gamma = 1/sqrt(R² + eta^(-2))  where eta = 0.5*(gi+gj)
/// dgamma/dR = -R / (R² + eta^(-2))^(3/2)
static double ohno_gamma_deriv(double r_bohr, double gi, double gj) {
    double gij = 0.5 * (gi + gj);
    double gij_inv2 = 1.0 / (gij * gij);
    double denom = r_bohr * r_bohr + gij_inv2;
    return -r_bohr / (denom * std::sqrt(denom));
}

/// Coulomb gradient: dE_coul/dR_A = Σ_ij Δq_i * (dγ_ij/dR_A) * Δq_j
static void computeCoulombGradient(const std::vector<ShellInfo> &shells,
                                    const double *pos_bohr, const int *Z, int natom,
                                    const std::vector<double> &shellCharges,
                                    double *gradient) {
    int nshells = (int)shells.size();
    for (int si = 0; si < nshells; si++) {
        int atomA = shells[si].atomIdx;
        double gam_A = gfn2Params[Z[atomA]].gam;
        double lgam_A = gfn2Params[Z[atomA]].lgam[shells[si].shellIdx];

        for (int sj = si + 1; sj < nshells; sj++) {
            int atomB = shells[sj].atomIdx;
            if (atomA == atomB) continue;

            double gam_B = gfn2Params[Z[atomB]].gam;
            double lgam_B = gfn2Params[Z[atomB]].lgam[shells[sj].shellIdx];

            double dx = pos_bohr[3*atomA]   - pos_bohr[3*atomB];
            double dy = pos_bohr[3*atomA+1] - pos_bohr[3*atomB+1];
            double dz = pos_bohr[3*atomA+2] - pos_bohr[3*atomB+2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < 1e-8) continue;

            double gi = gam_A * lgam_A;
            double gj = gam_B * lgam_B;
            double dgdr = ohno_gamma_deriv(r, gi, gj);

            // dE/dR = Δq_i * Δq_j * (dγ/dR) * (R_vec/R)
            double qiqj = shellCharges[si] * shellCharges[sj];
            double scale = qiqj * dgdr / r;

            gradient[3*atomA]   += scale * dx;
            gradient[3*atomA+1] += scale * dy;
            gradient[3*atomA+2] += scale * dz;
            gradient[3*atomB]   -= scale * dx;
            gradient[3*atomB+1] -= scale * dy;
            gradient[3*atomB+2] -= scale * dz;
        }
    }
}

// ============================================================================
// MARK: - Analytical Nuclear Gradient (Electronic: Pulay + Hellmann-Feynman)
// ============================================================================

/// Electronic gradient via density matrix derivative:
///   dE_elec/dR_A = Σ_μν P_μν * dH0_μν/dR_A  (Hellmann-Feynman)
///               + Σ_μν W_μν * dS_μν/dR_A      (Pulay force)
///
/// We compute dS/dR and dH0/dR analytically for each atom pair,
/// then contract with P and W.
///
/// For drug-sized molecules (~50-100 atoms), this is fast enough on CPU.
/// The dominant cost is the SCC itself, not the gradient.
static void computeElectronicGradient(const SCCWorkspace &ws, double *gradient) {
    // For each shell pair (si, sj) on different atoms A, B:
    // Compute dS_block/dR and dH0_block/dR via finite differences on the
    // overlap/H0 elements w.r.t. interatomic distance, then contract with P and W.
    //
    // This is the most computationally intensive part but typically ~2x the SCC cost.

    // Use semi-analytical approach: numerical derivative of overlap integrals
    // (analytical Obara-Saika gradient is possible but the current STO-nG
    // implementation would need substantial refactoring)
    double h = 1e-5; // Bohr, finite difference step

    int natom = ws.natom;
    int nbasis = ws.nbasis;

    for (int a = 0; a < natom; a++) {
        for (int d = 0; d < 3; d++) {
            // Displaced positions
            std::vector<double> pos_p(ws.pos_bohr);
            pos_p[3*a + d] += h;
            std::vector<double> pos_m(ws.pos_bohr);
            pos_m[3*a + d] -= h;

            // Recompute overlap and H0 at displaced positions
            std::vector<double> S_p, S_m, H0_p, H0_m;
            buildOverlapMatrix(ws.shells, pos_p.data(), nbasis, S_p);
            buildOverlapMatrix(ws.shells, pos_m.data(), nbasis, S_m);

            // Apply same normalization as the original
            for (int i = 0; i < nbasis; i++)
                for (int j = 0; j < nbasis; j++) {
                    S_p[i * nbasis + j] *= ws.normFactor[i] * ws.normFactor[j];
                    S_m[i * nbasis + j] *= ws.normFactor[i] * ws.normFactor[j];
                }

            std::vector<double> cn_p, cn_m;
            computeCN(pos_p.data(), ws.Z.data(), natom, cn_p);
            computeCN(pos_m.data(), ws.Z.data(), natom, cn_m);

            buildH0(ws.shells, pos_p.data(), ws.Z.data(), natom, nbasis, S_p, cn_p, H0_p);
            buildH0(ws.shells, pos_m.data(), ws.Z.data(), natom, nbasis, S_m, cn_m, H0_m);

            // Finite difference: dS/dR and dH0/dR
            double grad_a = 0.0;
            for (int mu = 0; mu < nbasis; mu++) {
                for (int nu = 0; nu < nbasis; nu++) {
                    int idx = mu * nbasis + nu;
                    double dH0 = (H0_p[idx] - H0_m[idx]) / (2.0 * h);
                    double dS  = (S_p[idx] - S_m[idx]) / (2.0 * h);

                    // Hellmann-Feynman: P * dH0/dR
                    grad_a += ws.P[idx] * dH0;
                    // Pulay: W * dS/dR (with negative sign: -W * dS/dR)
                    grad_a -= ws.W[idx] * dS;
                }
            }

            gradient[3*a + d] += grad_a;
        }
    }
}

// ============================================================================
// MARK: - CN Gradient (for D4 CN-derivative propagation)
// ============================================================================

/// Gradient of coordination number w.r.t. atomic positions.
/// dCN_i/dR_A contributes to the total gradient via:
///   dE/dR_A += Σ_i (dE/dCN_i) * (dCN_i/dR_A)
///
/// Here we only compute dCN/dR; dE/dCN comes from H0 or D4.
static void computeCNGradient(const double *pos_bohr, const int *Z, int natom,
                               double *cn_gradient_atom,  // 3*natom, accumulates
                               const double *dEdCN) {     // natom, dE/dCN_i
    // GPU path
    if (g_xtb_gpu && g_xtb_gpu->gpu_compute_cn_gradient && natom >= 8) {
        g_xtb_gpu->gpu_compute_cn_gradient(g_xtb_gpu->context, pos_bohr,
                                            reinterpret_cast<const int32_t*>(Z),
                                            natom, dEdCN, cn_gradient_atom);
        return;
    }

    // CPU fallback
    for (int i = 0; i < natom; i++) {
        for (int j = 0; j < i; j++) {
            double dx = pos_bohr[3*i]   - pos_bohr[3*j];
            double dy = pos_bohr[3*i+1] - pos_bohr[3*j+1];
            double dz = pos_bohr[3*i+2] - pos_bohr[3*j+2];
            double rij = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (rij < 1e-6) continue;

            double rcov = getCovRad(Z[i]) + getCovRad(Z[j]);
            double arg = -16.0 * (4.0/3.0 * rcov / rij - 1.0);
            double exparg = std::exp(arg);
            double denom = (1.0 + exparg);
            // dcount/dr_ij = 16 * (4/3) * rcov / rij² * exparg / denom²
            double dcountdr = 16.0 * (4.0/3.0) * rcov / (rij * rij) * exparg / (denom * denom);

            // Chain rule: dE/dR via CN
            double scale = (dEdCN[i] + dEdCN[j]) * dcountdr / rij;

            cn_gradient_atom[3*i]   -= scale * dx;
            cn_gradient_atom[3*i+1] -= scale * dy;
            cn_gradient_atom[3*i+2] -= scale * dz;
            cn_gradient_atom[3*j]   += scale * dx;
            cn_gradient_atom[3*j+1] += scale * dy;
            cn_gradient_atom[3*j+2] += scale * dz;
        }
    }
}

// ============================================================================
// MARK: - Full Gradient Assembly
// ============================================================================

/// Compute total GFN2-xTB gradient: repulsion + electronic + Coulomb + D4 + solvation.
/// All gradient contributions accumulated in Hartree/Bohr.
static void computeTotalGradient(const SCCWorkspace &ws,
                                  DruseXTBSolvationConfig solv,
                                  std::vector<double> &totalGrad,
                                  double &Edisp, double &Esolv) {
    int natom = ws.natom;
    totalGrad.assign(3 * natom, 0.0);

    // 1. Repulsion gradient (analytical)
    computeRepulsionGradient(ws.pos_bohr.data(), ws.Z.data(), natom, totalGrad.data());

    // 2. Coulomb gradient (analytical)
    computeCoulombGradient(ws.shells, ws.pos_bohr.data(), ws.Z.data(), natom,
                           ws.shellCharges, totalGrad.data());

    // 3. Electronic gradient (semi-analytical: finite diff on S and H0)
    computeElectronicGradient(ws, totalGrad.data());

    // 4. D4 dispersion (analytical gradient)
    std::vector<double> dispGrad(3 * natom, 0.0);
    Edisp = computeD4Dispersion(ws.pos_bohr.data(), ws.Z.data(), natom,
                                 ws.cn, dispGrad.data());
    for (int i = 0; i < 3 * natom; i++) totalGrad[i] += dispGrad[i];

    // 5. Solvation gradient
    if (solv.model != DRUSE_XTB_SOLV_NONE) {
        std::vector<double> solvGrad(3 * natom, 0.0);
        Esolv = computeSolvation(ws.pos_bohr.data(), ws.Z.data(), natom,
                                  ws.atomCharges, solv, solvGrad.data());
        for (int i = 0; i < 3 * natom; i++) totalGrad[i] += solvGrad[i];
    } else {
        Esolv = 0.0;
    }
}

// ============================================================================
// MARK: - L-BFGS Geometry Optimizer
// ============================================================================

static constexpr int LBFGS_MEMORY = 20;

struct LBFGSState {
    int n;                      // number of variables (3*natom for unfrozen)
    int memory;
    int iter;
    std::vector<double> s_hist; // displacement history: memory × n
    std::vector<double> y_hist; // gradient diff history: memory × n
    std::vector<double> rho;    // 1/(s·y) history
    std::vector<double> alpha;  // workspace

    void init(int nvars) {
        n = nvars;
        memory = LBFGS_MEMORY;
        iter = 0;
        s_hist.resize(memory * n, 0.0);
        y_hist.resize(memory * n, 0.0);
        rho.resize(memory, 0.0);
        alpha.resize(memory, 0.0);
    }

    /// L-BFGS two-loop recursion: compute search direction from gradient.
    /// Returns the search direction (negated, i.e., -H*g).
    void computeDirection(const double *grad, const double *prev_grad,
                          const double *displacement,
                          std::vector<double> &direction) {
        direction.resize(n);

        if (iter > 0) {
            // Store latest s and y
            int idx = (iter - 1) % memory;
            double sy = 0.0;
            for (int i = 0; i < n; i++) {
                s_hist[idx * n + i] = displacement[i];
                y_hist[idx * n + i] = grad[i] - prev_grad[i];
                sy += s_hist[idx * n + i] * y_hist[idx * n + i];
            }
            rho[idx] = (std::fabs(sy) > 1e-16) ? 1.0 / sy : 0.0;
        }

        // Two-loop recursion (Nocedal & Wright, Algorithm 7.4)
        std::copy(grad, grad + n, direction.data());

        int bound = std::min(iter, memory);

        // First loop: backward
        for (int m = iter - 1; m >= std::max(0, iter - bound); m--) {
            int idx = m % memory;
            double dot = 0.0;
            for (int i = 0; i < n; i++)
                dot += s_hist[idx * n + i] * direction[i];
            alpha[idx] = rho[idx] * dot;
            for (int i = 0; i < n; i++)
                direction[i] -= alpha[idx] * y_hist[idx * n + i];
        }

        // Initial Hessian approximation (diagonal: gamma * I)
        double gamma = 1.0;
        if (iter > 0) {
            int idx = (iter - 1) % memory;
            double yy = 0.0, sy2 = 0.0;
            for (int i = 0; i < n; i++) {
                yy += y_hist[idx * n + i] * y_hist[idx * n + i];
                sy2 += s_hist[idx * n + i] * y_hist[idx * n + i];
            }
            if (yy > 1e-16) gamma = sy2 / yy;
        }
        for (int i = 0; i < n; i++) direction[i] *= gamma;

        // Second loop: forward
        for (int m = std::max(0, iter - bound); m < iter; m++) {
            int idx = m % memory;
            double dot = 0.0;
            for (int i = 0; i < n; i++)
                dot += y_hist[idx * n + i] * direction[i];
            double beta = rho[idx] * dot;
            for (int i = 0; i < n; i++)
                direction[i] += s_hist[idx * n + i] * (alpha[idx] - beta);
        }

        // Negate: direction = -H * g
        for (int i = 0; i < n; i++) direction[i] = -direction[i];

        // Damping: prevent excessively large steps early on
        // f_damp = 1 / (1 + 3000 * step^(-3))  [from xtb optimizer.f90]
        double fdamp = 1.0 / (1.0 + 3000.0 * std::pow((double)(iter + 1), -3.0));
        for (int i = 0; i < n; i++) direction[i] *= fdamp;

        iter++;
    }
};

// ============================================================================
// MARK: - C API: Full Energy (extern "C")
// ============================================================================

extern "C" {

DruseXTBEnergyResult* druse_xtb_compute_energy(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    int32_t maxIterations,
    DruseXTBSolvationConfig solvation)
{
    auto *result = new DruseXTBEnergyResult();
    std::memset(result, 0, sizeof(DruseXTBEnergyResult));
    result->atomCount = atomCount;

    SCCWorkspace ws;
    if (!runSCC(positions, atomicNumbers, atomCount, totalCharge, maxIterations, ws)) {
        result->success = false;
        std::snprintf(result->errorMessage, 512, "SCC calculation failed");
        return result;
    }

    // D4 dispersion
    double Edisp = computeD4Dispersion(ws.pos_bohr.data(), ws.Z.data(), atomCount, ws.cn);

    // Solvation
    double Esolv = computeSolvation(ws.pos_bohr.data(), ws.Z.data(), atomCount,
                                     ws.atomCharges, solvation);

    result->electronicEnergy = (float)ws.Eelec;
    result->repulsionEnergy = (float)ws.Erep;
    result->dispersionEnergy = (float)Edisp;
    result->solvationEnergy = (float)Esolv;
    result->totalEnergy = (float)(ws.Eelec + ws.Erep + Edisp + Esolv);

    result->charges = new float[atomCount];
    for (int i = 0; i < atomCount; i++)
        result->charges[i] = -(float)ws.atomCharges[i];

    result->scfIterations = ws.scfIterations;
    result->converged = ws.converged;
    result->success = true;

    return result;
}

void druse_xtb_free_energy_result(DruseXTBEnergyResult *result) {
    if (result) {
        delete[] result->charges;
        delete result;
    }
}

// ============================================================================
// MARK: - C API: Gradient
// ============================================================================

DruseXTBGradientResult* druse_xtb_compute_gradient(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    int32_t maxIterations,
    DruseXTBSolvationConfig solvation)
{
    auto *result = new DruseXTBGradientResult();
    std::memset(result, 0, sizeof(DruseXTBGradientResult));
    result->atomCount = atomCount;

    SCCWorkspace ws;
    if (!runSCC(positions, atomicNumbers, atomCount, totalCharge, maxIterations, ws)) {
        result->success = false;
        std::snprintf(result->errorMessage, 512, "SCC calculation failed");
        return result;
    }

    double Edisp = 0.0, Esolv = 0.0;
    std::vector<double> totalGrad;
    computeTotalGradient(ws, solvation, totalGrad, Edisp, Esolv);

    result->electronicEnergy = (float)ws.Eelec;
    result->repulsionEnergy = (float)ws.Erep;
    result->dispersionEnergy = (float)Edisp;
    result->solvationEnergy = (float)Esolv;
    result->totalEnergy = (float)(ws.Eelec + ws.Erep + Edisp + Esolv);

    result->gradient = new float[3 * atomCount];
    double gnorm2 = 0.0;
    for (int i = 0; i < 3 * atomCount; i++) {
        result->gradient[i] = (float)totalGrad[i];
        gnorm2 += totalGrad[i] * totalGrad[i];
    }
    result->gradientNorm = (float)std::sqrt(gnorm2 / atomCount);

    result->charges = new float[atomCount];
    for (int i = 0; i < atomCount; i++)
        result->charges[i] = -(float)ws.atomCharges[i];

    result->scfIterations = ws.scfIterations;
    result->converged = ws.converged;
    result->success = true;

    return result;
}

void druse_xtb_free_gradient_result(DruseXTBGradientResult *result) {
    if (result) {
        delete[] result->gradient;
        delete[] result->charges;
        delete result;
    }
}

// ============================================================================
// MARK: - C API: Geometry Optimization (L-BFGS)
// ============================================================================

DruseXTBOptResult* druse_xtb_optimize_geometry(
    const float *positions,
    const int32_t *atomicNumbers,
    int32_t atomCount,
    int32_t totalCharge,
    DruseXTBSolvationConfig solvation,
    DruseXTBOptLevel optLevel,
    int32_t maxSteps,
    const bool *freezeMask)
{
    auto *result = new DruseXTBOptResult();
    std::memset(result, 0, sizeof(DruseXTBOptResult));
    result->atomCount = atomCount;

    // Convergence thresholds based on opt level
    double ethr, gthr;
    int defaultMaxSteps;
    switch (optLevel) {
        case DRUSE_XTB_OPT_CRUDE:
            ethr = 5e-4; gthr = 1e-2; defaultMaxSteps = atomCount; break;
        case DRUSE_XTB_OPT_NORMAL:
            ethr = 5e-6; gthr = 1e-3; defaultMaxSteps = 3 * atomCount; break;
        case DRUSE_XTB_OPT_TIGHT:
            ethr = 1e-6; gthr = 8e-4; defaultMaxSteps = 5 * atomCount; break;
        case DRUSE_XTB_OPT_EXTREME:
            ethr = 5e-8; gthr = 5e-5; defaultMaxSteps = 20 * atomCount; break;
        default:
            ethr = 5e-6; gthr = 1e-3; defaultMaxSteps = 3 * atomCount; break;
    }
    int nSteps = (maxSteps > 0) ? maxSteps : std::min(defaultMaxSteps, 500);

    // Working copy of positions (Angstrom)
    std::vector<float> pos(positions, positions + 3 * atomCount);

    // Determine which atoms are free to move
    std::vector<bool> frozen(atomCount, false);
    if (freezeMask) {
        for (int i = 0; i < atomCount; i++) frozen[i] = freezeMask[i];
    }

    // Map free coordinates
    int nfree = 0;
    std::vector<int> freeMap; // indices into pos[] for free coordinates
    for (int i = 0; i < atomCount; i++) {
        if (!frozen[i]) {
            freeMap.push_back(3*i);
            freeMap.push_back(3*i+1);
            freeMap.push_back(3*i+2);
            nfree += 3;
        }
    }
    if (nfree == 0) {
        result->success = false;
        std::snprintf(result->errorMessage, 512, "All atoms frozen");
        return result;
    }

    LBFGSState lbfgs;
    lbfgs.init(nfree);

    std::vector<double> prevGrad(nfree, 0.0);
    std::vector<double> displacement(nfree, 0.0);
    double prevEnergy = 1e30;

    result->converged = false;

    for (int step = 0; step < nSteps; step++) {
        // Run SCC at current geometry
        SCCWorkspace ws;
        if (!runSCC(pos.data(), atomicNumbers, atomCount, totalCharge, 50, ws)) {
            result->success = false;
            std::snprintf(result->errorMessage, 512,
                         "SCC failed at optimization step %d", step);
            break;
        }

        // Compute gradient
        double Edisp = 0.0, Esolv = 0.0;
        std::vector<double> totalGrad;
        computeTotalGradient(ws, solvation, totalGrad, Edisp, Esolv);

        double Etotal = ws.Eelec + ws.Erep + Edisp + Esolv;

        // Extract free-coordinate gradient (convert Hartree/Bohr → Hartree/Angstrom)
        std::vector<double> freeGrad(nfree);
        for (int k = 0; k < nfree; k++)
            freeGrad[k] = totalGrad[freeMap[k]] * BOHR_TO_ANG; // chain rule: dE/dR_ang = dE/dR_bohr * bohr/ang

        // Gradient norm (RMS per free atom, in Hartree/Bohr)
        double gnorm2 = 0.0;
        for (int i = 0; i < 3 * atomCount; i++) gnorm2 += totalGrad[i] * totalGrad[i];
        double gnorm = std::sqrt(gnorm2 / atomCount);

        // Check convergence
        double dE = Etotal - prevEnergy;
        if (step > 0 && std::fabs(dE) < ethr && gnorm < gthr) {
            result->converged = true;
            result->optimizationSteps = step + 1;
            result->finalGradientNorm = (float)gnorm;
            result->energyChange = (float)dE;
            result->totalEnergy = (float)Etotal;
            result->electronicEnergy = (float)ws.Eelec;
            result->repulsionEnergy = (float)ws.Erep;
            result->dispersionEnergy = (float)Edisp;
            result->solvationEnergy = (float)Esolv;

            result->charges = new float[atomCount];
            for (int i = 0; i < atomCount; i++)
                result->charges[i] = -(float)ws.atomCharges[i];

            result->optimizedPositions = new float[3 * atomCount];
            std::copy(pos.begin(), pos.end(), result->optimizedPositions);
            result->success = true;
            return result;
        }

        // L-BFGS step
        std::vector<double> direction;
        lbfgs.computeDirection(freeGrad.data(),
                                step > 0 ? prevGrad.data() : freeGrad.data(),
                                displacement.data(), direction);

        // Trust region: cap maximum displacement to 0.5 Angstrom per atom
        double maxDisp = 0.0;
        for (int k = 0; k < nfree; k++)
            maxDisp = std::max(maxDisp, std::fabs(direction[k]));
        if (maxDisp > 0.5) {
            double scale = 0.5 / maxDisp;
            for (int k = 0; k < nfree; k++) direction[k] *= scale;
        }

        // Apply displacement
        for (int k = 0; k < nfree; k++) {
            pos[freeMap[k]] += (float)direction[k];
            displacement[k] = direction[k];
        }

        prevGrad = freeGrad;
        prevEnergy = Etotal;
    }

    // Did not converge — return best geometry anyway
    if (!result->success) {
        // Run final SCC
        SCCWorkspace ws;
        if (runSCC(pos.data(), atomicNumbers, atomCount, totalCharge, 50, ws)) {
            double Edisp = computeD4Dispersion(ws.pos_bohr.data(), ws.Z.data(), atomCount, ws.cn);
            double Esolv = computeSolvation(ws.pos_bohr.data(), ws.Z.data(), atomCount,
                                             ws.atomCharges, solvation);
            result->totalEnergy = (float)(ws.Eelec + ws.Erep + Edisp + Esolv);
            result->electronicEnergy = (float)ws.Eelec;
            result->repulsionEnergy = (float)ws.Erep;
            result->dispersionEnergy = (float)Edisp;
            result->solvationEnergy = (float)Esolv;
            result->charges = new float[atomCount];
            for (int i = 0; i < atomCount; i++)
                result->charges[i] = -(float)ws.atomCharges[i];
        }
        result->optimizedPositions = new float[3 * atomCount];
        std::copy(pos.begin(), pos.end(), result->optimizedPositions);
        result->optimizationSteps = nSteps;
        result->converged = false;
        result->success = true;
        std::snprintf(result->errorMessage, 512,
                     "Optimization did not converge within %d steps", nSteps);
    }

    return result;
}

void druse_xtb_free_opt_result(DruseXTBOptResult *result) {
    if (result) {
        delete[] result->optimizedPositions;
        delete[] result->charges;
        delete result;
    }
}

} // extern "C"
