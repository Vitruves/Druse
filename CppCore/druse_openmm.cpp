// ============================================================================
// druse_openmm.cpp — Native C++ OpenMM pocket refinement
//
// Replaces the Python subprocess in OpenMMPocketRefiner.swift.
// Uses OpenMM C++ API directly:
//   - CustomExternalForce for positional restraints
//   - CustomNonbondedForce for protein-ligand interaction energy
//   - HarmonicBondForce to maintain protein bond geometry
//   - LocalEnergyMinimizer for L-BFGS minimization
// ============================================================================

#ifdef DRUSE_HAS_OPENMM

#include "druse_openmm.h"

// Include only the OpenMM headers we actually need (avoid umbrella OpenMM.h
// which pulls in ConstantPotentialForce with incomplete-type issues on newer clang)
#include "openmm/System.h"
#include "openmm/Context.h"
#include "openmm/State.h"
#include "openmm/Platform.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/LocalEnergyMinimizer.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/CustomNonbondedForce.h"
#include "openmm/Vec3.h"

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <set>

using namespace OpenMM;

// ============================================================================
// Helpers
// ============================================================================

static double estimateBondForceConstant(int atomicNum1, int atomicNum2) {
    // Approximate harmonic bond force constants (kJ/mol/nm²)
    // Based on AMBER ff14SB typical values for protein bonds.
    // These don't need to be exact — they just prevent bond stretching
    // during the small displacements of pocket refinement.
    int minZ = std::min(atomicNum1, atomicNum2);
    int maxZ = std::max(atomicNum1, atomicNum2);

    if (minZ == 6 && maxZ == 6)   return 259408;   // C-C  (~310 kcal/mol/Å²)
    if (minZ == 6 && maxZ == 7)   return 282001;   // C-N  (~337 kcal/mol/Å²)
    if (minZ == 6 && maxZ == 8)   return 301248;   // C-O  (~360 kcal/mol/Å²)
    if (minZ == 6 && maxZ == 16)  return 198320;   // C-S  (~237 kcal/mol/Å²)
    if (minZ == 7 && maxZ == 7)   return 282001;   // N-N
    if (minZ == 7 && maxZ == 8)   return 301248;   // N-O
    if (minZ == 8 && maxZ == 15)  return 230120;   // O-P
    if (minZ == 8 && maxZ == 16)  return 230120;   // O-S
    if (minZ == 16 && maxZ == 16) return 166000;    // S-S  (disulfide)
    return 250000;  // general fallback (~300 kcal/mol/Å²)
}

// ============================================================================
// Main refinement function
// ============================================================================

DruseOpenMMResult* druse_openmm_refine(
    const DruseOpenMMAtom *proteinAtoms,
    int32_t proteinAtomCount,
    const DruseOpenMMBond *proteinBonds,
    int32_t proteinBondCount,
    const DruseOpenMMLigandSite *ligandSites,
    int32_t ligandSiteCount,
    float pocketK,
    float backboneK,
    int32_t maxIterations
) {
    auto *result = new DruseOpenMMResult();
    result->success = false;
    result->interactionEnergyKcal = 0;
    result->refinedPositionsX = nullptr;
    result->refinedPositionsY = nullptr;
    result->refinedPositionsZ = nullptr;
    result->atomCount = 0;
    result->errorMessage[0] = '\0';

    if (!proteinAtoms || proteinAtomCount <= 0 || !ligandSites || ligandSiteCount <= 0) {
        std::strncpy(result->errorMessage, "Invalid input: no protein atoms or ligand sites", 255);
        return result;
    }

    try {
        // Load plugins (CPU platform)
        Platform::loadPluginsFromDirectory(Platform::getDefaultPluginsDirectory());

        // -------------------------------------------------------------------
        // 1. Create System
        // -------------------------------------------------------------------
        System system;

        // Total particles = protein atoms + ligand sites
        int totalParticles = proteinAtomCount + ligandSiteCount;

        // Add protein atoms
        for (int i = 0; i < proteinAtomCount; i++) {
            system.addParticle(proteinAtoms[i].mass);
        }

        // Add ligand sites as massless dummy particles (frozen via restraints)
        for (int i = 0; i < ligandSiteCount; i++) {
            system.addParticle(0.0);  // zero mass → infinite restraint effectively freezes them
        }

        // -------------------------------------------------------------------
        // 2. HarmonicBondForce — maintain protein geometry
        // -------------------------------------------------------------------
        auto *bondForce = new HarmonicBondForce();
        bondForce->setForceGroup(0);

        for (int i = 0; i < proteinBondCount; i++) {
            int a1 = proteinBonds[i].atom1;
            int a2 = proteinBonds[i].atom2;
            if (a1 < 0 || a1 >= proteinAtomCount || a2 < 0 || a2 >= proteinAtomCount) continue;

            double lengthNm = proteinBonds[i].lengthNm;
            double k = estimateBondForceConstant(
                proteinAtoms[a1].atomicNum,
                proteinAtoms[a2].atomicNum
            );
            bondForce->addBond(a1, a2, lengthNm, k);
        }
        system.addForce(bondForce);

        // -------------------------------------------------------------------
        // 3. CustomExternalForce — positional restraints
        // -------------------------------------------------------------------
        auto *restraint = new CustomExternalForce(
            "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        );
        restraint->addPerParticleParameter("k");
        restraint->addPerParticleParameter("x0");
        restraint->addPerParticleParameter("y0");
        restraint->addPerParticleParameter("z0");
        restraint->setForceGroup(30);

        // Protein atoms: pocket gets pocketK, non-pocket gets backboneK
        for (int i = 0; i < proteinAtomCount; i++) {
            double posNmX = proteinAtoms[i].x / 10.0;
            double posNmY = proteinAtoms[i].y / 10.0;
            double posNmZ = proteinAtoms[i].z / 10.0;
            double k = proteinAtoms[i].isPocket ? pocketK : backboneK;
            std::vector<double> params = {k, posNmX, posNmY, posNmZ};
            restraint->addParticle(i, params);
        }

        // Ligand sites: effectively frozen (very large k since mass=0 already does this,
        // but add restraints as safety)
        for (int i = 0; i < ligandSiteCount; i++) {
            double k = 50000.0;
            std::vector<double> params = {
                k,
                (double)ligandSites[i].x,
                (double)ligandSites[i].y,
                (double)ligandSites[i].z
            };
            restraint->addParticle(proteinAtomCount + i, params);
        }

        system.addForce(restraint);

        // -------------------------------------------------------------------
        // 4. CustomNonbondedForce — protein-ligand interaction
        //    Coulomb + Lennard-Jones (arithmetic mean sigma, geometric mean epsilon)
        // -------------------------------------------------------------------
        auto *interaction = new CustomNonbondedForce(
            "138.935456*charge1*charge2/r + "
            "4*sqrt(epsilon1*epsilon2)*(((sigma1+sigma2)/(2*r))^12 - ((sigma1+sigma2)/(2*r))^6)"
        );
        interaction->addPerParticleParameter("charge");
        interaction->addPerParticleParameter("sigma");
        interaction->addPerParticleParameter("epsilon");
        interaction->setNonbondedMethod(CustomNonbondedForce::NoCutoff);
        interaction->setForceGroup(31);

        // Add protein particles
        for (int i = 0; i < proteinAtomCount; i++) {
            std::vector<double> params = {
                (double)proteinAtoms[i].charge,
                (double)proteinAtoms[i].sigmaNm,
                (double)proteinAtoms[i].epsilonKJ
            };
            interaction->addParticle(params);
        }

        // Add ligand site particles
        for (int i = 0; i < ligandSiteCount; i++) {
            std::vector<double> params = {
                (double)ligandSites[i].charge,
                (double)ligandSites[i].sigmaNm,
                (double)ligandSites[i].epsilonKJ
            };
            interaction->addParticle(params);
        }

        // Only evaluate protein-ligand pairs (not protein-protein or ligand-ligand)
        std::set<int> proteinGroup, ligandGroup;
        for (int i = 0; i < proteinAtomCount; i++) proteinGroup.insert(i);
        for (int i = 0; i < ligandSiteCount; i++) ligandGroup.insert(proteinAtomCount + i);
        interaction->addInteractionGroup(proteinGroup, ligandGroup);

        system.addForce(interaction);

        // -------------------------------------------------------------------
        // 5. Set up integrator, context, positions
        // -------------------------------------------------------------------
        VerletIntegrator integrator(0.001);  // 1 fs (not used for dynamics, just required)

        // Try CPU platform explicitly
        Platform *platform = nullptr;
        try {
            platform = &Platform::getPlatformByName("CPU");
        } catch (...) {
            try {
                platform = &Platform::getPlatformByName("Reference");
            } catch (...) {
                // Fall through to default
            }
        }

        Context *context;
        if (platform) {
            context = new Context(system, integrator, *platform);
        } else {
            context = new Context(system, integrator);
        }

        // Set positions (convert protein from Å to nm; ligand already in nm)
        std::vector<Vec3> positions(totalParticles);
        for (int i = 0; i < proteinAtomCount; i++) {
            positions[i] = Vec3(
                proteinAtoms[i].x / 10.0,
                proteinAtoms[i].y / 10.0,
                proteinAtoms[i].z / 10.0
            );
        }
        for (int i = 0; i < ligandSiteCount; i++) {
            positions[proteinAtomCount + i] = Vec3(
                ligandSites[i].x,
                ligandSites[i].y,
                ligandSites[i].z
            );
        }
        context->setPositions(positions);

        // -------------------------------------------------------------------
        // 6. Minimize
        // -------------------------------------------------------------------
        LocalEnergyMinimizer::minimize(*context, 10.0, maxIterations);

        // -------------------------------------------------------------------
        // 7. Extract results
        // -------------------------------------------------------------------
        // Interaction energy (force group 31 only)
        State interactionState = context->getState(State::Energy, false, 1 << 31);
        double interactionKJ = interactionState.getPotentialEnergy();
        result->interactionEnergyKcal = (float)(interactionKJ * 0.239005736);

        // Refined protein positions (convert nm → Å)
        State posState = context->getState(State::Positions);
        const std::vector<Vec3> &finalPositions = posState.getPositions();

        result->refinedPositionsX = new float[proteinAtomCount];
        result->refinedPositionsY = new float[proteinAtomCount];
        result->refinedPositionsZ = new float[proteinAtomCount];
        result->atomCount = proteinAtomCount;

        for (int i = 0; i < proteinAtomCount; i++) {
            result->refinedPositionsX[i] = (float)(finalPositions[i][0] * 10.0);
            result->refinedPositionsY[i] = (float)(finalPositions[i][1] * 10.0);
            result->refinedPositionsZ[i] = (float)(finalPositions[i][2] * 10.0);
        }

        result->success = true;
        delete context;

    } catch (const std::exception &e) {
        std::strncpy(result->errorMessage, e.what(), 255);
        result->errorMessage[255] = '\0';
    } catch (...) {
        std::strncpy(result->errorMessage, "Unknown OpenMM error", 255);
    }

    return result;
}

void druse_free_openmm_result(DruseOpenMMResult *result) {
    if (!result) return;
    delete[] result->refinedPositionsX;
    delete[] result->refinedPositionsY;
    delete[] result->refinedPositionsZ;
    delete result;
}

bool druse_openmm_available(void) {
    return true;
}

#else // DRUSE_HAS_OPENMM not defined

#include "druse_openmm.h"
#include <cstring>

DruseOpenMMResult* druse_openmm_refine(
    const DruseOpenMMAtom *, int32_t,
    const DruseOpenMMBond *, int32_t,
    const DruseOpenMMLigandSite *, int32_t,
    float, float, int32_t
) {
    auto *result = new DruseOpenMMResult();
    result->success = false;
    result->interactionEnergyKcal = 0;
    result->refinedPositionsX = nullptr;
    result->refinedPositionsY = nullptr;
    result->refinedPositionsZ = nullptr;
    result->atomCount = 0;
    std::strncpy(result->errorMessage, "OpenMM not available (not compiled with DRUSE_HAS_OPENMM)", 255);
    return result;
}

void druse_free_openmm_result(DruseOpenMMResult *result) {
    if (!result) return;
    delete[] result->refinedPositionsX;
    delete[] result->refinedPositionsY;
    delete[] result->refinedPositionsZ;
    delete result;
}

bool druse_openmm_available(void) {
    return false;
}

#endif // DRUSE_HAS_OPENMM
