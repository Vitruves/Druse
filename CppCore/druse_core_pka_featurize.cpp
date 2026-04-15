// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core_internal.h"

// pKa GNN featurization — produces atom/bond feature matrices matching
// the Python training script (train_pka.py) exactly.

#include <GraphMol/RingInfo.h>
#include <GraphMol/MolOps.h>
#include <cstring>
#include <algorithm>

// Element vocabulary (must match train_pka.py ELEMENT_MAP)
static int elementIndex(int atomicNum) {
    switch (atomicNum) {
        case 1:  return 0;   // H
        case 6:  return 1;   // C
        case 7:  return 2;   // N
        case 8:  return 3;   // O
        case 9:  return 4;   // F
        case 15: return 5;   // P
        case 16: return 6;   // S
        case 17: return 7;   // Cl
        case 35: return 8;   // Br
        case 34: return 9;   // Se
        case 5:  return 10;  // B
        case 14: return 11;  // Si
        case 53: return 12;  // I
        default: return 13;  // other
    }
}

// Pauling electronegativity / 3.98  (must match train_pka.py)
static float scaledEN(int atomicNum) {
    constexpr float MAX_EN = 3.98f;
    switch (atomicNum) {
        case 1:  return 2.20f / MAX_EN;
        case 6:  return 2.55f / MAX_EN;
        case 7:  return 3.04f / MAX_EN;
        case 8:  return 3.44f / MAX_EN;
        case 9:  return 3.98f / MAX_EN;
        case 15: return 2.19f / MAX_EN;
        case 16: return 2.58f / MAX_EN;
        case 17: return 3.16f / MAX_EN;
        case 35: return 2.96f / MAX_EN;
        case 34: return 2.55f / MAX_EN;
        case 5:  return 2.04f / MAX_EN;
        case 14: return 1.90f / MAX_EN;
        case 53: return 2.66f / MAX_EN;
        default: return 2.50f / MAX_EN;
    }
}

DrusePKaGraphResult* druse_featurize_pka_graph(const char *smiles) {
    auto *result = new DrusePKaGraphResult();
    memset(result, 0, sizeof(DrusePKaGraphResult));
    result->atomFeatures = new float[PKA_MAX_ATOMS * PKA_ATOM_FEAT]();
    result->bondFeatures = new float[PKA_MAX_EDGES * PKA_BOND_FEAT]();
    result->edgeSrc = new int32_t[PKA_MAX_EDGES]();
    result->edgeDst = new int32_t[PKA_MAX_EDGES]();

    if (!smiles || !smiles[0]) {
        result->success = false;
        snprintf(result->errorMessage, 256, "Empty SMILES");
        return result;
    }

    try {
        // Parse and canonicalize for deterministic atom ordering
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) {
            result->success = false;
            snprintf(result->errorMessage, 256, "Invalid SMILES");
            return result;
        }

        std::string canon = MolToSmiles(*mol);
        mol.reset(SmilesToMol(canon));
        if (!mol) {
            result->success = false;
            snprintf(result->errorMessage, 256, "Failed to reparse canonical SMILES");
            return result;
        }

        int nAtoms = mol->getNumAtoms();
        if (nAtoms > PKA_MAX_ATOMS) {
            result->success = false;
            snprintf(result->errorMessage, 256, "Too many atoms: %d (max %d)", nAtoms, PKA_MAX_ATOMS);
            return result;
        }

        const auto *ri = mol->getRingInfo();

        // Atom features [N, 25] — row-major
        for (int i = 0; i < nAtoms; i++) {
            float *feat = &result->atomFeatures[i * PKA_ATOM_FEAT];
            const Atom *atom = mol->getAtomWithIdx(i);
            int an = atom->getAtomicNum();

            // Element one-hot (0-13)
            feat[elementIndex(an)] = 1.0f;

            // Degree / 6 (14)
            feat[14] = std::min((int)atom->getDegree(), 6) / 6.0f;

            // Hybridization (15-17): sp, sp2, sp3
            auto hyb = atom->getHybridization();
            if (hyb == Atom::SP)       feat[15] = 1.0f;
            else if (hyb == Atom::SP2) feat[16] = 1.0f;
            else if (hyb == Atom::SP3) feat[17] = 1.0f;

            // Aromaticity (18)
            feat[18] = atom->getIsAromatic() ? 1.0f : 0.0f;

            // Formal charge / 2, clipped to [-1, 1] (19)
            feat[19] = std::max(-2, std::min(2, atom->getFormalCharge())) / 2.0f;

            // In ring (20)
            feat[20] = (ri && ri->numAtomRings(i) > 0) ? 1.0f : 0.0f;

            // Total Hs / 4 (21)
            feat[21] = std::min((int)atom->getTotalNumHs(), 4) / 4.0f;

            // Electronegativity (22)
            feat[22] = scaledEN(an);

            // Radical electrons / 2 (23)
            feat[23] = std::min((int)atom->getNumRadicalElectrons(), 2) / 2.0f;

            // Is heteroatom (24)
            feat[24] = (an != 1 && an != 6) ? 1.0f : 0.0f;
        }

        // Build edges (both directions per bond)
        int nEdges = 0;
        for (auto bondIt = mol->beginBonds(); bondIt != mol->endBonds(); ++bondIt) {
            if (nEdges + 2 > PKA_MAX_EDGES) break;

            const Bond *bond = *bondIt;
            int a1 = bond->getBeginAtomIdx();
            int a2 = bond->getEndAtomIdx();

            // Bond features [10]
            float bf[PKA_BOND_FEAT] = {};

            // Bond type one-hot (0-3)
            auto bt = bond->getBondType();
            if (bt == Bond::SINGLE)       bf[0] = 1.0f;
            else if (bt == Bond::DOUBLE)  bf[1] = 1.0f;
            else if (bt == Bond::TRIPLE)  bf[2] = 1.0f;
            else if (bt == Bond::AROMATIC) bf[3] = 1.0f;

            // Conjugated (4)
            bf[4] = bond->getIsConjugated() ? 1.0f : 0.0f;

            // In ring (5)
            bool inRing = ri && ri->numBondRings(bond->getIdx()) > 0;
            bf[5] = inRing ? 1.0f : 0.0f;

            // Ring size (6-8): 5, 6, other
            if (inRing && ri) {
                for (const auto &ring : ri->atomRings()) {
                    bool hasA1 = false, hasA2 = false;
                    for (int idx : ring) {
                        if (idx == a1) hasA1 = true;
                        if (idx == a2) hasA2 = true;
                    }
                    if (hasA1 && hasA2) {
                        int rsize = (int)ring.size();
                        if (rsize == 5)      bf[6] = 1.0f;
                        else if (rsize == 6) bf[7] = 1.0f;
                        else                 bf[8] = 1.0f;
                        break;
                    }
                }
            }

            // Stereo (9)
            bf[9] = (bond->getStereo() != Bond::STEREONONE) ? 1.0f : 0.0f;

            // Forward edge
            result->edgeSrc[nEdges] = a1;
            result->edgeDst[nEdges] = a2;
            memcpy(&result->bondFeatures[nEdges * PKA_BOND_FEAT], bf, sizeof(bf));
            nEdges++;

            // Reverse edge
            result->edgeSrc[nEdges] = a2;
            result->edgeDst[nEdges] = a1;
            memcpy(&result->bondFeatures[nEdges * PKA_BOND_FEAT], bf, sizeof(bf));
            nEdges++;
        }

        // Self-loop fallback for single-atom molecules
        if (nEdges == 0 && nAtoms > 0) {
            result->edgeSrc[0] = 0;
            result->edgeDst[0] = 0;
            nEdges = 1;
        }

        result->numAtoms = nAtoms;
        result->numEdges = nEdges;
        result->success = true;

    } catch (const std::exception &e) {
        result->success = false;
        snprintf(result->errorMessage, 256, "%s", e.what());
    } catch (...) {
        result->success = false;
        snprintf(result->errorMessage, 256, "Unknown error");
    }

    return result;
}

void druse_free_pka_graph(DrusePKaGraphResult *result) {
    if (!result) return;
    delete[] result->atomFeatures;
    delete[] result->bondFeatures;
    delete[] result->edgeSrc;
    delete[] result->edgeDst;
    delete result;
}
