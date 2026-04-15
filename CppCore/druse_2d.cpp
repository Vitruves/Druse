// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/Depictor/RDDepictor.h>
#include <GraphMol/MolDraw2D/MolDraw2DSVG.h>
#include <GraphMol/MolDraw2D/MolDraw2DUtils.h>

#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace RDKit;

// ============================================================================
// 2D Coordinate Generation (RDDepict)
// ============================================================================

Druse2DResult* druse_compute_2d_coords(const char *smiles) {
    if (!smiles || !smiles[0]) return nullptr;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return nullptr;

        // Remove explicit Hs for a cleaner 2D depiction (heavy atoms only)
        MolOps::removeAllHs(*mol);

        // Generate 2D coordinates via CoordGen (RDKit's default depiction engine)
        RDDepict::compute2DCoords(*mol);

        if (mol->getNumConformers() == 0) return nullptr;

        const auto &conf = mol->getConformer(0);
        int natoms = (int)mol->getNumAtoms();

        auto *res = new Druse2DResult();
        res->atomCount = natoms;
        res->coords = new float[natoms * 2];   // flat [x0, y0, x1, y1, ...]
        res->atomicNums = new int32_t[natoms];
        res->isAromatic = new bool[natoms];

        // Capture aromaticity BEFORE Kekulization (which clears aromatic flags)
        for (int i = 0; i < natoms; i++) {
            const auto &pos = conf.getAtomPos(i);
            res->coords[i * 2]     = (float)pos.x;
            res->coords[i * 2 + 1] = (float)pos.y;
            res->atomicNums[i] = mol->getAtomWithIdx(i)->getAtomicNum();
            res->isAromatic[i] = mol->getAtomWithIdx(i)->getIsAromatic();
        }

        // Kekulize to get alternating single/double bonds for aromatic rings
        // (standard chemistry depiction instead of bond order 4)
        MolOps::Kekulize(*mol);

        // Bonds
        int nbonds = (int)mol->getNumBonds();
        res->bondCount = nbonds;
        res->bonds = new DruseBond[nbonds];

        for (int i = 0; i < nbonds; i++) {
            const auto *bond = mol->getBondWithIdx(i);
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

        return res;
    } catch (...) {
        return nullptr;
    }
}

void druse_free_2d_result(Druse2DResult *result) {
    if (!result) return;
    delete[] result->coords;
    delete[] result->atomicNums;
    delete[] result->isAromatic;
    delete[] result->bonds;
    delete result;
}

// ============================================================================
// SVG Depiction (MolDraw2DSVG)
// ============================================================================
// Produces publication-quality 2D depictions with:
// - Wedge/dash stereo bonds
// - Proper aromatic ring notation
// - Element coloring (CPK scheme)
// - Clean CoordGen layout

char* druse_mol_to_svg(const char *smiles, int32_t width, int32_t height) {
    if (!smiles || !smiles[0]) return nullptr;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return nullptr;

        // Remove explicit H for cleaner depiction, but preserve stereo info
        MolOps::removeAllHs(*mol);

        // Generate 2D coordinates
        RDDepict::compute2DCoords(*mol);

        // Assign stereo wedge/dash bonds from 3D/stereochemistry info
        MolOps::Kekulize(*mol);
        MolDraw2DUtils::prepareMolForDrawing(*mol);

        // Draw to SVG — classic RDKit publication style
        MolDraw2DSVG drawer(width, height);

        // White background for clean, readable depiction
        drawer.drawOptions().backgroundColour = DrawColour(1.0, 1.0, 1.0, 1.0);

        // Thicker bonds and larger font for clarity
        drawer.drawOptions().bondLineWidth = 2.0;
        drawer.drawOptions().multipleBondOffset = 0.15;
        drawer.drawOptions().minFontSize = 14;
        drawer.drawOptions().annotationFontScale = 0.75;

        // Use RDKit's default palette (CPK-style atom colors are already assigned)
        // The default palette from assignDefaultPalette() already has:
        // N=blue, O=red, F/Cl=green, S=dark yellow, Br=dark red, P=orange, etc.

        drawer.drawMolecule(*mol);
        drawer.finishDrawing();

        std::string svg = drawer.getDrawingText();
        char *result = new char[svg.size() + 1];
        memcpy(result, svg.c_str(), svg.size() + 1);
        return result;
    } catch (...) {
        return nullptr;
    }
}

// druse_free_string already defined in druse_core.cpp
