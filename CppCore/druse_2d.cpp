#include "druse_core.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/Depictor/RDDepictor.h>

#include <cstring>
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

        for (int i = 0; i < natoms; i++) {
            const auto &pos = conf.getAtomPos(i);
            res->coords[i * 2]     = (float)pos.x;
            res->coords[i * 2 + 1] = (float)pos.y;
            res->atomicNums[i] = mol->getAtomWithIdx(i)->getAtomicNum();
        }

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
    delete[] result->bonds;
    delete result;
}
