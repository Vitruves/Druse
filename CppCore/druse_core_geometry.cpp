// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core_internal.h"

// Torsion tree, spatial indices, alignment, energies, and ESP.

#include <GraphMol/RingInfo.h>
#include <queue>
#include <unordered_set>
#include <nanoflann.hpp>
#include <Eigen/Dense>

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
    if (rotBonds.empty()) return tree;

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

namespace {

struct PointCloud3f {
    const float *pts;
    int32_t count;

    inline size_t kdtree_get_point_count() const { return (size_t)count; }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx * 3 + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};

using KDTree3f = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud3f>,
    PointCloud3f, 3
>;

struct KDTreeHandle {
    PointCloud3f cloud;
    std::unique_ptr<KDTree3f> tree;
};

} // namespace

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
    size_t found = handle->tree->radiusSearch(queryPoint, radius * radius, matches, params);

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

float druse_kabsch_superpose(
    const float *mobile, const float *reference, int32_t n,
    float *rotation_out, float *translation_out
) {
    if (!mobile || !reference || n <= 0) return -1.0f;

    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> mob(mobile, n, 3);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>> ref(reference, n, 3);

    Eigen::Vector3f centroid_mob = mob.colwise().mean();
    Eigen::Vector3f centroid_ref = ref.colwise().mean();

    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> mob_c = mob.rowwise() - centroid_mob.transpose();
    Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> ref_c = ref.rowwise() - centroid_ref.transpose();

    Eigen::Matrix3f H = mob_c.transpose() * ref_c;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    float d = (V * U.transpose()).determinant();
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    if (d < 0.0f) S(2, 2) = -1.0f;

    Eigen::Matrix3f R = V * S * U.transpose();
    Eigen::Vector3f t = centroid_ref - R * centroid_mob;

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

DruseMoleculeResult* druse_minimize_lbfgs(const char *smiles, const char *name, int32_t maxIters) {
    if (!smiles || !smiles[0]) return make_error("Empty SMILES");
    if (maxIters <= 0) maxIters = 2000;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return make_error("Failed to parse SMILES");

        MolOps::addHs(*mol);
        if (embed_molecule(*mol) < 0) return make_error("3D embedding failed");

        auto res = MMFF::MMFFOptimizeMolecule(*mol, maxIters, "MMFF94", 1e-6);
        if (res.first < 0) {
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

double druse_mmff_strain_energy(const char *smiles, const float *heavyPositions, int32_t numHeavy) {
    if (!smiles || !heavyPositions || numHeavy <= 0) return NAN;
    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return NAN;

        MolOps::addHs(*mol);
        if (embed_molecule(*mol) < 0) return NAN;

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

        MMFF::MMFFMolProperties mmffProps(*mol);
        if (!mmffProps.isValid()) return NAN;

        auto *ff = MMFF::constructForceField(*mol, &mmffProps);
        if (!ff) return NAN;

        for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
            if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) {
                ff->fixedPoints().push_back(i);
            }
        }
        ff->minimize(200);

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

DruseMoleculeResult* druse_parse_mmcif(const char *content) {
    return druse_parse_structure(content);
}

void druse_compute_esp(
    const float *atomPositions, const float *charges, int32_t nAtoms,
    const float *surfacePoints, int32_t nSurface,
    float *outESP
) {
    if (!atomPositions || !charges || !surfacePoints || !outESP) return;
    if (nAtoms <= 0 || nSurface <= 0) return;

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
            if (r < 0.1f) r = 0.1f;

            esp += KE * charges[a] / (4.0f * r * r);
        }

        outESP[s] = esp;
    }
}
