// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core_internal.h"

// Torsion tree, spatial indices, alignment, energies, and ESP.

#include <GraphMol/RingInfo.h>
#include <GraphMol/Substruct/SubstructMatch.h>
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

static bool is_amide_like_bond(const ROMol &mol, const Bond *bond) {
    if (!bond || bond->getBondType() != Bond::SINGLE) return false;

    const auto *a1 = bond->getBeginAtom();
    const auto *a2 = bond->getEndAtom();
    if (!a1 || !a2) return false;

    const Atom *carbon = nullptr;
    const Atom *nitrogen = nullptr;
    if (a1->getAtomicNum() == 6 && a2->getAtomicNum() == 7) {
        carbon = a1;
        nitrogen = a2;
    } else if (a1->getAtomicNum() == 7 && a2->getAtomicNum() == 6) {
        carbon = a2;
        nitrogen = a1;
    } else {
        return false;
    }

    (void)nitrogen;
    for (const auto *neighborBond : mol.atomBonds(carbon)) {
        if (neighborBond == bond) continue;
        if (neighborBond->getBondType() != Bond::DOUBLE) continue;
        const auto *neighbor = neighborBond->getOtherAtom(carbon);
        if (!neighbor) continue;
        const int atomicNum = neighbor->getAtomicNum();
        if (atomicNum == 8 || atomicNum == 16) {
            return true;
        }
    }
    return false;
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
        if (is_amide_like_bond(mol, bond)) continue;

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

// MARK: - Ring conformation variants

namespace {

/// A ring dihedral: four full-molecule atom indices, endocyclic or with one exocyclic
/// heavy substituent (which separates axial from equatorial placements).
struct RingQuad {
    int a, b, c, d;
    bool endocyclic;
};

constexpr double kRingPlanarDeg = 15.0;
constexpr double kRingDihedralToleranceDeg = 30.0;

double dihedral_deg(const float *xyz, const RingQuad &q) {
    auto p = [xyz](int i) { return Eigen::Vector3d(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]); };
    const Eigen::Vector3d b1 = p(q.b) - p(q.a), b2 = p(q.c) - p(q.b), b3 = p(q.d) - p(q.c);
    const Eigen::Vector3d n1 = b1.cross(b2), n2 = b2.cross(b3);
    const double x = n1.dot(n2);
    const double y = n1.cross(n2).dot(b2.normalized());
    return std::atan2(y, x) * 180.0 / M_PI;
}

int planarity_class(double phi) {
    if (std::fabs(phi) < kRingPlanarDeg) return 0;
    return phi > 0 ? 1 : -1;
}

/// True when conformers `x` and `y` have the same ring conformation under some automorphism.
bool same_ring_conformation(const float *x, const float *y, const std::vector<RingQuad> &quads,
                            const std::vector<std::vector<int>> &automorphisms) {
    for (const auto &perm : automorphisms) {
        bool match = true;
        for (const auto &q : quads) {
            const RingQuad mapped{perm[q.a], perm[q.b], perm[q.c], perm[q.d], q.endocyclic};
            const double px = dihedral_deg(x, q), py = dihedral_deg(y, mapped);
            double diff = std::fabs(px - py);
            if (diff > 180.0) diff = 360.0 - diff;
            if (diff > kRingDihedralToleranceDeg
                || (q.endocyclic && planarity_class(px) != planarity_class(py))) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

} // namespace

DruseRingVariantSet* druse_ring_conformation_variants(
    const char *molBlock, const float *positions, int32_t numConformers, int32_t numAtoms,
    int32_t maxVariants, int32_t addMirror
) {
    if (!molBlock || !molBlock[0] || !positions || numConformers <= 0 || numAtoms <= 0 || maxVariants <= 0) {
        return nullptr;
    }
    try {
        std::unique_ptr<RWMol> mol(MolBlockToMol(std::string(molBlock), true, false, false));
        if (!mol || (int)mol->getNumAtoms() != numAtoms) return nullptr;
        MolOps::findSSSR(*mol);

        std::vector<RingQuad> quads;
        for (const auto &ring : mol->getRingInfo()->atomRings()) {
            const int k = (int)ring.size();
            if (k < 5) continue;
            if (std::all_of(ring.begin(), ring.end(),
                            [&](int i) { return mol->getAtomWithIdx(i)->getIsAromatic(); })) continue;
            const std::unordered_set<int> inRing(ring.begin(), ring.end());
            for (int i = 0; i < k; i++) {
                const int r0 = ring[i], r1 = ring[(i + 1) % k], r2 = ring[(i + 2) % k];
                quads.push_back({r0, r1, r2, ring[(i + 3) % k], true});
                for (const auto *nbr : mol->atomNeighbors(mol->getAtomWithIdx(r0))) {
                    const int s = (int)nbr->getIdx();
                    if (nbr->getAtomicNum() != 1 && !inRing.count(s)) quads.push_back({s, r0, r1, r2, false});
                }
            }
        }

        const size_t stride = (size_t)numAtoms * 3;
        auto conformer = [&](int i) { return positions + stride * (size_t)i; };
        bool nonPlanar = false;
        for (int i = 0; i < numConformers && !nonPlanar; i++) {
            for (const auto &q : quads) {
                if (q.endocyclic && planarity_class(dihedral_deg(conformer(i), q)) != 0) { nonPlanar = true; break; }
            }
        }

        std::vector<int> sourceIndex;
        std::vector<int> mirrored;
        std::vector<std::vector<float>> coords;
        auto addVariant = [&](int source, bool mirror, std::vector<float> xyz) {
            sourceIndex.push_back(source);
            mirrored.push_back(mirror ? 1 : 0);
            coords.push_back(std::move(xyz));
        };

        if (!nonPlanar) {
            addVariant(0, false, std::vector<float>(conformer(0), conformer(0) + stride));
        } else {
            // Graph automorphisms on heavy atoms, lifted to full-molecule indices
            // (hydrogens never appear in a ring quad).
            std::vector<int> heavyToFull;
            for (const auto *atom : mol->atoms()) {
                if (atom->getAtomicNum() != 1) heavyToFull.push_back((int)atom->getIdx());
            }
            std::unique_ptr<ROMol> heavy(MolOps::removeAllHs(static_cast<const ROMol &>(*mol), false));
            std::vector<std::vector<int>> automorphisms;
            if (heavy && heavy->getNumAtoms() == heavyToFull.size()) {
                SubstructMatchParameters params;
                params.uniquify = false;
                params.maxMatches = 1000;
                for (const auto &match : SubstructMatch(*heavy, *heavy, params)) {
                    std::vector<int> perm(numAtoms);
                    for (int i = 0; i < numAtoms; i++) perm[i] = i;
                    for (const auto &[queryIdx, molIdx] : match) perm[heavyToFull[queryIdx]] = heavyToFull[molIdx];
                    automorphisms.push_back(std::move(perm));
                }
            }
            if (automorphisms.empty()) {
                std::vector<int> identity(numAtoms);
                for (int i = 0; i < numAtoms; i++) identity[i] = i;
                automorphisms.push_back(std::move(identity));
            }

            // The mirror image is the same molecule only without tetrahedral stereocentres.
            bool mirrorAllowed = addMirror != 0;
            if (mirrorAllowed) {
                auto &conf = mol->getConformer();
                for (int i = 0; i < numAtoms; i++) {
                    const float *p = conformer(0) + 3 * i;
                    conf.setAtomPos(i, RDGeom::Point3D(p[0], p[1], p[2]));
                }
                conf.set3D(true);
                MolOps::assignStereochemistryFrom3D(*mol);
                for (const auto *atom : mol->atoms()) {
                    const auto tag = atom->getChiralTag();
                    if (tag == Atom::CHI_TETRAHEDRAL_CW || tag == Atom::CHI_TETRAHEDRAL_CCW) {
                        mirrorAllowed = false;
                        break;
                    }
                }
            }

            auto isNew = [&](const std::vector<float> &xyz) {
                for (const auto &kept : coords) {
                    if (same_ring_conformation(kept.data(), xyz.data(), quads, automorphisms)) return false;
                }
                return true;
            };
            for (int i = 0; i < numConformers && (int)coords.size() < maxVariants; i++) {
                std::vector<float> xyz(conformer(i), conformer(i) + stride);
                if (!isNew(xyz)) continue;
                std::vector<float> mirror;
                if (mirrorAllowed) {
                    double cx = 0;
                    for (int a = 0; a < numAtoms; a++) cx += xyz[3 * a];
                    cx /= numAtoms;
                    mirror = xyz;
                    for (int a = 0; a < numAtoms; a++) mirror[3 * a] = (float)(2 * cx - xyz[3 * a]);
                }
                addVariant(i, false, std::move(xyz));
                if (!mirror.empty() && (int)coords.size() < maxVariants && isNew(mirror)) {
                    addVariant(i, true, std::move(mirror));
                }
            }
        }

        auto *set = new DruseRingVariantSet();
        set->count = (int32_t)coords.size();
        set->numAtoms = numAtoms;
        set->sourceIndex = new int32_t[coords.size()];
        set->mirrored = new int32_t[coords.size()];
        set->positions = new float[coords.size() * stride];
        for (size_t v = 0; v < coords.size(); v++) {
            set->sourceIndex[v] = sourceIndex[v];
            set->mirrored[v] = mirrored[v];
            std::copy(coords[v].begin(), coords[v].end(), set->positions + v * stride);
        }
        return set;
    } catch (...) {
        return nullptr;
    }
}

void druse_free_ring_variant_set(DruseRingVariantSet *set) {
    if (!set) return;
    delete[] set->sourceIndex;
    delete[] set->mirrored;
    delete[] set->positions;
    delete set;
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

        std::unique_ptr<ForceFields::ForceField> ff(MMFF::constructForceField(*mol, &mmffProps));
        if (!ff) return NAN;

        for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
            if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) {
                ff->fixedPoints().push_back(i);
            }
        }
        ff->minimize(200);

        return ff->calcEnergy();
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
        if (!mmff_minimize_single(*mol)) return NAN;

        MMFF::MMFFMolProperties mmffProps(*mol);
        if (!mmffProps.isValid()) return NAN;

        std::unique_ptr<ForceFields::ForceField> ff(MMFF::constructForceField(*mol, &mmffProps));
        if (!ff) return NAN;

        return ff->calcEnergy();
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
