#include "druse_core_internal.h"

// Parallel batch helpers, fingerprints, fragments, and scaffold matching.

#include <tbb/parallel_for.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/BitOps.h>
#include <map>
#include <queue>
#include <set>

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
    for (int i = 0; i < count; ++i) results[i] = nullptr;

    tbb::parallel_for(0, (int)count, [&](int i) {
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

        std::vector<int> heavyIdx;
        for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
            if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) heavyIdx.push_back(i);
        }
        int nHeavy = (int)heavyIdx.size();
        if (nHeavy == 0) {
            std::strncpy(result->errorMessage, "No heavy atoms", 511);
            return result;
        }

        std::vector<int> origToHeavy(mol->getNumAtoms(), -1);
        for (int i = 0; i < nHeavy; i++) {
            origToHeavy[heavyIdx[i]] = i;
        }

        std::set<int> rotatableBondSet;
        for (auto bondIt = mol->beginBonds(); bondIt != mol->endBonds(); ++bondIt) {
            const Bond *bond = *bondIt;
            if (bond->getBondTypeAsDouble() != 1.0) continue;
            if (ringInfo.numBondRings(bond->getIdx()) > 0) continue;

            int a1 = bond->getBeginAtomIdx();
            int a2 = bond->getEndAtomIdx();
            if (mol->getAtomWithIdx(a1)->getAtomicNum() <= 1) continue;
            if (mol->getAtomWithIdx(a2)->getAtomicNum() <= 1) continue;

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

        std::vector<std::vector<int>> adj(nHeavy);
        struct RotBondInfo { int heavyA; int heavyB; int bondIdx; };
        std::vector<RotBondInfo> rotBonds;

        for (auto bondIt = mol->beginBonds(); bondIt != mol->endBonds(); ++bondIt) {
            const Bond *bond = *bondIt;
            int a1 = origToHeavy[bond->getBeginAtomIdx()];
            int a2 = origToHeavy[bond->getEndAtomIdx()];
            if (a1 < 0 || a2 < 0) continue;

            if (rotatableBondSet.count(bond->getIdx())) {
                rotBonds.push_back({a1, a2, (int)bond->getIdx()});
                continue;
            }
            adj[a1].push_back(a2);
            adj[a2].push_back(a1);
        }

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

        std::vector<int> fragSizes(numFrags, 0);
        for (int i = 0; i < nHeavy; i++) {
            fragSizes[fragMembership[i]]++;
        }

        int anchorIdx = 0;
        int maxSize = 0;
        for (int f = 0; f < numFrags; f++) {
            if (fragSizes[f] > maxSize) { maxSize = fragSizes[f]; anchorIdx = f; }
        }

        if (scaffoldSmarts && std::strlen(scaffoldSmarts) > 0) {
            std::unique_ptr<ROMol> pattern(SmartsToMol(scaffoldSmarts));
            if (pattern) {
                std::vector<MatchVectType> matches;
                SubstructMatch(*mol, *pattern, matches);
                if (!matches.empty()) {
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

        struct Connection { int parentFrag; int childFrag; int atomA; int atomB; };
        std::vector<Connection> connections;
        for (const auto &rb : rotBonds) {
            int fA = fragMembership[rb.heavyA];
            int fB = fragMembership[rb.heavyB];
            if (fA != fB) {
                connections.push_back({fA, fB, rb.heavyA, rb.heavyB});
            }
        }

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

DruseScaffoldMatch* druse_match_scaffold(const char *smiles, const char *scaffoldSmarts) {
    auto *result = new DruseScaffoldMatch();
    std::memset(result, 0, sizeof(DruseScaffoldMatch));
    result->hasMatch = false;
    result->tanimotoSimilarity = 0.0f;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return result;

        std::unique_ptr<ROMol> pattern(SmartsToMol(scaffoldSmarts));
        if (!pattern) return result;

        std::vector<MatchVectType> matches;
        SubstructMatch(*mol, *pattern, matches);

        if (!matches.empty()) {
            result->hasMatch = true;
            std::vector<int32_t> matchedHeavy;
            std::vector<int> heavyMap;
            for (unsigned i = 0; i < mol->getNumAtoms(); i++) {
                if (mol->getAtomWithIdx(i)->getAtomicNum() > 1) heavyMap.push_back(i);
            }
            std::map<int, int> origToHeavy;
            for (int i = 0; i < (int)heavyMap.size(); i++) {
                origToHeavy[heavyMap[i]] = i;
            }
            for (const auto &pair : matches[0]) {
                auto it = origToHeavy.find(pair.second);
                if (it != origToHeavy.end()) matchedHeavy.push_back(it->second);
            }
            result->matchCount = (int32_t)matchedHeavy.size();
            result->matchedAtomIndices = new int32_t[matchedHeavy.size()];
            std::memcpy(result->matchedAtomIndices, matchedHeavy.data(), matchedHeavy.size() * sizeof(int32_t));
        }

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
