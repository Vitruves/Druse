// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#include "druse_core.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gemmi/chemcomp.hpp>
#include <gemmi/metadata.hpp>
#include <gemmi/mmcif.hpp>
#include <gemmi/mmread.hpp>
#include <gemmi/neighbor.hpp>
#include <gemmi/polyheur.hpp>
#include <gemmi/read_cif.hpp>

namespace {

DruseMoleculeResult* make_structure_error(const char* msg) {
    auto* result = new DruseMoleculeResult();
    std::memset(result, 0, sizeof(DruseMoleculeResult));
    result->success = false;
    if (msg) {
        std::strncpy(result->errorMessage, msg, sizeof(result->errorMessage) - 1);
    }
    return result;
}

DruseResidueTopologyResult* make_topology_error(const char* msg) {
    auto* result = new DruseResidueTopologyResult();
    std::memset(result, 0, sizeof(DruseResidueTopologyResult));
    result->success = false;
    if (msg) {
        std::strncpy(result->errorMessage, msg, sizeof(result->errorMessage) - 1);
    }
    return result;
}

template <size_t N>
void copy_fixed(char (&dest)[N], const std::string& src) {
    std::strncpy(dest, src.c_str(), N - 1);
    dest[N - 1] = '\0';
}

gemmi::Structure read_structure_from_text(const char* content) {
    std::string buffer(content);
    return gemmi::read_structure_from_memory(
        buffer.data(),
        buffer.size(),
        "druse_structure",
        gemmi::CoorFormat::Detect
    );
}

int residue_sequence_number(const gemmi::Residue& residue) {
    return residue.seqid.num.has_value() ? *residue.seqid.num : 0;
}

int32_t bond_order_from_gemmi(gemmi::BondType type) {
    switch (type) {
        case gemmi::BondType::Double:
            return 2;
        case gemmi::BondType::Triple:
            return 3;
        case gemmi::BondType::Aromatic:
        case gemmi::BondType::Deloc:
            return 4;
        default:
            return 1;
    }
}

bool has_finite_position(const gemmi::Position& position) {
    return std::isfinite(position.x) && std::isfinite(position.y) && std::isfinite(position.z);
}

float distance_between(const gemmi::Position& a, const gemmi::Position& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return static_cast<float>(std::sqrt(dx * dx + dy * dy + dz * dz));
}

float angle_degrees(const gemmi::Position& a, const gemmi::Position& center, const gemmi::Position& b) {
    constexpr double pi = 3.14159265358979323846;
    const double ux = a.x - center.x;
    const double uy = a.y - center.y;
    const double uz = a.z - center.z;
    const double vx = b.x - center.x;
    const double vy = b.y - center.y;
    const double vz = b.z - center.z;

    const double ulen = std::sqrt(ux * ux + uy * uy + uz * uz);
    const double vlen = std::sqrt(vx * vx + vy * vy + vz * vz);
    if (ulen < 1e-6 || vlen < 1e-6) {
        return 0.0f;
    }

    const double cosine = std::clamp((ux * vx + uy * vy + uz * vz) / (ulen * vlen), -1.0, 1.0);
    return static_cast<float>(std::acos(cosine) * 180.0 / pi);
}

struct FlatAtomKey {
    int chainIndex;
    int residueIndex;
    int atomIndex;

    bool operator==(const FlatAtomKey& other) const = default;
};

struct FlatAtomKeyHash {
    size_t operator()(const FlatAtomKey& key) const {
        size_t value = static_cast<size_t>(key.chainIndex);
        value = value * 1315423911u ^ static_cast<size_t>(key.residueIndex);
        value = value * 2654435761u ^ static_cast<size_t>(key.atomIndex);
        return value;
    }
};

} // namespace

DruseMoleculeResult* druse_parse_structure(const char* content) {
    if (!content || !content[0]) {
        return make_structure_error("Empty structure content");
    }

    try {
        gemmi::Structure structure = read_structure_from_text(content);
        if (structure.models.empty()) {
            return make_structure_error("No coordinate model found");
        }

        gemmi::setup_entities(structure);
        gemmi::assign_het_flags(structure);

        gemmi::Model& model = structure.models.front();
        int32_t atomCount = 0;
        for (gemmi::Chain& chain : model.chains) {
            for (gemmi::Residue& residue : chain.residues) {
                atomCount += static_cast<int32_t>(residue.atoms.size());
            }
        }

        if (atomCount <= 0) {
            return make_structure_error("No atoms found in structure");
        }

        auto* result = new DruseMoleculeResult();
        std::memset(result, 0, sizeof(DruseMoleculeResult));
        result->success = true;
        copy_fixed(result->name, structure.name.empty() ? "structure" : structure.name);
        result->atomCount = atomCount;
        result->atoms = new DruseAtom[atomCount];

        int32_t atomCursor = 0;
        for (gemmi::Chain& chain : model.chains) {
            for (gemmi::Residue& residue : chain.residues) {
                for (gemmi::Atom& atom : residue.atoms) {
                    DruseAtom& out = result->atoms[atomCursor++];
                    std::memset(&out, 0, sizeof(DruseAtom));
                    out.x = static_cast<float>(atom.pos.x);
                    out.y = static_cast<float>(atom.pos.y);
                    out.z = static_cast<float>(atom.pos.z);
                    out.atomicNum = atom.element.atomic_number();
                    out.charge = 0.0f;
                    out.formalCharge = atom.charge;
                    copy_fixed(out.symbol, atom.element.name());
                    copy_fixed(out.name, atom.name);
                    copy_fixed(out.residueName, residue.name);
                    copy_fixed(out.chainID, chain.name.empty() ? "A" : chain.name);
                    out.residueSeq = residue_sequence_number(residue);
                    out.occupancy = atom.occ;
                    out.tempFactor = atom.b_iso;
                    if (atom.altloc != '\0') {
                        copy_fixed(out.altLoc, std::string(1, atom.altloc));
                    }
                    out.isHetAtom = residue.het_flag == 'H';
                }
            }
        }

        return result;
    } catch (const std::exception& error) {
        return make_structure_error(error.what());
    } catch (...) {
        return make_structure_error("Unknown error parsing structure with gemmi");
    }
}

DruseResidueTopologyResult* druse_parse_chemcomp_cif(const char* content) {
    if (!content || !content[0]) {
        return make_topology_error("Empty chemcomp CIF content");
    }

    try {
        gemmi::cif::Document document = gemmi::cif::read_string(content);
        if (document.blocks.empty()) {
            return make_topology_error("No CIF blocks found");
        }

        gemmi::ChemComp chemComp = gemmi::make_chemcomp_from_block(document.blocks.front());
        if (chemComp.name.empty()) {
            return make_topology_error("Failed to parse chemcomp block");
        }

        auto* result = new DruseResidueTopologyResult();
        std::memset(result, 0, sizeof(DruseResidueTopologyResult));
        result->success = true;
        copy_fixed(result->residueName, chemComp.name);

        const gemmi::cif::Block& block = document.blocks.front();
        std::unordered_map<std::string, gemmi::Position> atomPositions;
        if (block.has_any_value("_chem_comp_atom.pdbx_model_Cartn_x_ideal") ||
            block.has_any_value("_chem_comp_atom.model_Cartn_x") ||
            block.has_any_value("_chem_comp_atom.x")) {
            gemmi::ChemCompModel coordModel = gemmi::ChemCompModel::Xyz;
            if (block.has_any_value("_chem_comp_atom.pdbx_model_Cartn_x_ideal")) {
                coordModel = gemmi::ChemCompModel::Ideal;
            } else if (block.has_any_value("_chem_comp_atom.model_Cartn_x")) {
                coordModel = gemmi::ChemCompModel::Example;
            } else {
                coordModel = gemmi::ChemCompModel::Xyz;
            }

            gemmi::Residue coordResidue = gemmi::make_residue_from_chemcomp_block(block, coordModel);
            atomPositions.reserve(coordResidue.atoms.size());
            for (const gemmi::Atom& atom : coordResidue.atoms) {
                atomPositions.emplace(atom.name, atom.pos);
            }
        }

        result->atomCount = static_cast<int32_t>(chemComp.atoms.size());
        if (result->atomCount > 0) {
            result->atoms = new DruseResidueTopologyAtom[result->atomCount];
        }

        std::unordered_map<std::string, int32_t> atomIndices;
        atomIndices.reserve(chemComp.atoms.size());

        for (int32_t i = 0; i < result->atomCount; ++i) {
            const gemmi::ChemComp::Atom& atom = chemComp.atoms[i];
            DruseResidueTopologyAtom& out = result->atoms[i];
            std::memset(&out, 0, sizeof(DruseResidueTopologyAtom));
            copy_fixed(out.atomName, atom.id);
            out.atomicNum = atom.el.atomic_number();
            out.formalCharge = static_cast<int32_t>(std::lround(atom.charge));
            out.isHydrogen = atom.is_hydrogen();
            atomIndices.emplace(atom.id, i);
        }

        result->bondCount = static_cast<int32_t>(chemComp.rt.bonds.size());
        if (result->bondCount > 0) {
            result->bonds = new DruseResidueTopologyBond[result->bondCount];
        }

        std::vector<std::vector<int32_t>> adjacency(chemComp.atoms.size());
        for (int32_t i = 0; i < result->bondCount; ++i) {
            const gemmi::Restraints::Bond& bond = chemComp.rt.bonds[i];
            DruseResidueTopologyBond& out = result->bonds[i];
            std::memset(&out, 0, sizeof(DruseResidueTopologyBond));
            copy_fixed(out.atom1, bond.id1.atom);
            copy_fixed(out.atom2, bond.id2.atom);
            out.order = bond_order_from_gemmi(bond.type);
            if (std::isfinite(bond.value)) {
                out.idealLength = static_cast<float>(bond.value);
            } else if (std::isfinite(bond.value_nucleus)) {
                out.idealLength = static_cast<float>(bond.value_nucleus);
            }

            const auto atom1 = atomIndices.find(bond.id1.atom);
            const auto atom2 = atomIndices.find(bond.id2.atom);
            if (atom1 != atomIndices.end() && atom2 != atomIndices.end()) {
                adjacency[atom1->second].push_back(atom2->second);
                adjacency[atom2->second].push_back(atom1->second);
            }
            if (out.idealLength <= 0.0f &&
                atom1 != atomIndices.end() && atom2 != atomIndices.end()) {
                auto pos1It = atomPositions.find(bond.id1.atom);
                auto pos2It = atomPositions.find(bond.id2.atom);
                const gemmi::Position& pos1 = pos1It != atomPositions.end() ? pos1It->second : chemComp.atoms[atom1->second].xyz;
                const gemmi::Position& pos2 = pos2It != atomPositions.end() ? pos2It->second : chemComp.atoms[atom2->second].xyz;
                if (has_finite_position(pos1) && has_finite_position(pos2)) {
                    out.idealLength = distance_between(pos1, pos2);
                }
            }
        }

        std::vector<DruseResidueTopologyAngle> angleBuffer;
        angleBuffer.reserve(chemComp.rt.angles.size());
        for (const gemmi::Restraints::Angle& restraint : chemComp.rt.angles) {
            DruseResidueTopologyAngle angle{};
            copy_fixed(angle.atom1, restraint.id1.atom);
            copy_fixed(angle.atom2, restraint.id2.atom);
            copy_fixed(angle.atom3, restraint.id3.atom);
            if (std::isfinite(restraint.value)) {
                angle.idealAngleDegrees = static_cast<float>(restraint.value);
            }
            angleBuffer.push_back(angle);
        }

        if (angleBuffer.empty()) {
            for (size_t centerIndex = 0; centerIndex < adjacency.size(); ++centerIndex) {
                const auto& neighbors = adjacency[centerIndex];
                if (neighbors.size() < 2) {
                    continue;
                }

                const std::string& centerName = chemComp.atoms[centerIndex].id;
                auto centerIt = atomPositions.find(centerName);
                const gemmi::Position& center = centerIt != atomPositions.end()
                    ? centerIt->second
                    : chemComp.atoms[centerIndex].xyz;
                if (!has_finite_position(center)) {
                    continue;
                }

                for (size_t i = 0; i < neighbors.size(); ++i) {
                    for (size_t j = i + 1; j < neighbors.size(); ++j) {
                        const std::string& atom1Name = chemComp.atoms[neighbors[i]].id;
                        const std::string& atom3Name = chemComp.atoms[neighbors[j]].id;
                        auto pos1It = atomPositions.find(atom1Name);
                        auto pos3It = atomPositions.find(atom3Name);
                        const gemmi::Position& pos1 = pos1It != atomPositions.end()
                            ? pos1It->second
                            : chemComp.atoms[neighbors[i]].xyz;
                        const gemmi::Position& pos3 = pos3It != atomPositions.end()
                            ? pos3It->second
                            : chemComp.atoms[neighbors[j]].xyz;
                        if (!has_finite_position(pos1) || !has_finite_position(pos3)) {
                            continue;
                        }

                        DruseResidueTopologyAngle angle{};
                        copy_fixed(angle.atom1, atom1Name);
                        copy_fixed(angle.atom2, centerName);
                        copy_fixed(angle.atom3, atom3Name);
                        angle.idealAngleDegrees = angle_degrees(pos1, center, pos3);
                        angleBuffer.push_back(angle);
                    }
                }
            }
        }

        result->angleCount = static_cast<int32_t>(angleBuffer.size());
        if (result->angleCount > 0) {
            result->angles = new DruseResidueTopologyAngle[result->angleCount];
            std::memcpy(
                result->angles,
                angleBuffer.data(),
                angleBuffer.size() * sizeof(DruseResidueTopologyAngle)
            );
        }

        return result;
    } catch (const std::exception& error) {
        return make_topology_error(error.what());
    } catch (...) {
        return make_topology_error("Unknown error parsing chemcomp CIF with gemmi");
    }
}

void druse_free_residue_topology_result(DruseResidueTopologyResult* result) {
    if (!result) {
        return;
    }
    delete[] result->atoms;
    delete[] result->bonds;
    delete[] result->angles;
    delete result;
}

int32_t druse_find_structure_neighbors(
    const char* content,
    const float* queryPoint,
    float radius,
    bool includeHydrogens,
    int32_t* outIndices,
    int32_t maxResults
) {
    if (!content || !content[0] || !queryPoint || !outIndices || maxResults <= 0 || radius <= 0.0f) {
        return -1;
    }

    try {
        gemmi::Structure structure = read_structure_from_text(content);
        if (structure.models.empty()) {
            return -1;
        }

        gemmi::Model& model = structure.models.front();
        std::unordered_map<FlatAtomKey, int32_t, FlatAtomKeyHash> flatIndexByKey;

        int32_t flatIndex = 0;
        for (int chainIndex = 0; chainIndex < static_cast<int>(model.chains.size()); ++chainIndex) {
            gemmi::Chain& chain = model.chains[chainIndex];
            for (int residueIndex = 0; residueIndex < static_cast<int>(chain.residues.size()); ++residueIndex) {
                gemmi::Residue& residue = chain.residues[residueIndex];
                for (int atomIndex = 0; atomIndex < static_cast<int>(residue.atoms.size()); ++atomIndex) {
                    flatIndexByKey.emplace(FlatAtomKey{chainIndex, residueIndex, atomIndex}, flatIndex++);
                }
            }
        }

        gemmi::NeighborSearch neighborSearch(model, structure.cell, radius);
        neighborSearch.populate(includeHydrogens);

        const gemmi::Position query{
            static_cast<double>(queryPoint[0]),
            static_cast<double>(queryPoint[1]),
            static_cast<double>(queryPoint[2])
        };

        auto marks = neighborSearch.find_atoms(query, '\0', 0.0, radius);
        struct Hit {
            int32_t index;
            double distanceSquared;
        };
        std::vector<Hit> hits;
        std::unordered_set<int32_t> seen;
        hits.reserve(marks.size());

        for (const gemmi::NeighborSearch::Mark* mark : marks) {
            const auto iterator = flatIndexByKey.find(FlatAtomKey{
                mark->chain_idx,
                mark->residue_idx,
                mark->atom_idx
            });
            if (iterator == flatIndexByKey.end()) {
                continue;
            }
            if (!seen.insert(iterator->second).second) {
                continue;
            }
            hits.push_back(Hit{iterator->second, mark->pos.dist_sq(query)});
        }

        std::sort(hits.begin(), hits.end(), [](const Hit& lhs, const Hit& rhs) {
            return lhs.distanceSquared < rhs.distanceSquared;
        });

        const int32_t count = std::min<int32_t>(maxResults, static_cast<int32_t>(hits.size()));
        for (int32_t i = 0; i < count; ++i) {
            outIndices[i] = hits[i].index;
        }
        return count;
    } catch (...) {
        return -1;
    }
}

// ============================================================================
// Entity Sequence Extraction (SEQRES / entity_poly_seq)
// ============================================================================

DruseEntitySequenceResult* druse_get_entity_sequences(const char* content) {
    auto* result = new DruseEntitySequenceResult();
    std::memset(result, 0, sizeof(DruseEntitySequenceResult));

    if (!content || !content[0]) {
        result->success = false;
        std::strncpy(result->errorMessage, "Empty structure content", sizeof(result->errorMessage) - 1);
        return result;
    }

    try {
        gemmi::Structure structure = read_structure_from_text(content);
        if (structure.models.empty()) {
            result->success = false;
            std::strncpy(result->errorMessage, "No coordinate model found", sizeof(result->errorMessage) - 1);
            return result;
        }

        gemmi::setup_entities(structure);

        // Collect per-chain sequences: map author chain ID → sequence of 3-letter codes
        struct ChainSeq {
            std::string chainID;
            std::vector<std::string> residues;
        };
        std::vector<ChainSeq> chainSeqs;

        gemmi::Model& model = structure.models.front();
        for (const gemmi::Chain& chain : model.chains) {
            // Get the polymer subchain for this chain
            auto polymer = chain.get_polymer();
            if (polymer.size() == 0) continue;

            // Find the entity for this polymer's subchain
            std::string subchain = polymer.subchain_id();
            const gemmi::Entity* entity = gemmi::find_entity_of_subchain(subchain, structure.entities);
            if (!entity || entity->full_sequence.empty()) continue;
            if (entity->entity_type != gemmi::EntityType::Polymer) continue;

            ChainSeq cs;
            cs.chainID = chain.name.empty() ? "A" : chain.name;
            cs.residues.reserve(entity->full_sequence.size());
            for (const std::string& mon : entity->full_sequence) {
                cs.residues.push_back(gemmi::Entity::first_mon(mon));
            }
            chainSeqs.push_back(std::move(cs));
        }

        if (chainSeqs.empty()) {
            result->success = true;
            result->chainCount = 0;
            result->chains = nullptr;
            return result;
        }

        result->chainCount = static_cast<int32_t>(chainSeqs.size());
        result->chains = new DruseChainSequence[result->chainCount];

        for (int32_t i = 0; i < result->chainCount; ++i) {
            const auto& cs = chainSeqs[i];
            DruseChainSequence& out = result->chains[i];
            std::memset(&out, 0, sizeof(DruseChainSequence));
            copy_fixed(out.chainID, cs.chainID);
            out.residueCount = static_cast<int32_t>(cs.residues.size());

            if (out.residueCount > 0) {
                out.residueNames = new char[out.residueCount][8];
                for (int32_t j = 0; j < out.residueCount; ++j) {
                    std::memset(out.residueNames[j], 0, 8);
                    std::strncpy(out.residueNames[j], cs.residues[j].c_str(), 7);
                }
            }
        }

        result->success = true;
        return result;

    } catch (const std::exception& error) {
        result->success = false;
        std::strncpy(result->errorMessage, error.what(), sizeof(result->errorMessage) - 1);
        return result;
    } catch (...) {
        result->success = false;
        std::strncpy(result->errorMessage, "Unknown error extracting entity sequences", sizeof(result->errorMessage) - 1);
        return result;
    }
}

void druse_free_entity_sequence_result(DruseEntitySequenceResult* result) {
    if (!result) return;
    if (result->chains) {
        for (int32_t i = 0; i < result->chainCount; ++i) {
            delete[] result->chains[i].residueNames;
        }
        delete[] result->chains;
    }
    delete result;
}
