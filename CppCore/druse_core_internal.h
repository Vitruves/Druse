// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

#ifndef DRUSE_CORE_INTERNAL_H
#define DRUSE_CORE_INTERNAL_H

#include "druse_core.h"

#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolStandardize/Tautomer.h>
#include <GraphMol/PartialCharges/GasteigerCharges.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Depictor/RDDepictor.h>
#include <GraphMol/MolChemicalFeatures/MolChemicalFeature.h>
#include <GraphMol/MolChemicalFeatures/MolChemicalFeatureFactory.h>
#include <GraphMol/MonomerInfo.h>
#include <RDGeneral/versions.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <tuple>
#include <vector>

using namespace RDKit;

struct IonizableGroupDef {
    const char *name;
    const char *smarts;
    double pKa;
    bool isAcid;
};

extern const IonizableGroupDef kIonizableGroups[];
extern const int kNumIonizableGroups;

const std::vector<std::unique_ptr<ROMol>>& getCompiledPatterns();
std::vector<std::tuple<int, int, bool, double>> detectIonSitesInternal(const ROMol &mol);
unsigned neighborEnvironmentFingerprint(const ROMol &mol, int atomIdx);

DruseMoleculeResult* make_error(const char *msg);
/// Heavy-atom pairs more than two bonds apart (beyond 1-3) closer than this are a
/// collapsed geometry: MMFF can converge such a conformer to a deceptively low energy
/// (atoms passed through each other / a ring was pierced), so it must never win on energy.
constexpr double kCollapsedHeavyAtomDistance = 1.6;

/// True when `conf` has non-finite coordinates or two heavy atoms separated by more than
/// two bonds lie closer than kCollapsedHeavyAtomDistance.
bool conformer_is_collapsed(const ROMol &mol, const Conformer &conf);

/// Re-embeds `mol` (ETKDGv3 random-coordinate starts, fresh seeds), MMFF-minimizes the
/// trial conformers and writes the lowest-energy non-collapsed one into conformer
/// `confId`. `energy` receives its MMFF energy (NaN without MMFF parameters). Returns
/// false, leaving `mol` untouched, when every attempt collapsed or failed to embed.
bool reembed_valid_conformer(RWMol &mol, int confId, double *energy = nullptr);

/// MMFF-minimizes conformer `confId`; if the minimum is collapsed, replaces it via
/// reembed_valid_conformer. Returns false when no valid geometry could be produced.
bool mmff_minimize_single(RWMol &mol, int confId = -1, double *energy = nullptr);

/// MMFF-minimizes every conformer and copies the lowest-energy non-collapsed one into
/// the first conformer (re-embedding when all collapsed). Returns false when no valid
/// geometry could be produced.
bool mmff_minimize_confs_pick_best(RWMol &mol);
int embed_molecule(RWMol &mol);
void populate_pdb_metadata(const RWMol &mol, const Atom &atom, DruseAtom &da);
std::vector<int> embed_multiple(RWMol &mol, int numConfs);
DruseMoleculeResult* mol_to_result(RWMol &mol, const char *name);
DruseMoleculeResult* mol_to_result_conf(RWMol &mol, const char *name, int confId);

void compute_donor_acceptor_flags(
    const ROMol &mol,
    std::vector<bool> &donorFlags,
    std::vector<bool> &acceptorFlags
);

const MolChemicalFeatureFactory *vina_feature_factory();

int32_t vina_xs_type_for_atom(
    const ROMol &mol,
    const Atom &atom,
    const std::vector<bool> &donorFlags,
    const std::vector<bool> &acceptorFlags
);

/// Energy of conformer `ci` from MMFFOptimizeMoleculeConfs results. Status 0
/// (converged) and 1 (iteration cap reached) both carry a valid geometry and
/// energy; status -1 (no MMFF parameters) and missing entries yield NaN.
inline double conformer_mmff_energy(const std::vector<std::pair<int, double>> &mmffResults, int ci) {
    if (ci >= 0 && ci < (int)mmffResults.size() && mmffResults[ci].first >= 0) {
        return mmffResults[ci].second;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/// Strict-weak-ordering energy comparator that sorts NaN energies last.
inline bool energy_less_nan_last(const std::pair<int, double> &a, const std::pair<int, double> &b) {
    if (std::isnan(a.second)) return false;
    if (std::isnan(b.second)) return true;
    return a.second < b.second;
}

#endif
