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

#include <cstdint>
#include <memory>
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
void mmff_minimize_single(RWMol &mol, int confId = -1);
void mmff_minimize_confs_pick_best(RWMol &mol);
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

#endif
