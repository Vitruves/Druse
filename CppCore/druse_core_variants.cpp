#include "druse_core_internal.h"

// Tautomer, protomer, and ensemble preparation.

#include <GraphMol/MolStandardize/Tautomer.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <limits>
#include <set>

/// Helper: generate 3D for a molecule and return MMFF energy, or NaN on failure.
static double embed_and_minimize(RWMol &mol) {
    MolOps::addHs(mol, false, true);
    int cid = embed_molecule(mol);
    if (cid < 0) {
        try { RDDepict::compute2DCoords(mol); } catch (...) {}
        return NAN;
    }
    std::vector<std::pair<int, double>> results;
    MMFF::MMFFOptimizeMoleculeConfs(mol, results);
    if (!results.empty() && results[0].first >= 0) {
        return results[0].second;
    }
    return NAN;
}

DruseVariantSet* druse_enumerate_tautomers(
    const char *smiles, const char *name,
    int32_t maxTautomers, double energyCutoff
) {
    auto *set = new DruseVariantSet();
    memset(set, 0, sizeof(DruseVariantSet));

    (void)energyCutoff;

    if (!smiles || !smiles[0]) return set;

    try {
        std::unique_ptr<ROMol> mol(SmilesToMol(smiles));
        if (!mol) return set;

        struct TautEntry {
            std::string smi;
            std::unique_ptr<RWMol> rwmol;
            double energy;
            int score;
        };
        std::vector<TautEntry> entries;
        std::set<std::string> seen;

        try {
            MolStandardize::TautomerEnumerator enumerator;
            enumerator.setMaxTautomers(std::max(maxTautomers * 2, (int32_t)4));
            enumerator.setMaxTransforms(500);
            auto result = enumerator.enumerate(*mol);

            for (const auto &taut : result) {
                std::string dedupSmi;
                try {
                    RWMol kekulized(*taut);
                    MolOps::Kekulize(kekulized);
                    dedupSmi = MolToSmiles(kekulized);
                } catch (...) {
                    dedupSmi = MolToSmiles(*taut);
                }
                if (seen.count(dedupSmi)) continue;
                seen.insert(dedupSmi);

                auto rw = std::make_unique<RWMol>(*taut);
                int score = MolStandardize::TautomerScoringFunctions::scoreTautomer(*rw);
                double energy = -score;

                std::string displaySmi = MolToSmiles(*taut);
                entries.push_back({displaySmi, std::move(rw), energy, score});
                if ((int)entries.size() >= maxTautomers) break;
            }
        } catch (const std::exception &e) {
            fprintf(stderr, "[druse] tautomer enumeration failed: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[druse] tautomer enumeration failed: unknown error\n");
        }

        if (entries.empty()) {
            auto rw = std::make_unique<RWMol>(*mol);
            int score = MolStandardize::TautomerScoringFunctions::scoreTautomer(*rw);
            entries.push_back({MolToSmiles(*mol), std::move(rw), (double)-score, score});
        }

        std::sort(entries.begin(), entries.end(), [](const TautEntry &a, const TautEntry &b) {
            return a.energy < b.energy;
        });

        int count = (int)entries.size();
        if (count == 0) return set;

        set->count = count;
        set->variants = new DruseMoleculeResult*[count];
        set->scores = new double[count];
        set->infos = new DruseVariantInfo[count];

        for (int i = 0; i < count; i++) {
            set->variants[i] = mol_to_result(*entries[i].rwmol, name);
            set->scores[i] = entries[i].energy;
            set->infos[i].kind = 0;
            snprintf(set->infos[i].label, sizeof(set->infos[i].label),
                     "Tautomer %d (score %d)", i + 1, entries[i].score);
        }
    } catch (...) {
    }

    return set;
}

DruseVariantSet* druse_enumerate_protomers(
    const char *smiles, const char *name,
    int32_t maxProtomers, double pH, double pkaThreshold
) {
    auto *set = new DruseVariantSet();
    memset(set, 0, sizeof(DruseVariantSet));

    if (!smiles || !smiles[0]) return set;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return set;

        struct IonizableSite {
            int groupIdx;
            int atomIdx;
            std::string label;
        };
        std::vector<IonizableSite> ambiguousSites;
        std::set<int> seenAtomIdx;

        const auto &patterns = getCompiledPatterns();
        for (int g = 0; g < kNumIonizableGroups; g++) {
            if (!patterns[g]) continue;

            std::vector<MatchVectType> matches;
            SubstructMatch(*mol, *patterns[g], matches);
            for (const auto &match : matches) {
                if (match.empty()) continue;
                int aidx = match[0].second;
                if (seenAtomIdx.count(aidx)) continue;

                double deltaPKa = std::abs(pH - kIonizableGroups[g].pKa);
                if (deltaPKa < pkaThreshold) {
                    ambiguousSites.push_back({g, aidx, kIonizableGroups[g].name});
                    seenAtomIdx.insert(aidx);
                }
            }
        }

        int nSites = std::min((int)ambiguousSites.size(), 10);
        int nCombinations = 1 << nSites;

        struct ComboPopulation { int combo; double population; };
        std::vector<ComboPopulation> combosByPop;
        combosByPop.reserve(nCombinations);
        for (int combo = 0; combo < nCombinations; combo++) {
            double pop = 1.0;
            for (int s = 0; s < nSites; s++) {
                bool toggle = (combo >> s) & 1;
                const auto &grp = kIonizableGroups[ambiguousSites[s].groupIdx];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - grp.pKa));
                double fracDeprot = 1.0 - fracProt;
                pop *= toggle ? (grp.isAcid ? fracDeprot : fracProt)
                              : (grp.isAcid ? fracProt : fracDeprot);
            }
            combosByPop.push_back({combo, pop});
        }
        std::sort(combosByPop.begin(), combosByPop.end(),
            [](const ComboPopulation &a, const ComboPopulation &b) {
                return a.population > b.population;
            });
        int nToTry = std::min((int)combosByPop.size(), (int)maxProtomers);

        struct ProtomerEntry {
            std::string smi;
            std::unique_ptr<RWMol> rwmol;
            double energy;
            std::string label;
        };
        std::vector<ProtomerEntry> entries;
        std::set<std::string> seen;

        for (int ci = 0; ci < nToTry; ci++) {
            int combo = combosByPop[ci].combo;
            auto rw = std::make_unique<RWMol>(*mol);

            std::string comboLabel;
            for (int s = 0; s < nSites; s++) {
                bool toggle = (combo >> s) & 1;
                if (!toggle) continue;

                const auto &site = ambiguousSites[s];
                const auto &grp = kIonizableGroups[site.groupIdx];
                Atom *atom = rw->getAtomWithIdx(site.atomIdx);

                if (grp.isAcid) {
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                } else {
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                }

                if (!comboLabel.empty()) comboLabel += ", ";
                comboLabel += std::string(grp.isAcid ? "deprot " : "prot ") + grp.name;
            }

            try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }

            std::string protSmi = MolToSmiles(*rw);
            if (seen.count(protSmi)) continue;
            seen.insert(protSmi);

            if (comboLabel.empty()) comboLabel = "Parent (as-drawn)";

            double energy = embed_and_minimize(*rw);
            entries.push_back({protSmi, std::move(rw), energy, comboLabel});
        }

        std::sort(entries.begin(), entries.end(), [](const ProtomerEntry &a, const ProtomerEntry &b) {
            if (std::isnan(a.energy)) return false;
            if (std::isnan(b.energy)) return true;
            return a.energy < b.energy;
        });

        int count = std::min((int)entries.size(), (int)maxProtomers);
        if (count == 0) return set;

        set->count = count;
        set->variants = new DruseMoleculeResult*[count];
        set->scores = new double[count];
        set->infos = new DruseVariantInfo[count];

        for (int i = 0; i < count; i++) {
            set->variants[i] = mol_to_result(*entries[i].rwmol, name);
            set->scores[i] = entries[i].energy;
            set->infos[i].kind = 1;
            snprintf(set->infos[i].label, sizeof(set->infos[i].label), "%s", entries[i].label.c_str());
        }
    } catch (...) {
    }

    return set;
}

void druse_free_variant_set(DruseVariantSet *set) {
    if (!set) return;
    if (set->variants) {
        for (int i = 0; i < set->count; i++) {
            druse_free_molecule_result(set->variants[i]);
        }
        delete[] set->variants;
    }
    delete[] set->scores;
    delete[] set->infos;
    delete set;
}

// ============================================================================
// Unified Ligand Ensemble Preparation
// ============================================================================
//
// Pipeline: SMILES → protomers(pH) → tautomers(each) → dedup → conformers(each)
//           → addH + MMFF minimize + Gasteiger charges → Boltzmann weights

DruseEnsembleResult* druse_prepare_ligand_ensemble(
    const char *smiles, const char *name,
    double pH, double pkaThreshold,
    int32_t maxTautomers, int32_t maxProtomers,
    double energyCutoff, int32_t conformersPerForm,
    double temperature
) {
    auto *result = new DruseEnsembleResult();
    memset(result, 0, sizeof(DruseEnsembleResult));
    result->numConformersPerForm = conformersPerForm;

    if (!smiles || !smiles[0]) {
        result->success = false;
        snprintf(result->errorMessage, 512, "Empty SMILES");
        return result;
    }

    try {
        std::unique_ptr<RWMol> parentMol(SmilesToMol(smiles));
        if (!parentMol) {
            result->success = false;
            snprintf(result->errorMessage, 512, "Invalid SMILES: %s", smiles);
            return result;
        }

        struct ChemicalForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            int kind;
            double hhPopulation = 1.0;
        };
        std::vector<ChemicalForm> allForms;
        std::set<std::string> seenSMILES;

        struct IonSite { int groupIdx; int atomIdx; bool isAcid; double groupPKa; };
        std::vector<IonSite> allSites;
        std::set<int> seenAtomIdx;
        const auto &cachedPats = getCompiledPatterns();
        for (int g = 0; g < kNumIonizableGroups; g++) {
            if (!cachedPats[g]) continue;
            std::vector<MatchVectType> matches;
            SubstructMatch(*parentMol, *cachedPats[g], matches);
            for (const auto &m : matches) {
                if (m.empty()) continue;
                int aIdx = m[0].second;

                if (!seenAtomIdx.count(aIdx)) {
                    seenAtomIdx.insert(aIdx);
                    allSites.push_back({g, aIdx, kIonizableGroups[g].isAcid, kIonizableGroups[g].pKa});
                }

                if (m.size() > 2) {
                    const Atom *primaryAtom = parentMol->getAtomWithIdx(aIdx);
                    const auto *ri = parentMol->getRingInfo();
                    if (ri && ri->numAtomRings(aIdx) > 0) {
                        int primaryAN = primaryAtom->getAtomicNum();
                        bool primaryArom = primaryAtom->getIsAromatic();
                        unsigned primaryDeg = primaryAtom->getDegree();
                        unsigned primaryH = primaryAtom->getTotalNumHs();
                        unsigned primaryFP = neighborEnvironmentFingerprint(*parentMol, aIdx);
                        for (size_t mi = 1; mi < m.size(); mi++) {
                            int otherIdx = m[mi].second;
                            if (seenAtomIdx.count(otherIdx)) continue;
                            const Atom *other = parentMol->getAtomWithIdx(otherIdx);
                            if (other->getAtomicNum() == primaryAN &&
                                ri->numAtomRings(otherIdx) > 0 &&
                                other->getIsAromatic() == primaryArom &&
                                other->getDegree() == primaryDeg &&
                                other->getTotalNumHs() == primaryH &&
                                neighborEnvironmentFingerprint(*parentMol, otherIdx) == primaryFP) {
                                seenAtomIdx.insert(otherIdx);
                                allSites.push_back({g, otherIdx, kIonizableGroups[g].isAcid,
                                                    kIonizableGroups[g].pKa});
                            }
                        }
                    }
                }

                if (kIonizableGroups[g].isAcid) {
                    const Atom *primary = parentMol->getAtomWithIdx(aIdx);
                    if (primary->getTotalNumHs() == 0) {
                        for (size_t mi = 1; mi < m.size(); mi++) {
                            int otherIdx = m[mi].second;
                            if (seenAtomIdx.count(otherIdx)) continue;
                            const Atom *other = parentMol->getAtomWithIdx(otherIdx);
                            if (other->getTotalNumHs() > 0 &&
                                (other->getAtomicNum() == 8 ||
                                 other->getAtomicNum() == 16 ||
                                 other->getAtomicNum() == 7)) {
                                seenAtomIdx.insert(otherIdx);
                                allSites.push_back({g, otherIdx, true, kIonizableGroups[g].pKa});
                                break;
                            }
                        }
                    }
                }
            }
        }

        {
            auto *ringInfo = parentMol->getRingInfo();
            auto distMatrix = MolOps::getDistanceMat(*parentMol);
            int nAtoms = parentMol->getNumAtoms();

            for (size_t i = 0; i < allSites.size(); i++) {
                if (allSites[i].isAcid) continue;
                for (size_t j = i + 1; j < allSites.size(); j++) {
                    if (allSites[j].isAcid) continue;

                    int ai = allSites[i].atomIdx;
                    int aj = allSites[j].atomIdx;
                    int topoDist = (int)distMatrix[ai * nAtoms + aj];
                    if (topoDist > 6) continue;

                    bool sameRing = false;
                    if (ringInfo) {
                        for (const auto &ring : ringInfo->atomRings()) {
                            bool hasI = false, hasJ = false;
                            for (int idx : ring) {
                                if (idx == ai) hasI = true;
                                if (idx == aj) hasJ = true;
                            }
                            if (hasI && hasJ) { sameRing = true; break; }
                        }
                    }

                    // Electrostatic pKa depression for nearby basic sites.
                    // Values calibrated against Caron & Bhatt 2022 polyamine data.
                    // Previous values (4.5/3.0/1.5/0.5) were too aggressive and
                    // eliminated legitimate protomers for drug-like molecules.
                    double depression;
                    if (sameRing) {
                        depression = 3.0;
                    } else if (topoDist <= 3) {
                        depression = 2.0;
                    } else if (topoDist <= 5) {
                        depression = 1.0;
                    } else {
                        depression = 0.3;
                    }

                    if (allSites[i].groupPKa >= allSites[j].groupPKa) {
                        allSites[j].groupPKa -= depression;
                    } else {
                        allSites[i].groupPKa -= depression;
                    }
                }
            }
        }

        auto washedMol = std::make_unique<RWMol>(*parentMol);
        for (const auto &site : allSites) {
            Atom *atom = washedMol->getAtomWithIdx(site.atomIdx);
            double deltaPH = pH - site.groupPKa;

            if (site.isAcid && deltaPH > pkaThreshold) {
                int nH = atom->getTotalNumHs();
                if (nH > 0) {
                    atom->setNumExplicitHs(nH - 1);
                    atom->setFormalCharge(atom->getFormalCharge() - 1);
                }
            } else if (!site.isAcid && deltaPH < -pkaThreshold) {
                int nH = atom->getTotalNumHs();
                atom->setNumExplicitHs(nH + 1);
                atom->setFormalCharge(atom->getFormalCharge() + 1);
            }
        }
        try { MolOps::sanitizeMol(*washedMol); } catch (...) {
            washedMol = std::make_unique<RWMol>(*parentMol);
        }

        std::vector<IonSite> ambSites;
        for (const auto &site : allSites) {
            double deltaPH = std::abs(pH - site.groupPKa);
            if (deltaPH <= pkaThreshold) {
                ambSites.push_back(site);
            }
        }

        // Diagnostic: log all detected sites and which passed the ambiguity filter
        if (allSites.size() > 0) {
            fprintf(stderr, "[druse] %s: %zu ionizable sites detected, %zu ambiguous (pH=%.1f, threshold=%.1f)\n",
                    name, allSites.size(), ambSites.size(), pH, pkaThreshold);
            for (const auto &site : allSites) {
                double deltaPH = std::abs(pH - site.groupPKa);
                fprintf(stderr, "  atom %d: %s pKa=%.1f %s %s\n",
                        site.atomIdx, kIonizableGroups[site.groupIdx].name,
                        site.groupPKa, site.isAcid ? "(acid)" : "(base)",
                        deltaPH <= pkaThreshold ? "** AMBIGUOUS **" : "");
            }
        }

        // Cap ambiguous sites — 2^N combinations grow exponentially
        // Use 16 to handle large molecules without combinatorial explosion
        int nAmbSites = std::min((int)ambSites.size(), 16);
        int nTotalCombos = 1 << nAmbSites;

        struct ComboPopulation { int combo; double population; };
        std::vector<ComboPopulation> combosByPop;
        combosByPop.reserve(nTotalCombos);
        for (int combo = 0; combo < nTotalCombos; combo++) {
            double pop = 1.0;
            for (int s = 0; s < nAmbSites; s++) {
                const auto &site = ambSites[s];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.groupPKa));
                double fracDeprot = 1.0 - fracProt;
                bool toggle = (combo >> s) & 1;
                pop *= !toggle ? (site.isAcid ? fracProt : fracDeprot)
                               : (site.isAcid ? fracDeprot : fracProt);
            }
            combosByPop.push_back({combo, pop});
        }
        std::sort(combosByPop.begin(), combosByPop.end(),
            [](const ComboPopulation &a, const ComboPopulation &b) {
                return a.population > b.population;
            });
        int nCombos = std::min((int)combosByPop.size(), (int)maxProtomers);

        struct ProtomerForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            double hhPopulation;
        };
        std::vector<ProtomerForm> protomers;

        for (int ci = 0; ci < nCombos; ci++) {
            int combo = combosByPop[ci].combo;
            auto rw = std::make_unique<RWMol>(*washedMol);
            std::string comboLabel;
            double hhPop = 1.0;

            for (int s = 0; s < nAmbSites; s++) {
                const auto &site = ambSites[s];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.groupPKa));
                double fracDeprot = 1.0 - fracProt;

                bool toggle = (combo >> s) & 1;
                if (!toggle) {
                    hhPop *= site.isAcid ? fracProt : fracDeprot;
                    continue;
                }

                Atom *atom = rw->getAtomWithIdx(site.atomIdx);
                if (site.isAcid) {
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                    hhPop *= fracDeprot;
                } else {
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                    hhPop *= fracProt;
                }
                if (!comboLabel.empty()) comboLabel += "+";
                comboLabel += std::string(site.isAcid ? "deprot_" : "prot_") +
                    kIonizableGroups[site.groupIdx].name;
            }

            try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
            std::string pSmi = MolToSmiles(*rw);
            if (seenSMILES.count(pSmi)) continue;
            seenSMILES.insert(pSmi);

            if (comboLabel.empty()) comboLabel = "Parent";
            protomers.push_back({pSmi, std::move(rw), comboLabel, hhPop});
        }

        if (protomers.empty()) {
            auto rw = std::make_unique<RWMol>(*washedMol);
            std::string wSmi = MolToSmiles(*rw);
            seenSMILES.insert(wSmi);
            protomers.push_back({wSmi, std::move(rw), "Parent", 1.0});
        }

        int maxTotalForms = maxProtomers * maxTautomers;
        if (maxTotalForms > 500) maxTotalForms = 500;

        for (auto &prot : protomers) {
            if ((int)allForms.size() >= maxTotalForms) break;

            if (!seenSMILES.count(prot.smi)) {
                seenSMILES.insert(prot.smi);
            }
            {
                auto rw = std::make_unique<RWMol>(*prot.mol);
                int kind = (prot.label != "Parent") ? 2 : 0;
                allForms.push_back({prot.smi, std::move(rw), prot.label, kind, prot.hhPopulation});
            }

            try {
                MolStandardize::TautomerEnumerator enumerator;
                // Request more tautomers than needed so filtering still leaves enough
                enumerator.setMaxTautomers(std::max(maxTautomers * 2, 4));
                enumerator.setMaxTransforms(500);
                auto tautResult = enumerator.enumerate(*prot.mol);

                int tautCount = 0;
                for (const auto &taut : tautResult) {
                    if (tautCount >= maxTautomers || (int)allForms.size() >= maxTotalForms) break;
                    auto rw = std::make_unique<RWMol>(*taut);
                    try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
                    std::string tSmi = MolToSmiles(*rw);

                    if (seenSMILES.count(tSmi)) continue;
                    seenSMILES.insert(tSmi);

                    bool isFromProtomer = (prot.label != "Parent");
                    int kind = isFromProtomer ? 3 : 1;

                    std::string label = prot.label;
                    if (label != "Parent") label += "_";
                    else label = "";
                    label += "Taut" + std::to_string(tautCount + 1);

                    allForms.push_back({tSmi, std::move(rw), label, kind, prot.hhPopulation});
                    tautCount++;
                }
            } catch (const std::exception &e) {
                // Log tautomer enumeration failure — don't swallow silently
                fprintf(stderr, "[druse] tautomer enumeration failed for %s: %s\n",
                        prot.label.c_str(), e.what());
            } catch (...) {
                fprintf(stderr, "[druse] tautomer enumeration failed for %s: unknown error\n",
                        prot.label.c_str());
            }
        }

        if (allForms.empty()) {
            result->success = false;
            snprintf(result->errorMessage, 512, "No valid chemical forms generated");
            return result;
        }

        result->numForms = (int)allForms.size();

        struct PreparedConformer {
            std::unique_ptr<RWMol> mol;
            double energy;
            int formIdx;
            int confIdx;
            std::string label;
            std::string smi;
            int kind;
            double hhPopulation;
        };
        std::vector<PreparedConformer> allConformers;

        for (int fi = 0; fi < (int)allForms.size(); fi++) {
            auto &form = allForms[fi];
            auto mol = std::make_unique<RWMol>(*form.mol);
            MolOps::addHs(*mol, false, true);

            DGeomHelpers::EmbedParameters embedParams = DGeomHelpers::ETKDGv3;
            embedParams.randomSeed = 42;
            embedParams.pruneRmsThresh = 0.25;
            embedParams.numThreads = 0;
            auto cids = DGeomHelpers::EmbedMultipleConfs(*mol, conformersPerForm, embedParams);
            if (cids.empty()) {
                int cid = DGeomHelpers::EmbedMolecule(*mol, embedParams);
                if (cid >= 0) cids.push_back(cid);
                else {
                    cid = embed_molecule(*mol);
                    if (cid >= 0) cids.push_back(cid);
                }
            }

            if (cids.empty()) {
                fprintf(stderr, "[druse] conformer generation failed for form %d (%s), skipping\n",
                        fi, form.smi.c_str());
                continue;
            }

            std::vector<std::pair<int, double>> mmffResults;
            try {
                MMFF::MMFFOptimizeMoleculeConfs(
                    *mol, mmffResults, /*numThreads=*/0, /*maxIters=*/1000, /*mmffVariant=*/"MMFF94"
                );
            } catch (...) {
            }

            try { computeGasteigerCharges(*mol); } catch (...) {}

            std::vector<std::pair<int, double>> indexed;
            for (int ci = 0; ci < (int)cids.size(); ci++) {
                double e = std::numeric_limits<double>::quiet_NaN();
                if (ci < (int)mmffResults.size()) {
                    if (mmffResults[ci].first == 0) {
                        e = mmffResults[ci].second;
                    } else {
                        continue;
                    }
                }

                if (std::isnan(e) || e > 1e6) continue;

                bool hasClash = false;
                const Conformer &conf = mol->getConformer(cids[ci]);
                int nAtoms = mol->getNumAtoms();
                for (int a = 0; a < nAtoms && !hasClash; a++) {
                    if (mol->getAtomWithIdx(a)->getAtomicNum() == 1) continue;
                    const auto &pa = conf.getAtomPos(a);
                    if (pa.x == 0.0 && pa.y == 0.0 && pa.z == 0.0) { hasClash = true; break; }
                    for (int b = a + 1; b < nAtoms && !hasClash; b++) {
                        if (mol->getAtomWithIdx(b)->getAtomicNum() == 1) continue;
                        if (mol->getBondBetweenAtoms(a, b)) continue;
                        const auto &pb = conf.getAtomPos(b);
                        double d2 = (pa.x-pb.x)*(pa.x-pb.x) + (pa.y-pb.y)*(pa.y-pb.y) + (pa.z-pb.z)*(pa.z-pb.z);
                        if (d2 < 1.0) hasClash = true;
                    }
                }
                if (hasClash) continue;

                indexed.push_back({ci, e});
            }

            if (indexed.empty() && !cids.empty()) {
                for (int ci = 0; ci < (int)cids.size(); ci++) {
                    if (ci < (int)mmffResults.size() && mmffResults[ci].first == 0) {
                        indexed.push_back({ci, mmffResults[ci].second});
                        break;
                    }
                }
            }

            std::sort(indexed.begin(), indexed.end(),
                      [](const auto &a, const auto &b) { return a.second < b.second; });

            for (int ri = 0; ri < (int)indexed.size(); ri++) {
                int confId = cids[indexed[ri].first];
                double energy = indexed[ri].second;

                auto confMol = std::make_unique<RWMol>(*mol);
                if (confMol->getNumConformers() > 1) {
                    std::vector<unsigned int> toRemove;
                    for (auto it = confMol->beginConformers(); it != confMol->endConformers(); ++it) {
                        if ((int)(*it)->getId() != confId) {
                            toRemove.push_back((*it)->getId());
                        }
                    }
                    for (auto cid : toRemove) {
                        confMol->removeConformer(cid);
                    }
                }

                std::string confLabel = form.label;
                if (indexed.size() > 1) {
                    confLabel += "_Conf" + std::to_string(ri + 1);
                }

                allConformers.push_back({
                    std::move(confMol), energy, fi, ri,
                    confLabel, form.smi, form.kind, form.hhPopulation
                });
            }
        }

        if (energyCutoff > 0 && !allConformers.empty()) {
            double bestE = 1e30;
            for (const auto &c : allConformers) {
                if (!std::isnan(c.energy) && c.energy < bestE) bestE = c.energy;
            }
            allConformers.erase(
                std::remove_if(allConformers.begin(), allConformers.end(),
                    [&](const PreparedConformer &c) {
                        return !std::isnan(c.energy) && c.energy > bestE + energyCutoff;
                    }),
                allConformers.end()
            );
        }

        double kBT = 0.001987204 * temperature;
        if (kBT < 1e-10) kBT = 0.593;

        double minE = 1e30;
        for (const auto &c : allConformers) {
            if (!std::isnan(c.energy) && c.energy < minE) minE = c.energy;
        }

        std::vector<double> boltzWeights(allConformers.size(), 0.0);
        double partitionZ = 0.0;
        for (size_t i = 0; i < allConformers.size(); i++) {
            double dE = std::isnan(allConformers[i].energy) ? 50.0 : (allConformers[i].energy - minE);
            double boltzFactor = std::exp(-dE / kBT);
            double hhFactor = allConformers[i].hhPopulation;
            boltzWeights[i] = hhFactor * boltzFactor;
            partitionZ += boltzWeights[i];
        }
        if (partitionZ > 0) {
            for (auto &w : boltzWeights) w /= partitionZ;
        }

        int count = (int)allConformers.size();
        result->count = count;
        result->members = new DruseEnsembleMember[count];
        result->success = true;

        for (int i = 0; i < count; i++) {
            auto &c = allConformers[i];
            auto &m = result->members[i];

            m.molecule = mol_to_result(*c.mol, name);
            m.mmffEnergy = c.energy;
            m.boltzmannWeight = boltzWeights[i];
            m.kind = c.kind;
            m.conformerIndex = c.confIdx;
            m.formIndex = c.formIdx;
            snprintf(m.label, sizeof(m.label), "%s", c.label.c_str());
            snprintf(m.smiles, sizeof(m.smiles), "%s", c.smi.c_str());
        }

    } catch (const std::exception &e) {
        result->success = false;
        snprintf(result->errorMessage, 512, "Ensemble preparation failed: %s", e.what());
    } catch (...) {
        result->success = false;
        snprintf(result->errorMessage, 512, "Ensemble preparation failed (unknown error)");
    }

    return result;
}

void druse_free_ensemble_result(DruseEnsembleResult *result) {
    if (!result) return;
    if (result->members) {
        for (int i = 0; i < result->count; i++) {
            druse_free_molecule_result(result->members[i].molecule);
        }
        delete[] result->members;
    }
    delete result;
}
