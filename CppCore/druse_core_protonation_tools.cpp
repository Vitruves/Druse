#include "druse_core_internal.h"

// Ionizable-site inspection, per-site protomer generation, and ensemble v2.

#include <limits>
#include <map>
#include <set>

DruseIonSiteResult* druse_detect_ionizable_sites(const char *smiles) {
    auto *result = new DruseIonSiteResult();
    result->sites = nullptr;
    result->count = 0;

    if (!smiles || !smiles[0]) return result;

    try {
        std::unique_ptr<RWMol> mol(SmilesToMol(smiles));
        if (!mol) return result;

        auto sites = detectIonSitesInternal(*mol);
        if (sites.empty()) return result;

        result->count = (int32_t)sites.size();
        result->sites = new DruseIonSite[sites.size()];

        for (size_t i = 0; i < sites.size(); i++) {
            auto &[atomIdx, groupIdx, isAcid, defaultPKa] = sites[i];
            auto &out = result->sites[i];
            out.atomIdx = atomIdx;
            out.isAcid = isAcid;
            out.defaultPKa = defaultPKa;
            snprintf(out.groupName, sizeof(out.groupName), "%s", kIonizableGroups[groupIdx].name);
        }
    } catch (...) {
    }

    return result;
}

void druse_free_ion_sites(DruseIonSiteResult *result) {
    if (!result) return;
    delete[] result->sites;
    delete result;
}

DruseSiteProtomerPair* druse_generate_site_protomers(
    const char *smiles, int32_t atomIdx, bool isAcid
) {
    auto *result = new DruseSiteProtomerPair();
    memset(result, 0, sizeof(DruseSiteProtomerPair));

    if (!smiles || !smiles[0]) {
        result->success = false;
        snprintf(result->errorMessage, sizeof(result->errorMessage), "Empty SMILES");
        return result;
    }

    try {
        std::unique_ptr<RWMol> parentMol(SmilesToMol(smiles));
        if (!parentMol) {
            result->success = false;
            snprintf(result->errorMessage, sizeof(result->errorMessage), "Invalid SMILES");
            return result;
        }

        int parentCharge = 0;
        for (auto atom : parentMol->atoms()) parentCharge += atom->getFormalCharge();

        auto protMol = std::make_unique<RWMol>(*parentMol);
        auto deprotMol = std::make_unique<RWMol>(*parentMol);

        if (isAcid) {
            Atom *depAtom = deprotMol->getAtomWithIdx(atomIdx);
            int nH = depAtom->getTotalNumHs();
            if (nH > 0) {
                depAtom->setNumExplicitHs(nH - 1);
                depAtom->setFormalCharge(depAtom->getFormalCharge() - 1);
            }
            result->protonatedCharge = parentCharge;
            result->deprotonatedCharge = parentCharge - 1;
        } else {
            Atom *protAtom = protMol->getAtomWithIdx(atomIdx);
            int nH = protAtom->getTotalNumHs();
            protAtom->setNumExplicitHs(nH + 1);
            protAtom->setFormalCharge(protAtom->getFormalCharge() + 1);
            result->protonatedCharge = parentCharge + 1;
            result->deprotonatedCharge = parentCharge;
        }

        try { MolOps::sanitizeMol(*protMol); } catch (...) {}
        try { MolOps::sanitizeMol(*deprotMol); } catch (...) {}

        auto prepare3D = [](RWMol &mol) {
            MolOps::addHs(mol, false, true);
            int cid = embed_molecule(mol);
            if (cid < 0) {
                try { RDDepict::compute2DCoords(mol); } catch (...) {}
            }
            std::vector<std::pair<int,double>> mmffRes;
            try { MMFF::MMFFOptimizeMoleculeConfs(mol, mmffRes, 0, 500); } catch (...) {}
            try { computeGasteigerCharges(mol); } catch (...) {}
        };

        prepare3D(*protMol);
        prepare3D(*deprotMol);

        result->protonated = mol_to_result(*protMol, "protonated");
        result->deprotonated = mol_to_result(*deprotMol, "deprotonated");
        result->success = true;

    } catch (const std::exception &e) {
        result->success = false;
        snprintf(result->errorMessage, sizeof(result->errorMessage), "%s", e.what());
    } catch (...) {
        result->success = false;
        snprintf(result->errorMessage, sizeof(result->errorMessage), "Unknown error");
    }

    return result;
}

void druse_free_site_protomer_pair(DruseSiteProtomerPair *result) {
    if (!result) return;
    druse_free_molecule_result(result->protonated);
    druse_free_molecule_result(result->deprotonated);
    delete result;
}

DruseEnsembleResult* druse_prepare_ligand_ensemble_ex(
    const char *smiles, const char *name,
    double pH, double pkaThreshold,
    int32_t maxTautomers, int32_t maxProtomers,
    double energyCutoff, int32_t conformersPerForm,
    double temperature,
    const DruseIonSiteDef *sites, int32_t nSites
) {
    // No explicit sites → fall back to SMARTS detection
    if (!sites || nSites <= 0) {
        return druse_prepare_ligand_ensemble(smiles, name, pH, pkaThreshold,
            maxTautomers, maxProtomers, energyCutoff, conformersPerForm, temperature);
    }

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

        // GNN-provided sites — cross-validate against SMARTS lookup table
        // to catch systematic mispredictions (e.g. aminopyridazine vs piperazine).
        // No proximity depression is applied (GNN already accounts for environment).
        auto smartsSites = detectIonSitesInternal(*parentMol);
        struct SmartsHit { double pKa; bool isAcid; };
        std::map<int, SmartsHit> smartsLookup;
        for (const auto &[aIdx, gIdx, isA, pk] : smartsSites) {
            smartsLookup[aIdx] = {pk, isA};
        }

        // Build corrected site list: use GNN pKa but clamp against SMARTS
        // when they disagree by more than 3 pKa units.
        struct CorrectedSite { int atomIdx; bool isAcid; double pKa; };
        std::vector<CorrectedSite> correctedSites;
        std::set<int> gnnAtomIdx;
        for (int32_t i = 0; i < nSites; i++) {
            int atomIdx = sites[i].atomIdx;
            if (atomIdx < 0 || atomIdx >= (int)parentMol->getNumAtoms()) continue;
            double gnnPKa = sites[i].pKa;
            bool isAcid = sites[i].isAcid;

            auto it = smartsLookup.find(atomIdx);
            if (it != smartsLookup.end()) {
                double smartsPk = it->second.pKa;
                bool smartsIsAcid = it->second.isAcid;
                double diff = gnnPKa - smartsPk;
                if (smartsIsAcid != isAcid) {
                    // GNN and SMARTS disagree on acid/base classification —
                    // SMARTS is chemically grounded, override.
                    fprintf(stderr, "[druse] pKa class override: atom %d GNN=%s/%.1f → SMARTS=%s/%.1f\n",
                            atomIdx, isAcid ? "acid" : "base", gnnPKa,
                            smartsIsAcid ? "acid" : "base", smartsPk);
                    gnnPKa = smartsPk;
                    isAcid = smartsIsAcid;
                } else if (std::abs(diff) > 1.5) {
                    // Strong disagreement: SMARTS is the curated reference for
                    // drug-like functional groups (piperazines, anilines,
                    // carboxylic acids, etc). GNN underprediction on common
                    // motifs would otherwise drop legitimate ionizable sites
                    // from enumeration. Snap to SMARTS for these cases.
                    fprintf(stderr, "[druse] pKa override: atom %d GNN=%.1f → SMARTS=%.1f\n",
                            atomIdx, gnnPKa, smartsPk);
                    gnnPKa = smartsPk;
                }
            }
            correctedSites.push_back({atomIdx, isAcid, gnnPKa});
            gnnAtomIdx.insert(atomIdx);
        }

        // Backfill: GNN's ionizable-probability threshold can drop legitimate
        // basic sites (e.g. tertiary piperazine N next to an aminopyridazine
        // distractor). For any SMARTS hit GNN did not predict, add it at its
        // tabulated pKa so it still participates in protomer enumeration.
        for (const auto &[aIdx, gIdx, isA, pk] : smartsSites) {
            if (gnnAtomIdx.count(aIdx)) continue;
            correctedSites.push_back({aIdx, isA, pk});
            fprintf(stderr, "[druse] SMARTS backfill: atom %d pKa=%.1f %s (missed by GNN)\n",
                    aIdx, pk, isA ? "(acid)" : "(base)");
        }

        struct AmbSite { int atomIdx; bool isAcid; double pKa; };
        std::vector<AmbSite> ambSites;
        for (size_t i = 0; i < correctedSites.size(); i++) {
            double deltaPH = pH - correctedSites[i].pKa;
            if (std::abs(deltaPH) <= pkaThreshold) {
                ambSites.push_back({correctedSites[i].atomIdx,
                                    correctedSites[i].isAcid,
                                    correctedSites[i].pKa});
            }
        }

        // Helper: apply "clearly protonated/deprotonated" sites to a molecule.
        // Skips atoms whose H count differs from the parent — those have been
        // moved by tautomerization and the protonation rule no longer applies.
        auto applyClearProtonation = [&](RWMol &mol) {
            for (const auto &cs : correctedSites) {
                double deltaPH = pH - cs.pKa;
                Atom *parentAtom = parentMol->getAtomWithIdx(cs.atomIdx);
                Atom *atom = mol.getAtomWithIdx(cs.atomIdx);
                if (parentAtom->getTotalNumHs() != atom->getTotalNumHs()) continue;
                if (cs.isAcid && deltaPH > pkaThreshold) {
                    int nH = atom->getTotalNumHs();
                    if (nH > 0) {
                        atom->setNumExplicitHs(nH - 1);
                        atom->setFormalCharge(atom->getFormalCharge() - 1);
                    }
                } else if (!cs.isAcid && deltaPH < -pkaThreshold) {
                    int nH = atom->getTotalNumHs();
                    atom->setNumExplicitHs(nH + 1);
                    atom->setFormalCharge(atom->getFormalCharge() + 1);
                }
            }
        };

        // === Tautomer-first enumeration ===
        // Run tautomer enumeration on the *unprotonated* parent so that ring
        // NH ↔ ring N=C rearrangements (e.g. aminopyridazine ↔ pyridazinone)
        // aren't blocked by adjacent charged sites. Atom indices are preserved
        // by RDKit's TautomerEnumerator transformations, so the same site list
        // can be applied to each tautomer.
        std::vector<std::unique_ptr<RWMol>> baseTautomers;
        {
            auto rw = std::make_unique<RWMol>(*parentMol);
            baseTautomers.push_back(std::move(rw));
        }
        std::set<std::string> tautSeen;
        tautSeen.insert(MolToSmiles(*parentMol));
        try {
            MolStandardize::TautomerEnumerator tautEnum;
            tautEnum.setMaxTautomers(std::max(maxTautomers, 2));
            tautEnum.setMaxTransforms(200);
            auto tres = tautEnum.enumerate(*parentMol);
            for (const auto &t : tres) {
                if ((int)baseTautomers.size() >= maxTautomers) break;
                auto rw = std::make_unique<RWMol>(*t);
                try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
                std::string tSmi = MolToSmiles(*rw);
                if (tautSeen.count(tSmi)) continue;
                tautSeen.insert(tSmi);
                baseTautomers.push_back(std::move(rw));
            }
        } catch (...) {}

        struct ChemicalForm {
            std::string smi;
            std::unique_ptr<RWMol> mol;
            std::string label;
            int kind;
            double hhPopulation = 1.0;
        };
        std::vector<ChemicalForm> allForms;
        std::set<std::string> seenSMILES;

        // Combo enumeration is shared across tautomers — same ambiguous sites,
        // same Henderson-Hasselbalch populations.
        int nAmb = std::min((int)ambSites.size(), 10);
        int nTotalCombos = 1 << nAmb;

        struct ComboPop { int combo; double population; };
        std::vector<ComboPop> combosByPop;
        combosByPop.reserve(nTotalCombos);
        for (int combo = 0; combo < nTotalCombos; combo++) {
            double pop = 1.0;
            for (int s = 0; s < nAmb; s++) {
                const auto &site = ambSites[s];
                double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.pKa));
                double fracDeprot = 1.0 - fracProt;
                bool toggle = (combo >> s) & 1;
                pop *= !toggle ? (site.isAcid ? fracProt : fracDeprot)
                               : (site.isAcid ? fracDeprot : fracProt);
            }
            combosByPop.push_back({combo, pop});
        }
        std::sort(combosByPop.begin(), combosByPop.end(),
            [](const ComboPop &a, const ComboPop &b) { return a.population > b.population; });
        int nCombos = std::min((int)combosByPop.size(), (int)maxProtomers);

        int maxTotalForms = maxProtomers * maxTautomers;
        if (maxTotalForms > 200) maxTotalForms = 200;

        // For each tautomer × each protomer combo, generate one form
        for (size_t ti = 0; ti < baseTautomers.size(); ti++) {
            if ((int)allForms.size() >= maxTotalForms) break;

            auto washedTaut = std::make_unique<RWMol>(*baseTautomers[ti]);
            applyClearProtonation(*washedTaut);
            try { MolOps::sanitizeMol(*washedTaut); } catch (...) {
                washedTaut = std::make_unique<RWMol>(*baseTautomers[ti]);
            }

            std::string tautPrefix = (ti == 0) ? "" : ("Taut" + std::to_string(ti) + "_");

            for (int ci = 0; ci < nCombos; ci++) {
                if ((int)allForms.size() >= maxTotalForms) break;

                int combo = combosByPop[ci].combo;
                auto rw = std::make_unique<RWMol>(*washedTaut);
                std::string comboLabel;
                double hhPop = 1.0;

                for (int s = 0; s < nAmb; s++) {
                    const auto &site = ambSites[s];
                    double fracProt = 1.0 / (1.0 + std::pow(10.0, pH - site.pKa));
                    double fracDeprot = 1.0 - fracProt;

                    bool toggle = (combo >> s) & 1;
                    if (!toggle) {
                        hhPop *= site.isAcid ? fracProt : fracDeprot;
                        continue;
                    }

                    Atom *atom = rw->getAtomWithIdx(site.atomIdx);
                    // Skip if tautomerization moved the H away from this atom
                    Atom *parentAtom = parentMol->getAtomWithIdx(site.atomIdx);
                    if (parentAtom->getTotalNumHs() != atom->getTotalNumHs()) {
                        hhPop *= site.isAcid ? fracProt : fracDeprot;
                        continue;
                    }
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
                    comboLabel += site.isAcid ? "deprot" : "prot";
                    comboLabel += "_atom" + std::to_string(site.atomIdx);
                }

                try { MolOps::sanitizeMol(*rw); } catch (...) { continue; }
                std::string pSmi = MolToSmiles(*rw);
                if (seenSMILES.count(pSmi)) continue;
                seenSMILES.insert(pSmi);

                std::string label = tautPrefix + (comboLabel.empty() ? "Parent" : comboLabel);
                int kind = (ti == 0)
                    ? (comboLabel.empty() ? 0 : 2)   // 0=parent, 2=protomer
                    : (comboLabel.empty() ? 1 : 3); // 1=tautomer, 3=both
                allForms.push_back({pSmi, std::move(rw), label, kind, hhPop});
            }
        }

        if (allForms.empty()) {
            // Fallback: at least output the parent
            auto rw = std::make_unique<RWMol>(*parentMol);
            std::string wSmi = MolToSmiles(*rw);
            allForms.push_back({wSmi, std::move(rw), "Parent", 0, 1.0});
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

            DGeomHelpers::EmbedParameters ep = DGeomHelpers::ETKDGv3;
            ep.randomSeed = 42;
            ep.pruneRmsThresh = 0.25;
            ep.numThreads = 0;
            auto cids = DGeomHelpers::EmbedMultipleConfs(*mol, conformersPerForm, ep);
            if (cids.empty()) {
                int cid = DGeomHelpers::EmbedMolecule(*mol, ep);
                if (cid >= 0) cids.push_back(cid);
                else {
                    cid = embed_molecule(*mol);
                    if (cid >= 0) cids.push_back(cid);
                }
            }
            if (cids.empty()) continue;

            std::vector<std::pair<int, double>> mmffResults;
            try { MMFF::MMFFOptimizeMoleculeConfs(*mol, mmffResults, 0, 1000, "MMFF94"); } catch (...) {}
            try { computeGasteigerCharges(*mol); } catch (...) {}

            std::vector<std::pair<int, double>> indexed;
            for (int ci = 0; ci < (int)cids.size(); ci++) {
                double e = std::numeric_limits<double>::quiet_NaN();
                if (ci < (int)mmffResults.size()) {
                    if (mmffResults[ci].first == 0) e = mmffResults[ci].second;
                    else continue;
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
                        if ((int)(*it)->getId() != confId) toRemove.push_back((*it)->getId());
                    }
                    for (auto cid : toRemove) confMol->removeConformer(cid);
                }

                std::string confLabel = form.label;
                if (indexed.size() > 1) confLabel += "_Conf" + std::to_string(ri + 1);

                allConformers.push_back({
                    std::move(confMol), energy, fi, ri, confLabel, form.smi, form.kind, form.hhPopulation
                });
            }
        }

        if (energyCutoff > 0 && !allConformers.empty()) {
            double bestE = 1e30;
            for (const auto &c : allConformers)
                if (!std::isnan(c.energy) && c.energy < bestE) bestE = c.energy;
            allConformers.erase(
                std::remove_if(allConformers.begin(), allConformers.end(),
                    [&](const PreparedConformer &c) {
                        return !std::isnan(c.energy) && c.energy > bestE + energyCutoff;
                    }),
                allConformers.end());
        }

        double kBT = 0.001987204 * temperature;
        if (kBT < 1e-10) kBT = 0.593;
        double minE = 1e30;
        for (const auto &c : allConformers)
            if (!std::isnan(c.energy) && c.energy < minE) minE = c.energy;

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
