#include "druse_core_internal.h"

// Ionizable-site inspection, per-site protomer generation, and ensemble v2.

#include <limits>
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

        // Use GNN-provided sites directly — no SMARTS re-detection needed.
        // GNN pKa values already account for electronic environment, so no
        // proximity depression is applied (would double-count).
        auto washedMol = std::make_unique<RWMol>(*parentMol);

        struct AmbSite { int atomIdx; bool isAcid; double pKa; };
        std::vector<AmbSite> ambSites;

        for (int32_t i = 0; i < nSites; i++) {
            int atomIdx = sites[i].atomIdx;
            if (atomIdx < 0 || atomIdx >= (int)parentMol->getNumAtoms()) continue;
            bool isAcid = sites[i].isAcid;
            double pKa = sites[i].pKa;
            double deltaPH = pH - pKa;

            if (isAcid && deltaPH > pkaThreshold) {
                // Clearly deprotonated at this pH
                Atom *atom = washedMol->getAtomWithIdx(atomIdx);
                int nH = atom->getTotalNumHs();
                if (nH > 0) {
                    atom->setNumExplicitHs(nH - 1);
                    atom->setFormalCharge(atom->getFormalCharge() - 1);
                }
            } else if (!isAcid && deltaPH < -pkaThreshold) {
                // Clearly protonated at this pH
                Atom *atom = washedMol->getAtomWithIdx(atomIdx);
                int nH = atom->getTotalNumHs();
                atom->setNumExplicitHs(nH + 1);
                atom->setFormalCharge(atom->getFormalCharge() + 1);
            } else if (std::abs(deltaPH) <= pkaThreshold) {
                // Ambiguous — enumerate both states
                ambSites.push_back({atomIdx, isAcid, pKa});
            }
        }
        try { MolOps::sanitizeMol(*washedMol); } catch (...) {
            washedMol = std::make_unique<RWMol>(*parentMol);
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
        if (maxTotalForms > 200) maxTotalForms = 200;

        for (auto &prot : protomers) {
            if ((int)allForms.size() >= maxTotalForms) break;

            if (!seenSMILES.count(prot.smi)) seenSMILES.insert(prot.smi);
            {
                auto rw = std::make_unique<RWMol>(*prot.mol);
                int kind = (prot.label != "Parent") ? 2 : 0;
                allForms.push_back({prot.smi, std::move(rw), prot.label, kind, prot.hhPopulation});
            }

            try {
                MolStandardize::TautomerEnumerator enumerator;
                enumerator.setMaxTautomers(std::max(maxTautomers, 2));
                enumerator.setMaxTransforms(200);
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
            } catch (...) {
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
