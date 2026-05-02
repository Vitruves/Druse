// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - File Loading, PDB Fetching, Search, Test Data

extension AppViewModel {

    // MARK: - Load Test Data

    func loadCaffeine() {
        let mol = TestMolecules.caffeine()
        setLigandForDocking(mol)
        log.debug("[Loading] Loaded test caffeine ligand (\(mol.atoms.count) atoms)", category: .molecule)
    }

    func loadAlanineDipeptide() {
        let mol = TestMolecules.alanineDipeptide()
        molecules.protein = mol
        pushToRenderer()
        renderer?.fitToContent()
        log.debug("[Loading] Loaded test alanine dipeptide (\(mol.atoms.count) atoms)", category: .molecule)
    }

    func loadBoth() {
        loadAlanineDipeptide()
        loadCaffeine()
    }

    // MARK: - PDB Loading

    func loadFromPDB(id: String) {
        workspace.isLoading = true
        workspace.loadingMessage = "Fetching \(id.uppercased())..."
        log.info("Fetching PDB \(id.uppercased()) from RCSB...", category: .network)

        Task {
            do {
                let content = try await PDBService.shared.fetchPDBFile(id: id)
                molecules.rawPDBContent = content
                workspace.loadingMessage = "Parsing..."

                let result = await Task.detached { PDBParser.parse(content) }.value

                if let protData = result.protein {
                    let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                       bonds: protData.bonds, title: protData.title)
                    mol.secondaryStructureAssignments = protData.ssRanges.map {
                        (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                    }
                    molecules.protein = mol
                } else {
                    molecules.protein = nil
                    log.info("[PDB] No protein data found in \(id.uppercased())", category: .pdb)
                }

                if let firstLig = result.ligands.first {
                    let ligMol = Molecule(name: firstLig.name, atoms: firstLig.atoms,
                                          bonds: firstLig.bonds, title: firstLig.title)
                    setLigandForDocking(ligMol)

                    for ligData in result.ligands {
                        let entry = LigandEntry(
                            name: ligData.name,
                            smiles: ligData.smiles ?? "",
                            atoms: ligData.atoms,
                            bonds: ligData.bonds,
                            isPrepared: true,
                            conformerCount: 1
                        )
                        ligandDB.add(entry)
                    }
                } else {
                    molecules.ligand = nil
                    log.info("[PDB] No ligands found in \(id.uppercased())", category: .pdb)
                }

                molecules.preparationReport = ProteinPreparation.analyze(
                    atoms: result.protein?.atoms ?? [],
                    bonds: result.protein?.bonds ?? [],
                    waterCount: result.waterCount
                )

                workspace.hiddenChainIDs.removeAll()
                pushToRenderer()
                renderer?.fitToContent()

                let atomCount = molecules.protein?.atomCount ?? 0
                let ligCount = result.ligands.count
                log.success("Loaded \(id.uppercased()): \(atomCount) protein atoms, \(ligCount) ligand(s), \(result.waterCount) waters removed", category: .pdb)

                if let prot = molecules.protein {
                    let chains = Set(prot.atoms.map(\.chainID)).sorted()
                    let residueCount = prot.residues.filter(\.isStandard).count
                    let hetCount = prot.atoms.filter(\.isHetAtom).count
                    log.info("  Chains: \(chains.joined(separator: ", ")) — \(residueCount) residues, \(hetCount) het atoms", category: .pdb)
                }

                for w in result.warnings.prefix(5) {
                    log.warn(w, category: .pdb)
                }

                workspace.statusMessage = "\(id.uppercased()) loaded"
            } catch {
                log.error("Failed to load \(id): \(error.localizedDescription)", category: .pdb)
                workspace.statusMessage = "Failed to load \(id)"
            }

            workspace.isLoading = false
            workspace.loadingMessage = ""
        }
    }

    // MARK: - File Loading

    func loadFromFile(url: URL) {
        guard let format = FileImportHandler.detectFormat(url: url) else {
            log.error("[Loading] Unsupported file format: \(url.pathExtension) (\(url.lastPathComponent))", category: .system)
            return
        }

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        log.info("[Loading] Importing \(format) file: \(url.lastPathComponent) (\(fileSize / 1024) KB)", category: .molecule)

        if format != .pdb {
            molecules.rawPDBContent = nil
        }

        workspace.isLoading = true
        workspace.loadingMessage = "Loading \(url.lastPathComponent)..."

        Task {
            do {
                switch format {
                case .pdb:
                    let (content, result) = try await Task.detached {
                        let content = try String(contentsOf: url, encoding: .utf8)
                        return (content, PDBParser.parse(content))
                    }.value
                    molecules.rawPDBContent = content

                    if let protData = result.protein {
                        let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                           bonds: protData.bonds, title: protData.title)
                        mol.secondaryStructureAssignments = protData.ssRanges.map {
                            (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                        }
                        molecules.protein = mol
                    } else {
                        log.info("[PDB] No protein data found in file", category: .pdb)
                    }
                    molecules.ligand = result.ligands.first.map {
                        Molecule(name: $0.name, atoms: $0.atoms, bonds: $0.bonds, title: $0.title)
                    }
                    if molecules.ligand == nil {
                        log.debug("[PDB] No ligands extracted from PDB file", category: .pdb)
                    }

                    molecules.preparationReport = ProteinPreparation.analyze(
                        atoms: result.protein?.atoms ?? [],
                        bonds: result.protein?.bonds ?? [],
                        waterCount: result.waterCount
                    )

                    log.success("Loaded PDB: \(molecules.protein?.atomCount ?? 0) atoms, \(result.waterCount) waters removed", category: .pdb)
                    if let prot = molecules.protein {
                        let chains = Set(prot.atoms.map(\.chainID)).sorted()
                        let residueCount = prot.residues.filter(\.isStandard).count
                        log.info("  Chains: \(chains.joined(separator: ", ")) — \(residueCount) residues", category: .pdb)
                    }

                case .sdf, .mol:
                    let molecules = try await Task.detached { try SDFParser.parse(url: url) }.value

                    if let first = molecules.first {
                        self.molecules.ligand = Molecule(name: first.name, atoms: first.atoms,
                                          bonds: first.bonds, title: first.title)
                        log.success("Loaded \(molecules.count) molecule(s) from SDF", category: .molecule)
                    } else {
                        log.warn("[Loading] SDF/MOL file contained no parseable molecules", category: .molecule)
                    }

                case .smi:
                    try ligandDB.importSMIFile(url: url, prepare: false)
                    log.success("Imported SMILES file into ligand database", category: .molecule)

                case .csv:
                    try ligandDB.importCSV(url: url, prepare: false)
                    log.success("Imported CSV file into ligand database", category: .molecule)

                case .mol2:
                    let mols = try await Task.detached { try MOL2Parser.parse(url: url) }.value
                    if let first = mols.first {
                        molecules.ligand = Molecule(name: first.name, atoms: first.atoms,
                                          bonds: first.bonds, title: first.title)
                        log.success("Loaded \(mols.count) molecule(s) from MOL2", category: .molecule)
                    } else {
                        log.warn("[Loading] MOL2 file contained no parseable molecules", category: .molecule)
                    }

                case .mmcif:
                    let content = try await Task.detached {
                        try String(contentsOf: url, encoding: .utf8)
                    }.value
                    do {
                        let mmcifResult = try await Task.detached { try MMCIFParser.parse(content: content) }.value
                        let mol = Molecule(name: mmcifResult.name, atoms: mmcifResult.atoms,
                                           bonds: mmcifResult.bonds, title: mmcifResult.title)
                        molecules.protein = mol
                        molecules.preparationReport = ProteinPreparation.analyze(
                            atoms: mmcifResult.atoms,
                            bonds: mmcifResult.bonds
                        )
                    } catch {
                        log.warn("[Loading] mmCIF parser failed, falling back to PDB parser: \(error.localizedDescription)", category: .pdb)
                        let result = await Task.detached { PDBParser.parse(content) }.value
                        if let protData = result.protein {
                            let mol = Molecule(name: protData.name, atoms: protData.atoms,
                                               bonds: protData.bonds, title: protData.title)
                            mol.secondaryStructureAssignments = protData.ssRanges.map {
                                (start: $0.start, end: $0.end, type: $0.type, chain: $0.chain)
                            }
                            molecules.protein = mol
                        }
                        molecules.preparationReport = ProteinPreparation.analyze(
                            atoms: result.protein?.atoms ?? [],
                            bonds: result.protein?.bonds ?? [],
                            waterCount: result.waterCount
                        )
                    }
                    log.success("Loaded mmCIF: \(molecules.protein?.atomCount ?? 0) atoms", category: .pdb)
                }

                workspace.hiddenChainIDs.removeAll()
                pushToRenderer()
                renderer?.fitToContent()
                workspace.statusMessage = "\(url.lastPathComponent) loaded"
            } catch {
                log.error("Failed to load file: \(error.localizedDescription)", category: .system)
                workspace.statusMessage = "Failed to load file"
            }

            workspace.isLoading = false
            workspace.loadingMessage = ""
        }
    }

    func importFile() {
        guard let url = FileImportHandler.showOpenPanel() else {
            log.debug("[Loading] File import cancelled by user", category: .molecule)
            return
        }
        loadFromFile(url: url)
    }

    // MARK: - PDB Search

    func searchPDB(query: String) {
        guard !query.trimmingCharacters(in: .whitespaces).isEmpty else {
            log.debug("[PDB] Search skipped: empty query", category: .network)
            return
        }
        workspace.isSearching = true
        workspace.searchResults = []
        log.info("Searching RCSB for '\(query)'...", category: .network)

        Task {
            do {
                let results = try await PDBService.shared.search(query: query)
                workspace.searchResults = results
                log.success("Found \(results.count) results for '\(query)'", category: .network)
            } catch {
                log.error("Search failed: \(error.localizedDescription)", category: .network)
            }
            workspace.isSearching = false
        }
    }

    // MARK: - Open Project (for welcome screen)

    func openProject() {
        let panel = NSOpenPanel()
        panel.title = "Open Druse Project"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowedContentTypes = [.folder]
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor [weak self] in
                guard let self else { return }
                do {
                    try await DruseProjectIO.load(from: url, into: self)
                } catch {
                    self.log.error("Load failed: \(error.localizedDescription)", category: .system)
                }
            }
        }
    }
}
