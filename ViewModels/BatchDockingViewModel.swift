import Foundation
import simd

/// ViewModel for batch docking (virtual screening) of multiple ligands from the database.
@Observable
@MainActor
final class BatchDockingViewModel {
    var isRunning = false
    var currentLigandIndex = 0
    var totalLigands = 0
    var currentLigandName = ""
    var completedResults: [(name: String, bestEnergy: Float, poseCount: Int)] = []
    var progress: Double = 0
    var statusMessage = ""

    private var dockingEngine: DockingEngine?
    private var shouldStop = false

    /// Dock all entries from the ligand database against the given pocket.
    func dockAll(
        entries: [LigandEntry],
        pocket: BindingPocket,
        protein: Molecule,
        config: DockingConfig,
        engine: DockingEngine
    ) async {
        guard !entries.isEmpty else { return }

        isRunning = true
        shouldStop = false
        totalLigands = entries.count
        currentLigandIndex = 0
        completedResults = []
        dockingEngine = engine

        // Compute grid maps once (shared across all ligands)
        statusMessage = "Computing grid maps..."
        engine.computeGridMaps(protein: protein, pocket: pocket, spacing: config.gridSpacing)
        statusMessage = "Grid maps ready"

        for (i, entry) in entries.enumerated() {
            guard !shouldStop else { break }

            currentLigandIndex = i
            currentLigandName = entry.name
            progress = Double(i) / Double(totalLigands)
            statusMessage = "Docking \(entry.name) (\(i + 1)/\(totalLigands))..."

            // Create molecule from entry
            guard !entry.atoms.isEmpty else {
                completedResults.append((name: entry.name, bestEnergy: .infinity, poseCount: 0))
                continue
            }

            let mol = Molecule(name: entry.name, atoms: entry.atoms, bonds: entry.bonds, title: entry.smiles, smiles: entry.smiles)

            // Use fewer generations for batch screening (speed vs accuracy tradeoff)
            var batchConfig = config
            batchConfig.numRuns = 1
            batchConfig.generationsPerRun = 50
            batchConfig.populationSize = min(config.populationSize, 100)
            batchConfig.liveUpdateFrequency = 999 // no live viz for batch

            let results = await engine.runDocking(ligand: mol, pocket: pocket, config: batchConfig)

            let bestEnergy = results.first?.energy ?? .infinity
            completedResults.append((name: entry.name, bestEnergy: bestEnergy, poseCount: results.count))

            ActivityLog.shared.info(
                String(format: "  %@: %.1f kcal/mol (%d poses)", entry.name, bestEnergy, results.count),
                category: .dock
            )
        }

        progress = 1.0
        isRunning = false

        // Sort by energy
        completedResults.sort { $0.bestEnergy < $1.bestEnergy }

        let completed = completedResults.count
        let bestHit = completedResults.first
        statusMessage = String(format: "Batch complete: %d ligands, best: %@ (%.1f kcal/mol)",
                               completed, bestHit?.name ?? "?", bestHit?.bestEnergy ?? 0)

        ActivityLog.shared.success(statusMessage, category: .dock)
    }

    func stop() {
        shouldStop = true
        dockingEngine?.stopDocking()
    }

    /// Export batch results as CSV
    func exportCSV() -> String {
        var csv = "Rank,Name,Best_Energy,Pose_Count\n"
        for (i, r) in completedResults.enumerated() {
            csv += "\(i + 1),\(r.name),\(String(format: "%.2f", r.bestEnergy)),\(r.poseCount)\n"
        }
        return csv
    }
}
