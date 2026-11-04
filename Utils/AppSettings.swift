// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Metal
import Observation
import Synchronization

/// Verbosity of the activity log mirrored to stderr in headless runs (`--log-level`, `log level`,
/// `settings set log-level=`). Each level shows itself and everything more severe.
enum LogVerbosity: String, CaseIterable, Sendable {
    case debug, info, warn, error, none

    /// The `LogLevel`s an `ActivityLog` stderr mirror should pass through at this verbosity.
    var levels: Set<LogLevel> {
        switch self {
        case .debug: Set(LogLevel.allCases)
        case .info: [.info, .success, .warning, .error]
        case .warn: [.warning, .error]
        case .error: [.error]
        case .none: []
        }
    }

    /// Accepts the canonical names plus `warning` as an alias of `warn`.
    init?(name: String) {
        let lowered = name.lowercased()
        self.init(rawValue: lowered == "warning" ? "warn" : lowered)
    }
}

/// The user's software settings, shared by the GUI, the in-app console and headless runs.
///
/// Values live in `UserDefaults` (the app's defaults domain, which `Druse --headless` shares), so a
/// change made in Settings applies to the next terminal run and vice versa. The type is
/// `@Observable` so SwiftUI views bind to it directly, and main-actor isolated because every
/// consumer (views, view-model operations, console commands) runs there.
///
/// `maxCPUCores` is applied for real: `applyThreadLimits()` caps the C++ core's TBB pool and the
/// OpenMM CPU platform and publishes the limit to `CPUBudget`, which bounds Swift fan-out
/// (ligand preparation, pocket detection probes, pose clustering, GFN2 pose refinement).
@Observable
@MainActor
final class AppSettings {
    static let shared = AppSettings()

    /// `UserDefaults` keys.
    private enum Key {
        static let maxCPUCores = "settings.maxCPUCores"
        static let pdbCacheDirectory = "settings.pdbCacheDirectory"
        static let defaultProjectDirectory = "settings.defaultProjectDirectory"
        static let consoleHistoryLength = "settings.consoleHistoryLength"
        static let consoleAutosuggestions = "settings.consoleAutosuggestions"
        static let headlessLogLevel = "settings.headlessLogLevel"
    }

    // MARK: - Machine

    /// Logical CPUs macOS currently lets this process use: the upper bound for `maxCPUCores`.
    nonisolated static var availableCores: Int { max(ProcessInfo.processInfo.activeProcessorCount, 1) }

    /// Performance cores (Apple Silicon P-cluster), or every core on machines without perf levels.
    nonisolated static var performanceCores: Int {
        sysctlInt("hw.perflevel0.physicalcpu") ?? availableCores
    }

    /// Efficiency cores (Apple Silicon E-cluster); 0 when the machine reports none.
    nonisolated static var efficiencyCores: Int {
        sysctlInt("hw.perflevel1.physicalcpu") ?? 0
    }

    /// Default core budget: all performance cores but one, so the UI and the window server keep a
    /// fast core while a long job runs.
    nonisolated static var defaultCPUCores: Int {
        min(max(performanceCores - 1, 1), availableCores)
    }

    /// `$HOME`, which shells expand `~` to (the account's home folder when unset). Files meant for
    /// the user's terminal (the `druse` command, completions, the prompt's history) go under it,
    /// so a run with another `HOME` stays inside that folder.
    nonisolated static var shellHome: URL {
        ProcessInfo.processInfo.environment["HOME"].map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.homeDirectoryForCurrentUser
    }

    static let historyLengthRange = 10...10_000
    static let defaultHistoryLength = 500

    nonisolated static var defaultPDBCacheDirectory: String {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".druse/pdb_cache").path
    }

    nonisolated static var defaultProjectDirectoryPath: String {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Documents").path
    }

    // MARK: - Stored settings

    /// CPU cores Druse may keep busy at once. Persisted and applied immediately; writers keep it
    /// within 1…`availableCores` (see `clampCores`).
    var maxCPUCores: Int {
        didSet {
            defaults.set(maxCPUCores, forKey: Key.maxCPUCores)
            sessionCoreOverride = nil
            applyThreadLimits()
        }
    }

    /// Where downloaded PDB/mmCIF entries are cached (7-day retention).
    var pdbCacheDirectory: String {
        didSet { defaults.set(pdbCacheDirectory, forKey: Key.pdbCacheDirectory) }
    }

    /// Folder the Save Project panel starts in.
    var defaultProjectDirectory: String {
        didSet { defaults.set(defaultProjectDirectory, forKey: Key.defaultProjectDirectory) }
    }

    /// Commands kept in the console history (GUI console and the terminal prompt); writers keep
    /// it within `historyLengthRange`.
    var consoleHistoryLength: Int {
        didSet { defaults.set(consoleHistoryLength, forKey: Key.consoleHistoryLength) }
    }

    /// Show inline completion suggestions while typing in the in-app console.
    var consoleAutosuggestions: Bool {
        didSet { defaults.set(consoleAutosuggestions, forKey: Key.consoleAutosuggestions) }
    }

    /// Activity-log verbosity mirrored to stderr by `Druse --headless` when `--log-level` is not given.
    var headlessLogLevel: LogVerbosity {
        didSet { defaults.set(headlessLogLevel.rawValue, forKey: Key.headlessLogLevel) }
    }

    // MARK: - Session state (not persisted)

    /// Core limit for this process only (`Druse --headless --cores N`); cleared when the stored
    /// setting changes so the latest explicit choice wins.
    private(set) var sessionCoreOverride: Int?

    /// Cores in effect for this process: the session override, else `maxCPUCores`.
    var effectiveCores: Int { sessionCoreOverride ?? maxCPUCores }

    /// Width for Swift task groups of CPU-bound work (ligand preparation, GFN2 pose refinement).
    /// Equal to `effectiveCores`.
    var maxConcurrentTasks: Int { effectiveCores }

    /// The limit the C++ core reported after the last `applyThreadLimits()`.
    private(set) var appliedThreadLimit: Int = 0

    /// Verbosity of the activity log mirrored to stderr in this headless process; nil in the GUI,
    /// where activity-log entries appear in the Logs panel instead.
    private(set) var sessionLogVerbosity: LogVerbosity?

    private let defaults: UserDefaults

    private init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let storedCores = defaults.object(forKey: Key.maxCPUCores) as? Int
        maxCPUCores = Self.clampCores(storedCores ?? Self.defaultCPUCores)
        pdbCacheDirectory = defaults.string(forKey: Key.pdbCacheDirectory) ?? Self.defaultPDBCacheDirectory
        defaultProjectDirectory = defaults.string(forKey: Key.defaultProjectDirectory) ?? Self.defaultProjectDirectoryPath
        let storedHistory = defaults.object(forKey: Key.consoleHistoryLength) as? Int
        consoleHistoryLength = min(max(storedHistory ?? Self.defaultHistoryLength, Self.historyLengthRange.lowerBound),
                                   Self.historyLengthRange.upperBound)
        consoleAutosuggestions = defaults.object(forKey: Key.consoleAutosuggestions) as? Bool ?? true
        headlessLogLevel = defaults.string(forKey: Key.headlessLogLevel).flatMap(LogVerbosity.init(name:)) ?? .warn
    }

    // MARK: - Applying

    /// Pushes `effectiveCores` into the C++ core (TBB global limit, OpenMM CPU threads). Called at
    /// launch (GUI and headless) and whenever the limit changes.
    func applyThreadLimits() {
        appliedThreadLimit = Int(druse_set_max_threads(Int32(effectiveCores)))
        CPUBudget.setWorkers(effectiveCores)
    }

    /// Limits this process to `cores` without changing the stored setting (`--cores N`).
    /// Returns the clamped value that was applied.
    @discardableResult
    func overrideCores(forSession cores: Int) -> Int {
        sessionCoreOverride = Self.clampCores(cores)
        applyThreadLimits()
        return effectiveCores
    }

    /// Mirrors activity-log entries at `verbosity` (and more severe) to stderr for the rest of this
    /// headless run (`--log-level`, `log level`).
    func applyHeadlessLogVerbosity(_ verbosity: LogVerbosity) {
        sessionLogVerbosity = verbosity
        ActivityLog.shared.enableStderrMirroring(levels: verbosity.levels)
    }

    /// Restores every stored setting to its default (the session core override is dropped too).
    func resetToDefaults() {
        maxCPUCores = Self.defaultCPUCores
        pdbCacheDirectory = Self.defaultPDBCacheDirectory
        defaultProjectDirectory = Self.defaultProjectDirectoryPath
        consoleHistoryLength = Self.defaultHistoryLength
        consoleAutosuggestions = true
        headlessLogLevel = .warn
        sessionCoreOverride = nil
        applyThreadLimits()
    }

    /// Folder URL for `defaultProjectDirectory`, or nil when it no longer exists (panels then
    /// fall back to their own last location).
    var defaultProjectDirectoryURL: URL? {
        let url = URL(fileURLWithPath: (defaultProjectDirectory as NSString).expandingTildeInPath, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            return nil
        }
        return url
    }

    // MARK: - Helpers

    nonisolated static func clampCores(_ cores: Int) -> Int {
        min(max(cores, 1), availableCores)
    }

    private nonisolated static func sysctlInt(_ name: String) -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0, value > 0 else { return nil }
        return Int(value)
    }
}

/// Read-only description of the machine's compute resources, for Settings ▸ Performance and the
/// `settings` command.
struct MachineInfo {
    let cpuSummary: String
    let gpuSummary: String
    let memorySummary: String

    static var current: MachineInfo {
        let performance = AppSettings.performanceCores
        let efficiency = AppSettings.efficiencyCores
        let cpu = efficiency > 0
            ? "\(AppSettings.availableCores) cores (\(performance) performance + \(efficiency) efficiency)"
            : "\(AppSettings.availableCores) cores"
        let gpu: String
        if let device = MTLCreateSystemDefaultDevice() {
            let workingSet = ByteSize.string(device.recommendedMaxWorkingSetSize, style: .memory)
            gpu = "\(device.name), \(device.hasUnifiedMemory ? "unified memory" : "discrete memory"), \(workingSet) recommended working set"
        } else {
            gpu = "no Metal device"
        }
        let memory = ByteSize.string(ProcessInfo.processInfo.physicalMemory, style: .memory)
        return MachineInfo(cpuSummary: cpu, gpuSummary: gpu, memorySummary: memory)
    }
}

/// Byte counts with a "." decimal separator on every system locale, to match the English UI.
enum ByteSize {
    static func string<I: BinaryInteger>(_ bytes: I, style: ByteCountFormatStyle.Style = .file) -> String {
        Int64(bytes).formatted(.byteCount(style: style).locale(Locale(identifier: "en_US_POSIX")))
    }
}

/// The core limit as seen from any thread.
///
/// `AppSettings` is main-actor isolated, but CPU-bound loops (pocket grids, RMSD clustering) run on
/// background threads and cannot hop to the main actor per call. `applyThreadLimits()` publishes the
/// limit here so those loops honour Settings and `--cores N` without capturing it by hand.
enum CPUBudget {
    private static let workerCount = Atomic<Int>(ProcessInfo.processInfo.activeProcessorCount)

    /// Number of worker threads CPU-bound fan-out may use (at least 1).
    static var workers: Int { max(workerCount.load(ordering: .relaxed), 1) }

    /// Publishes a new limit; called by `AppSettings.applyThreadLimits()`.
    static func setWorkers(_ count: Int) {
        workerCount.store(max(count, 1), ordering: .relaxed)
    }

    /// `DispatchQueue.concurrentPerform` limited to `workers` threads: each worker strides through
    /// the iteration space, so `body` still runs exactly once per index.
    static func concurrentPerform(iterations: Int, _ body: (Int) -> Void) {
        guard iterations > 0 else { return }
        let width = min(workers, iterations)
        DispatchQueue.concurrentPerform(iterations: width) { worker in
            var index = worker
            while index < iterations {
                body(index)
                index += width
            }
        }
    }
}
