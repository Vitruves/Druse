import Foundation
import SwiftUI

// MARK: - Log Level

enum LogLevel: String, CaseIterable, Sendable {
    case info    = "Info"
    case success = "Success"
    case warning = "Warning"
    case error   = "Error"
    case debug   = "Debug"

    var icon: String {
        switch self {
        case .info:    "info.circle.fill"
        case .success: "checkmark.circle.fill"
        case .warning: "exclamationmark.triangle.fill"
        case .error:   "xmark.octagon.fill"
        case .debug:   "ant.fill"
        }
    }

    var color: Color {
        switch self {
        case .info:    .secondary
        case .success: .green
        case .warning: .orange
        case .error:   .red
        case .debug:   .purple
        }
    }

    /// Short tag for file log lines.
    var tag: String {
        switch self {
        case .info:    "INFO"
        case .success: "OK"
        case .warning: "WARN"
        case .error:   "ERR"
        case .debug:   "DBG"
        }
    }
}

// MARK: - Log Category

enum LogCategory: String, CaseIterable, Sendable {
    case system    = "System"
    case render    = "Render"
    case pdb       = "PDB"
    case prep      = "Prep"
    case dock      = "Dock"
    case smiles    = "SMILES"
    case selection = "Selection"
    case network   = "Network"
    case molecule  = "Molecule"
}

// MARK: - Log Entry

struct LogEntry: Identifiable, Sendable {
    let id = UUID()
    let timestamp: Date
    let level: LogLevel
    let category: LogCategory
    let message: String

    var formattedTime: String {
        Self.formatter.string(from: timestamp)
    }

    /// Full-precision timestamp for file logging: `2026-03-23 14:05:12.345`
    var fileFormattedTime: String {
        Self.fileFormatter.string(from: timestamp)
    }

    /// Single formatted line for file output.
    var fileLine: String {
        "\(fileFormattedTime) [\(level.tag)] [\(category.rawValue)] \(message)"
    }

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private static let fileFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()
}

// MARK: - File Log Writer

/// Async file writer that appends log lines to a session log file.
/// Thread-safe: all writes go through a serial actor.
private actor FileLogWriter {
    private var fileHandle: FileHandle?
    private let fileURL: URL

    init(fileURL: URL) {
        self.fileURL = fileURL
    }

    func open() {
        let dir = fileURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        if !FileManager.default.fileExists(atPath: fileURL.path) {
            FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }
        fileHandle = try? FileHandle(forWritingTo: fileURL)
        fileHandle?.seekToEndOfFile()

        // Write session header
        let header = "=== Druse session started at \(LogEntry.fileFormatterStatic.string(from: Date())) ===\n"
        if let data = header.data(using: .utf8) {
            fileHandle?.write(data)
        }
    }

    func write(_ lines: [String]) {
        guard let fh = fileHandle else { return }
        let block = lines.joined(separator: "\n") + "\n"
        if let data = block.data(using: .utf8) {
            fh.write(data)
        }
    }

    func close() {
        fileHandle?.closeFile()
        fileHandle = nil
    }

    deinit {
        fileHandle?.closeFile()
    }
}

// MARK: - Crash-Safe Synchronous Writer

/// Minimal synchronous file writer for crash handlers.
/// Uses POSIX write() — safe to call from signal handlers (no locks, no allocations).
/// Shared mutable state is unavoidable here: signal handlers cannot use Swift actors.
private enum CrashLogWriter {
    nonisolated(unsafe) static var fd: Int32 = -1
    nonisolated(unsafe) static var logPath: String = ""

    static func open(path: String) {
        logPath = path
        fd = Darwin.open(path, O_WRONLY | O_APPEND | O_CREAT, 0o644)
    }

    /// Synchronous write — safe from signal handlers.
    static func writeSync(_ message: String) {
        guard fd >= 0 else { return }
        message.withCString { ptr in
            let len = strlen(ptr)
            _ = Darwin.write(fd, ptr, len)
        }
    }

    static func close() {
        guard fd >= 0 else { return }
        Darwin.close(fd)
        fd = -1
    }
}

// MARK: - Crash Handler

/// Installs signal handlers and uncaught exception handler to capture crash info
/// into the log file before the process terminates.
enum CrashReporter {

    /// Install crash handlers. Call once at app startup.
    static func install() {
        NSSetUncaughtExceptionHandler { exception in
            let msg = """
            \n=== UNCAUGHT EXCEPTION ===
            Name: \(exception.name.rawValue)
            Reason: \(exception.reason ?? "unknown")
            Stack:\n\(exception.callStackSymbols.joined(separator: "\n"))
            === END EXCEPTION ===\n
            """
            CrashLogWriter.writeSync(msg)
        }

        // Catch fatal signals: SIGABRT (Metal asserts), SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP
        // Use sigaction instead of signal() — more reliable, not overridden by frameworks.
        let signals: [Int32] = [SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP]
        for sig in signals {
            var action = sigaction()
            action.__sigaction_u.__sa_handler = { signum in
                // Signal-safe: only POSIX write, no malloc, no ObjC
                guard CrashLogWriter.fd >= 0 else {
                    signal(signum, SIG_DFL)
                    raise(signum)
                    return
                }
                // Write crash marker — all strings are literals, signal-safe
                let msg: String
                switch signum {
                case SIGABRT: msg = "\n=== CRASH: SIGABRT ===\n"
                case SIGSEGV: msg = "\n=== CRASH: SIGSEGV ===\n"
                case SIGBUS:  msg = "\n=== CRASH: SIGBUS ===\n"
                case SIGILL:  msg = "\n=== CRASH: SIGILL ===\n"
                case SIGFPE:  msg = "\n=== CRASH: SIGFPE ===\n"
                case SIGTRAP: msg = "\n=== CRASH: SIGTRAP ===\n"
                default:      msg = "\n=== CRASH: SIGNAL \(signum) ===\n"
                }
                CrashLogWriter.writeSync(msg)
                CrashLogWriter.writeSync("Check log entries above for the last operation before crash.\n=== END CRASH ===\n")
                _ = fsync(CrashLogWriter.fd)
                Darwin.close(CrashLogWriter.fd)
                CrashLogWriter.fd = -1

                // Re-raise with default handler so macOS generates the .crash report
                signal(signum, SIG_DFL)
                raise(signum)
            }
            action.sa_flags = 0
            sigemptyset(&action.sa_mask)
            sigaction(sig, &action, nil)
        }
    }

    /// Check if the most recent log file from a previous session ended with a crash marker.
    /// Returns the crash summary if found, nil otherwise.
    static func checkPreviousSessionCrash() -> String? {
        let dir = URL(fileURLWithPath: LogSettings.logDirectory, isDirectory: true)
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        ) else { return nil }

        // Find the most recent .log file that is NOT the current session
        let currentFile = LogSettings.sessionFileURL().lastPathComponent
        let logFiles = files
            .filter { $0.pathExtension == "log" && $0.lastPathComponent != currentFile }
            .sorted {
                let d1 = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                let d2 = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return d1 > d2
            }

        guard let lastLog = logFiles.first,
              let content = try? String(contentsOf: lastLog, encoding: .utf8),
              content.contains("=== CRASH:") || content.contains("=== UNCAUGHT EXCEPTION ===")
        else { return nil }

        // Extract the crash block
        if let crashRange = content.range(of: "=== CRASH:", options: .backwards) {
            let crashBlock = String(content[crashRange.lowerBound...])
            let filename = lastLog.lastPathComponent
            return "Previous session (\(filename)) ended with a crash:\n\(crashBlock.prefix(500))"
        }
        if let exRange = content.range(of: "=== UNCAUGHT EXCEPTION ===", options: .backwards) {
            let exBlock = String(content[exRange.lowerBound...])
            let filename = lastLog.lastPathComponent
            return "Previous session (\(filename)) ended with an exception:\n\(exBlock.prefix(500))"
        }
        return nil
    }
}

private extension LogEntry {
    static let fileFormatterStatic: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss.SSS"
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()
}

// MARK: - Log Settings

/// Persistent log settings stored in UserDefaults.
struct LogSettings {
    private static let enabledKey = "logFileEnabled"
    private static let directoryKey = "logFileDirectory"
    private static let retentionKey = "logFileRetentionDays"

    static var isFileLoggingEnabled: Bool {
        get { UserDefaults.standard.object(forKey: enabledKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: enabledKey) }
    }

    static var logDirectory: String {
        get { UserDefaults.standard.string(forKey: directoryKey) ?? defaultLogDirectory }
        set { UserDefaults.standard.set(newValue, forKey: directoryKey) }
    }

    static var retentionDays: Int {
        get {
            let v = UserDefaults.standard.integer(forKey: retentionKey)
            return v > 0 ? v : 7
        }
        set { UserDefaults.standard.set(max(newValue, 1), forKey: retentionKey) }
    }

    static var defaultLogDirectory: String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.druse/logs"
    }

    /// Generate the log file URL for the current session.
    static func sessionFileURL() -> URL {
        let dir = URL(fileURLWithPath: logDirectory, isDirectory: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let filename = "druse_\(formatter.string(from: Date())).log"
        return dir.appendingPathComponent(filename)
    }

    /// Delete log files older than retention period.
    static func pruneOldLogs() {
        let dir = URL(fileURLWithPath: logDirectory, isDirectory: true)
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.creationDateKey], options: .skipsHiddenFiles
        ) else { return }

        let cutoff = Date().addingTimeInterval(-Double(retentionDays) * 86400)
        for file in files where file.pathExtension == "log" {
            if let attrs = try? file.resourceValues(forKeys: [.creationDateKey]),
               let created = attrs.creationDate, created < cutoff {
                try? FileManager.default.removeItem(at: file)
            }
        }
    }
}

// MARK: - Activity Log

@Observable
@MainActor
final class ActivityLog {
    static let shared = ActivityLog()

    private(set) var entries: [LogEntry] = []
    private let maxEntries = 2000

    // Throttling: buffer logs and flush at most 10x/sec
    private var pendingEntries: [LogEntry] = []
    private var flushTask: Task<Void, Never>?
    private let flushInterval: Duration = .milliseconds(100)

    // File logging
    private var fileWriter: FileLogWriter?
    private(set) var currentLogFileURL: URL?

    /// When true, every log entry is also printed to stdout (for benchmark CLI monitoring).
    /// Activated by BenchmarkRunner when it reads stdoutLogs=true from .bench_config.json.
    private var mirrorToStdout = false

    /// Enable stdout log mirroring (called by benchmark runner).
    func enableStdoutMirroring() {
        mirrorToStdout = true
    }

    private init() {
        startFileLogging()
        CrashReporter.install()

        // Check if previous session crashed
        if let crashSummary = CrashReporter.checkPreviousSessionCrash() {
            // Defer to avoid logging during init
            Task { @MainActor in
                ActivityLog.shared.warn("Previous session crashed — check log file for details", category: .system)
                ActivityLog.shared.debug(crashSummary, category: .system)
            }
        }
    }

    // MARK: - File Logging Lifecycle

    /// Start (or restart) file logging based on current settings.
    func startFileLogging() {
        guard LogSettings.isFileLoggingEnabled else {
            stopFileLogging()
            return
        }

        // Close existing writer if any
        if let old = fileWriter {
            Task { await old.close() }
        }
        CrashLogWriter.close()

        let url = LogSettings.sessionFileURL()
        currentLogFileURL = url

        // Ensure directory exists BEFORE opening POSIX fd
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Synchronous POSIX writer for crash-time logging — must open first
        // so crash handler can write even if async writer hasn't started yet
        CrashLogWriter.open(path: url.path)

        // Async writer for normal (buffered) logs
        let writer = FileLogWriter(fileURL: url)
        fileWriter = writer
        Task { await writer.open() }

        // Prune old logs on startup
        Task.detached(priority: .utility) {
            LogSettings.pruneOldLogs()
        }
    }

    func stopFileLogging() {
        CrashLogWriter.writeSync("\n=== Druse session ended cleanly at \(LogEntry.fileFormatterStatic.string(from: Date())) ===\n")
        if let writer = fileWriter {
            Task { await writer.close() }
        }
        CrashLogWriter.close()
        fileWriter = nil
        currentLogFileURL = nil
    }

    /// Call on app termination to mark a clean shutdown in the log.
    func shutdown() {
        // Flush any pending entries synchronously
        flush()
        stopFileLogging()
    }

    // MARK: - Logging

    func log(_ message: String, level: LogLevel = .info, category: LogCategory = .system) {
        let entry = LogEntry(timestamp: Date(), level: level, category: category, message: message)
        pendingEntries.append(entry)
        scheduleFlush()

        // Mirror to stdout for benchmark CLI monitoring (--druse-logs flag)
        if mirrorToStdout {
            print(entry.fileLine)
            fflush(stdout)
        }

        // Write-through: errors and warnings are always flushed immediately to survive crashes.
        // Docking category is also write-through since crashes most often occur during GPU docking.
        if level == .error || level == .warning || category == .dock {
            CrashLogWriter.writeSync(entry.fileLine + "\n")
        }
    }

    private func scheduleFlush() {
        guard flushTask == nil else { return }
        flushTask = Task { [weak self] in
            try? await Task.sleep(for: self?.flushInterval ?? .milliseconds(100))
            guard !Task.isCancelled else { return }
            self?.flush()
        }
    }

    private func flush() {
        flushTask = nil
        guard !pendingEntries.isEmpty else { return }

        let batch = pendingEntries
        pendingEntries.removeAll()

        // Append to in-memory entries
        entries.append(contentsOf: batch)
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }

        // Write to file asynchronously
        if let writer = fileWriter {
            let lines = batch.map(\.fileLine)
            Task { await writer.write(lines) }
        }
    }

    // MARK: - Convenience Methods

    func info(_ message: String, category: LogCategory = .system) {
        log(message, level: .info, category: category)
    }

    func success(_ message: String, category: LogCategory = .system) {
        log(message, level: .success, category: category)
    }

    func warn(_ message: String, category: LogCategory = .system) {
        log(message, level: .warning, category: category)
    }

    func error(_ message: String, category: LogCategory = .system) {
        log(message, level: .error, category: category)
    }

    func debug(_ message: String, category: LogCategory = .system) {
        log(message, level: .debug, category: category)
    }

    func clear() {
        entries.removeAll()
        pendingEntries.removeAll()
        flushTask?.cancel()
        flushTask = nil
    }

    /// Reveal the current log file in Finder.
    func revealLogInFinder() {
        guard let url = currentLogFileURL else { return }
        NSWorkspace.shared.selectFile(url.path, inFileViewerRootedAtPath: url.deletingLastPathComponent().path)
    }

    /// Reveal the log directory in Finder.
    func revealLogDirectoryInFinder() {
        let dir = URL(fileURLWithPath: LogSettings.logDirectory, isDirectory: true)
        NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: dir.path)
    }
}
