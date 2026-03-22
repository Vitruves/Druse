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

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()
}

// MARK: - Activity Log

@Observable
@MainActor
final class ActivityLog {
    static let shared = ActivityLog()

    private(set) var entries: [LogEntry] = []
    private let maxEntries = 500

    // Throttling: buffer logs and flush at most 10x/sec
    private var pendingEntries: [LogEntry] = []
    private var flushTask: Task<Void, Never>?
    private let flushInterval: Duration = .milliseconds(100)

    private init() {}

    func log(_ message: String, level: LogLevel = .info, category: LogCategory = .system) {
        let entry = LogEntry(timestamp: Date(), level: level, category: category, message: message)
        pendingEntries.append(entry)
        scheduleFlush()
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
        entries.append(contentsOf: pendingEntries)
        pendingEntries.removeAll()
        if entries.count > maxEntries {
            entries.removeFirst(entries.count - maxEntries)
        }
    }

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
}
