// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import simd

// MARK: - Shared pieces of the Ligand Database and Results Database tables
//
// Both windows use a native SwiftUI `Table` (sortable, user-customizable columns
// persisted per window). This file holds what they share: one sort comparator,
// one number formatting vocabulary, the numeric/score/status cells, ligand
// efficiency and pose RMSD.

// MARK: - Sorting

/// A cell's sort key. Missing values always sort last, whatever the direction,
/// so unprepared compounds or undocked ligands never crowd the top of a sorted column.
enum TableSortValue: Sendable {
    case numeric(Double)
    case text(String)
    case missing

    static func number(_ value: Float?) -> TableSortValue { value.map { .numeric(Double($0)) } ?? .missing }
    static func number(_ value: Double?) -> TableSortValue { value.map { .numeric($0) } ?? .missing }
    static func number(_ value: Int?) -> TableSortValue { value.map { .numeric(Double($0)) } ?? .missing }

    /// Imported free-form values: numeric when they parse, text otherwise.
    static func parsed(_ raw: String?) -> TableSortValue {
        guard let raw = raw?.trimmingCharacters(in: .whitespaces), !raw.isEmpty else { return .missing }
        return Double(raw).map { .numeric($0) } ?? .text(raw)
    }
}

/// One comparator type for every column of a table (a `Table` needs a single
/// `SortComparator` type). Identity is the column id plus the direction.
struct TableSort<Row>: SortComparator {
    let id: String
    var order: SortOrder
    private let key: @Sendable (Row) -> TableSortValue

    init(_ id: String, order: SortOrder = .forward, key: @escaping @Sendable (Row) -> TableSortValue) {
        self.id = id
        self.order = order
        self.key = key
    }

    func compare(_ lhs: Row, _ rhs: Row) -> ComparisonResult {
        let base: ComparisonResult
        switch (key(lhs), key(rhs)) {
        case (.missing, .missing): return .orderedSame
        case (.missing, _): return .orderedDescending
        case (_, .missing): return .orderedAscending
        case let (.numeric(a), .numeric(b)): base = a < b ? .orderedAscending : a > b ? .orderedDescending : .orderedSame
        case let (.text(a), .text(b)): base = a.localizedStandardCompare(b)
        case (.numeric, .text): base = .orderedAscending
        case (.text, .numeric): base = .orderedDescending
        }
        guard order == .reverse else { return base }
        switch base {
        case .orderedAscending: return .orderedDescending
        case .orderedDescending: return .orderedAscending
        case .orderedSame: return .orderedSame
        }
    }

    static func == (lhs: TableSort, rhs: TableSort) -> Bool { lhs.id == rhs.id && lhs.order == rhs.order }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(order)
    }
}

// MARK: - Column customization persistence

/// Column order, widths and visibility of a `Table`, persisted in user defaults.
enum TableColumnStore {
    static func load<Row: Identifiable>(_ key: String) -> TableColumnCustomization<Row> {
        guard let data = UserDefaults.standard.data(forKey: key),
              let decoded = try? JSONDecoder().decode(TableColumnCustomization<Row>.self, from: data)
        else { return TableColumnCustomization<Row>() }
        return decoded
    }

    static func save<Row: Identifiable>(_ customization: TableColumnCustomization<Row>, key: String) {
        guard let data = try? JSONEncoder().encode(customization) else { return }
        UserDefaults.standard.set(data, forKey: key)
    }
}

// MARK: - Number formatting

/// One formatting vocabulary for both tables and their inspectors.
enum TableFormat {
    static func mw(_ value: Float) -> String { String(format: "%.1f", value) }
    static func tpsa(_ value: Float) -> String { String(format: "%.0f", value) }
    static func score(_ value: Float) -> String { String(format: "%.1f", value) }
    static func energyTerm(_ value: Float) -> String { String(format: "%.1f", value) }
    static func ligandEfficiency(_ value: Float) -> String { String(format: "%.2f", value) }
    static func rmsd(_ value: Float) -> String { String(format: "%.1f", value) }
    static func pK(_ value: Float) -> String { String(format: "%.1f", value) }
    /// Whole percents from 10 % up keep "100%" inside a narrow column.
    static func percent(_ fraction: Double) -> String {
        String(format: fraction >= 0.0995 ? "%.0f%%" : "%.1f%%", fraction * 100)
    }
}

// MARK: - Derived metrics

enum LigandMetrics {
    /// Ligand efficiency in kcal/mol per heavy atom: −ΔG / HA for energy scores,
    /// 1.37 · pK / HA (ΔG at 298 K) for affinity scores.
    static func ligandEfficiency(score: Float, method: ScoringMethod, heavyAtoms: Int) -> Float? {
        guard heavyAtoms > 0 else { return nil }
        let deltaG = method.isAffinityScore ? -1.37 * score : score
        return -deltaG / Float(heavyAtoms)
    }

    /// Lipophilic ligand efficiency (LLE / LipE): pKi − cLogP.
    static func lipophilicEfficiency(pK: Float, logP: Float) -> Float { pK - logP }

    /// Heavy-atom RMSD between two poses of the same ligand (same atom order, no symmetry correction).
    static func rmsd(_ a: [SIMD3<Float>], _ b: [SIMD3<Float>]) -> Float? {
        guard !a.isEmpty, a.count == b.count else { return nil }
        var sumSq: Float = 0
        for k in 0..<a.count {
            let d = a[k] - b[k]
            sumSq += simd_dot(d, d)
        }
        return (sumSq / Float(a.count)).squareRoot()
    }
}

extension DSScore {
    /// Ligand efficiency (kcal/mol/HA); ≥ 0.3 is the usual lead-like bar.
    static func tone(ligandEfficiency le: Float) -> DSTone { le >= 0.3 ? .success : le >= 0.2 ? .warning : .danger }
    /// Lipophilic ligand efficiency; ≥ 5 is the usual optimized-compound bar.
    static func tone(lipophilicEfficiency lle: Float) -> DSTone { lle >= 5 ? .success : lle >= 3 ? .warning : .danger }
}

extension DSTone {
    /// Text colour of a toned number in a table cell: the poor tone reads orange, not alarm red.
    var cellColor: Color { self == .danger ? DS.Color.scorePoor : color }
}

// MARK: - Cells

/// Numeric cell: monospaced digits, optional tone, em dash when empty.
struct TableNumberCell: View {
    let text: String?
    var tone: DSTone? = nil
    var help: String? = nil

    var body: some View {
        if let text {
            Text(text)
                .font(DS.Font.secondary)
                .monospacedDigit()
                .foregroundStyle(tone.map { AnyShapeStyle($0.cellColor) } ?? AnyShapeStyle(.primary))
                .help(help ?? "")
        } else {
            TableDashCell()
        }
    }
}

/// Placeholder for a missing value.
struct TableDashCell: View {
    var body: some View {
        Text("—")
            .font(DS.Font.secondary)
            .foregroundStyle(.tertiary)
            .accessibilityLabel("No value")
    }
}

/// Docking score without its unit (the unit lives in the column header), toned
/// by the method's convention.
struct TableScoreCell: View {
    let value: Float?
    let method: ScoringMethod

    var body: some View {
        if let value {
            let tone = method.isAffinityScore ? DSScore.tone(pK: value) : DSScore.tone(energy: value)
            Text(TableFormat.score(value))
                .font(DS.Font.secondary.weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(tone.cellColor)
                .accessibilityLabel("\(TableFormat.score(value)) \(method.unitLabel)")
        } else {
            TableDashCell()
        }
    }
}

/// Header title of the score column for the active scoring method.
extension ScoringMethod {
    var scoreColumnTitle: String { isAffinityScore ? "Score (\(unitLabel))" : "Score (kcal/mol)" }
}

// MARK: - Ligand preparation status

/// Preparation state of a library compound, worded identically in the sidebar,
/// the Ligand Database table and its inspector.
enum LigandPrepStatus {
    case prepared, unprepared, failed

    init(_ entry: LigandEntry) {
        if entry.preparationError != nil { self = .failed }
        else if entry.isPrepared { self = .prepared }
        else { self = .unprepared }
    }

    var label: String {
        switch self {
        case .prepared:   "Prepared"
        case .unprepared: "Not prepared"
        case .failed:     "Failed"
        }
    }

    var tone: DSTone {
        switch self {
        case .prepared:   .success
        case .unprepared: .neutral
        case .failed:     .danger
        }
    }

    var systemImage: String {
        switch self {
        case .prepared:   "checkmark.circle.fill"
        case .unprepared: "circle.dashed"
        case .failed:     "exclamationmark.triangle.fill"
        }
    }

    var sortRank: Int {
        switch self {
        case .prepared:   0
        case .unprepared: 1
        case .failed:     2
        }
    }

    var help: String {
        switch self {
        case .prepared:   "Prepared: 3D coordinates, protonation, charges and conformers are ready for docking"
        case .unprepared: "Not prepared yet: run Prepare before docking"
        case .failed:     "Preparation failed"
        }
    }

    @MainActor var tag: DSTag { DSTag(label, tone: tone) }
}

/// Status column cell: tinted icon + word.
struct LigandPrepStatusCell: View {
    let entry: LigandEntry

    var body: some View {
        let status = LigandPrepStatus(entry)
        Label(status.label, systemImage: status.systemImage)
            .font(DS.Font.secondary)
            .foregroundStyle(status == .unprepared ? AnyShapeStyle(.secondary) : AnyShapeStyle(status.tone.color))
            .help(entry.preparationError ?? status.help)
    }
}

// MARK: - Docking results per ligand

extension AppViewModel {
    /// Best pose per ligand name across the batch results and the current run,
    /// so the ligand table can show each compound's docking score.
    var bestDockingResultByLigand: [String: DockingResult] {
        var best: [String: DockingResult] = [:]
        for entry in docking.batchResults {
            if let top = entry.results.first { best[entry.ligandName] = top }
        }
        if docking.batchResults.isEmpty,
           let name = (docking.originalDockingLigand ?? molecules.ligand)?.name,
           let top = docking.dockingResults.first {
            best[name] = top
        }
        return best
    }
}

// MARK: - Status bar

/// Small tinted icon + secondary title, sized for `DSWindowStatusBar`.
struct StatusLabelStyle: LabelStyle {
    let tint: Color

    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: DS.Space.xs) {
            configuration.icon
                .font(DS.Font.caption)
                .imageScale(.small)
                .foregroundStyle(tint)
            configuration.title
        }
    }
}

/// Middle-dot separator between status bar items.
struct StatusSeparator: View {
    var body: some View {
        Text("·").foregroundStyle(.tertiary)
    }
}
