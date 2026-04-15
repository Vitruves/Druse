// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import Foundation
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Ligand Table Column Manifest
//
// Drives the dynamic columns of the Ligand Database table:
//   - Built-in columns (Name, SMILES, descriptors, enumeration metadata)
//   - User-imported custom columns (one per unique key in entry.userProperties)
//
// State (column order + visibility) is persisted via @AppStorage as JSON.
// Discovery: when the database loads, ColumnState.discoverUserColumns(from:)
// scans entries for property keys that aren't already registered, appends
// them as visible columns, and saves.

/// What kind of data a column displays. `userProperty` carries the property
/// key the column is bound to.
enum LigandColumnKind: Equatable, Hashable, Codable {
    case name
    case smiles
    case popPercent
    case deltaE
    case conf
    case mw
    case logP
    case hbd
    case hba
    case tpsa
    case rotB
    case lipinski
    case atoms
    case userProperty(String)

    /// Stable string id used for ordering, persistence, and drag-drop.
    var stableID: String {
        switch self {
        case .name: return "builtin.name"
        case .smiles: return "builtin.smiles"
        case .popPercent: return "builtin.popPercent"
        case .deltaE: return "builtin.deltaE"
        case .conf: return "builtin.conf"
        case .mw: return "builtin.mw"
        case .logP: return "builtin.logP"
        case .hbd: return "builtin.hbd"
        case .hba: return "builtin.hba"
        case .tpsa: return "builtin.tpsa"
        case .rotB: return "builtin.rotB"
        case .lipinski: return "builtin.lipinski"
        case .atoms: return "builtin.atoms"
        case .userProperty(let key): return "user.\(key)"
        }
    }

    var title: String {
        switch self {
        case .name: return "Name"
        case .smiles: return "SMILES"
        case .popPercent: return "Pop%"
        case .deltaE: return "ΔE"
        case .conf: return "Conf"
        case .mw: return "MW"
        case .logP: return "LogP"
        case .hbd: return "HBD"
        case .hba: return "HBA"
        case .tpsa: return "TPSA"
        case .rotB: return "RotB"
        case .lipinski: return "Lip."
        case .atoms: return "Atoms"
        case .userProperty(let key): return key
        }
    }

    /// Fixed pixel width. `nil` means flexible (fills remaining space).
    /// Only the SMILES column returns nil. Widths are kept tight so more
    /// columns fit on screen — content is monospaced and centered, so the
    /// values fill the column nicely.
    var defaultWidth: CGFloat? {
        switch self {
        case .name: return 150
        case .smiles: return nil
        case .popPercent: return 46
        case .deltaE: return 40
        case .conf: return 38
        case .mw: return 44
        case .logP: return 38
        case .hbd: return 32
        case .hba: return 32
        case .tpsa: return 40
        case .rotB: return 34
        case .lipinski: return 28
        case .atoms: return 38
        case .userProperty: return 70
        }
    }

    /// Cell content alignment. Name and SMILES are left-aligned (text-like
    /// data); everything else is center-aligned for consistency in a tight
    /// numeric grid.
    var alignment: Alignment {
        switch self {
        case .name, .smiles: return .leading
        default: return .center
        }
    }

    /// The sort field this column uses, or nil if it's not currently sortable
    /// via the existing SortField enum. (User-property and enumeration columns
    /// are not yet sortable; that's a follow-up.)
    var sortField: LigandDatabaseWindow.SortField? {
        switch self {
        case .name: return .name
        case .smiles: return .smiles
        case .mw: return .mw
        case .logP: return .logP
        case .hbd: return .hbd
        case .hba: return .hba
        case .tpsa: return .tpsa
        case .rotB: return .rotB
        case .atoms: return .atoms
        default: return nil
        }
    }

    /// Whether this column may be hidden by the user. Name + SMILES are kept
    /// always-visible because hiding both would leave rows unrecognizable.
    var isHideable: Bool {
        switch self {
        case .name, .smiles: return false
        default: return true
        }
    }
}

/// One column entry — kind + visibility. Position in `ColumnState.columns`
/// determines display order.
struct LigandColumn: Identifiable, Equatable, Codable {
    let kind: LigandColumnKind
    var visible: Bool

    var id: String { kind.stableID }
}

// MARK: - Column State (observable)

/// Ordered list of columns shown in the Ligand Database table. Persists to
/// `@AppStorage` via JSON encoding so user choices (order, visibility) survive
/// app relaunches.
@Observable
@MainActor
final class LigandColumnState {

    private(set) var columns: [LigandColumn]

    /// AppStorage-backed JSON snapshot. Updated whenever `columns` changes.
    private static let storageKey = "ligand-table-columns-v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([LigandColumn].self, from: data),
           !decoded.isEmpty {
            self.columns = Self.mergeWithDefaults(decoded)
        } else {
            self.columns = Self.defaultColumns
        }
    }

    /// The columns to render, in display order, filtered to visible only.
    var visibleColumns: [LigandColumn] { columns.filter(\.visible) }

    /// All hideable columns (for the show/hide context menu).
    var hideableColumns: [LigandColumn] { columns.filter(\.kind.isHideable) }

    /// Set visibility on a column by id.
    func setVisible(_ visible: Bool, id: String) {
        guard let idx = columns.firstIndex(where: { $0.id == id }) else { return }
        guard columns[idx].visible != visible else { return }
        columns[idx].visible = visible
        save()
    }

    /// Toggle visibility on a column by id.
    func toggleVisible(id: String) {
        guard let idx = columns.firstIndex(where: { $0.id == id }) else { return }
        columns[idx].visible.toggle()
        save()
    }

    /// Move a column to a new position. Both indices are in `columns` array
    /// space (full list including hidden columns).
    func move(fromID: String, beforeID: String) {
        guard let from = columns.firstIndex(where: { $0.id == fromID }),
              let to = columns.firstIndex(where: { $0.id == beforeID }),
              from != to
        else { return }
        let item = columns.remove(at: from)
        let adjusted = from < to ? to - 1 : to
        columns.insert(item, at: adjusted)
        save()
    }

    /// Restore default order and make all defaults visible. User-property
    /// columns are kept (appended at the end).
    func resetToDefaults() {
        let userCols = columns.filter {
            if case .userProperty = $0.kind { return true }
            return false
        }
        columns = Self.defaultColumns + userCols
        save()
    }

    /// Synchronize user-property columns with the entries:
    ///   - Append columns for keys that exist in entries but not yet in the manifest
    ///   - Remove columns whose keys no longer appear in any entry (cleanup
    ///     after a re-import where the user mapped a previously-custom column
    ///     to a built-in target like Name)
    func discoverUserColumns(from entries: [LigandEntry]) {
        var seen = Set<String>()
        for entry in entries {
            for key in entry.userProperties.keys {
                seen.insert(key)
            }
        }

        let existingKeys: Set<String> = Set(columns.compactMap { col in
            if case .userProperty(let key) = col.kind { return key }
            return nil
        })

        // Append new
        let newKeys = seen.subtracting(existingKeys).sorted()
        for key in newKeys {
            columns.append(LigandColumn(kind: .userProperty(key), visible: true))
        }

        // Remove orphans (user-property columns whose key is no longer in any entry)
        let orphanKeys = existingKeys.subtracting(seen)
        if !orphanKeys.isEmpty {
            columns.removeAll { col in
                if case .userProperty(let key) = col.kind {
                    return orphanKeys.contains(key)
                }
                return false
            }
        }

        if !newKeys.isEmpty || !orphanKeys.isEmpty {
            save()
        }
    }

    /// Discard a user-property column. Used when an import is undone or
    /// a key is no longer present in any entry.
    func removeUserColumn(key: String) {
        columns.removeAll {
            if case .userProperty(let k) = $0.kind { return k == key }
            return false
        }
        save()
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(columns) else { return }
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }

    // MARK: - Defaults

    /// The built-in column set, in default display order. SMILES is the
    /// flexible column (no fixed width).
    static let defaultColumns: [LigandColumn] = [
        LigandColumn(kind: .name,       visible: true),
        LigandColumn(kind: .smiles,     visible: true),
        LigandColumn(kind: .popPercent, visible: true),
        LigandColumn(kind: .deltaE,     visible: true),
        LigandColumn(kind: .conf,       visible: true),
        LigandColumn(kind: .mw,         visible: true),
        LigandColumn(kind: .logP,       visible: true),
        LigandColumn(kind: .hbd,        visible: true),
        LigandColumn(kind: .hba,        visible: true),
        LigandColumn(kind: .tpsa,       visible: true),
        LigandColumn(kind: .rotB,       visible: true),
        LigandColumn(kind: .lipinski,   visible: true),
        LigandColumn(kind: .atoms,      visible: true),
    ]

    /// Merge a persisted state with the current defaults. New built-in
    /// columns added in code after the user's last save are appended visibly;
    /// removed built-in columns are dropped from the user's state. User-
    /// property columns are preserved as-is.
    private static func mergeWithDefaults(_ persisted: [LigandColumn]) -> [LigandColumn] {
        let defaultIDs = Set(defaultColumns.map(\.id))
        let persistedIDs = Set(persisted.map(\.id))

        // Drop built-in columns the code no longer defines.
        var merged = persisted.filter { col in
            if case .userProperty = col.kind { return true }
            return defaultIDs.contains(col.id)
        }

        // Append any built-in columns the persisted state didn't have.
        for col in defaultColumns where !persistedIDs.contains(col.id) {
            merged.append(col)
        }

        return merged
    }
}

// MARK: - Drag-to-Reorder Drop Delegate

/// SwiftUI `DropDelegate` that handles header-cell drops for column reordering.
/// Loads the dragged column's stable id from the `NSItemProvider`, then asks
/// `LigandColumnState` to move it before the target column. Also fires
/// `onDragEnter` / `onDragExit` callbacks so the parent view can show a live
/// insertion indicator while the drag is in progress, and `onDrop` to clear
/// drag-source state.
struct ColumnDropDelegate: DropDelegate {
    let targetID: String
    let columnState: LigandColumnState
    var onDragEnter: (() -> Void)? = nil
    var onDragExit: (() -> Void)? = nil
    var onDrop: (() -> Void)? = nil

    func validateDrop(info: DropInfo) -> Bool {
        info.hasItemsConforming(to: [.text])
    }

    func dropEntered(info: DropInfo) {
        onDragEnter?()
    }

    func dropExited(info: DropInfo) {
        onDragExit?()
    }

    func performDrop(info: DropInfo) -> Bool {
        guard let provider = info.itemProviders(for: [.text]).first else {
            onDrop?()
            return false
        }
        provider.loadObject(ofClass: NSString.self) { item, _ in
            guard let droppedID = item as? String else { return }
            Task { @MainActor in
                columnState.move(fromID: droppedID, beforeID: targetID)
                onDrop?()
            }
        }
        return true
    }
}
