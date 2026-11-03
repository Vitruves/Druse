// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Column Mapping Model

/// Target fields that an imported column can be assigned to.
///
/// `customColumn` is for free-form user-imported properties — the actual
/// column key the user chooses lives on `ImportColumnMapping.customColumnName`,
/// not in the enum, so the enum stays `CaseIterable` for the picker.
enum ImportTargetField: String, CaseIterable, Identifiable {
    case doNotImport = "Do not import"
    case name = "Name"
    case smiles = "SMILES"
    case ki = "Ki (nM)"
    case pKi = "pKi"
    case ic50 = "IC50 (nM)"
    case customColumn = "Custom column…"

    var id: String { rawValue }
}

/// A detected column from the imported file with a user-assigned target.
struct ImportColumnMapping: Identifiable {
    let id = UUID()
    let sourceHeader: String
    let sampleValues: [String]
    var target: ImportTargetField = .customColumn
    /// User-editable name for the custom column. Defaults to `sourceHeader`,
    /// only used when `target == .customColumn`.
    var customColumnName: String = ""
}

/// Detected file contents ready for the mapping sheet.
struct ImportPreview {
    var columns: [ImportColumnMapping]
    var rowCount: Int
    var fileURL: URL
    var fileType: ImportFileType
}

enum ImportFileType {
    case csv, smi, sdf
}

// MARK: - Mapping Sheet View

struct ImportMappingSheet: View {
    @Binding var preview: ImportPreview?
    let onImport: (ImportPreview) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        DSSheetScaffold(
            title: "Column Mapping",
            subtitle: subtitle,
            systemImage: "tablecells"
        ) {
            Form {
                if let preview {
                    Section {
                        ForEach(preview.columns.indices, id: \.self) { idx in
                            columnRow(idx)
                        }
                    } header: {
                        HStack {
                            Text("Source Column")
                            Spacer()
                            Text("Import As")
                        }
                    } footer: {
                        HStack(alignment: .firstTextBaseline) {
                            Text(requirementText)
                                .foregroundStyle(hasRequiredMapping ? AnyShapeStyle(.secondary) : AnyShapeStyle(DS.Color.warning))
                            Spacer()
                            Text("\(assignedCount) of \(preview.columns.count) columns will be imported")
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }
                        .font(DS.Font.caption)
                    }
                }
            }
            .formStyle(.grouped)
        } footer: {
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Import") {
                if let p = preview {
                    onImport(p)
                }
                dismiss()
            }
            .keyboardShortcut(.defaultAction)
            .disabled(preview == nil || !hasRequiredMapping)
        }
        .frame(minWidth: 600, idealWidth: 680, minHeight: 360, idealHeight: 520)
    }

    private var subtitle: String {
        let rows = preview?.rowCount ?? 0
        let file = preview?.fileURL.lastPathComponent ?? "file"
        return "\(rows) \(rows == 1 ? "row" : "rows") from \(file). Choose how each column is stored; unmatched columns become custom table columns."
    }

    private var assignedCount: Int {
        preview?.columns.filter { $0.target != .doNotImport }.count ?? 0
    }

    private var requirementText: String {
        guard let preview else { return "" }
        if preview.fileType == .sdf {
            return "SDF records carry their own structures; property mappings are optional."
        }
        return hasRequiredMapping
            ? "Rows without a SMILES or name are skipped."
            : "Assign at least one column to SMILES or Name."
    }

    private var hasRequiredMapping: Bool {
        guard let preview else { return false }
        let targets = Set(preview.columns.map(\.target))
        // SDF doesn't need SMILES (has 3D coords). CSV/SMI need SMILES.
        if preview.fileType == .sdf {
            return true // SDF always has structure, mappings are optional extras
        }
        return targets.contains(.smiles) || targets.contains(.name)
    }

    @ViewBuilder
    private func columnRow(_ idx: Int) -> some View {
        if let preview, idx < preview.columns.count {
            let col = preview.columns[idx]
            LabeledContent {
                HStack(spacing: DS.Space.s) {
                    if col.target == .customColumn {
                        TextField("Column name", text: customNameBinding(idx))
                            .textFieldStyle(.roundedBorder)
                            .font(DS.Font.mono)
                            .frame(width: 150)
                    }
                    Picker("Import As", selection: bindingForColumn(idx)) {
                        ForEach(ImportTargetField.allCases) { field in
                            Text(field.rawValue).tag(field)
                        }
                    }
                    .labelsHidden()
                    .fixedSize()
                }
            } label: {
                VStack(alignment: .leading, spacing: DS.Space.xxs) {
                    Text(col.sourceHeader)
                        .font(DS.Font.mono.weight(.medium))
                        .foregroundStyle(col.target == .doNotImport ? .secondary : .primary)
                        .lineLimit(1)
                    Text(col.sampleValues.isEmpty ? "No sample values" : col.sampleValues.prefix(3).joined(separator: ", "))
                        .font(DS.Font.monoSmall)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.tail)
                }
            }
        }
    }

    private func bindingForColumn(_ idx: Int) -> Binding<ImportTargetField> {
        Binding(
            get: { preview?.columns[idx].target ?? .doNotImport },
            set: { newValue in
                preview?.columns[idx].target = newValue
            }
        )
    }

    private func customNameBinding(_ idx: Int) -> Binding<String> {
        Binding(
            get: { preview?.columns[idx].customColumnName ?? "" },
            set: { newValue in
                preview?.columns[idx].customColumnName = newValue
            }
        )
    }
}
