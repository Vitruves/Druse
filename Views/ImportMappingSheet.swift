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
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "tablecells")
                    .foregroundStyle(.blue)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Column Mapping")
                        .font(.system(size: 13, weight: .semibold))
                    Text("\(preview?.rowCount ?? 0) rows from \(preview?.fileURL.lastPathComponent ?? "file")")
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding()

            Divider()

            // Column list
            if let preview {
                ScrollView {
                    VStack(spacing: 1) {
                        // Table header
                        HStack(spacing: 0) {
                            Text("Source Column")
                                .frame(width: 140, alignment: .leading)
                            Text("Sample Values")
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Text("Assign To")
                                .frame(width: 140, alignment: .trailing)
                        }
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)

                        ForEach(preview.columns.indices, id: \.self) { idx in
                            columnRow(idx)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .frame(maxHeight: 300)
            }

            Divider()

            // Actions
            HStack {
                let assignedCount = preview?.columns.filter { $0.target != .doNotImport }.count ?? 0
                Text("\(assignedCount) column(s) will be imported")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)

                Spacer()

                Button("Cancel") {
                    dismiss()
                }
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
            .padding()
        }
        .frame(width: 640)
        .frame(minHeight: 200)
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
            HStack(spacing: 0) {
                Text(col.sourceHeader)
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .frame(width: 140, alignment: .leading)
                    .lineLimit(1)

                Text(col.sampleValues.prefix(3).joined(separator: ", "))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .lineLimit(1)

                Picker("", selection: bindingForColumn(idx)) {
                    ForEach(ImportTargetField.allCases) { field in
                        Text(field.rawValue).tag(field)
                    }
                }
                .frame(width: 130)
                .controlSize(.small)

                // Editable name appears only for custom-column targets
                if col.target == .customColumn {
                    TextField("Column name", text: customNameBinding(idx))
                        .textFieldStyle(.roundedBorder)
                        .controlSize(.small)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(width: 140)
                        .padding(.leading, 6)
                } else {
                    Color.clear.frame(width: 146, height: 1)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 4)
            .background(idx % 2 == 0 ? Color.clear : Color.primary.opacity(0.03))
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
