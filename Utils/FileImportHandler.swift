// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import AppKit
import UniformTypeIdentifiers

enum MoleculeFileFormat: String {
    case pdb
    case sdf
    case mol
    case mol2
    case smi
    case csv
    case mmcif
}

enum FileImportHandler {

    static var supportedExtensions: Set<String> {
        ["pdb", "ent", "sdf", "sd", "mol", "mol2", "smi", "csv", "tsv", "txt", "cif", "mmcif"]
    }

    /// General open panel for all supported molecular file types.
    @MainActor
    static func showOpenPanel() -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [
            UTType(filenameExtension: "pdb"),
            UTType(filenameExtension: "ent"),
            UTType(filenameExtension: "sdf"),
            UTType(filenameExtension: "mol"),
            UTType(filenameExtension: "smi"),
            UTType(filenameExtension: "csv"),
            UTType(filenameExtension: "tsv"),
            UTType(filenameExtension: "txt"),
        ].compactMap { $0 }
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select a molecular file"
        panel.prompt = "Open"

        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    /// Open panel specifically for SMILES/CSV batch files.
    @MainActor
    static func showBatchOpenPanel(fileType: ImportFileType? = nil) -> URL? {
        let panel = NSOpenPanel()
        switch fileType {
        case .smi:
            panel.allowedContentTypes = [
                UTType(filenameExtension: "smi"),
                UTType(filenameExtension: "smiles"),
            ].compactMap { $0 }
            panel.message = "Select a SMILES file (.smi)"
        case .csv:
            panel.allowedContentTypes = [
                UTType(filenameExtension: "csv"),
                UTType(filenameExtension: "tsv"),
                UTType(filenameExtension: "txt"),
            ].compactMap { $0 }
            panel.message = "Select a CSV or TSV file"
        case .sdf:
            panel.allowedContentTypes = [
                UTType(filenameExtension: "sdf"),
                UTType(filenameExtension: "sd"),
            ].compactMap { $0 }
            panel.message = "Select an SDF file (.sdf)"
        case nil:
            panel.allowedContentTypes = [
                UTType(filenameExtension: "smi"),
                UTType(filenameExtension: "csv"),
                UTType(filenameExtension: "tsv"),
                UTType(filenameExtension: "txt"),
                UTType(filenameExtension: "sdf"),
            ].compactMap { $0 }
            panel.message = "Select a SMILES (.smi), CSV, or SDF file"
        }
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.prompt = "Import"

        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    static func detectFormat(url: URL) -> MoleculeFileFormat? {
        switch url.pathExtension.lowercased() {
        case "pdb", "ent":    return .pdb
        case "sdf", "sd":     return .sdf
        case "mol":           return .mol
        case "mol2":          return .mol2
        case "smi":           return .smi
        case "csv", "tsv":    return .csv
        case "cif", "mmcif":  return .mmcif
        case "txt":           return .smi
        default:              return nil
        }
    }

    static func canHandle(url: URL) -> Bool {
        supportedExtensions.contains(url.pathExtension.lowercased())
    }
}
