import Foundation

enum ProteinResidueTemplateStore {

    static func normalizeAtomName(_ atomName: String) -> String {
        atomName
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .uppercased()
    }

    static func hydrogenGroupName(for atomName: String) -> String? {
        let normalized = normalizeAtomName(atomName)
        guard !normalized.isEmpty else { return nil }

        if normalized.first?.isNumber == true {
            let ordinalPrefix = normalized.prefix { $0.isNumber }
            let remainder = normalized.dropFirst(ordinalPrefix.count)
            guard remainder.hasPrefix("H") else { return nil }
            return String(remainder)
        }

        guard normalized.hasPrefix("H") else { return nil }
        if normalized.count > 1, normalized.last?.isNumber == true {
            return String(normalized.dropLast())
        }
        return normalized
    }
}
