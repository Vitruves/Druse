import SwiftUI

/// Sheet for entering a scaffold SMARTS/SMILES pattern for fragment-based docking.
/// The scaffold determines which fragment of the ligand is placed first as the anchor.
/// Ligands not containing the scaffold are docked last, sorted by Tanimoto similarity.
struct ScaffoldInputSheet: View {
    @Binding var scaffoldSMARTS: String?
    @Binding var scaffoldMode: FragmentDockingConfig.ScaffoldMode
    @Binding var isPresented: Bool

    @State private var inputText: String = ""
    @State private var validationResult: String? = nil
    @State private var isValid: Bool = false

    // Current ligand SMILES for "use current ligand" button
    var currentLigandSmiles: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: "puzzlepiece.extension")
                    .font(.system(size: 14))
                    .foregroundStyle(.orange)
                Text("Enforce Scaffold Anchor")
                    .font(.system(size: 13, weight: .semibold))
                Spacer()
                Button {
                    isPresented = false
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }

            Text("Enter a SMARTS or SMILES pattern. The fragment-based docking engine will use the matching fragment as the anchor, placing it first in the binding site.")
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            // Input field
            HStack(spacing: 4) {
                TextField("SMARTS or SMILES (e.g. c1ccncc1)", text: $inputText)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 11, design: .monospaced))
                    .onSubmit { validateInput() }

                Button("Validate") { validateInput() }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
            }

            // Validation result
            if let result = validationResult {
                HStack(spacing: 4) {
                    Image(systemName: isValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(isValid ? .green : .red)
                        .font(.system(size: 10))
                    Text(result)
                        .font(.system(size: 10))
                        .foregroundStyle(isValid ? .green : .red)
                }
            }

            // Quick actions
            HStack(spacing: 6) {
                if let smiles = currentLigandSmiles {
                    Button {
                        inputText = smiles
                        validateInput()
                    } label: {
                        HStack(spacing: 3) {
                            Image(systemName: "arrow.right.circle")
                                .font(.system(size: 9))
                            Text("Use Current Ligand")
                                .font(.system(size: 10))
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }

                Button {
                    if let clipboard = NSPasteboard.general.string(forType: .string) {
                        inputText = clipboard.trimmingCharacters(in: .whitespacesAndNewlines)
                        validateInput()
                    }
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "doc.on.clipboard")
                            .font(.system(size: 9))
                        Text("Paste")
                            .font(.system(size: 10))
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // Common scaffolds
            VStack(alignment: .leading, spacing: 2) {
                Text("Common Scaffolds")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)

                let commonScaffolds: [(String, String)] = [
                    ("Pyridine", "c1ccncc1"),
                    ("Benzimidazole", "c1ccc2[nH]cnc2c1"),
                    ("Indole", "c1ccc2[nH]ccc2c1"),
                    ("Piperazine", "C1CNCCN1"),
                    ("Quinoline", "c1ccc2ncccc2c1"),
                    ("Benzene", "c1ccccc1"),
                ]

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 3) {
                    ForEach(commonScaffolds, id: \.0) { (name, smarts) in
                        Button {
                            inputText = smarts
                            validateInput()
                        } label: {
                            Text(name)
                                .font(.system(size: 9))
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 3)
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                }
            }

            Divider()

            // Action buttons
            HStack {
                Button("Cancel") {
                    isPresented = false
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Apply") {
                    if isValid && !inputText.isEmpty {
                        scaffoldSMARTS = inputText
                        scaffoldMode = .manual
                    }
                    isPresented = false
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!isValid)
            }
        }
        .padding(16)
        .frame(width: 380)
        .onAppear {
            if let existing = scaffoldSMARTS {
                inputText = existing
                validateInput()
            }
        }
    }

    private func validateInput() {
        guard !inputText.isEmpty else {
            validationResult = nil
            isValid = false
            return
        }

        // Try to match against a simple test molecule (aspirin)
        if let match = RDKitBridge.matchScaffold(smiles: "CC(=O)OC1=CC=CC=C1C(=O)O", scaffoldSMARTS: inputText) {
            if match.hasMatch {
                validationResult = "Valid pattern (\(match.matchedAtomIndices.count) atoms matched in test molecule)"
                isValid = true
            } else {
                // Pattern parsed but didn't match test — that's OK, it may match the actual ligand
                validationResult = "Valid pattern (no match in test molecule, may match your ligand)"
                isValid = true
            }
        } else {
            validationResult = "Invalid SMARTS/SMILES pattern"
            isValid = false
        }
    }
}
