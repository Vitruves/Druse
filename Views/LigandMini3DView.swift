import SwiftUI
import MetalKit

// MARK: - LigandMini3DView
//
// Self-contained 3D viewport with isolated state. This struct exists to keep
// the embedded MetalView shielded from the database window's state churn —
// when the parent window has many @State / @Observable properties (db.entries,
// conformers, ligand2DImage, inspectedEntry, ...), repeated re-evaluation of
// the parent's body can starve a deeply-nested MetalMTKView's display link
// and make 3D rotation feel multi-second-laggy.
//
// By packaging the renderer + MetalView + conformer navigation into a separate
// View struct with its own @State, the only re-evaluation triggers for this
// view are: a new ligand passed in (atoms changed), or a conformer switch.
// Everything else in the database window is invisible to this struct.

struct LigandMini3DView: View {
    let atoms: [Atom]
    let bonds: [Bond]
    let conformers: [LigandDatabaseWindow.ConformerEntry]
    @Binding var selectedConformerIndex: Int

    @State private var renderer: Renderer?

    var body: some View {
        VStack(spacing: 0) {
            if let renderer = renderer {
                MetalView(renderer: renderer)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if atoms.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "cube")
                        .font(.title)
                        .foregroundStyle(.secondary)
                    Text("Prepare ligand for 3D view")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Color.clear
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onAppear { initRenderer() }
            }

            // Conformer navigation bar — chevrons + counter + energy
            if !conformers.isEmpty {
                Divider()
                HStack(spacing: 8) {
                    Button(action: { stepConformer(by: -1) }) {
                        Image(systemName: "chevron.left")
                            .font(.footnote.weight(.semibold))
                    }
                    .buttonStyle(.borderless)
                    .help("Previous conformer")

                    Text("Conformer \(selectedConformerIndex + 1) / \(conformers.count)")
                        .font(.footnote.monospaced().weight(.medium))

                    Button(action: { stepConformer(by: 1) }) {
                        Image(systemName: "chevron.right")
                            .font(.footnote.weight(.semibold))
                    }
                    .buttonStyle(.borderless)
                    .help("Next conformer")

                    Spacer()

                    if selectedConformerIndex < conformers.count {
                        Text(String(format: "%.1f kcal/mol", conformers[selectedConformerIndex].energy))
                            .font(.footnote.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
            }
        }
        .onChange(of: atoms.count) { _, _ in
            // New molecule (or atom-count change) — re-fit camera.
            updateRenderer(atoms: atoms, bonds: bonds, fit: true)
        }
        .onChange(of: selectedConformerIndex) { _, newIdx in
            guard newIdx >= 0, newIdx < conformers.count else { return }
            let conf = conformers[newIdx]
            // Geometry-only update — preserve camera angle.
            updateRenderer(atoms: conf.molecule.atoms, bonds: conf.molecule.bonds, fit: false)
        }
    }

    private func initRenderer() {
        guard renderer == nil, !atoms.isEmpty else { return }
        guard let device = MTLCreateSystemDefaultDevice() else { return }

        // Bootstrap MTKView for pipeline construction. Discarded after init —
        // the real MTKView is created by MetalView.makeNSView and connected
        // via renderer.mtkView weak ref.
        let tempView = MTKView(frame: CGRect(x: 0, y: 0, width: 400, height: 300), device: device)
        tempView.colorPixelFormat = .bgra8Unorm
        tempView.depthStencilPixelFormat = .depth32Float
        tempView.sampleCount = 4

        guard let r = Renderer(mtkView: tempView) else { return }
        r.updateMoleculeData(atoms: atoms, bonds: bonds)
        r.fitToContent()
        renderer = r
    }

    private func updateRenderer(atoms: [Atom], bonds: [Bond], fit: Bool) {
        guard let r = renderer else {
            initRenderer()
            return
        }
        r.updateMoleculeData(atoms: atoms, bonds: bonds)
        if fit { r.fitToContent() }
    }

    private func stepConformer(by delta: Int) {
        guard !conformers.isEmpty else { return }
        let n = conformers.count
        selectedConformerIndex = ((selectedConformerIndex + delta) % n + n) % n
    }
}
