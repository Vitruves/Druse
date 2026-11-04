// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

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
    /// A preparation is running (the empty state shows progress instead of its button).
    var isPreparing: Bool = false
    /// Prepares the compound from the empty state; nil hides the button.
    var onPrepare: (() -> Void)? = nil

    @State private var renderer: Renderer?

    var body: some View {
        VStack(spacing: 0) {
            if let renderer = renderer {
                MetalView(renderer: renderer)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if atoms.isEmpty {
                VStack(spacing: DS.Space.m) {
                    Image(systemName: "cube.transparent")
                        .font(.title2)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(.tertiary)
                    Text(isPreparing ? "Preparing…" : "No 3D structure yet")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                    if isPreparing {
                        ProgressView().controlSize(.small)
                    } else if let onPrepare {
                        Button(action: onPrepare) {
                            Label("Prepare", systemImage: "wand.and.stars")
                        }
                        .controlSize(.small)
                        .help("Generate 3D coordinates, charges and conformers for this compound")
                    }
                }
                .padding(DS.Space.l)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(DS.Color.cardFill)
            } else {
                Color.clear
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onAppear { initRenderer() }
            }

            // Conformer navigation bar — chevrons + counter + energy
            if !conformers.isEmpty {
                Divider()
                HStack(spacing: DS.Space.s) {
                    Button(action: { stepConformer(by: -1) }) {
                        Image(systemName: "chevron.left")
                    }
                    .buttonStyle(.borderless)
                    .help("Previous conformer")
                    .accessibilityLabel("Previous conformer")

                    Text("Conformer \(selectedConformerIndex + 1) of \(conformers.count)")
                        .font(DS.Font.secondary.weight(.medium))
                        .monospacedDigit()

                    Button(action: { stepConformer(by: 1) }) {
                        Image(systemName: "chevron.right")
                    }
                    .buttonStyle(.borderless)
                    .help("Next conformer")
                    .accessibilityLabel("Next conformer")

                    Spacer(minLength: DS.Space.s)

                    if selectedConformerIndex < conformers.count,
                       let minEnergy = conformers.map(\.energy).min() {
                        // Absolute force-field energies have no physical
                        // meaning on their own — report strain vs the lowest conformer.
                        let energy = conformers[selectedConformerIndex].energy
                        let delta = energy - minEnergy
                        Text(delta < 0.05 ? "lowest energy" : String(format: "ΔE +%.1f kcal/mol", delta))
                            .font(DS.Font.monoSmall)
                            .foregroundStyle(delta < 0.05 ? DS.Color.success : .secondary)
                            .help(String(format: "Force-field energy %.1f kcal/mol (relative to the lowest of %d conformers)",
                                         energy, conformers.count))
                    }
                }
                .font(DS.Font.secondary)
                .padding(.horizontal, DS.Space.m)
                .frame(height: DS.Size.rowHeight + DS.Space.xs)
                .background(.bar)
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
