import SwiftUI
import MetalKit

// MARK: - MetalView (SwiftUI Integration)

struct MetalView: NSViewRepresentable {
    let renderer: Renderer
    var onAtomSelected: ((Int?, Bool) -> Void)?  // (atomIndex, isOptionClick)
    var onAtomDoubleClicked: ((Int) -> Void)?
    var onRenderModeChanged: ((RenderMode) -> Void)?
    var onToggleHydrogens: (() -> Void)?
    var onToggleLighting: (() -> Void)?

    // Box selection callback: (atomIndices, addToSelection)
    var onBoxSelection: (([Int], Bool) -> Void)?

    // Ribbon residue selection callback: (proteinAtomID, isOptionClick)
    var onRibbonResidueSelected: ((Int, Bool) -> Void)?

    // Fit-to-view callback (context-aware)
    var onFitToView: (() -> Void)?

    // Context menu callbacks
    var onContextMenu: ((NSEvent, NSView) -> Void)?

    // Deselect all (Escape key)
    var onDeselectAll: (() -> Void)?

    func makeNSView(context: Context) -> MetalMTKView {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        let mtkView = MetalMTKView(frame: .zero, device: device)
        mtkView.delegate = context.coordinator
        mtkView.coordinator = context.coordinator

        // Configure for molecular rendering
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.sampleCount = 4 // 4x MSAA
        // Match renderer theme so the very first frame doesn't flash dark on light-theme relaunch
        if renderer.themeMode == 1 {
            mtkView.clearColor = MTLClearColor(red: 0.96, green: 0.96, blue: 0.98, alpha: 1.0)
        } else {
            mtkView.clearColor = MTLClearColor(red: 0.08, green: 0.09, blue: 0.12, alpha: 1.0)
        }
        mtkView.preferredFramesPerSecond = 60
        mtkView.isPaused = true
        mtkView.enableSetNeedsDisplay = true

        // Give the renderer a weak ref so it can request on-demand redraws
        renderer.mtkView = mtkView

        // Accept touch events (modern API)
        mtkView.allowedTouchTypes = .indirect

        // Trigger the initial frame
        mtkView.needsDisplay = true

        return mtkView
    }

    func updateNSView(_ nsView: MetalMTKView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        let coord = Coordinator(renderer: renderer, onAtomSelected: onAtomSelected, onAtomDoubleClicked: onAtomDoubleClicked)
        coord.onRenderModeChanged = onRenderModeChanged
        coord.onToggleHydrogens = onToggleHydrogens
        coord.onToggleLighting = onToggleLighting
        coord.onBoxSelection = onBoxSelection
        coord.onRibbonResidueSelected = onRibbonResidueSelected
        coord.onFitToView = onFitToView
        coord.onContextMenu = onContextMenu
        coord.onDeselectAll = onDeselectAll
        return coord
    }

    // MARK: - Coordinator

    @MainActor
    class Coordinator: NSObject, MTKViewDelegate {
        let renderer: Renderer
        var onAtomSelected: ((Int?, Bool) -> Void)?
        var onAtomDoubleClicked: ((Int) -> Void)?

        private var mouseDownLocation: NSPoint = .zero
        private var isDragging = false
        private var isShiftDrag = false
        private let dragThreshold: CGFloat = 3.0

        // Box selection state
        private var isBoxSelecting = false
        private var boxStartPoint: SIMD2<Float> = .zero
        private var boxCurrentPoint: SIMD2<Float> = .zero
        private var selectionOverlay: NSView?

        init(renderer: Renderer, onAtomSelected: ((Int?, Bool) -> Void)?, onAtomDoubleClicked: ((Int) -> Void)?) {
            self.renderer = renderer
            self.onAtomSelected = onAtomSelected
            self.onAtomDoubleClicked = onAtomDoubleClicked
        }

        nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            MainActor.assumeIsolated {
                renderer.camera.aspectRatio = Float(size.width / size.height)
            }
        }

        nonisolated func draw(in view: MTKView) {
            MainActor.assumeIsolated {
                renderer.draw(in: view)
            }
        }

        // MARK: - Mouse Events

        func handleMouseDown(_ event: NSEvent, in view: NSView) {
            let loc = view.convert(event.locationInWindow, from: nil)
            mouseDownLocation = loc
            isDragging = false

            let hasShift = event.modifierFlags.contains(.shift)
            let hasOption = event.modifierFlags.contains(.option)
            // Option+drag => box selection
            if hasOption {
                isBoxSelecting = true
                isShiftDrag = false
                let screenPt = screenPoint(from: loc, in: view)
                boxStartPoint = screenPt
                boxCurrentPoint = screenPt
                showSelectionOverlay(in: view, from: loc, to: loc)
                return
            }

            // Shift+drag => pan (translate)
            if hasShift {
                isShiftDrag = true
                isBoxSelecting = false
                let screenPt = screenPoint(from: loc, in: view)
                renderer.camera.beginPan(at: screenPt)
                return
            }

            // Plain drag => orbit rotation, Ctrl+drag => Z-roll (spin around view axis)
            let hasControl = event.modifierFlags.contains(.control)
            isShiftDrag = false
            isBoxSelecting = false
            let screenPt = screenPoint(from: loc, in: view)
            renderer.camera.beginRotation(at: screenPt, zRoll: hasControl)
        }

        func handleMouseDragged(_ event: NSEvent, in view: NSView) {
            let loc = view.convert(event.locationInWindow, from: nil)
            let distance = hypot(loc.x - mouseDownLocation.x, loc.y - mouseDownLocation.y)
            if distance > dragThreshold { isDragging = true }

            if isBoxSelecting {
                let screenPt = screenPoint(from: loc, in: view)
                boxCurrentPoint = screenPt
                updateSelectionOverlay(in: view, from: mouseDownLocation, to: loc)
                return
            }

            if isShiftDrag {
                let screenPt = screenPoint(from: loc, in: view)
                let vpSize = viewportSize(of: view)
                renderer.camera.updatePan(to: screenPt, viewportSize: vpSize)
                return
            }

            let screenPt = screenPoint(from: loc, in: view)
            let vpSize = viewportSize(of: view)
            renderer.camera.updateRotation(to: screenPt, viewportSize: vpSize)
        }

        func handleMouseUp(_ event: NSEvent, in view: NSView) {
            if isBoxSelecting {
                isBoxSelecting = false
                removeSelectionOverlay()

                if isDragging {
                    // Complete box selection
                    let vpSize = viewportSize(of: view)
                    let minX = min(boxStartPoint.x, boxCurrentPoint.x)
                    let maxX = max(boxStartPoint.x, boxCurrentPoint.x)
                    let minY = min(boxStartPoint.y, boxCurrentPoint.y)
                    let maxY = max(boxStartPoint.y, boxCurrentPoint.y)

                    let rectMin = SIMD2<Float>(minX, minY)
                    let rectMax = SIMD2<Float>(maxX, maxY)
                    let atomIndices = renderer.atomsInRect(rectMin: rectMin, rectMax: rectMax, viewportSize: vpSize)

                    let addToExisting = event.modifierFlags.contains(.shift)
                    onBoxSelection?(atomIndices, addToExisting)

                    isDragging = false
                    return
                }
                // Option+click without drag → fall through to single-pick logic
                // so that option-click toggles selection in ribbon mode (and BAS).
            }

            if isShiftDrag {
                renderer.camera.endPan()
                isShiftDrag = false
                isDragging = false
                return
            }

            renderer.camera.endRotation()

            if !isDragging {
                let loc = view.convert(event.locationInWindow, from: nil)
                let screenPt = screenPoint(from: loc, in: view)
                let vpSize = viewportSize(of: view)
                let pickedAtom = renderer.pickAtom(at: screenPt, viewportSize: vpSize)
                let isOption = event.modifierFlags.contains(.option)

                if event.clickCount == 2, let idx = pickedAtom {
                    onAtomDoubleClicked?(idx)
                } else if let idx = pickedAtom {
                    onAtomSelected?(idx, isOption)
                } else if let ribbonAtomID = renderer.pickRibbonResidue(at: screenPt, viewportSize: vpSize) {
                    // Ribbon mode: picked a CA control point, select the residue
                    onRibbonResidueSelected?(ribbonAtomID, isOption)
                } else {
                    onAtomSelected?(nil, isOption)
                }
            }
            isDragging = false
        }

        func handleRightMouseDown(_ event: NSEvent, in view: NSView) {
            onContextMenu?(event, view)
        }

        func handleRightMouseDragged(_ event: NSEvent, in view: NSView) {}

        func handleRightMouseUp(_ event: NSEvent, in view: NSView) {}

        func handleMiddleMouseDown(_ event: NSEvent, in view: NSView) {
            let loc = view.convert(event.locationInWindow, from: nil)
            renderer.camera.beginPan(at: screenPoint(from: loc, in: view))
        }

        func handleMiddleMouseDragged(_ event: NSEvent, in view: NSView) {
            let loc = view.convert(event.locationInWindow, from: nil)
            renderer.camera.updatePan(to: screenPoint(from: loc, in: view), viewportSize: viewportSize(of: view))
        }

        func handleMiddleMouseUp(_ event: NSEvent, in view: NSView) {
            renderer.camera.endPan()
        }

        func handleScrollWheel(_ event: NSEvent, in view: NSView) {
            if event.momentumPhase != [] || abs(event.scrollingDeltaX) > abs(event.scrollingDeltaY) * 0.5 {
                let sensitivity: Float = 0.02 * renderer.camera.distance / 15.0
                let dx = Float(event.scrollingDeltaX) * sensitivity
                let dy = Float(event.scrollingDeltaY) * sensitivity
                renderer.camera.target -= renderer.camera.rightVector * dx
                renderer.camera.target += renderer.camera.upVector * dy
            } else {
                renderer.camera.zoom(by: Float(event.scrollingDeltaY) * 0.5)
            }
        }

        func handleMagnify(_ event: NSEvent, in view: NSView) {
            renderer.camera.pinchZoom(magnification: 1.0 + Float(event.magnification))
        }

        var onRenderModeChanged: ((RenderMode) -> Void)?
        var onToggleHydrogens: (() -> Void)?
        var onToggleLighting: (() -> Void)?
        var onBoxSelection: (([Int], Bool) -> Void)?
        var onRibbonResidueSelected: ((Int, Bool) -> Void)?
        var onFitToView: (() -> Void)?
        var onContextMenu: ((NSEvent, NSView) -> Void)?

        var onDeselectAll: (() -> Void)?

        func handleKeyDown(_ event: NSEvent) {
            // Escape key
            if event.keyCode == 53 {
                onDeselectAll?()
                return
            }
            switch event.charactersIgnoringModifiers {
            case " ":
                onFitToView?()
            case "1":
                onRenderModeChanged?(.ballAndStick)
            case "2":
                onRenderModeChanged?(.spaceFilling)
            case "3":
                onRenderModeChanged?(.wireframe)
            case "4":
                onRenderModeChanged?(.ribbon)
            case "h", "H":
                onToggleHydrogens?()
            case "l", "L":
                onToggleLighting?()
            default:
                break
            }
        }

        // MARK: - Selection Overlay

        private func showSelectionOverlay(in view: NSView, from startLoc: NSPoint, to currentLoc: NSPoint) {
            if selectionOverlay == nil {
                let overlay = SelectionRectView(frame: .zero)
                view.addSubview(overlay)
                selectionOverlay = overlay
            }
            updateOverlayFrame(in: view, from: startLoc, to: currentLoc)
        }

        private func updateSelectionOverlay(in view: NSView, from startLoc: NSPoint, to currentLoc: NSPoint) {
            updateOverlayFrame(in: view, from: startLoc, to: currentLoc)
        }

        private func updateOverlayFrame(in view: NSView, from startLoc: NSPoint, to currentLoc: NSPoint) {
            let minX = min(startLoc.x, currentLoc.x)
            let minY = min(startLoc.y, currentLoc.y)
            let width = abs(currentLoc.x - startLoc.x)
            let height = abs(currentLoc.y - startLoc.y)
            selectionOverlay?.frame = NSRect(x: minX, y: minY, width: width, height: height)
        }

        private func removeSelectionOverlay() {
            selectionOverlay?.removeFromSuperview()
            selectionOverlay = nil
        }

        // MARK: - Coordinate Conversion

        private func screenPoint(from loc: NSPoint, in view: NSView) -> SIMD2<Float> {
            let scale = Float(view.window?.backingScaleFactor ?? 2.0)
            return SIMD2<Float>(Float(loc.x) * scale, Float(view.bounds.height - loc.y) * scale)
        }

        private func viewportSize(of view: NSView) -> SIMD2<Float> {
            let scale = Float(view.window?.backingScaleFactor ?? 2.0)
            return SIMD2<Float>(Float(view.bounds.width) * scale, Float(view.bounds.height) * scale)
        }
    }
}

// MARK: - Selection Rectangle Overlay View

private class SelectionRectView: NSView {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.backgroundColor = NSColor(calibratedRed: 0.3, green: 0.6, blue: 1.0, alpha: 0.15).cgColor
        layer?.borderColor = NSColor(calibratedRed: 0.3, green: 0.6, blue: 1.0, alpha: 0.7).cgColor
        layer?.borderWidth = 1.5
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

// MARK: - MetalMTKView (NSView subclass for event handling)

class MetalMTKView: MTKView {
    weak var coordinator: MetalView.Coordinator?

    override var acceptsFirstResponder: Bool { true }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    /// Request a redraw after any user interaction that changes the scene.
    /// Calls `draw()` directly instead of `needsDisplay = true` so the frame
    /// is rendered immediately on the main thread, bypassing the per-window
    /// display link. CADisplayLink in secondary `Window` scenes (e.g. the
    /// Ligand Database window) is sometimes throttled or slow to fire,
    /// causing multi-second drag latency in on-demand mode.
    private func requestRedraw() {
        draw()
    }

    override func mouseDown(with event: NSEvent) {
        coordinator?.handleMouseDown(event, in: self)
        requestRedraw()
    }

    override func mouseDragged(with event: NSEvent) {
        coordinator?.handleMouseDragged(event, in: self)
        requestRedraw()
    }

    override func mouseUp(with event: NSEvent) {
        coordinator?.handleMouseUp(event, in: self)
        requestRedraw()
    }

    override func rightMouseDown(with event: NSEvent) {
        coordinator?.handleRightMouseDown(event, in: self)
    }

    override func rightMouseDragged(with event: NSEvent) {
        coordinator?.handleRightMouseDragged(event, in: self)
    }

    override func rightMouseUp(with event: NSEvent) {
        coordinator?.handleRightMouseUp(event, in: self)
    }

    override func otherMouseDown(with event: NSEvent) {
        coordinator?.handleMiddleMouseDown(event, in: self)
        requestRedraw()
    }

    override func otherMouseDragged(with event: NSEvent) {
        coordinator?.handleMiddleMouseDragged(event, in: self)
        requestRedraw()
    }

    override func otherMouseUp(with event: NSEvent) {
        coordinator?.handleMiddleMouseUp(event, in: self)
        requestRedraw()
    }

    override func scrollWheel(with event: NSEvent) {
        coordinator?.handleScrollWheel(event, in: self)
        requestRedraw()
    }

    override func magnify(with event: NSEvent) {
        coordinator?.handleMagnify(event, in: self)
        requestRedraw()
    }

    override func keyDown(with event: NSEvent) {
        coordinator?.handleKeyDown(event)
        requestRedraw()
    }

    override func becomeFirstResponder() -> Bool { true }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        window?.makeFirstResponder(self)
    }
}
