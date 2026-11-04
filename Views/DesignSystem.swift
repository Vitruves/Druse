// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI
import AppKit

// MARK: - Tokens

enum DS {
    // MARK: Spacing (4-pt grid)
    enum Space {
        static let xxs: CGFloat = 2, xs: CGFloat = 4, s: CGFloat = 6, m: CGFloat = 8
        static let l: CGFloat = 12, xl: CGFloat = 16, xxl: CGFloat = 24, xxxl: CGFloat = 32
    }

    // MARK: Corner radii (always style: .continuous)
    enum Radius {
        static let xs: CGFloat = 4        // chips, tags, swatches
        static let s: CGFloat = 6         // buttons, fields, canvas tool buttons
        static let m: CGFloat = 10        // cards, embedded lists, popovers
        static let floating: CGFloat = 14 // glass clusters, HUD, legends
        /// Buttons nested in a floating cluster (floating radius minus the cluster's `xs` inset), so corners stay concentric.
        static let floatingInner: CGFloat = floating - Space.xs
        static let l: CGFloat = 20        // welcome cards / hero
    }

    // MARK: Sizes
    enum Size {
        static let toolButton: CGFloat = 28, toolIcon: CGFloat = 14
        static let stepBarButton: CGFloat = 30
        /// Step-number disc in the toolbar step bar.
        static let stepNumber: CGFloat = 18
        /// Removal / clear buttons on rows and chips: never smaller than this hit target.
        static let removeButton: CGFloat = 16
        static let rowHeight: CGFloat = 22, buttonHeight: CGFloat = 26, primaryButtonHeight: CGFloat = 30
        static let labelColumn: CGFloat = 96, valueColumn: CGFloat = 60, numberField: CGFloat = 72
        static let navigatorMin: CGFloat = 300, navigatorIdeal: CGFloat = 340, navigatorMax: CGFloat = 440
        static let inspectorMin: CGFloat = 240, inspectorIdeal: CGFloat = 280, inspectorMax: CGFloat = 380
        static let debugBar: CGFloat = 24
        /// Minimum window frame, toolbar included.
        static let windowMin = CGSize(width: 1180, height: 720)
        /// Height of the unified toolbar above the window content.
        static let toolbarHeight: CGFloat = 52
        static let windowDefault = CGSize(width: 1440, height: 900)
    }

    // MARK: Typography (semantic, macOS sizes in comments)
    enum Font {
        static let display      = SwiftUI.Font.largeTitle.weight(.semibold)               // 26
        static let stepTitle    = SwiftUI.Font.title3.weight(.semibold)                   // 15
        static let sectionTitle = SwiftUI.Font.body.weight(.semibold)                     // 13
        static let body         = SwiftUI.Font.body                                       // 13
        static let label        = SwiftUI.Font.callout                                    // 12  (panel rows)
        static let secondary    = SwiftUI.Font.subheadline                                // 11
        static let caption      = SwiftUI.Font.caption                                    // 10  (UI floor)
        static let eyebrow      = SwiftUI.Font.caption.weight(.semibold)                  // 10, used with .tracking(0.6)
        static let metric       = SwiftUI.Font.title2.weight(.semibold).monospacedDigit() // 17
        static let mono         = SwiftUI.Font.system(.callout, design: .monospaced)      // 12
        static let monoSmall    = SwiftUI.Font.system(.caption, design: .monospaced)      // 10
        static let canvasLabel  = SwiftUI.Font.system(size: 9, weight: .medium)           // ONLY inside zoomable canvases
    }

    // MARK: Opacity
    enum Opacity {
        /// Disabled controls drawn by hand (plain-style canvas buttons, choice tiles).
        static let disabled: Double = 0.32
    }

    // MARK: Motion
    enum Motion {
        static let snappy = Animation.snappy(duration: 0.22)
        static let panel  = Animation.smooth(duration: 0.25)
        /// Returns nil when Reduce Motion is on (use with withAnimation(DS.Motion.respecting(...))).
        static func respecting(_ reduceMotion: Bool, _ animation: Animation = panel) -> Animation? { reduceMotion ? nil : animation }
    }

    // MARK: Colors (all dynamic light/dark)
    enum Color {
        static let accent                 = SwiftUI.Color.accentColor
        /// Text/icons on a solid accent fill: white, or near-black when the effective accent is light
        /// (the app's teal in Dark mode). Follows the user's system accent colour.
        static let accentOnFill           = SwiftUI.Color(nsColor: NSColor(name: nil) { appearance in
            var luminance: CGFloat = 0
            appearance.performAsCurrentDrawingAppearance {
                let accent = NSColor.controlAccentColor.usingColorSpace(.sRGB) ?? .systemBlue
                luminance = 0.2126 * accent.redComponent + 0.7152 * accent.greenComponent + 0.0722 * accent.blueComponent
            }
            return luminance > 0.6 ? srgb(0x04262B, alpha: 1) : .white
        })
        /// Tinted background for selected tiles and toggled tools; always derived from the effective
        /// accent so it never disagrees with `accent` when the user picks a system accent colour.
        static let accentSoft             = SwiftUI.Color(nsColor: NSColor(name: nil) { appearance in
            let isDark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            var color = NSColor.controlAccentColor
            appearance.performAsCurrentDrawingAppearance {
                color = (NSColor.controlAccentColor.usingColorSpace(.sRGB) ?? .systemBlue)
                    .withAlphaComponent(isDark ? 0.22 : 0.13)
            }
            return color
        })
        static let cardFill               = dynamic(light: 0x000000, dark: 0xFFFFFF, lightAlpha: 0.035, darkAlpha: 0.05)
        static let hairline               = dynamic(light: 0x000000, dark: 0xFFFFFF, lightAlpha: 0.08, darkAlpha: 0.10)
        static let controlFill            = dynamic(light: 0x000000, dark: 0xFFFFFF, lightAlpha: 0.05, darkAlpha: 0.08)
        static let canvasBackground       = dynamic(light: 0xF2F4F6, dark: 0x0E1114)

        static let protein                = dynamic(light: 0x2F6FD1, dark: 0x6EA8FF)
        static let ligand                 = dynamic(light: 0xC2620A, dark: 0xFFA94D)
        static let pocket                 = dynamic(light: 0x7A4FD6, dark: 0xB392F0)
        static let water                  = dynamic(light: 0x0E8FB8, dark: 0x6CD4F5)
        static let metal                  = dynamic(light: 0x8A6D3B, dark: 0xD4B07A)

        static let success                = dynamic(light: 0x1A7F37, dark: 0x3FB950)
        static let warning                = dynamic(light: 0x9A6700, dark: 0xD29922)
        static let danger                 = dynamic(light: 0xCF222E, dark: 0xF85149)
        static let info                   = dynamic(light: 0x0969DA, dark: 0x58A6FF)
        static let scorePoor              = dynamic(light: 0xBC4C00, dark: 0xF0883E)

        static let interactionHBond       = dynamic(light: 0x2F6FD1, dark: 0x6EA8FF)
        static let interactionSaltBridge  = dynamic(light: 0xCF222E, dark: 0xFF7B72)
        static let interactionHydrophobic = dynamic(light: 0x7D7D3A, dark: 0xC9C97A)
        static let interactionPiStack     = dynamic(light: 0x2DA44E, dark: 0x56D364)
        static let interactionPiCation    = dynamic(light: 0xBF3989, dark: 0xF778BA)
        static let interactionHalogen     = dynamic(light: 0x0E8FB8, dark: 0x6CD4F5)
        static let interactionMetal       = dynamic(light: 0x8A6D3B, dark: 0xD4B07A)
        static let interactionClash       = dynamic(light: 0xE5534B, dark: 0xE5534B)

        static let diagramInk             = dynamic(light: 0x1C1C1E, dark: 0xF2F2F7)
        static let diagramPaper           = dynamic(light: 0xFFFFFF, dark: 0x1C1C1E)
    }

    /// Appearance-adaptive color resolved by AppKit for aqua / darkAqua.
    static func dynamic(light: UInt32, dark: UInt32, lightAlpha: CGFloat = 1, darkAlpha: CGFloat = 1) -> SwiftUI.Color {
        let lightColor = srgb(light, alpha: lightAlpha)
        let darkColor = srgb(dark, alpha: darkAlpha)
        return SwiftUI.Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? darkColor : lightColor
        })
    }

    /// Bridge for AppKit and CoreGraphics renderers.
    static func nsColor(_ c: SwiftUI.Color) -> NSColor { NSColor(c) }

    private static func srgb(_ hex: UInt32, alpha: CGFloat) -> NSColor {
        NSColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
                green: CGFloat((hex >> 8) & 0xFF) / 255,
                blue: CGFloat(hex & 0xFF) / 255,
                alpha: alpha)
    }
}

// MARK: - Tones, scores, step status

enum DSTone: Sendable {
    case neutral, accent, success, warning, danger, info, protein, ligand, pocket

    var color: Color {
        switch self {
        case .neutral: .secondary
        case .accent:  DS.Color.accent
        case .success: DS.Color.success
        case .warning: DS.Color.warning
        case .danger:  DS.Color.danger
        case .info:    DS.Color.info
        case .protein: DS.Color.protein
        case .ligand:  DS.Color.ligand
        case .pocket:  DS.Color.pocket
        }
    }
}

enum DSScore {
    /// Energy (kcal/mol or the method's energy unit; lower is better). Thresholds preserved from current UI.
    static func tone(energy: Float) -> DSTone { energy < -6 ? .success : energy < -3 ? .warning : .danger }
    /// Affinity (pKi/pKd; higher is better).
    static func tone(pK: Float) -> DSTone { pK > 6 ? .success : pK > 4 ? .warning : .danger }
}

enum DSStepStatus: Sendable { case blocked, ready, running, attention, done }

// MARK: - Buttons

struct DSPrimaryButtonStyle: ButtonStyle {
    var tint: Color
    var fullWidth: Bool

    init(tint: Color = DS.Color.accent, fullWidth: Bool = true) {
        self.tint = tint
        self.fullWidth = fullWidth
    }

    func makeBody(configuration: Configuration) -> some View {
        DSPrimaryButtonBody(configuration: configuration, tint: tint, fullWidth: fullWidth)
    }
}

private struct DSPrimaryButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let tint: Color
    let fullWidth: Bool
    @Environment(\.isEnabled) private var isEnabled

    var body: some View {
        configuration.label
            .font(DS.Font.sectionTitle)
            .lineLimit(1)
            .foregroundStyle(tint == DS.Color.accent ? DS.Color.accentOnFill : .white)
            .padding(.horizontal, DS.Space.l)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .frame(height: DS.Size.primaryButtonHeight)
            .background(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous).fill(tint))
            .brightness(configuration.isPressed ? -0.06 : 0)
            .opacity(isEnabled ? 1 : 0.4)
            .contentShape(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous))
    }
}

extension ButtonStyle where Self == DSPrimaryButtonStyle {
    static var dsPrimary: Self { DSPrimaryButtonStyle() }
}

struct DSSecondaryButtonStyle: ButtonStyle {
    var tint: Color?
    var fullWidth: Bool

    init(tint: Color? = nil, fullWidth: Bool = true) {
        self.tint = tint
        self.fullWidth = fullWidth
    }

    func makeBody(configuration: Configuration) -> some View {
        DSSecondaryButtonBody(configuration: configuration, tint: tint, fullWidth: fullWidth)
    }
}

private struct DSSecondaryButtonBody: View {
    let configuration: ButtonStyleConfiguration
    let tint: Color?
    let fullWidth: Bool
    @Environment(\.isEnabled) private var isEnabled
    @State private var isHovering = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        configuration.label
            .font(DS.Font.label)
            .lineLimit(1)
            .foregroundStyle(tint ?? .primary)
            .padding(.horizontal, DS.Space.m)
            .frame(maxWidth: fullWidth ? .infinity : nil)
            .frame(height: DS.Size.buttonHeight)
            .background(shape.fill(DS.Color.controlFill).opacity(isHovering && isEnabled ? 1.6 : 1))
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
            .brightness(configuration.isPressed ? -0.04 : 0)
            .opacity(isEnabled ? 1 : 0.4)
            .contentShape(shape)
            .onHover { isHovering = $0 }
    }
}

extension ButtonStyle where Self == DSSecondaryButtonStyle {
    static var dsSecondary: Self { DSSecondaryButtonStyle() }
}

// MARK: - Step header / scaffold / navigation

struct DSStepHeader<Accessory: View>: View {
    let tab: SidebarTab
    let summary: String?
    let accessory: Accessory

    init(tab: SidebarTab, summary: String? = nil, @ViewBuilder accessory: () -> Accessory) {
        self.tab = tab
        self.summary = summary
        self.accessory = accessory()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.xxs) {
            Text(tab.stepNumber.map { "STEP \($0)" } ?? "TOOL")
                .font(DS.Font.eyebrow)
                .tracking(0.6)
                .foregroundStyle(.tertiary)
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.m) {
                Text(tab.rawValue)
                    .font(DS.Font.stepTitle)
                Spacer(minLength: DS.Space.xs)
                accessory
            }
            Text(tab.subtitle)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
            // Live data, so it gets readable contrast (not `.tertiary`) and may wrap instead of clipping.
            if let summary {
                Text(summary)
                    .font(DS.Font.secondary.weight(.medium))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.top, DS.Space.xxs)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, DS.Space.l)
        .padding(.top, DS.Space.l)
        .padding(.bottom, DS.Space.m)
    }
}

extension DSStepHeader where Accessory == EmptyView {
    init(tab: SidebarTab, summary: String? = nil) {
        self.init(tab: tab, summary: summary) { EmptyView() }
    }
}

struct DSStepScaffold<Content: View, Footer: View>: View {
    let content: Content
    let footer: Footer

    init(@ViewBuilder content: () -> Content, @ViewBuilder footer: () -> Footer) {
        self.content = content()
        self.footer = footer()
    }

    var body: some View {
        if Footer.self == EmptyView.self {
            scrollContent
        } else {
            scrollContent.safeAreaInset(edge: .bottom, spacing: 0) {
                VStack(spacing: 0) {
                    Divider()
                    VStack(spacing: DS.Space.s) { footer }
                        .padding(DS.Space.l)
                }
                .background(.bar)
            }
        }
    }

    private var scrollContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DS.Space.m) { content }
                .padding(.horizontal, DS.Space.l)
                .padding(.vertical, DS.Space.m)
        }
        .scrollIndicators(.automatic)
        // Content scrolling under the pinned footer fades out instead of reading as clipped.
        .scrollEdgeEffectStyle(.soft, for: .vertical)
    }
}

extension DSStepScaffold where Footer == EmptyView {
    init(@ViewBuilder content: () -> Content) {
        self.init(content: content) { EmptyView() }
    }
}

struct DSNextStepButton: View {
    @Environment(AppViewModel.self) private var viewModel
    let tab: SidebarTab

    init(_ tab: SidebarTab) { self.tab = tab }

    var body: some View {
        Button {
            viewModel.workspace.requestedTab = tab
            viewModel.workspace.requestedTabToken &+= 1
        } label: {
            HStack(spacing: DS.Space.xs) {
                Text("Continue to \(tab.rawValue)")
                Image(systemName: "arrow.right")
            }
            .font(DS.Font.label)
            .foregroundStyle(DS.Color.accent)
        }
        .buttonStyle(.borderless)
        .help("Go to step: \(tab.rawValue)")
    }
}

// MARK: - Empty state

struct DSEmptyState<Actions: View>: View {
    let title: String
    let systemImage: String
    let description: String?
    let actions: Actions

    init(_ title: String, systemImage: String, description: String? = nil, @ViewBuilder actions: () -> Actions) {
        self.title = title
        self.systemImage = systemImage
        self.description = description
        self.actions = actions()
    }

    var body: some View {
        ContentUnavailableView {
            Label {
                Text(title)
            } icon: {
                Image(systemName: systemImage)
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.secondary)
            }
        } description: {
            if let description { Text(description) }
        } actions: {
            actions
        }
    }
}

extension DSEmptyState where Actions == EmptyView {
    init(_ title: String, systemImage: String, description: String? = nil) {
        self.init(title, systemImage: systemImage, description: description) { EmptyView() }
    }
}

// MARK: - Badges, tags, scores, metrics

struct DSStatusBadge: View {
    let status: DSStepStatus
    let size: CGFloat

    init(_ status: DSStepStatus, size: CGFloat = 8) {
        self.status = status
        self.size = size
    }

    var body: some View {
        Group {
            switch status {
            case .done:
                Image(systemName: "checkmark.circle.fill")
                    .resizable().scaledToFit()
                    .frame(width: size + 4, height: size + 4)
                    .foregroundStyle(DS.Color.success)
            case .ready:
                Circle().fill(DS.Color.accent).frame(width: size, height: size)
            case .blocked:
                Circle().strokeBorder(.tertiary, lineWidth: 1).frame(width: size, height: size)
            case .running:
                ProgressView().controlSize(.mini)
            case .attention:
                Image(systemName: "exclamationmark.circle.fill")
                    .resizable().scaledToFit()
                    .frame(width: size + 4, height: size + 4)
                    .foregroundStyle(DS.Color.warning)
            }
        }
        .accessibilityLabel(accessibilityText)
    }

    private var accessibilityText: String {
        switch status {
        case .done:      "Done"
        case .ready:     "Ready"
        case .blocked:   "Not available yet"
        case .running:   "Running"
        case .attention: "Needs attention"
        }
    }
}

struct DSTag: View {
    let text: String
    let systemImage: String?
    let monospaced: Bool
    private let tint: Color?

    init(_ text: String, tone: DSTone = .neutral, systemImage: String? = nil, monospaced: Bool = false) {
        self.text = text
        self.systemImage = systemImage
        self.monospaced = monospaced
        self.tint = tone == .neutral ? nil : tone.color
    }

    /// Arbitrary tint (e.g. data-palette colors); same chrome as a toned tag.
    init(_ text: String, tint: Color, systemImage: String? = nil, monospaced: Bool = false) {
        self.text = text
        self.systemImage = systemImage
        self.monospaced = monospaced
        self.tint = tint
    }

    var body: some View {
        HStack(spacing: DS.Space.xxs) {
            if let systemImage { Image(systemName: systemImage) }
            Text(text)
                .monospacedDigit()
                .lineLimit(1)
        }
        .font(monospaced ? DS.Font.monoSmall.weight(.medium) : DS.Font.caption.weight(.medium))
        .foregroundStyle(tint ?? .secondary)
        .padding(.horizontal, DS.Space.s)
        .padding(.vertical, DS.Space.xxs)
        .background(
            RoundedRectangle(cornerRadius: DS.Radius.xs, style: .continuous)
                .fill(tint.map { $0.opacity(0.14) } ?? DS.Color.controlFill)
        )
    }
}

struct DSScoreBadge: View {
    private let value: Float
    private let unit: String
    private let tone: DSTone

    init(energy: Float, unit: String) {
        self.value = energy
        self.unit = unit
        self.tone = DSScore.tone(energy: energy)
    }

    init(pK: Float, label: String = "pKi") {
        self.value = pK
        self.unit = label
        self.tone = DSScore.tone(pK: pK)
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: DS.Space.xxs) {
            Text(String(format: "%.1f", value))
                .font(DS.Font.label.weight(.semibold).monospacedDigit())
                .foregroundStyle(tone == .danger ? DS.Color.scorePoor : tone.color)
            Text(unit)
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(String(format: "%.1f", value)) \(unit), \(quality)")
    }

    private var quality: String {
        switch tone {
        case .success: "good"
        case .warning: "fair"
        default:       "weak"
        }
    }
}

struct DSMetric: View {
    let title: String
    let value: String
    let unit: String?
    let tone: DSTone

    init(_ title: String, value: String, unit: String? = nil, tone: DSTone = .neutral) {
        self.title = title
        self.value = value
        self.unit = unit
        self.tone = tone
    }

    var body: some View {
        VStack(spacing: DS.Space.xxs) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.xxs) {
                Text(value)
                    .font(DS.Font.metric)
                    .foregroundStyle(tone == .neutral ? Color.primary : tone.color)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                if let unit {
                    Text(unit)
                        .font(DS.Font.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text(title)
                .font(DS.Font.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
        .padding(DS.Space.m)
        .frame(maxWidth: .infinity)
        .background(RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous).fill(DS.Color.cardFill))
        .accessibilityElement(children: .combine)
    }
}

struct DSMetricGrid<Content: View>: View {
    let columns: Int
    let content: Content

    init(columns: Int = 2, @ViewBuilder content: () -> Content) {
        self.columns = columns
        self.content = content()
    }

    var body: some View {
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: DS.Space.s), count: max(1, columns)),
            spacing: DS.Space.s
        ) {
            content
        }
    }
}

// MARK: - Floating (Liquid Glass) surfaces

/// How `dsFloating` surfaces are drawn. Liquid Glass refracts what is behind its edges, which
/// smears atoms under wide surfaces such as legends; `.solid` draws an opaque material card instead.
enum DSFloatingStyle: Sendable { case glass, solid }

extension EnvironmentValues {
    @Entry var dsFloatingStyle: DSFloatingStyle = .glass
}

private struct DSFloatingModifier<S: Shape>: ViewModifier {
    let shape: S
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @Environment(\.dsFloatingStyle) private var style

    func body(content: Content) -> some View {
        if reduceTransparency || style == .solid {
            content
                .background(reduceTransparency ? AnyShapeStyle(.thickMaterial) : AnyShapeStyle(.regularMaterial), in: shape)
                .overlay(shape.stroke(DS.Color.hairline, lineWidth: 0.5))
                .shadow(color: .black.opacity(0.12), radius: 6, y: 2)
        } else {
            content.glassEffect(.regular, in: shape)
        }
    }
}

extension View {
    func dsFloating(cornerRadius: CGFloat = DS.Radius.floating) -> some View {
        modifier(DSFloatingModifier(shape: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)))
    }

    func dsFloatingCapsule() -> some View {
        modifier(DSFloatingModifier(shape: Capsule()))
    }

    /// Sets how the `dsFloating` surfaces inside this view are drawn (glass by default).
    func dsFloatingStyle(_ style: DSFloatingStyle) -> some View {
        environment(\.dsFloatingStyle, style)
    }

    /// Card chrome (cardFill + hairline, radius `m`) for custom containers.
    func dsCard(padding: CGFloat = DS.Space.l) -> some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.m, style: .continuous)
        return self
            .padding(padding)
            .background(shape.fill(DS.Color.cardFill))
            .overlay(shape.strokeBorder(DS.Color.hairline, lineWidth: 0.5))
    }
}

// MARK: - Canvas chips & tools

struct DSMoleculeChip: View {
    enum Kind { case protein, ligand, pocket }

    let kind: Kind
    let name: String
    let detail: String?
    let onClear: (() -> Void)?
    @State private var isHovering = false

    init(kind: Kind, name: String, detail: String? = nil, onClear: (() -> Void)? = nil) {
        self.kind = kind
        self.name = name
        self.detail = detail
        self.onClear = onClear
    }

    private var entityColor: Color {
        switch kind {
        case .protein: DS.Color.protein
        case .ligand:  DS.Color.ligand
        case .pocket:  DS.Color.pocket
        }
    }

    var body: some View {
        HStack(spacing: DS.Space.s) {
            Circle().fill(entityColor).frame(width: 8, height: 8)
            Text(name)
                .font(DS.Font.label.weight(.medium))
                .lineLimit(1)
            if let detail {
                Text(detail)
                    .font(DS.Font.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            if let onClear, isHovering {
                Button(action: onClear) {
                    Image(systemName: "xmark.circle.fill")
                        .font(DS.Font.secondary)
                        .foregroundStyle(.secondary)
                        .frame(width: DS.Size.removeButton, height: DS.Size.removeButton)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Remove \(name)")
                .accessibilityLabel("Remove \(name)")
            }
        }
        .padding(.horizontal, DS.Space.m)
        .padding(.vertical, DS.Space.xs)
        .dsFloatingCapsule()
        .onHover { isHovering = $0 }
    }
}

struct DSCanvasToolButton: View {
    let title: String
    let systemImage: String
    let isOn: Bool
    let help: String
    let action: () -> Void
    @State private var isHovering = false
    @Environment(\.isEnabled) private var isEnabled

    init(_ title: String, systemImage: String, isOn: Bool = false, help: String, action: @escaping () -> Void) {
        self.title = title
        self.systemImage = systemImage
        self.isOn = isOn
        self.help = help
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.system(size: DS.Size.toolIcon, weight: .medium))
                .dsCanvasToolChrome(isOn: isOn, isHovering: isHovering, isEnabled: isEnabled)
        }
        .buttonStyle(.plain)
        .onHover { isHovering = $0 }
        .help(help)
        .accessibilityLabel(title)
        .accessibilityAddTraits(isOn ? .isSelected : [])
    }
}

extension View {
    /// Shared look of canvas tool buttons (symbol or text glyph): concentric with the cluster,
    /// accent when on, hover fill, clearly dimmed when disabled.
    func dsCanvasToolChrome(isOn: Bool, isHovering: Bool, isEnabled: Bool) -> some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.floatingInner, style: .continuous)
        return self
            .foregroundStyle(isOn ? AnyShapeStyle(DS.Color.accent) : AnyShapeStyle(Color.primary.opacity(0.85)))
            .frame(width: DS.Size.toolButton, height: DS.Size.toolButton)
            .background(shape.fill(isOn ? DS.Color.accentSoft : (isHovering && isEnabled ? DS.Color.controlFill : Color.clear)))
            .opacity(isEnabled ? 1 : DS.Opacity.disabled)
            .contentShape(shape)
    }
}

/// Hairline between groups of a horizontal `DSCanvasToolCluster`.
struct DSCanvasToolDivider: View {
    var body: some View {
        Divider().frame(height: DS.Size.toolButton - DS.Space.m).padding(.horizontal, DS.Space.xxs)
    }
}

struct DSCanvasToolCluster<Content: View>: View {
    let axis: Axis
    let content: Content

    init(axis: Axis = .horizontal, @ViewBuilder content: () -> Content) {
        self.axis = axis
        self.content = content()
    }

    var body: some View {
        let layout = axis == .horizontal
            ? AnyLayout(HStackLayout(spacing: DS.Space.xxs))
            : AnyLayout(VStackLayout(spacing: DS.Space.xxs))
        GlassEffectContainer(spacing: DS.Space.xs) {
            layout { content }
                .padding(DS.Space.xs)
                .dsFloating()
        }
    }
}

// MARK: - Step mark

/// Step number in a disc whose colour carries the step's status, so status reads without a
/// separate dot: done = success, attention = warning, running = accent ring, ready = neutral,
/// blocked = outlined and dimmed. Used by the toolbar step bar.
struct DSStepMark: View {
    let number: Int
    let status: DSStepStatus?
    let isSelected: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Text("\(number)")
            .font(DS.Font.caption.weight(.bold))
            .monospacedDigit()
            .foregroundStyle(numberStyle)
            .frame(width: DS.Size.stepNumber, height: DS.Size.stepNumber)
            .background(Circle().fill(fill))
            .overlay {
                switch status {
                case .blocked:
                    Circle().strokeBorder(DS.Color.hairline, lineWidth: 1)
                case .running:
                    Circle()
                        .strokeBorder(DS.Color.accent, lineWidth: 1.5)
                        .phaseAnimator(reduceMotion ? [1.0] : [1.0, 0.3]) { ring, phase in ring.opacity(phase) }
                            animation: { _ in .easeInOut(duration: 0.8) }
                default:
                    EmptyView()
                }
            }
            .accessibilityHidden(true)
    }

    private var numberStyle: AnyShapeStyle {
        switch status {
        case .done:      AnyShapeStyle(DS.Color.success)
        case .attention: AnyShapeStyle(DS.Color.warning)
        case .running:   AnyShapeStyle(DS.Color.accent)
        case .blocked:   AnyShapeStyle(.tertiary)
        case .ready, nil: isSelected ? AnyShapeStyle(DS.Color.accent) : AnyShapeStyle(.primary)
        }
    }

    private var fill: AnyShapeStyle {
        switch status {
        case .done:      AnyShapeStyle(DS.Color.success.opacity(0.16))
        case .attention: AnyShapeStyle(DS.Color.warning.opacity(0.18))
        case .running:   AnyShapeStyle(DS.Color.accentSoft)
        case .blocked:   AnyShapeStyle(.clear)
        case .ready, nil: AnyShapeStyle(DS.Color.controlFill)
        }
    }
}

// MARK: - Secondary windows & sheets

struct DSWindowStatusBar<Leading: View, Trailing: View>: View {
    let leading: Leading
    let trailing: Trailing

    init(@ViewBuilder leading: () -> Leading, @ViewBuilder trailing: () -> Trailing) {
        self.leading = leading()
        self.trailing = trailing()
    }

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: DS.Space.m) {
                leading
                Spacer(minLength: DS.Space.m)
                trailing
            }
            .font(DS.Font.secondary)
            .monospacedDigit()
            .foregroundStyle(.secondary)
            .lineLimit(1)
            .padding(.horizontal, DS.Space.l)
            .frame(height: DS.Size.debugBar)
        }
        .background(.bar)
    }
}

struct DSSheetScaffold<Content: View, Footer: View>: View {
    let title: String
    let subtitle: String?
    let systemImage: String?
    let content: Content
    let footer: Footer

    init(title: String, subtitle: String? = nil, systemImage: String? = nil,
         @ViewBuilder content: () -> Content, @ViewBuilder footer: () -> Footer) {
        self.title = title
        self.subtitle = subtitle
        self.systemImage = systemImage
        self.content = content()
        self.footer = footer()
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .center, spacing: DS.Space.l) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.title3)
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(DS.Color.accent)
                }
                VStack(alignment: .leading, spacing: DS.Space.xxs) {
                    Text(title).font(DS.Font.stepTitle)
                    if let subtitle {
                        Text(subtitle)
                            .font(DS.Font.secondary)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(DS.Space.xl)

            content
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)

            Divider()
            HStack(spacing: DS.Space.m) {
                Spacer()
                footer
            }
            .padding(DS.Space.l)
        }
    }
}

struct DSSectionHeader<Accessory: View>: View {
    let title: String
    let count: Int?
    let accessory: Accessory

    init(_ title: String, count: Int? = nil, @ViewBuilder accessory: () -> Accessory) {
        self.title = title
        self.count = count
        self.accessory = accessory()
    }

    var body: some View {
        HStack(spacing: DS.Space.s) {
            Text(title).font(DS.Font.sectionTitle)
            if let count { DSTag("\(count)") }
            Spacer(minLength: DS.Space.m)
            accessory
        }
    }
}

extension DSSectionHeader where Accessory == EmptyView {
    init(_ title: String, count: Int? = nil) {
        self.init(title, count: count) { EmptyView() }
    }
}
