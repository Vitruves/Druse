// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0

import SwiftUI

// MARK: - Layout constants

/// Legacy panel constants; every value forwards to `DS` tokens.
enum PanelStyle {
    static let cardCornerRadius: CGFloat = DS.Radius.m
    static let cardPadding: CGFloat = DS.Space.l
    static let cardSpacing: CGFloat = DS.Space.m
    static let sectionSpacing: CGFloat = DS.Space.l
    static let rowSpacing: CGFloat = DS.Space.s
    static let labelColumnWidth: CGFloat = DS.Size.labelColumn
    static let valueColumnWidth: CGFloat = DS.Size.valueColumn
    static let numberFieldWidth: CGFloat = DS.Size.numberField
    static let buttonHeight: CGFloat = DS.Size.buttonHeight
    static let runButtonHeight: CGFloat = DS.Size.primaryButtonHeight
    static let cardFillOpacity: Double = 0.045
    static let chipFillOpacity: Double = 0.06

    static let titleFont: Font = DS.Font.sectionTitle
    static let bodyFont: Font = DS.Font.label
    static let smallFont: Font = DS.Font.secondary
    static let captionFont: Font = DS.Font.caption
    static let monoSmall: Font = DS.Font.monoSmall
    static let monoBody: Font = DS.Font.mono
}

/// Maps legacy low-contrast literals (.yellow / .cyan / .green) onto tone-safe tokens.
private func dsToneSafe(_ color: Color) -> Color {
    switch color {
    case .yellow: DS.Color.warning
    case .cyan:   DS.Color.accent
    case .green:  DS.Color.success
    default:      color
    }
}

// MARK: - Card

struct PanelCard<Content: View>: View {
    let title: String
    let icon: String
    let accessory: AnyView?
    @ViewBuilder let content: () -> Content

    init(_ title: String, icon: String,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.icon = icon
        self.accessory = nil
        self.content = content
    }

    init<A: View>(_ title: String, icon: String,
                  @ViewBuilder accessory: () -> A,
                  @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.icon = icon
        self.accessory = AnyView(accessory())
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            HStack(spacing: DS.Space.s) {
                Image(systemName: icon)
                    .font(.system(size: DS.Size.toolIcon, weight: .medium))
                    .symbolRenderingMode(.hierarchical)
                    .foregroundStyle(.secondary)
                    .frame(width: 16, height: 16)
                Text(title)
                    .font(DS.Font.sectionTitle)
                Spacer(minLength: DS.Space.xs)
                if let accessory { accessory }
            }
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .dsCard(padding: PanelStyle.cardPadding)
    }
}

// MARK: - Subheader (uppercase caption used inside a card)

struct PanelSubheader: View {
    let title: String
    var icon: String? = nil
    var body: some View {
        HStack(spacing: DS.Space.xs) {
            if let icon {
                Image(systemName: icon)
            }
            Text(title)
                .textCase(.uppercase)
                .tracking(0.6)
        }
        .font(DS.Font.eyebrow)
        .foregroundStyle(.tertiary)
    }
}

// MARK: - Slider row

/// Quantises slider writes to `step` (anchored at the range's lower bound) without
/// passing `step:` to `Slider`. On macOS `Slider(step:)` draws one tick mark per step,
/// which on ranges like ±100 Å at 0.5 Å collapses into a solid bar under the track.
func snappedSliderBinding(_ value: Binding<Float>, range: ClosedRange<Float>, step: Float) -> Binding<Float> {
    Binding(
        get: { value.wrappedValue },
        set: { raw in
            guard step > 0 else { value.wrappedValue = raw; return }
            let snapped = range.lowerBound + ((raw - range.lowerBound) / step).rounded() * step
            value.wrappedValue = min(max(snapped, range.lowerBound), range.upperBound)
        }
    )
}

struct PanelSliderRow: View {
    let label: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var step: Float = 0.01
    var format: (Float) -> String = { String(format: "%.2f", $0) }
    var labelWidth: CGFloat = PanelStyle.labelColumnWidth
    var valueWidth: CGFloat = PanelStyle.valueColumnWidth
    var valueColor: Color = .secondary
    var onChange: ((Float) -> Void)? = nil

    var body: some View {
        HStack(spacing: DS.Space.m) {
            Text(label)
                .font(DS.Font.label)
                .foregroundStyle(.secondary)
                .frame(width: labelWidth, alignment: .leading)
                .lineLimit(1)
            Slider(value: snappedSliderBinding($value, range: range, step: step), in: range)
                .controlSize(.small)
                .onChange(of: value) { _, newValue in onChange?(newValue) }
            Text(format(value))
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(valueColor)
                .frame(width: valueWidth, alignment: .trailing)
        }
    }
}

// MARK: - Single-axis slider (XYZ grid rows)

struct PanelAxisSlider: View {
    let axis: String
    @Binding var value: Float
    let range: ClosedRange<Float>
    var step: Float = 0.5
    var format: (Float) -> String = { String(format: "%.1f Å", $0) }
    var onChange: ((Float) -> Void)? = nil

    var body: some View {
        HStack(spacing: DS.Space.s) {
            Text(axis)
                .font(DS.Font.monoSmall.weight(.bold))
                .foregroundStyle(.secondary)
                .frame(width: 10, alignment: .trailing)
            Slider(value: snappedSliderBinding($value, range: range, step: step), in: range)
                .controlSize(.small)
                .onChange(of: value) { _, newValue in onChange?(newValue) }
            Text(format(value))
                .font(DS.Font.monoSmall)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 56, alignment: .trailing)
        }
    }
}

// MARK: - Number field

struct PanelNumberField: View {
    @Binding var value: Int
    var width: CGFloat = PanelStyle.numberFieldWidth
    var minimum: Int = 1

    var body: some View {
        TextField("", value: Binding(
            get: { value },
            set: { value = max(minimum, $0) }
        ), format: .number)
            .textFieldStyle(.roundedBorder)
            .controlSize(.small)
            .font(DS.Font.monoSmall)
            .monospacedDigit()
            .multilineTextAlignment(.trailing)
            .frame(width: width)
    }
}

// MARK: - LabeledRow (label left, content right)

struct PanelLabeledRow<Content: View>: View {
    let label: String
    var help: String? = nil
    @ViewBuilder let content: () -> Content

    init(_ label: String, help: String? = nil,
         @ViewBuilder content: @escaping () -> Content) {
        self.label = label
        self.help = help
        self.content = content
    }

    var body: some View {
        HStack(spacing: DS.Space.m) {
            Text(label)
                .font(DS.Font.label)
                .foregroundStyle(.secondary)
                .frame(minWidth: DS.Size.labelColumn, alignment: .leading)
            Spacer(minLength: DS.Space.xs)
            content()
        }
        .help(help ?? "")
    }
}

// MARK: - Choice button (segment-style)

struct PanelChoiceButton: View {
    let title: String
    var icon: String? = nil
    var badge: String? = nil
    let isSelected: Bool
    var isDisabled: Bool = false
    var help: String? = nil
    let action: () -> Void

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
        Button(action: action) {
            HStack(spacing: DS.Space.xs) {
                if let icon {
                    Image(systemName: icon)
                        .font(DS.Font.secondary.weight(.medium))
                }
                Text(title)
                    .font(DS.Font.label.weight(isSelected ? .semibold : .regular))
                    .lineLimit(1)
                if let badge {
                    DSTag(badge, tone: .info)
                }
            }
            .padding(.horizontal, DS.Space.xs)
            .frame(maxWidth: .infinity)
            .frame(height: PanelStyle.buttonHeight)
            .background(shape.fill(isSelected ? DS.Color.accentSoft : DS.Color.controlFill))
            .overlay(
                shape.strokeBorder(isSelected ? DS.Color.accent : DS.Color.hairline,
                                   lineWidth: isSelected ? 1 : 0.5)
            )
            .foregroundStyle(isSelected ? DS.Color.accent : Color.primary)
            .opacity(isDisabled ? 0.4 : 1.0)
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .disabled(isDisabled)
        .help(help ?? "")
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

// MARK: - Choice grid wrapper

struct PanelChoiceGrid<Content: View>: View {
    var columns: Int = 2
    var spacing: CGFloat = DS.Space.s
    @ViewBuilder let content: () -> Content

    var body: some View {
        LazyVGrid(
            columns: Array(repeating: GridItem(.flexible(), spacing: spacing), count: columns),
            spacing: spacing
        ) {
            content()
        }
    }
}

// MARK: - Toggle row

struct PanelToggleRow: View {
    let title: String
    var subtitle: String? = nil
    var icon: String? = nil
    @Binding var isOn: Bool

    var body: some View {
        HStack(alignment: .center, spacing: DS.Space.m) {
            if let icon {
                Image(systemName: icon)
                    .font(DS.Font.secondary)
                    .foregroundStyle(.secondary)
                    .frame(width: 14)
            }
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(DS.Font.label)
                if let subtitle {
                    Text(subtitle)
                        .font(DS.Font.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
            Toggle("", isOn: $isOn)
                .toggleStyle(.switch)
                .controlSize(.mini)
                .labelsHidden()
        }
    }
}

// MARK: - Hint / inline message

struct PanelHint: View {
    let text: String
    var icon: String? = nil
    var color: Color = .secondary
    var body: some View {
        let tint = dsToneSafe(color)
        HStack(alignment: .top, spacing: DS.Space.xs) {
            if let icon {
                Image(systemName: icon)
                    .font(DS.Font.caption)
                    .foregroundStyle(tint)
                    .padding(.top, 1)
            }
            Text(text)
                .font(DS.Font.secondary)
                .foregroundStyle(tint)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

// MARK: - Stat cell (dashboard-style number + caption)

struct PanelStat: View {
    let label: String
    let value: String
    var unit: String? = nil
    var color: Color = .primary

    var body: some View {
        let tint = dsToneSafe(color)
        VStack(spacing: DS.Space.xxs) {
            HStack(alignment: .firstTextBaseline, spacing: DS.Space.xxs) {
                Text(value)
                    .font(DS.Font.metric)
                    .foregroundStyle(tint)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                if let unit {
                    Text(unit)
                        .font(DS.Font.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text(label)
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

// MARK: - Run button (primary action)

struct PanelRunButton: View {
    let title: String
    var icon: String = "play.fill"
    var color: Color = .accentColor
    var isDisabled: Bool = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: DS.Space.s) {
                Image(systemName: icon)
                Text(title)
            }
        }
        .buttonStyle(DSPrimaryButtonStyle(tint: color))
        .disabled(isDisabled)
    }
}

// MARK: - Secondary button (compact, neutral)

struct PanelSecondaryButton: View {
    let title: String
    var icon: String? = nil
    var tint: Color? = nil
    var isDisabled: Bool = false
    var help: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: DS.Space.xs) {
                if let icon {
                    Image(systemName: icon)
                        .font(DS.Font.secondary.weight(.medium))
                }
                Text(title)
            }
        }
        .buttonStyle(DSSecondaryButtonStyle(tint: tint))
        .disabled(isDisabled)
        .help(help ?? "")
    }
}

// MARK: - Labeled divider (uppercase label with thin horizontal line)

/// Thin horizontal rule with an uppercase caption label on the left.
/// Used to separate non-essential / advanced sections inside a card without
/// hiding them behind a disclosure.
struct PanelLabeledDivider: View {
    let title: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: DS.Space.s) {
            if let icon {
                Image(systemName: icon)
            }
            Text(title)
                .textCase(.uppercase)
                .tracking(0.6)
                .fixedSize()
            Rectangle()
                .fill(DS.Color.hairline)
                .frame(height: 0.5)
        }
        .font(DS.Font.eyebrow)
        .foregroundStyle(.tertiary)
        .padding(.top, DS.Space.xxs)
    }
}

// MARK: - Disclosure section (collapsible group inside a card)

struct PanelDisclosure<Content: View>: View {
    let title: String
    var icon: String? = nil
    @Binding var isExpanded: Bool
    @ViewBuilder let content: () -> Content
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(_ title: String,
         icon: String? = nil,
         isExpanded: Binding<Bool>,
         @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.icon = icon
        self._isExpanded = isExpanded
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: DS.Space.m) {
            Button {
                withAnimation(DS.Motion.respecting(reduceMotion, DS.Motion.snappy)) { isExpanded.toggle() }
            } label: {
                HStack(spacing: DS.Space.xs) {
                    Image(systemName: "chevron.right")
                        .font(DS.Font.caption.weight(.semibold))
                        .foregroundStyle(.tertiary)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0))
                    if let icon {
                        Image(systemName: icon)
                            .font(DS.Font.secondary)
                            .foregroundStyle(.secondary)
                    }
                    Text(title)
                        .font(DS.Font.label.weight(.medium))
                    Spacer()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityValue(isExpanded ? "Expanded" : "Collapsed")
            if isExpanded {
                content()
            }
        }
    }
}

// MARK: - Highlight box (used for batch ligand info, single-ligand chip, etc.)

struct PanelHighlightRow<Content: View>: View {
    let color: Color
    @ViewBuilder let content: () -> Content

    var body: some View {
        content()
            .padding(.horizontal, DS.Space.m)
            .padding(.vertical, DS.Space.s)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: DS.Radius.s, style: .continuous)
                    .fill(color.opacity(0.10))
            )
    }
}

// MARK: - Tag chip (residue names, etc.)

struct PanelChip: View {
    let text: String
    var color: Color = .accentColor
    var body: some View {
        DSTag(text, tint: dsToneSafe(color), monospaced: true)
    }
}

// MARK: - Requirement label (red x + text)

struct PanelRequirement: View {
    let text: String
    var body: some View {
        HStack(spacing: DS.Space.xs) {
            Image(systemName: "xmark.circle.fill")
                .font(DS.Font.secondary)
                .foregroundStyle(DS.Color.danger)
            Text(text)
                .font(DS.Font.secondary)
                .foregroundStyle(.secondary)
        }
    }
}
