// SPDX-FileCopyrightText: 2026 Johan H.G. Natter
// SPDX-License-Identifier: Apache-2.0
//
// Renders the Druse "crystal pocket" app icon natively at every AppIcon size.
//
//   swift Tools/render_icon.swift Resources/Assets.xcassets/AppIcon.appiconset [Resources/Branding/druse-icon.svg]
//
// Geometry lives on a 1024² canvas (origin top-left, y down). Angles are clockwise from 12 o'clock and
// R(φ) = (sin φ, −cos φ). Each size is drawn by scaling the CTM — nothing is resampled — and the small
// sizes use hand-tuned simplified variants. The optional second argument also writes the SVG master.

import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Geometry

let center = CGPoint(x: 512, y: 532)
let bodyRect = CGRect(x: 100, y: 100, width: 824, height: 824)

/// Tip radius per slot (degrees). 45° is skipped: that gap is the pocket mouth.
let tipRadius: [Int: CGFloat] = [0: 160, 90: 176, 135: 164, 180: 150, 225: 170, 270: 158, 315: 172]
let fullSlots = [270, 315, 0, 90, 135, 180, 225]
let smallSlots = [270, 0, 90, 135, 180]
let upperSlots: Set<Int> = [270, 315, 0, 90]
/// Wall gaps between primaries, excluding the two flanking the mouth.
let secondarySlots: [CGFloat] = [112.5, 157.5, 202.5, 247.5, 292.5, 337.5]

func dir(_ degrees: CGFloat) -> CGPoint {
    let r = degrees * .pi / 180
    return CGPoint(x: sin(r), y: -cos(r))
}

func polar(_ degrees: CGFloat, _ radius: CGFloat, from origin: CGPoint = center) -> CGPoint {
    let d = dir(degrees)
    return CGPoint(x: origin.x + d.x * radius, y: origin.y + d.y * radius)
}

/// Kite-shaped crystal pointing at the pocket centre, split along back→tip into two facets.
struct Shard {
    let theta: CGFloat
    let tip: CGPoint, back: CGPoint, outerLeft: CGPoint, outerRight: CGPoint
    let colors: (lit: UInt32, shaded: UInt32)

    init(theta: CGFloat, tipRadius tipR: CGFloat, length: CGFloat, halfAngle: CGFloat, shoulder: CGFloat,
         colors: (lit: UInt32, shaded: UInt32)) {
        let backR = tipR + length
        self.theta = theta
        self.colors = colors
        tip = polar(theta, tipR)
        back = polar(theta, backR)
        outerLeft = polar(theta - halfAngle, backR - shoulder)
        outerRight = polar(theta + halfAngle, backR - shoulder)
    }

    /// One of the seven pocket-lining crystals.
    static func primary(_ slot: Int) -> Shard {
        let upper = upperSlots.contains(slot)
        return Shard(theta: CGFloat(slot), tipRadius: tipRadius[slot]!, length: 184, halfAngle: 19, shoulder: 34,
                     colors: upper ? (Palette.upperLeft, Palette.upperRight) : (Palette.lowerLeft, Palette.lowerRight))
    }

    /// Smaller, recessed crystal filling the wall between two primaries (large renditions only).
    static func secondary(_ theta: CGFloat) -> Shard {
        Shard(theta: theta, tipRadius: 214, length: 100, halfAngle: 7.5, shoulder: 20,
              colors: (Palette.recessedLit, Palette.recessedShaded))
    }

    var outline: [CGPoint] { [back, outerLeft, tip, outerRight] }

    /// True when the facet on the outerLeft side faces the light.
    var leftFacetIsLit: Bool {
        let side = dir(theta - 90)
        return side.x * toLight.x + side.y * toLight.y > 0
    }

    var leftColor: UInt32 { leftFacetIsLit ? colors.lit : colors.shaded }
    var rightColor: UInt32 { leftFacetIsLit ? colors.shaded : colors.lit }
}

func hexagonVertices(radius: CGFloat) -> [CGPoint] {
    stride(from: 30, to: 360, by: 60).map { polar(CGFloat($0), radius) }
}

func quadBezier(_ p0: CGPoint, _ c: CGPoint, _ p1: CGPoint, _ t: CGFloat) -> CGPoint {
    let u = 1 - t
    return CGPoint(x: u * u * p0.x + 2 * u * t * c.x + t * t * p1.x,
                   y: u * u * p0.y + 2 * u * t * c.y + t * t * p1.y)
}

let trajectoryStart = polar(45, 330)
let trajectoryControl = polar(60, 250)
let trajectoryEnd = polar(45, 120)
/// Six dots easing in from the mouth: they grow (r 4 → 7) and brighten (α 0.15 → 0.7) as they approach the pocket.
let trajectoryDots: [(point: CGPoint, radius: CGFloat, alpha: CGFloat)] = (0..<6).map { i in
    let t = CGFloat(i) / 5
    return (quadBezier(trajectoryStart, trajectoryControl, trajectoryEnd, t), 4 + 3 * t, 0.15 + 0.55 * t)
}

enum PathOp {
    case move(CGPoint), line(CGPoint), curve(CGPoint, CGPoint, CGPoint)
}

/// Continuous-curvature ("squircle") rounded rect — the corner profile of Apple's icon grid.
/// Each corner blends over 1.5287·r of its edges; offsets are (along-edge back from the corner, along the next edge).
func continuousRoundedRect(_ rect: CGRect, radius r: CGFloat) -> [PathOp] {
    let corners: [(corner: CGPoint, e1: CGPoint, e2: CGPoint)] = [
        (CGPoint(x: rect.maxX, y: rect.minY), CGPoint(x: 1, y: 0), CGPoint(x: 0, y: 1)),
        (CGPoint(x: rect.maxX, y: rect.maxY), CGPoint(x: 0, y: 1), CGPoint(x: -1, y: 0)),
        (CGPoint(x: rect.minX, y: rect.maxY), CGPoint(x: -1, y: 0), CGPoint(x: 0, y: -1)),
        (CGPoint(x: rect.minX, y: rect.minY), CGPoint(x: 0, y: -1), CGPoint(x: 1, y: 0)),
    ]
    func at(_ c: (corner: CGPoint, e1: CGPoint, e2: CGPoint), _ u: CGFloat, _ v: CGFloat) -> CGPoint {
        CGPoint(x: c.corner.x - u * r * c.e1.x + v * r * c.e2.x, y: c.corner.y - u * r * c.e1.y + v * r * c.e2.y)
    }
    var ops: [PathOp] = [.move(at(corners[3], 0, 1.52866483))]
    for c in corners {
        ops.append(.line(at(c, 1.52866483, 0)))
        ops.append(.curve(at(c, 1.08849323, 0), at(c, 0.86840689, 0), at(c, 0.63149399, 0.07491100)))
        ops.append(.curve(at(c, 0.37282392, 0.16905899), at(c, 0.16905899, 0.37282392), at(c, 0.07491100, 0.63149399)))
        ops.append(.curve(at(c, 0, 0.86840689), at(c, 0, 1.08849323), at(c, 0, 1.52866483)))
    }
    return ops
}

let bodyRadius: CGFloat = 185.4
let bodyOps = continuousRoundedRect(bodyRect, radius: bodyRadius)

/// Facet lighting: light falls from the upper left, so the facet whose side faces it takes the bright colour.
let toLight = CGPoint(x: -0.6, y: -0.8)

// MARK: - Variants

struct Variant {
    let slots: [Int]
    let secondary: Bool
    let faceted: Bool
    let glint: Bool
    let trajectory: Bool
    let hexScale: CGFloat
    let innerRing: Bool
    let minRingPx: CGFloat
    let snap: Bool

    static func forPixels(_ px: Int) -> Variant {
        switch px {
        case ..<48:
            return Variant(slots: smallSlots, secondary: false, faceted: false, glint: false, trajectory: false,
                           hexScale: 1.25, innerRing: false, minRingPx: 0, snap: true)
        case ..<160:
            return Variant(slots: fullSlots, secondary: false, faceted: true, glint: false, trajectory: false,
                           hexScale: 1, innerRing: true, minRingPx: 1.5, snap: false)
        default:
            return Variant(slots: fullSlots, secondary: true, faceted: true, glint: true, trajectory: true,
                           hexScale: 1, innerRing: true, minRingPx: 0, snap: false)
        }
    }
}

// MARK: - Colour

func rgb(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255,
            alpha: alpha)
}

let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

func gradient(_ stops: [(CGColor, CGFloat)]) -> CGGradient {
    CGGradient(colorsSpace: srgb, colors: stops.map(\.0) as CFArray, locations: stops.map(\.1))!
}

enum Palette {
    static let bodyTop: UInt32 = 0x12384A
    static let bodyBottom: UInt32 = 0x0B2530
    static let bodyGlow: UInt32 = 0x1E6A7F
    static let upperLeft: UInt32 = 0x5FD3DF
    static let upperRight: UInt32 = 0x2A9BB0
    static let lowerLeft: UInt32 = 0x4AA6AF
    static let lowerRight: UInt32 = 0x21798A
    static let recessedLit: UInt32 = 0x2C8C9C
    static let recessedShaded: UInt32 = 0x195A68
    static let glint: UInt32 = 0xBFF3F8
    static let ligandLight: UInt32 = 0xFFC56B
    static let ligandDark: UInt32 = 0xF08A24
    static let ligandGlow: UInt32 = 0xFFA94D
    static let ring: UInt32 = 0xFFF1D6
}

// MARK: - Raster rendering

final class IconRenderer {
    let scale: CGFloat
    let variant: Variant
    let ctx: CGContext

    init(px: Int) {
        scale = CGFloat(px) / 1024
        variant = .forPixels(px)
        ctx = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0, space: srgb,
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.setShouldAntialias(true)
        // y-down, 1024-unit canvas.
        ctx.translateBy(x: 0, y: CGFloat(px))
        ctx.scaleBy(x: scale, y: -scale)
    }

    /// Rounds a canvas point to the device pixel grid (small variants only).
    func snapped(_ p: CGPoint) -> CGPoint {
        guard variant.snap else { return p }
        return CGPoint(x: (p.x * scale).rounded() / scale, y: (p.y * scale).rounded() / scale)
    }

    func polygon(_ points: [CGPoint]) -> CGPath {
        let path = CGMutablePath()
        path.addLines(between: points.map(snapped))
        path.closeSubpath()
        return path
    }

    /// Small variants grow the body outwards to whole pixels so its edges are crisp, not a grey fringe.
    func snappedBodyOps() -> [PathOp] {
        guard variant.snap else { return bodyOps }
        let minEdge = (bodyRect.minX * scale).rounded(.down) / scale
        let maxEdge = (bodyRect.maxX * scale).rounded(.up) / scale
        let rect = CGRect(x: minEdge, y: minEdge, width: maxEdge - minEdge, height: maxEdge - minEdge)
        return continuousRoundedRect(rect, radius: bodyRadius * rect.width / bodyRect.width)
    }

    /// Shadow parameters are in device space (unaffected by the CTM, y up), so convert from canvas units.
    func setShadow(dy: CGFloat, sigma: CGFloat, color: CGColor) {
        ctx.setShadow(offset: CGSize(width: 0, height: -dy * scale), blur: 2 * sigma * scale, color: color)
    }

    func render() -> CGImage {
        let body = CGMutablePath()
        for op in snappedBodyOps() {
            switch op {
            case .move(let p): body.move(to: p)
            case .line(let p): body.addLine(to: p)
            case .curve(let c1, let c2, let p): body.addCurve(to: p, control1: c1, control2: c2)
            }
        }
        body.closeSubpath()
        drawBody(body)
        ctx.saveGState()
        ctx.addPath(body)
        ctx.clip()
        drawShards()
        if variant.trajectory { drawTrajectory() }
        drawLigand()
        ctx.restoreGState()
        return ctx.makeImage()!
    }

    func drawBody(_ body: CGPath) {
        ctx.saveGState()
        setShadow(dy: 12, sigma: 14, color: rgb(0x000000, 0.30))
        ctx.addPath(body)
        ctx.setFillColor(rgb(Palette.bodyBottom))
        ctx.fillPath()
        ctx.restoreGState()

        ctx.saveGState()
        ctx.addPath(body)
        ctx.clip()
        ctx.drawLinearGradient(gradient([(rgb(Palette.bodyTop), 0), (rgb(Palette.bodyBottom), 1)]),
                               start: CGPoint(x: 512, y: bodyRect.minY), end: CGPoint(x: 512, y: bodyRect.maxY),
                               options: [])
        let glowCenter = CGPoint(x: 512, y: 470)
        ctx.drawRadialGradient(gradient([(rgb(Palette.bodyGlow, 0.55), 0), (rgb(Palette.bodyGlow, 0), 1)]),
                               startCenter: glowCenter, startRadius: 0, endCenter: glowCenter, endRadius: 420,
                               options: [])
        // Cavity: the crystals line a bowl that darkens towards its wall (too fine for the snapped sizes).
        if !variant.snap {
            ctx.drawRadialGradient(gradient([(rgb(0x000000, 0), 0), (rgb(0x000000, 0), 0.55),
                                             (rgb(0x000000, 0.24), 0.88), (rgb(0x000000, 0), 1)]),
                                   startCenter: center, startRadius: 0, endCenter: center, endRadius: 400, options: [])
        }
        // Inner edge: the stroke is clipped to the body, leaving its inner half (2 units, at least 1 device pixel).
        ctx.addPath(body)
        ctx.setStrokeColor(rgb(0xFFFFFF, 0.10))
        ctx.setLineWidth(max(4, 2 / scale))
        ctx.strokePath()
        ctx.restoreGState()
    }

    func drawShards() {
        let shards = (variant.secondary ? secondarySlots.map(Shard.secondary) : []) + variant.slots.map(Shard.primary)

        // Soft contact shadow so the crystals sit on the cavity wall.
        ctx.saveGState()
        setShadow(dy: 6, sigma: 8, color: rgb(0x000000, 0.35))
        ctx.beginTransparencyLayer(auxiliaryInfo: nil)
        for shard in shards {
            ctx.addPath(polygon(shard.outline))
        }
        ctx.setFillColor(rgb(Palette.upperLeft))
        ctx.fillPath()
        ctx.endTransparencyLayer()
        ctx.restoreGState()

        for shard in shards {
            if variant.faceted {
                ctx.addPath(polygon([shard.back, shard.outerLeft, shard.tip]))
                ctx.setFillColor(rgb(shard.leftColor))
                ctx.fillPath()
                ctx.addPath(polygon([shard.back, shard.tip, shard.outerRight]))
                ctx.setFillColor(rgb(shard.rightColor))
                ctx.fillPath()
            } else {
                ctx.addPath(polygon(shard.outline))
                ctx.setFillColor(rgb(Palette.upperLeft))
                ctx.fillPath()
            }
            if variant.glint {
                ctx.move(to: shard.back)
                ctx.addLine(to: shard.tip)
                ctx.setStrokeColor(rgb(Palette.glint, 0.45))
                ctx.setLineWidth(1.5)
                ctx.setLineCap(.round)
                ctx.strokePath()
            }
        }
    }

    func drawTrajectory() {
        for dot in trajectoryDots {
            ctx.setFillColor(rgb(Palette.ligandGlow, dot.alpha))
            ctx.fillEllipse(in: CGRect(x: dot.point.x - dot.radius, y: dot.point.y - dot.radius,
                                       width: 2 * dot.radius, height: 2 * dot.radius))
        }
    }

    func drawLigand() {
        let hex = polygon(hexagonVertices(radius: 86 * variant.hexScale))

        // Under-glow: the opaque fill is covered by the hexagon below, leaving only the blurred halo.
        ctx.saveGState()
        setShadow(dy: 0, sigma: 22, color: rgb(Palette.ligandGlow, 0.55))
        ctx.addPath(hex)
        ctx.setFillColor(rgb(Palette.ligandGlow))
        ctx.fillPath()
        ctx.restoreGState()

        ctx.saveGState()
        ctx.addPath(hex)
        ctx.clip()
        let box = hex.boundingBoxOfPath
        ctx.drawLinearGradient(gradient([(rgb(Palette.ligandLight), 0), (rgb(Palette.ligandDark), 1)]),
                               start: CGPoint(x: box.minX, y: box.minY), end: CGPoint(x: box.maxX, y: box.maxY),
                               options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
        ctx.restoreGState()

        if variant.innerRing {
            ctx.setStrokeColor(rgb(Palette.ring, 0.85))
            ctx.setLineWidth(max(10, variant.minRingPx / scale))
            ctx.strokeEllipse(in: CGRect(x: center.x - 44, y: center.y - 44, width: 88, height: 88))
        }
    }
}

func writePNG(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        throw CocoaError(.fileWriteUnknown)
    }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
}

// MARK: - SVG master (documentation only; same geometry as the 1024 rendition)

func svgPoints(_ points: [CGPoint]) -> String {
    points.map { String(format: "%.1f,%.1f", $0.x, $0.y) }.joined(separator: " ")
}

func hexString(_ hex: UInt32) -> String { String(format: "#%06X", hex) }

func makeSVG() -> String {
    func xy(_ p: CGPoint) -> String { String(format: "%.2f %.2f", p.x, p.y) }
    let bodyPath = bodyOps.map { op -> String in
        switch op {
        case .move(let p): return "M\(xy(p))"
        case .line(let p): return "L\(xy(p))"
        case .curve(let c1, let c2, let p): return "C\(xy(c1)) \(xy(c2)) \(xy(p))"
        }
    }.joined(separator: " ") + " Z"
    let hex = hexagonVertices(radius: 86)
    var shards = ""
    for s in secondarySlots.map(Shard.secondary) + fullSlots.map(Shard.primary) {
        shards += """
            <polygon points="\(svgPoints([s.back, s.outerLeft, s.tip]))" fill="\(hexString(s.leftColor))"/>
            <polygon points="\(svgPoints([s.back, s.tip, s.outerRight]))" fill="\(hexString(s.rightColor))"/>
            <line x1="\(String(format: "%.1f", s.back.x))" y1="\(String(format: "%.1f", s.back.y))" x2="\(String(format: "%.1f", s.tip.x))" y2="\(String(format: "%.1f", s.tip.y))" stroke="\(hexString(Palette.glint))" stroke-opacity="0.45" stroke-width="1.5" stroke-linecap="round"/>

        """
    }
    let dots = trajectoryDots.map {
        String(format: "    <circle cx=\"%.1f\" cy=\"%.1f\" r=\"%.1f\" fill=\"%@\" fill-opacity=\"%.2f\"/>\n",
               $0.point.x, $0.point.y, $0.radius, hexString(Palette.ligandGlow), $0.alpha)
    }.joined()
    return """
    <svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 1024 1024">
      <!-- Druse "crystal pocket" app icon master. Generated by Tools/render_icon.swift; do not edit by hand. -->
      <defs>
        <linearGradient id="body" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="\(hexString(Palette.bodyTop))"/>
          <stop offset="1" stop-color="\(hexString(Palette.bodyBottom))"/>
        </linearGradient>
        <radialGradient id="bodyGlow" gradientUnits="userSpaceOnUse" cx="512" cy="470" r="420">
          <stop offset="0" stop-color="\(hexString(Palette.bodyGlow))" stop-opacity="0.55"/>
          <stop offset="1" stop-color="\(hexString(Palette.bodyGlow))" stop-opacity="0"/>
        </radialGradient>
        <radialGradient id="cavity" gradientUnits="userSpaceOnUse" cx="\(Int(center.x))" cy="\(Int(center.y))" r="400">
          <stop offset="0.55" stop-color="#000000" stop-opacity="0"/>
          <stop offset="0.88" stop-color="#000000" stop-opacity="0.24"/>
          <stop offset="1" stop-color="#000000" stop-opacity="0"/>
        </radialGradient>
        <linearGradient id="ligand" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="\(hexString(Palette.ligandLight))"/>
          <stop offset="1" stop-color="\(hexString(Palette.ligandDark))"/>
        </linearGradient>
        <clipPath id="bodyClip"><path d="\(bodyPath)"/></clipPath>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="12" stdDeviation="14" flood-color="#000000" flood-opacity="0.30"/>
        </filter>
        <filter id="shardShadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000000" flood-opacity="0.35"/>
        </filter>
        <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="22"/>
        </filter>
      </defs>
      <path d="\(bodyPath)" fill="url(#body)" filter="url(#shadow)"/>
      <g clip-path="url(#bodyClip)">
        <rect x="100" y="100" width="824" height="824" fill="url(#bodyGlow)"/>
        <rect x="100" y="100" width="824" height="824" fill="url(#cavity)"/>
        <path d="\(bodyPath)" fill="none" stroke="#FFFFFF" stroke-opacity="0.10" stroke-width="4"/>
        <g filter="url(#shardShadow)">
    \(shards)    </g>
    \(dots)    <polygon points="\(svgPoints(hex))" fill="\(hexString(Palette.ligandGlow))" fill-opacity="0.55" filter="url(#glow)"/>
        <polygon points="\(svgPoints(hex))" fill="url(#ligand)"/>
        <circle cx="\(Int(center.x))" cy="\(Int(center.y))" r="44" fill="none" stroke="\(hexString(Palette.ring))" stroke-opacity="0.85" stroke-width="10"/>
      </g>
    </svg>

    """
}

// MARK: - Main

let args = CommandLine.arguments
guard args.count >= 2 else {
    FileHandle.standardError.write("usage: swift Tools/render_icon.swift <appiconset-dir> [svg-path]\n".data(using: .utf8)!)
    exit(64)
}
let outDir = URL(fileURLWithPath: args[1], isDirectory: true)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
for px in [16, 32, 64, 128, 256, 512, 1024] {
    let url = outDir.appendingPathComponent("icon_\(px)x\(px).png")
    try writePNG(IconRenderer(px: px).render(), to: url)
    print("wrote \(url.path)")
}
if args.count >= 3 {
    let svgURL = URL(fileURLWithPath: args[2])
    try FileManager.default.createDirectory(at: svgURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try makeSVG().write(to: svgURL, atomically: true, encoding: .utf8)
    print("wrote \(svgURL.path)")
}
