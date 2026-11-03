import CoreGraphics
import CoreVideo
import Foundation

public struct TracerColor: Sendable, Equatable {
    public let red: Double
    public let green: Double
    public let blue: Double

    public init(red: Double, green: Double, blue: Double) {
        self.red = red
        self.green = green
        self.blue = blue
    }
}

public struct TrajectoryRenderingStyle: Sendable, Equatable {
    public var observedColor = TracerColor(red: 35 / 255, green: 230 / 255, blue: 1)
    public var estimatedGapColor = TracerColor(red: 1, green: 185 / 255, blue: 0)
    public var inferredColor = TracerColor(red: 1, green: 80 / 255, blue: 210 / 255)
    public var glowAlpha = 0.22
    public var lineAlpha = 0.92

    public init() {}
}

public enum TrajectoryRendererError: Error, Equatable {
    case unsupportedPixelFormat(OSType)
    case unavailablePixelBufferMemory
    case couldNotCreateGraphicsContext
}

private struct RenderPoint: Sendable {
    let frameIndex: Int
    let x: Double
    let y: Double
    let opacity: Double
    let provenance: PointProvenance
}

/// Draws presentation-only tracer styling into portrait, top-left-origin BGRA
/// frames. Coordinates and provenance are never modified.
public struct TrajectoryRenderer: Sendable {
    private let points: [RenderPoint]
    private let style: TrajectoryRenderingStyle

    public init(
        trajectory: [TrajectoryPoint],
        style: TrajectoryRenderingStyle = TrajectoryRenderingStyle()
    ) {
        let ordered = trajectory.sorted { $0.frameIndex < $1.frameIndex }
        let coordinates = TrajectoryPresentationSmoother.coordinates(for: ordered)
        points = zip(ordered, coordinates).map { point, coordinate in
            RenderPoint(
                frameIndex: point.frameIndex,
                x: coordinate.x,
                y: coordinate.y,
                opacity: min(1, max(0, point.opacity ?? 1)),
                provenance: point.provenance
                    ?? ((point.estimated ?? false) ? .estimatedGap : .observed)
            )
        }
        self.style = style
    }

    public func render(into pixelBuffer: CVPixelBuffer, frameIndex: Int) throws {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard pixelFormat == kCVPixelFormatType_32BGRA else {
            throw TrajectoryRendererError.unsupportedPixelFormat(pixelFormat)
        }
        let visibleCount = points.partitioningIndex { $0.frameIndex > frameIndex }
        guard visibleCount > 0 else { return }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw TrajectoryRendererError.unavailablePixelBufferMemory
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard let context = CGContext(
            data: baseAddress,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { throw TrajectoryRendererError.couldNotCreateGraphicsContext }

        context.setAllowsAntialiasing(true)
        context.setShouldAntialias(true)
        context.translateBy(x: 0, y: CGFloat(height))
        context.scaleBy(x: 1, y: -1)
        let scale = Double(min(width, height)) / 1080
        let innerWidth = max(3, (5 * scale).rounded())
        let glowWidth = max(innerWidth + 4, (13 * scale).rounded())
        let visible = points.prefix(visibleCount)

        for (previous, point) in zip(visible, visible.dropFirst()) {
            let segment = segmentStyle(previous: previous, point: point)
            drawSegment(
                context: context,
                from: previous,
                to: point,
                color: segment.color,
                alpha: style.glowAlpha * segment.opacity,
                width: glowWidth,
                dashed: segment.dashed
            )
            drawSegment(
                context: context,
                from: previous,
                to: point,
                color: segment.color,
                alpha: style.lineAlpha * segment.opacity,
                width: innerWidth,
                dashed: segment.dashed
            )
        }

        if let head = visible.last {
            let headColor = color(for: head.provenance)
            let headOpacity = head.provenance == .inferred ? head.opacity : 1
            fillCircle(
                context: context,
                center: head,
                radius: max(7, (11 * scale).rounded()),
                color: headColor,
                alpha: style.glowAlpha * headOpacity
            )
            fillCircle(
                context: context,
                center: head,
                radius: max(3, (5 * scale).rounded()),
                color: TracerColor(red: 1, green: 1, blue: 1),
                alpha: style.lineAlpha
            )
        }
    }

    private func segmentStyle(
        previous: RenderPoint, point: RenderPoint
    ) -> (color: TracerColor, opacity: Double, dashed: Bool) {
        if previous.provenance == .inferred || point.provenance == .inferred {
            return (style.inferredColor, min(previous.opacity, point.opacity), true)
        }
        if previous.provenance == .estimatedGap || point.provenance == .estimatedGap {
            return (style.estimatedGapColor, 1, true)
        }
        return (style.observedColor, 1, false)
    }

    private func color(for provenance: PointProvenance) -> TracerColor {
        switch provenance {
        case .observed: style.observedColor
        case .estimatedGap: style.estimatedGapColor
        case .inferred: style.inferredColor
        }
    }

    private func drawSegment(
        context: CGContext,
        from start: RenderPoint,
        to end: RenderPoint,
        color: TracerColor,
        alpha: Double,
        width: Double,
        dashed: Bool
    ) {
        context.saveGState()
        defer { context.restoreGState() }
        context.setBlendMode(.plusLighter)
        context.setStrokeColor(
            red: color.red, green: color.green, blue: color.blue, alpha: alpha
        )
        context.setLineWidth(width)
        context.setLineCap(.round)
        context.setLineJoin(.round)
        if dashed {
            context.setLineDash(
                phase: 0,
                lengths: [max(8, width * 2.5), max(5, width * 1.5)]
            )
        }
        context.move(to: CGPoint(x: start.x, y: start.y))
        context.addLine(to: CGPoint(x: end.x, y: end.y))
        context.strokePath()
    }

    private func fillCircle(
        context: CGContext,
        center: RenderPoint,
        radius: Double,
        color: TracerColor,
        alpha: Double
    ) {
        context.saveGState()
        defer { context.restoreGState() }
        context.setBlendMode(.plusLighter)
        context.setFillColor(
            red: color.red, green: color.green, blue: color.blue, alpha: alpha
        )
        context.fillEllipse(in: CGRect(
            x: center.x - radius,
            y: center.y - radius,
            width: 2 * radius,
            height: 2 * radius
        ))
    }
}

private extension RandomAccessCollection {
    func partitioningIndex(where belongsInSecondPartition: (Element) -> Bool) -> Index {
        var low = startIndex
        var high = endIndex
        while low != high {
            let distance = self.distance(from: low, to: high)
            let middle = index(low, offsetBy: distance / 2)
            if belongsInSecondPartition(self[middle]) {
                high = middle
            } else {
                low = index(after: middle)
            }
        }
        return low
    }
}
