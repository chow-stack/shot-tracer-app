import Foundation

struct PresentationCoordinate: Sendable, Equatable {
    let x: Double
    let y: Double
}

/// Produces display coordinates only. The trajectory document, timestamps,
/// confidence, and provenance remain untouched for fitting and diagnostics.
enum TrajectoryPresentationSmoother {
    static func coordinates(for points: [TrajectoryPoint]) -> [PresentationCoordinate] {
        guard points.count >= 5 else {
            return points.map { PresentationCoordinate(x: $0.x, y: $0.y) }
        }
        var result = points.map { PresentationCoordinate(x: $0.x, y: $0.y) }
        var start = 0
        while start < points.count {
            let inferred = provenance(of: points[start]) == .inferred
            var end = start + 1
            while end < points.count,
                  (provenance(of: points[end]) == .inferred) == inferred {
                end += 1
            }
            if !inferred, end - start >= 5 {
                smooth(points: points, range: start..<end, into: &result)
            }
            start = end
        }
        return result
    }

    private static func smooth(
        points: [TrajectoryPoint],
        range: Range<Int>,
        into result: inout [PresentationCoordinate]
    ) {
        // A symmetric five-sample kernel reduces single-frame Vision jitter.
        // The endpoints stay exact so the address and physics join do not move.
        let weights = [1.0, 2.0, 3.0, 2.0, 1.0]
        for index in (range.lowerBound + 2)..<(range.upperBound - 2) {
            var x = 0.0
            var y = 0.0
            for offset in -2...2 {
                let weight = weights[offset + 2]
                x += weight * points[index + offset].x
                y += weight * points[index + offset].y
            }
            result[index] = PresentationCoordinate(x: x / 9, y: y / 9)
        }
    }

    private static func provenance(of point: TrajectoryPoint) -> PointProvenance {
        point.provenance ?? ((point.estimated ?? false) ? .estimatedGap : .observed)
    }
}
