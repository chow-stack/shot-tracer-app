import Foundation

/// Conservative evidence gates for presentation-only prediction. A rejected
/// prediction still allows export of the measured tracer; it never invents a
/// continuation from a weak or discontinuous track.
public enum TrajectoryQualityGate {
    public static func permitsPrediction(
        document: TrajectoryDocument,
        associationScore: Double
    ) -> Bool {
        guard associationScore.isFinite, associationScore > 0 else { return false }
        let observed = document.trajectory.filter {
            ($0.provenance ?? (($0.estimated ?? false) ? .estimatedGap : .observed))
                == .observed
        }.sorted { $0.frameIndex < $1.frameIndex }
        guard observed.count >= 12,
              let first = observed.first,
              let last = observed.last,
              last.frameIndex > first.frameIndex else { return false }

        let maximumGap = zip(observed, observed.dropFirst()).map {
            $1.frameIndex - $0.frameIndex
        }.max() ?? 0
        guard maximumGap <= 3 else { return false }

        // The final observations drive the continuation join. Require a
        // consistent direction so a one-frame Vision wobble cannot launch an
        // implausible physics curve.
        let tail = Array(observed.suffix(6))
        guard tail.count >= 4 else { return false }
        let velocities = zip(tail, tail.dropFirst()).map {
            (x: $1.x - $0.x, y: $1.y - $0.y)
        }.filter { hypot($0.x, $0.y) >= 1 }
        guard velocities.count >= 3 else { return false }
        for (firstVelocity, secondVelocity) in zip(velocities, velocities.dropFirst()) {
            let denominator = hypot(firstVelocity.x, firstVelocity.y)
                * hypot(secondVelocity.x, secondVelocity.y)
            guard denominator > 0 else { return false }
            let cosine = (firstVelocity.x * secondVelocity.x
                + firstVelocity.y * secondVelocity.y) / denominator
            if cosine < 0.35 { return false }
        }
        return true
    }
}
