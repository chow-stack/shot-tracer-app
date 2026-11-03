import Foundation

public struct AssociationConfiguration: Sendable {
    public var beamWidth = 600
    public var maxGap = 2
    public var maxLaunchRadius = 190.0
    public var spatialBin = 10.0
    public var minimumVisibleBeforeTermination = 8

    public init(
        beamWidth: Int = 600,
        maxGap: Int = 2,
        maxLaunchRadius: Double = 190,
        spatialBin: Double = 10,
        minimumVisibleBeforeTermination: Int = 8
    ) {
        self.beamWidth = beamWidth
        self.maxGap = maxGap
        self.maxLaunchRadius = maxLaunchRadius
        self.spatialBin = spatialBin
        self.minimumVisibleBeforeTermination = minimumVisibleBeforeTermination
    }
}

public enum AssociationError: Error, Equatable {
    case invalidConfiguration
    case insufficientFrames
    case noPlausibleLaunch
    case noQualifiedTrack
    case missingTimestamp(Int)
}

public struct AssociatedTrajectoryPoint: Sendable, Equatable {
    public let frameIndex: Int
    public let timestampSeconds: Double
    public let x: Double
    public let y: Double
    public let confidence: Double
    public let sources: [String]
    public let sourceRanks: [String: Int]
    public let provenance: PointProvenance
}

public struct AssociationResult: Sendable {
    public let launchFrame: Int
    public let terminationFrame: Int
    public let finalScore: Double
    public let trajectory: [AssociatedTrajectoryPoint]
}

private enum BeamPhase: String, Sendable {
    case active
    case gap
    case terminated
}

private struct Observation: Sendable {
    let frameIndex: Int
    let x: Double
    let y: Double
}

private struct BeamNode: Sendable {
    let score: Double
    let phase: BeamPhase
    let frameIndex: Int
    let candidate: AssociationCandidate?
    let parentIndex: Int?
    let observations: [Observation]
    let gapCount: Int
    let visibleCount: Int
    let sequence: Int
}

private enum SpatialKey: Hashable {
    case terminated
    case gap(Int, Int, Int)
    case active(Int, Int)
}

public enum TrajectoryAssociator {
    public static func associateFullClip(
        candidatesByFrame: [[AssociationCandidate]],
        timestampsSeconds: [Double],
        address: (x: Double, y: Double),
        configuration: AssociationConfiguration = AssociationConfiguration()
    ) throws -> AssociationResult {
        guard configuration.beamWidth > 0,
              configuration.maxGap >= 0,
              configuration.maxLaunchRadius > 0,
              configuration.spatialBin > 0,
              configuration.minimumVisibleBeforeTermination > 0 else {
            throw AssociationError.invalidConfiguration
        }
        let frameCount = min(candidatesByFrame.count, timestampsSeconds.count)
        guard frameCount >= 4 else { throw AssociationError.insufficientFrames }
        let frameIndexes = Array(1..<(frameCount - 1))
        var arena: [BeamNode] = []
        var beams: [Int] = []
        var nextSequence = 0

        for position in 1..<frameIndexes.count {
            let previousIndex = frameIndexes[position - 1]
            let frameIndex = frameIndexes[position]
            var expanded = expand(
                beams: beams,
                arena: &arena,
                frameIndex: frameIndex,
                candidates: candidatesByFrame[frameIndex],
                configuration: configuration,
                nextSequence: &nextSequence
            )
            expanded.append(contentsOf: initialStates(
                firstIndex: previousIndex,
                secondIndex: frameIndex,
                candidatesByFrame: candidatesByFrame,
                address: address,
                configuration: configuration,
                arena: &arena,
                nextSequence: &nextSequence
            ))
            if !expanded.isEmpty {
                beams = prune(
                    expanded,
                    arena: arena,
                    beamWidth: configuration.beamWidth,
                    spatialBin: configuration.spatialBin
                )
            }
        }
        guard !beams.isEmpty else { throw AssociationError.noPlausibleLaunch }
        let qualified = beams.filter {
            arena[$0].visibleCount >= configuration.minimumVisibleBeforeTermination
        }
        guard let finalIndex = best(qualified, arena: arena) else {
            throw AssociationError.noQualifiedTrack
        }
        let stateIndexes = unwind(finalIndex, arena: arena)
        let trajectory = try trajectory(
            from: stateIndexes,
            arena: arena,
            timestampsSeconds: timestampsSeconds
        )
        guard let first = trajectory.first, let last = trajectory.last else {
            throw AssociationError.noQualifiedTrack
        }
        return AssociationResult(
            launchFrame: first.frameIndex,
            terminationFrame: last.frameIndex,
            finalScore: arena[finalIndex].score,
            trajectory: trajectory
        )
    }

    private static func initialStates(
        firstIndex: Int,
        secondIndex: Int,
        candidatesByFrame: [[AssociationCandidate]],
        address: (x: Double, y: Double),
        configuration: AssociationConfiguration,
        arena: inout [BeamNode],
        nextSequence: inout Int
    ) -> [Int] {
        let first = candidatesByFrame[firstIndex].filter {
            hypot($0.x - address.x, $0.y - address.y) <= configuration.maxLaunchRadius
                && $0.y <= address.y + 40
        }
        let second = candidatesByFrame[secondIndex]
        var states: [Int] = []
        for start in first {
            let launchDistance = hypot(start.x - address.x, start.y - address.y)
            for following in second {
                let dx = following.x - start.x
                let dy = following.y - start.y
                let displacement = hypot(dx, dy)
                guard displacement >= 40, displacement <= 320,
                      dy < -35, abs(dx) <= 170 else { continue }
                let score = start.quality + following.quality + 1.8
                    - 0.003 * launchDistance - 0.002 * abs(dx)
                let rootIndex = append(BeamNode(
                    score: score,
                    phase: .active,
                    frameIndex: firstIndex,
                    candidate: start,
                    parentIndex: nil,
                    observations: [Observation(frameIndex: firstIndex, x: start.x, y: start.y)],
                    gapCount: 0,
                    visibleCount: 1,
                    sequence: nextSequence
                ), to: &arena, nextSequence: &nextSequence)
                let childIndex = append(BeamNode(
                    score: score,
                    phase: .active,
                    frameIndex: secondIndex,
                    candidate: following,
                    parentIndex: rootIndex,
                    observations: [
                        Observation(frameIndex: firstIndex, x: start.x, y: start.y),
                        Observation(frameIndex: secondIndex, x: following.x, y: following.y),
                    ],
                    gapCount: 0,
                    visibleCount: 2,
                    sequence: nextSequence
                ), to: &arena, nextSequence: &nextSequence)
                states.append(childIndex)
            }
        }
        return Array(stableScoreSort(states, arena: arena).prefix(configuration.beamWidth))
    }

    private static func expand(
        beams: [Int],
        arena: inout [BeamNode],
        frameIndex: Int,
        candidates: [AssociationCandidate],
        configuration: AssociationConfiguration,
        nextSequence: inout Int
    ) -> [Int] {
        var expanded: [Int] = []
        for stateIndex in beams {
            let state = arena[stateIndex]
            if state.phase == .terminated {
                expanded.append(append(BeamNode(
                    score: state.score, phase: .terminated, frameIndex: frameIndex,
                    candidate: nil, parentIndex: stateIndex, observations: state.observations,
                    gapCount: state.gapCount, visibleCount: state.visibleCount,
                    sequence: nextSequence
                ), to: &arena, nextSequence: &nextSequence))
                continue
            }
            for candidate in candidates {
                guard let increment = transitionScore(
                    state: state, candidate: candidate, frameIndex: frameIndex
                ) else { continue }
                let newObservation = Observation(
                    frameIndex: frameIndex, x: candidate.x, y: candidate.y
                )
                expanded.append(append(BeamNode(
                    score: state.score + increment,
                    phase: .active,
                    frameIndex: frameIndex,
                    candidate: candidate,
                    parentIndex: stateIndex,
                    observations: Array((state.observations + [newObservation]).suffix(3)),
                    gapCount: 0,
                    visibleCount: state.visibleCount + 1,
                    sequence: nextSequence
                ), to: &arena, nextSequence: &nextSequence))
            }
            if state.gapCount < configuration.maxGap {
                expanded.append(append(BeamNode(
                    score: state.score - 0.55 - 0.25 * Double(state.gapCount),
                    phase: .gap, frameIndex: frameIndex, candidate: nil,
                    parentIndex: stateIndex, observations: state.observations,
                    gapCount: state.gapCount + 1, visibleCount: state.visibleCount,
                    sequence: nextSequence
                ), to: &arena, nextSequence: &nextSequence))
            }
            if state.visibleCount >= configuration.minimumVisibleBeforeTermination {
                expanded.append(append(BeamNode(
                    score: state.score - 0.35,
                    phase: .terminated, frameIndex: frameIndex, candidate: nil,
                    parentIndex: stateIndex, observations: state.observations,
                    gapCount: state.gapCount, visibleCount: state.visibleCount,
                    sequence: nextSequence
                ), to: &arena, nextSequence: &nextSequence))
            }
        }
        return expanded
    }

    private static func transitionScore(
        state: BeamNode,
        candidate: AssociationCandidate,
        frameIndex: Int
    ) -> Double? {
        let prediction = predictNext(state.observations, frameIndex: frameIndex)
        let error = hypot(candidate.x - prediction.x, candidate.y - prediction.y)
        let last = state.observations[state.observations.count - 1]
        let gap = max(0, frameIndex - last.frameIndex - 1)
        let gate = max(18, 0.34 * prediction.speed + 9) * (1 + 0.65 * Double(gap))
        guard error <= 2.2 * gate else { return nil }

        let dx = candidate.x - last.x
        let dy = candidate.y - last.y
        let observationDelta = max(1, frameIndex - last.frameIndex)
        guard dy <= 12 * Double(observationDelta) else { return nil }
        let displacement = hypot(dx, dy) / Double(observationDelta)
        guard displacement <= 340 else { return nil }
        if prediction.speed < 15,
           displacement > max(12, 2 * prediction.speed) { return nil }

        let previous = state.observations[state.observations.count - 2]
        let previousDX = last.x - previous.x
        let previousDY = last.y - previous.y
        let previousNorm = hypot(previousDX, previousDY)
        var directionPenalty = 0.0
        if displacement > 1e-6, previousNorm > 1e-6 {
            let cosine = (dx * previousDX + dy * previousDY) / (displacement * previousNorm)
            guard cosine >= -0.15 else { return nil }
            if prediction.speed > 15, cosine < 0.35 { return nil }
            directionPenalty = 0.7 * (1 - max(-1, min(1, cosine)))
        }
        let expectedSpeed = max(1, prediction.speed * prediction.decay)
        let speedPenalty = 0.35 * abs(log(max(1, displacement) / expectedSpeed))
        return 0.17 + candidate.quality - 1.65 * error / gate
            - speedPenalty - directionPenalty - 0.18 * Double(gap)
    }

    private static func predictNext(
        _ observations: [Observation], frameIndex: Int
    ) -> (x: Double, y: Double, speed: Double, decay: Double) {
        let previous = observations[observations.count - 2]
        let last = observations[observations.count - 1]
        let frameDelta = max(1, last.frameIndex - previous.frameIndex)
        let velocityX = (last.x - previous.x) / Double(frameDelta)
        let velocityY = (last.y - previous.y) / Double(frameDelta)
        let speed = hypot(velocityX, velocityY)
        let decay: Double
        if observations.count >= 3 {
            let older = observations[observations.count - 3]
            let olderDelta = max(1, previous.frameIndex - older.frameIndex)
            let olderSpeed = hypot(
                (previous.x - older.x) / Double(olderDelta),
                (previous.y - older.y) / Double(olderDelta)
            )
            decay = olderSpeed > 1e-6
                ? max(0.55, min(1.08, speed / olderSpeed)) : 0.9
        } else {
            decay = speed > 30 ? 0.72 : 0.9
        }
        let ahead = max(1, frameIndex - last.frameIndex)
        let multiplier: Double
        if abs(1 - decay) < 1e-6 {
            multiplier = Double(ahead)
        } else {
            multiplier = decay * (1 - pow(decay, Double(ahead))) / (1 - decay)
        }
        return (
            last.x + velocityX * multiplier,
            last.y + velocityY * multiplier,
            speed,
            decay
        )
    }

    private static func prune(
        _ states: [Int], arena: [BeamNode], beamWidth: Int, spatialBin: Double
    ) -> [Int] {
        var kept: [Int] = []
        var perBin: [SpatialKey: Int] = [:]
        for stateIndex in stableScoreSort(states, arena: arena) {
            let state = arena[stateIndex]
            let key: SpatialKey
            if state.phase == .terminated {
                key = .terminated
            } else if state.candidate == nil {
                let last = state.observations[state.observations.count - 1]
                key = .gap(
                    Int((last.x / spatialBin).rounded(.toNearestOrEven)),
                    Int((last.y / spatialBin).rounded(.toNearestOrEven)),
                    state.gapCount
                )
            } else {
                let candidate = state.candidate!
                key = .active(
                    Int((candidate.x / spatialBin).rounded(.toNearestOrEven)),
                    Int((candidate.y / spatialBin).rounded(.toNearestOrEven))
                )
            }
            if (perBin[key] ?? 0) >= 3 { continue }
            perBin[key, default: 0] += 1
            kept.append(stateIndex)
            if kept.count >= beamWidth { break }
        }
        return kept
    }

    private static func trajectory(
        from stateIndexes: [Int], arena: [BeamNode], timestampsSeconds: [Double]
    ) throws -> [AssociatedTrajectoryPoint] {
        var points: [Int: AssociatedTrajectoryPoint] = [:]
        for index in stateIndexes {
            let state = arena[index]
            guard state.phase == .active, let candidate = state.candidate else { continue }
            guard timestampsSeconds.indices.contains(state.frameIndex) else {
                throw AssociationError.missingTimestamp(state.frameIndex)
            }
            points[state.frameIndex] = AssociatedTrajectoryPoint(
                frameIndex: state.frameIndex,
                timestampSeconds: timestampsSeconds[state.frameIndex],
                x: candidate.x,
                y: candidate.y,
                confidence: candidate.confidence,
                sources: candidate.sources,
                sourceRanks: candidate.sourceRanks,
                provenance: .observed
            )
        }
        for (position, index) in stateIndexes.enumerated() {
            let state = arena[index]
            guard state.phase == .gap else { continue }
            let left = stateIndexes[..<position].reversed().lazy
                .map { arena[$0] }
                .first { $0.phase == .active && $0.candidate != nil }
            let right = stateIndexes[(position + 1)...].lazy
                .map { arena[$0] }
                .first { $0.phase == .active && $0.candidate != nil }
            guard let left, let right,
                  let leftCandidate = left.candidate,
                  let rightCandidate = right.candidate,
                  right.frameIndex - left.frameIndex <= 3 else { continue }
            let leftTime = timestampsSeconds[left.frameIndex]
            let rightTime = timestampsSeconds[right.frameIndex]
            let currentTime = timestampsSeconds[state.frameIndex]
            let ratio = rightTime > leftTime
                ? (currentTime - leftTime) / (rightTime - leftTime)
                : Double(state.frameIndex - left.frameIndex) / Double(right.frameIndex - left.frameIndex)
            points[state.frameIndex] = AssociatedTrajectoryPoint(
                frameIndex: state.frameIndex,
                timestampSeconds: currentTime,
                x: leftCandidate.x + ratio * (rightCandidate.x - leftCandidate.x),
                y: leftCandidate.y + ratio * (rightCandidate.y - leftCandidate.y),
                confidence: 0.5 * min(leftCandidate.confidence, rightCandidate.confidence),
                sources: ["short_gap_interpolation"],
                sourceRanks: [:],
                provenance: .estimatedGap
            )
        }
        return points.keys.sorted().compactMap { points[$0] }
    }

    private static func unwind(_ finalIndex: Int, arena: [BeamNode]) -> [Int] {
        var values: [Int] = []
        var current: Int? = finalIndex
        while let index = current {
            values.append(index)
            current = arena[index].parentIndex
        }
        return values.reversed()
    }

    private static func append(
        _ node: BeamNode, to arena: inout [BeamNode], nextSequence: inout Int
    ) -> Int {
        let index = arena.count
        arena.append(node)
        nextSequence += 1
        return index
    }

    private static func stableScoreSort(_ indexes: [Int], arena: [BeamNode]) -> [Int] {
        indexes.sorted {
            let first = arena[$0]
            let second = arena[$1]
            if first.score != second.score { return first.score > second.score }
            return first.sequence < second.sequence
        }
    }

    private static func best(_ indexes: [Int], arena: [BeamNode]) -> Int? {
        indexes.max {
            let first = arena[$0]
            let second = arena[$1]
            if first.score != second.score { return first.score < second.score }
            return first.sequence > second.sequence
        }
    }
}
