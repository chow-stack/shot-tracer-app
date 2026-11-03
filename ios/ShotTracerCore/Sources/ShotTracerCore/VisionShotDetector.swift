import AVFoundation
import Foundation
import ImageIO
import OSLog
import Vision

public enum VisionShotDetectionError: LocalizedError, Sendable {
    case noFrames
    case noConfidentTrajectory

    public var errorDescription: String? {
        switch self {
        case .noFrames:
            "The selected video has no readable frames."
        case .noConfidentTrajectory:
            "The ball could not be found confidently at that tap. Use a stable rear-view video and re-check the tap position, then try again."
        }
    }
}

public struct VisionShotDetectionResult: Sendable {
    public let document: TrajectoryDocument
    public let address: (x: Double, y: Double)
    public let associationScore: Double
    public let stageTimings: StageTimings
}

/// A deliberately small native MVP detector for stable-camera clips. Vision
/// supplies image-supported trajectory measurements; the existing physics
/// continuation remains a separate, presentation-only stage.
public actor VisionShotDetector {
    private static let logger = Logger(
        subsystem: "com.caio.ShotTracer",
        category: "tracking"
    )
    private struct Sample: Sendable {
        let frameIndex: Int
        let timestampSeconds: Double
        let x: Double
        let y: Double
        let confidence: Double
    }

    private struct Track {
        var samplesByFrame: [Int: Sample] = [:]
        var confidenceTotal = 0.0
        var observationCount = 0
        var radiusTotal = 0.0

        var meanConfidence: Double {
            observationCount == 0 ? 0 : confidenceTotal / Double(observationCount)
        }

        var meanRadius: Double {
            observationCount == 0 ? 0 : radiusTotal / Double(observationCount)
        }
    }

    public init() {}

    public func detect(
        sourceURL: URL,
        manualAddress: CGPoint,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> VisionShotDetectionResult {
        let asset = AVURLAsset(url: sourceURL)
        let duration = try await asset.load(.duration)
        let durationSeconds = max(0, duration.seconds)
        let reader = try await SequentialVideoFrameReader(url: sourceURL)
        let shortestSide = Double(min(reader.width, reader.height))
        let request = VNDetectTrajectoriesRequest(
            frameAnalysisSpacing: .zero,
            trajectoryLength: 6
        )
        request.objectMinimumNormalizedRadius = Float(0.75 / shortestSide)
        request.objectMaximumNormalizedRadius = Float(18.0 / shortestSide)

        var timestamps: [Double] = []
        var tracks: [UUID: Track] = [:]
        var motionExtractor = ThreeFrameMotionCandidateExtractor(
            width: reader.width,
            height: reader.height
        )
        var motionCandidatesByFrame: [Int: [AssociationCandidate]] = [:]
        let clock = ContinuousClock()
        var decodeSeconds = 0.0
        var visionSeconds = 0.0
        do {
            while true {
                let decodeStart = clock.now
                guard let frame = try reader.nextFrame() else { break }
                decodeSeconds += (clock.now - decodeStart).secondsDouble
                try Task.checkCancellation()
                timestamps.append(frame.presentationTimeSeconds)
                if let motion = try motionExtractor.consume(
                    pixelBuffer: frame.pixelBuffer,
                    frameIndex: frame.frameIndex
                ) {
                    motionCandidatesByFrame[motion.frameIndex] = motion.candidates
                }
                let visionStart = clock.now
                let handler = VNImageRequestHandler(
                    cmSampleBuffer: frame.sampleBuffer,
                    orientation: .up,
                    options: [:]
                )
                try handler.perform([request])
                visionSeconds += (clock.now - visionStart).secondsDouble
                for observation in request.results ?? [] where observation.confidence >= 0.70 {
                    update(
                        track: &tracks[observation.uuid, default: Track()],
                        observation: observation,
                        currentFrame: frame.frameIndex,
                        timestamps: timestamps,
                        width: reader.width,
                        height: reader.height
                    )
                }
                if durationSeconds > 0 {
                    progress(min(0.98, frame.presentationTimeSeconds / durationSeconds))
                }
            }
        } catch {
            reader.cancel()
            throw error
        }
        guard !timestamps.isEmpty else { throw VisionShotDetectionError.noFrames }
        let mergeStart = clock.now
        let candidates = makeCandidates(
            tracks: tracks,
            motionCandidatesByFrame: motionCandidatesByFrame,
            frameCount: timestamps.count,
            width: reader.width,
            height: reader.height
        )
        let mergeSeconds = (clock.now - mergeStart).secondsDouble
        let visionCount = candidates.flatMap { $0 }.filter {
            $0.sources.contains("apple_vision_trajectory")
        }.count
        let motionCount = candidates.flatMap { $0 }.filter {
            $0.sources.contains("three_frame_motion")
        }.count
        Self.logger.info(
            "Tracking candidates: \(visionCount) Vision-supported, \(motionCount) motion-supported across \(timestamps.count) frames"
        )
        // Every production caller supplies a user tap; there is no automatic
        // address fallback here. `AutomaticShotSetup.initialize` remains as
        // unreachable research code exercised only by its own unit tests.
        let address = (Double(manualAddress.x), Double(manualAddress.y))
        let associationStart = clock.now
        guard let association = try? TrajectoryAssociator.associateFullClip(
            candidatesByFrame: candidates,
            timestampsSeconds: timestamps,
            address: address
        ) else { throw VisionShotDetectionError.noConfidentTrajectory }
        let associationSeconds = (clock.now - associationStart).secondsDouble
        Self.logger.info(
            "Accepted track frames \(association.launchFrame)-\(association.terminationFrame), points \(association.trajectory.count), score \(association.finalScore)"
        )
        let fps: Double
        if timestamps.count > 1, let first = timestamps.first, let last = timestamps.last,
           last > first {
            fps = Double(timestamps.count - 1) / (last - first)
        } else {
            fps = 0
        }
        let points = association.trajectory.map { sample in
            TrajectoryPoint(
                frameIndex: sample.frameIndex,
                timestampSeconds: sample.timestampSeconds,
                x: sample.x,
                y: sample.y,
                confidence: sample.confidence,
                uncertaintyPixels: nil,
                opacity: 1,
                provenance: sample.provenance,
                estimated: sample.provenance != .observed,
                source: sample.sources.first,
                sources: sample.sources,
                sourceRanks: sample.sourceRanks
            )
        }
        let document = TrajectoryDocument(
            trajectorySchemaVersion: 2,
            launchFrame: association.launchFrame,
            terminationFrame: association.terminationFrame,
            observationEndFrame: association.trajectory.last?.frameIndex,
            continuationMode: "observed_only",
            video: VideoMetadata(
                path: sourceURL.path,
                width: reader.width,
                height: reader.height,
                fps: fps,
                frameCount: timestamps.count,
                timestampsSeconds: timestamps
            ),
            trajectory: points,
            continuation: nil
        )
        var stageTimings = StageTimings()
        stageTimings.record("decode", seconds: decodeSeconds)
        stageTimings.record(
            "vimage_downscale_change",
            seconds: motionExtractor.vImageDownscaleChangeSeconds
        )
        stageTimings.record(
            "connected_components",
            seconds: motionExtractor.connectedComponentsSeconds
        )
        stageTimings.record("vision_requests", seconds: visionSeconds)
        stageTimings.record("candidate_merge", seconds: mergeSeconds)
        stageTimings.record("association", seconds: associationSeconds)
        progress(1)
        return VisionShotDetectionResult(
            document: document,
            address: address,
            associationScore: association.finalScore,
            stageTimings: stageTimings
        )
    }

    private func update(
        track: inout Track,
        observation: VNTrajectoryObservation,
        currentFrame: Int,
        timestamps: [Double],
        width: Int,
        height: Int
    ) {
        let points = observation.detectedPoints
        guard !points.isEmpty else { return }
        track.confidenceTotal += Double(observation.confidence)
        track.radiusTotal += Double(observation.movingAverageRadius)
        track.observationCount += 1

        if track.samplesByFrame.isEmpty {
            let rangeStart = observation.timeRange.start.seconds
            let rangeEnd = CMTimeRangeGetEnd(observation.timeRange).seconds
            for (index, point) in points.enumerated() {
                let fraction = points.count == 1
                    ? 1 : Double(index) / Double(points.count - 1)
                let targetTime: Double
                if rangeStart.isFinite, rangeEnd.isFinite, rangeEnd > rangeStart {
                    targetTime = rangeStart + fraction * (rangeEnd - rangeStart)
                } else {
                    let fallback = max(0, currentFrame - points.count + 1 + index)
                    targetTime = timestamps[min(fallback, timestamps.count - 1)]
                }
                let frameIndex = nearestFrame(to: targetTime, timestamps: timestamps)
                insert(
                    point: point,
                    frameIndex: frameIndex,
                    timestamp: timestamps[frameIndex],
                    confidence: observation.confidence,
                    width: width,
                    height: height,
                    into: &track
                )
            }
        } else if let point = points.last {
            insert(
                point: point,
                frameIndex: currentFrame,
                timestamp: timestamps[currentFrame],
                confidence: observation.confidence,
                width: width,
                height: height,
                into: &track
            )
        }
    }

    private func insert(
        point: VNPoint,
        frameIndex: Int,
        timestamp: Double,
        confidence: VNConfidence,
        width: Int,
        height: Int,
        into track: inout Track
    ) {
        let sample = Sample(
            frameIndex: frameIndex,
            timestampSeconds: timestamp,
            x: Double(point.x) * Double(width),
            y: (1 - Double(point.y)) * Double(height),
            confidence: Double(confidence)
        )
        if let existing = track.samplesByFrame[frameIndex],
           existing.confidence > sample.confidence { return }
        track.samplesByFrame[frameIndex] = sample
    }

    private func nearestFrame(to target: Double, timestamps: [Double]) -> Int {
        guard timestamps.count > 1 else { return 0 }
        var low = 0
        var high = timestamps.count
        while low < high {
            let middle = (low + high) / 2
            if timestamps[middle] < target { low = middle + 1 } else { high = middle }
        }
        if low == 0 { return 0 }
        if low == timestamps.count { return timestamps.count - 1 }
        return abs(timestamps[low] - target) < abs(timestamps[low - 1] - target)
            ? low : low - 1
    }

    private func makeCandidates(
        tracks: [UUID: Track],
        motionCandidatesByFrame: [Int: [AssociationCandidate]],
        frameCount: Int,
        width: Int,
        height: Int
    ) -> [[AssociationCandidate]] {
        let diagonal = hypot(Double(width), Double(height))
        let eligible = tracks.values.compactMap { track -> Track? in
            let samples = track.samplesByFrame.values.sorted {
                $0.frameIndex < $1.frameIndex
            }
            guard samples.count >= 6,
                  let first = samples.first else { return nil }
            let highestY = samples.map(\.y).min() ?? first.y
            let upwardTravel = first.y - highestY
            let totalTravel = zip(samples, samples.dropFirst()).reduce(0.0) {
                $0 + hypot($1.1.x - $1.0.x, $1.1.y - $1.0.y)
            }
            guard upwardTravel >= max(24, 0.018 * Double(height)),
                  totalTravel >= max(45, 0.025 * diagonal),
                  track.meanConfidence >= 0.72,
                  track.meanRadius <= 0.015 else { return nil }
            return track
        }

        var result = Array(repeating: [AssociationCandidate](), count: frameCount)
        for (frameIndex, candidates) in motionCandidatesByFrame
        where result.indices.contains(frameIndex) {
            result[frameIndex].append(contentsOf: candidates)
        }
        for track in eligible {
            for sample in track.samplesByFrame.values
            where result.indices.contains(sample.frameIndex) {
                result[sample.frameIndex].append(AssociationCandidate(
                    frameIndex: sample.frameIndex,
                    x: sample.x,
                    y: sample.y,
                    confidence: sample.confidence,
                    sources: ["apple_vision_trajectory"],
                    sourceRanks: ["apple_vision_trajectory": 1]
                ))
            }
        }
        for frameIndex in result.indices {
            let ordered = result[frameIndex].sorted {
                if $0.quality != $1.quality { return $0.quality > $1.quality }
                return $0.confidence > $1.confidence
            }
            var merged: [AssociationCandidate] = []
            for candidate in ordered {
                if let existing = merged.firstIndex(where: {
                    hypot($0.x - candidate.x, $0.y - candidate.y) <= 8
                }) {
                    let old = merged[existing]
                    let coordinateSource = candidate.confidence > old.confidence
                        ? candidate : old
                    var ranks = old.sourceRanks
                    for (source, rank) in candidate.sourceRanks {
                        ranks[source] = min(ranks[source] ?? rank, rank)
                    }
                    merged[existing] = AssociationCandidate(
                        frameIndex: frameIndex,
                        x: coordinateSource.x,
                        y: coordinateSource.y,
                        confidence: max(old.confidence, candidate.confidence),
                        sources: Array(Set(old.sources + candidate.sources)).sorted(),
                        sourceRanks: ranks
                    )
                } else if merged.count < 100 {
                    merged.append(candidate)
                }
            }
            result[frameIndex] = merged
        }
        return result
    }
}
