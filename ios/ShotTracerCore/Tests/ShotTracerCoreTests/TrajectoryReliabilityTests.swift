import Foundation
import Testing
@testable import ShotTracerCore

private func reliabilityPoint(
    frame: Int,
    x: Double,
    y: Double,
    provenance: PointProvenance = .observed
) -> TrajectoryPoint {
    TrajectoryPoint(
        frameIndex: frame,
        timestampSeconds: Double(frame) / 60,
        x: x,
        y: y,
        confidence: 0.9,
        uncertaintyPixels: nil,
        opacity: 1,
        provenance: provenance,
        estimated: provenance != .observed,
        source: "test",
        sources: ["test"],
        sourceRanks: ["test": 1]
    )
}

@Test func presentationSmoothingReducesJitterWithoutMovingEndpoints() {
    let points = (0..<9).map { frame in
        reliabilityPoint(
            frame: frame,
            x: Double(frame * 20),
            y: 200 - Double(frame * 12) + (frame.isMultiple(of: 2) ? 9 : -9)
        )
    }
    let smoothed = TrajectoryPresentationSmoother.coordinates(for: points)
    #expect(smoothed.first?.x == points.first?.x)
    #expect(smoothed.first?.y == points.first?.y)
    #expect(smoothed.last?.x == points.last?.x)
    #expect(smoothed.last?.y == points.last?.y)
    #expect(abs(smoothed[4].y - (200 - Double(4 * 12))) < 9)
}

@Test func predictionGateRejectsAReversingTerminalTrack() {
    var points = (0..<12).map {
        reliabilityPoint(frame: $0, x: Double($0 * 20), y: 900 - Double($0 * 30))
    }
    points.append(reliabilityPoint(frame: 12, x: 180, y: 570))
    let timestamps = (0..<60).map { Double($0) / 60 }
    let document = TrajectoryDocument(
        trajectorySchemaVersion: 2,
        launchFrame: 0,
        terminationFrame: 12,
        observationEndFrame: 12,
        continuationMode: "observed_only",
        video: VideoMetadata(
            path: nil,
            width: 1080,
            height: 1920,
            fps: 60,
            frameCount: timestamps.count,
            timestampsSeconds: timestamps
        ),
        trajectory: points,
        continuation: nil
    )
    #expect(!TrajectoryQualityGate.permitsPrediction(
        document: document,
        associationScore: 4
    ))
}

@Test func predictionGateAcceptsAConsistentAssociatedTrack() {
    let points = (0..<14).map {
        reliabilityPoint(frame: $0, x: 300 + Double($0 * 18), y: 1_500 - Double($0 * 42))
    }
    let timestamps = (0..<90).map { Double($0) / 60 }
    let document = TrajectoryDocument(
        trajectorySchemaVersion: 2,
        launchFrame: 0,
        terminationFrame: 13,
        observationEndFrame: 13,
        continuationMode: "observed_only",
        video: VideoMetadata(
            path: nil,
            width: 1080,
            height: 1920,
            fps: 60,
            frameCount: timestamps.count,
            timestampsSeconds: timestamps
        ),
        trajectory: points,
        continuation: nil
    )
    #expect(TrajectoryQualityGate.permitsPrediction(
        document: document,
        associationScore: 4
    ))
}

@Test func visionCandidatesUseLaunchAnchoringAndIgnoreAnUnrelatedTrack() throws {
    let frameCount = 32
    var candidates = Array(repeating: [AssociationCandidate](), count: frameCount)
    for frame in 5...18 {
        let step = Double(frame - 5)
        candidates[frame].append(AssociationCandidate(
            frameIndex: frame,
            x: 500 + 46 * step,
            y: 1_500 - 62 * step,
            confidence: 0.92,
            sources: ["apple_vision_trajectory"],
            sourceRanks: ["apple_vision_trajectory": 1]
        ))
        candidates[frame].append(AssociationCandidate(
            frameIndex: frame,
            x: 900 - 24 * step,
            y: 500 + 16 * step,
            confidence: 0.98,
            sources: ["apple_vision_trajectory"],
            sourceRanks: ["apple_vision_trajectory": 2]
        ))
    }
    let timestamps = (0..<frameCount).map { Double($0) / 60 }
    let result = try TrajectoryAssociator.associateFullClip(
        candidatesByFrame: candidates,
        timestampsSeconds: timestamps,
        address: (500, 1_500)
    )
    #expect(result.launchFrame == 5)
    #expect(result.trajectory.count >= 12)
    #expect(result.trajectory.last?.x ?? 0 > 900)
}
