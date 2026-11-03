import Foundation
import OSLog
import ShotTracerCore

/// The production offline workflow: detect, optionally continue, render, and
/// export. Keeping orchestration outside the app target makes it reusable and
/// testable without UI lifecycle dependencies.
public struct OfflineShotProcessor: ShotProcessingClient {
    private static let performanceLogger = Logger(
        subsystem: "com.caio.ShotTracer",
        category: "performance"
    )

    public init() {}

    public func process(
        _ request: ShotProcessingRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ShotProcessingOutput {
        let clock = ContinuousClock()
        let totalStart = clock.now
        let detector = VisionShotDetector()
        let detection = try await detector.detect(
            sourceURL: request.sourceURL,
            manualAddress: request.manualAddress
        ) { value in
            progress(0.52 * value)
        }
        try Task.checkCancellation()

        var timings = detection.stageTimings
        let continuationStart = clock.now
        let profile: ShotProfile?
        let continuation: PhysicsTemplateContinuationResult?
        let predictionReady = TrajectoryQualityGate.permitsPrediction(
            document: detection.document,
            associationScore: detection.associationScore
        )
        if let override = request.profileOverride, predictionReady {
            profile = override
            continuation = try? PhysicsTemplateContinuation.generate(
                from: detection.document,
                durationSeconds: 6,
                address: detection.address,
                profile: override,
                profileSource: "explicit_user_input"
            )
        } else if predictionReady {
            switch AutomaticShotSetup.selectProfile(
                document: detection.document,
                address: detection.address
            ) {
            case let .accepted(selection) where selection.confidence >= 0.70:
                profile = selection.profile
                continuation = selection.continuation
            case .manualRequired:
                profile = nil
                continuation = nil
            case .accepted:
                profile = nil
                continuation = nil
            }
        } else {
            profile = nil
            continuation = nil
        }
        timings.record("continuation", seconds: (clock.now - continuationStart).secondsDouble)

        let document = continuation.map(detection.document.appendingPhysicsContinuation)
            ?? detection.document
        let outputURL = FileManager.default.temporaryDirectory
            .appending(path: "shot-tracer-output-\(UUID().uuidString)")
            .appendingPathExtension("mp4")
        let exporter = TrajectoryVideoExporter()
        let exportStart = clock.now
        _ = try await exporter.export(
            sourceURL: request.sourceURL,
            outputURL: outputURL,
            trajectory: document
        ) { value in
            progress(0.55 + 0.45 * value)
        }
        timings.record("export", seconds: (clock.now - exportStart).secondsDouble)

        let wallTotal = (clock.now - totalStart).secondsDouble
        Self.performanceLogger.notice(
            "Stage timings (frames=\(detection.document.video.frameCount, privacy: .public), \(detection.document.video.width, privacy: .public)x\(detection.document.video.height, privacy: .public)@\(detection.document.video.fps, privacy: .public)fps): \(timings.summary, privacy: .public) wall_total=\(wallTotal, privacy: .public)s"
        )

        return ShotProcessingOutput(
            videoURL: outputURL,
            selectedProfile: profile
        )
    }
}
