import Foundation

public enum PointProvenance: String, Codable, Sendable {
    case observed
    case estimatedGap = "estimated_gap"
    case inferred
}

public struct VideoMetadata: Codable, Sendable {
    public let path: String?
    public let width: Int
    public let height: Int
    public let fps: Double
    public let frameCount: Int
    public let timestampsSeconds: [Double]

    enum CodingKeys: String, CodingKey {
        case path, width, height, fps
        case frameCount = "frame_count"
        case timestampsSeconds = "timestamps_seconds"
    }
}

public struct TrajectoryPoint: Codable, Sendable {
    public let frameIndex: Int
    public let timestampSeconds: Double?
    public let x: Double
    public let y: Double
    public let confidence: Double?
    public let uncertaintyPixels: Double?
    public let opacity: Double?
    public let provenance: PointProvenance?
    public let estimated: Bool?
    public let source: String?
    public let sources: [String]?
    public let sourceRanks: [String: Int]?

    enum CodingKeys: String, CodingKey {
        case x, y, confidence, opacity, provenance, estimated, source, sources
        case frameIndex = "frame_index"
        case timestampSeconds = "timestamp_seconds"
        case uncertaintyPixels = "uncertainty_px"
        case sourceRanks = "source_ranks"
    }
}

public struct ContinuationEndpoint: Codable, Sendable {
    public let source: String
    public let x: Double
    public let y: Double
    public let isMeasuredLanding: Bool?

    enum CodingKeys: String, CodingKey {
        case source, x, y
        case isMeasuredLanding = "is_measured_landing"
    }
}

public struct ContinuationShotProfile: Codable, Sendable {
    public let value: ShotProfile
    public let source: String
}

public struct ContinuationMetadata: Codable, Sendable {
    public let mode: String
    public let qualityStatus: String
    public let shotProfile: ContinuationShotProfile?
    public let templateBankVersion: String?
    public let aerodynamicModelVersion: String?
    public let selectedTemplateID: String?
    public let closeAlternativeCount: Int?
    public let visibleRMSErrorPixels: Double?
    public let stabilityEndpointShiftPixels: Double?
    public let uncertaintyModelVersion: String?
    public let endpoint: ContinuationEndpoint?

    enum CodingKeys: String, CodingKey {
        case mode, endpoint
        case qualityStatus = "quality_status"
        case shotProfile = "shot_profile"
        case templateBankVersion = "template_bank_version"
        case aerodynamicModelVersion = "aerodynamic_model_version"
        case selectedTemplateID = "selected_template_id"
        case closeAlternativeCount = "close_alternative_count"
        case visibleRMSErrorPixels = "visible_fit_rms_error_px"
        case stabilityEndpointShiftPixels = "stability_endpoint_shift_px"
        case uncertaintyModelVersion = "uncertainty_model_version"
    }
}

public struct TrajectoryDocument: Codable, Sendable {
    public let trajectorySchemaVersion: Int?
    public let launchFrame: Int?
    public let terminationFrame: Int?
    public let observationEndFrame: Int?
    public let continuationMode: String?
    public let video: VideoMetadata
    public let trajectory: [TrajectoryPoint]
    public let continuation: ContinuationMetadata?

    enum CodingKeys: String, CodingKey {
        case video, trajectory, continuation
        case trajectorySchemaVersion = "trajectory_schema_version"
        case launchFrame = "launch_frame"
        case terminationFrame = "termination_frame"
        case observationEndFrame = "observation_end_frame"
        case continuationMode = "continuation_mode"
    }

    public static func decode(from data: Data) throws -> TrajectoryDocument {
        try JSONDecoder().decode(TrajectoryDocument.self, from: data)
    }

    public func appendingPhysicsContinuation(
        _ result: PhysicsTemplateContinuationResult
    ) -> TrajectoryDocument {
        let inferred = result.points.map { point in
            TrajectoryPoint(
                frameIndex: point.frameIndex,
                timestampSeconds: point.timestampSeconds,
                x: point.x,
                y: point.y,
                confidence: nil,
                uncertaintyPixels: point.uncertaintyPixels,
                opacity: point.opacity,
                provenance: point.provenance,
                estimated: false,
                source: point.source,
                sources: nil,
                sourceRanks: nil
            )
        }
        return TrajectoryDocument(
            trajectorySchemaVersion: 2,
            launchFrame: launchFrame,
            terminationFrame: terminationFrame,
            observationEndFrame: observationEndFrame,
            continuationMode: "physics_template",
            video: video,
            trajectory: trajectory + inferred,
            continuation: ContinuationMetadata(
                mode: "physics_template",
                qualityStatus: "accepted",
                shotProfile: ContinuationShotProfile(
                    value: result.profile,
                    source: result.profileSource
                ),
                templateBankVersion: PhysicsTemplateContinuation.templateBankVersion,
                aerodynamicModelVersion: PhysicsTemplateContinuation.aerodynamicModelVersion,
                selectedTemplateID: result.fit.selectedTemplateID,
                closeAlternativeCount: result.fit.closeAlternativeCount,
                visibleRMSErrorPixels: result.fit.visibleRMSErrorPixels,
                stabilityEndpointShiftPixels: result.fit.stabilityEndpointShiftPixels,
                uncertaintyModelVersion: PhysicsTemplateContinuation.uncertaintyModelVersion,
                endpoint: ContinuationEndpoint(
                    source: "physics_template_projection",
                    x: result.fit.endpointX,
                    y: result.fit.endpointY,
                    isMeasuredLanding: false
                )
            )
        )
    }
}
