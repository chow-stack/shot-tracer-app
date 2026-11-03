import Foundation
import Testing
@testable import ShotTracerCore

private func repositoryRoot() -> URL {
    var directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0..<4 {
        directory.deleteLastPathComponent()
    }
    return directory
}

private func decode(_ relativePath: String) throws -> TrajectoryDocument {
    let data = try Data(contentsOf: repositoryRoot().appending(path: relativePath))
    return try TrajectoryDocument.decode(from: data)
}

private func decodeCandidates(_ relativePath: String) throws -> CandidateDocument {
    let data = try Data(contentsOf: repositoryRoot().appending(path: relativePath))
    return try CandidateDocument.decode(from: data)
}

private struct PhysicsReferenceMetadata: Decodable {
    struct Continuation: Decodable {
        struct Projection: Decodable {
            let timeScale: Double
            let timeOffsetSeconds: Double
            let cameraDistanceMeters: Double
            let horizontalLateralScale: Double
            let horizontalDepthScale: Double
            let verticalHeightScale: Double
            let verticalDepthScale: Double
            let anchorTemplateTimeSeconds: Double

            enum CodingKeys: String, CodingKey {
                case timeScale = "time_scale"
                case timeOffsetSeconds = "time_offset_seconds"
                case cameraDistanceMeters = "camera_distance_m"
                case horizontalLateralScale = "horizontal_lateral_scale"
                case horizontalDepthScale = "horizontal_depth_scale"
                case verticalHeightScale = "vertical_height_scale"
                case verticalDepthScale = "vertical_depth_scale"
                case anchorTemplateTimeSeconds = "anchor_template_time_seconds"
            }
        }

        struct VisibleFit: Decodable {
            let rms: Double
            let joinVelocityError: Double
            let addressError: Double

            enum CodingKeys: String, CodingKey {
                case rms = "rms_error_px"
                case joinVelocityError = "join_velocity_error_px_per_second"
                case addressError = "address_error_px"
            }
        }

        struct Stability: Decodable {
            let endpointShift: Double
            enum CodingKeys: String, CodingKey {
                case endpointShift = "endpoint_shift_px"
            }
        }

        struct Endpoint: Decodable {
            let x: Double
            let y: Double
        }

        let selectedTemplateID: String
        let closeAlternativeCount: Int
        let projection: Projection
        let visibleFit: VisibleFit
        let stability: Stability
        let endpoint: Endpoint

        enum CodingKeys: String, CodingKey {
            case selectedTemplateID = "selected_template_id"
            case closeAlternativeCount = "close_alternative_count"
            case projection
            case visibleFit = "visible_fit"
            case stability, endpoint
        }
    }

    let continuation: Continuation
}

private func expectPhysicsTemplateMatchesFrozenPython(
    sourcePath: String,
    referencePath: String,
    profile: ShotProfile,
    address: (x: Double, y: Double)
) throws {
    let source = try decode(sourcePath)
    let reference = try decode(referencePath)
    let metadataData = try Data(contentsOf: repositoryRoot().appending(path: referencePath))
    let metadata = try JSONDecoder().decode(PhysicsReferenceMetadata.self, from: metadataData)
    let actual = try PhysicsTemplateContinuation.generate(
        from: source,
        durationSeconds: 6,
        address: address,
        profile: profile
    )
    let expectedPoints = reference.trajectory.filter { $0.provenance == .inferred }
    #expect(actual.points.count == expectedPoints.count)
    #expect(actual.fit.selectedTemplateID == metadata.continuation.selectedTemplateID)
    #expect(actual.fit.closeAlternativeCount == metadata.continuation.closeAlternativeCount)
    #expect(abs(actual.fit.visibleRMSErrorPixels - metadata.continuation.visibleFit.rms) <= 1e-6)
    #expect(abs(actual.fit.joinVelocityErrorPixelsPerSecond
        - metadata.continuation.visibleFit.joinVelocityError) <= 1e-6)
    #expect(abs(actual.fit.addressErrorPixels
        - metadata.continuation.visibleFit.addressError) <= 1e-6)
    #expect(abs(actual.fit.stabilityEndpointShiftPixels
        - metadata.continuation.stability.endpointShift) <= 1e-6)
    #expect(abs(actual.fit.endpointX - metadata.continuation.endpoint.x) <= 1e-6)
    #expect(abs(actual.fit.endpointY - metadata.continuation.endpoint.y) <= 1e-6)
    let actualProjection = actual.fit.projection
    let expectedProjection = metadata.continuation.projection
    #expect(abs(actualProjection.timeScale - expectedProjection.timeScale) <= 1e-12)
    #expect(abs(actualProjection.timeOffsetSeconds - expectedProjection.timeOffsetSeconds) <= 1e-12)
    #expect(abs(actualProjection.cameraDistanceMeters
        - expectedProjection.cameraDistanceMeters) <= 1e-12)
    #expect(abs(actualProjection.horizontalLateralScale
        - expectedProjection.horizontalLateralScale) <= 1e-6)
    #expect(abs(actualProjection.horizontalDepthScale
        - expectedProjection.horizontalDepthScale) <= 1e-6)
    #expect(abs(actualProjection.verticalHeightScale
        - expectedProjection.verticalHeightScale) <= 1e-12)
    #expect(abs(actualProjection.verticalDepthScale
        - expectedProjection.verticalDepthScale) <= 1e-6)
    #expect(abs(actualProjection.anchorTemplateTimeSeconds
        - expectedProjection.anchorTemplateTimeSeconds) <= 1e-9)
    for (point, expected) in zip(actual.points, expectedPoints) {
        #expect(point.frameIndex == expected.frameIndex)
        #expect(abs(point.timestampSeconds - (expected.timestampSeconds ?? 0)) <= 1e-9)
        #expect(abs(point.x - expected.x) <= 1e-6)
        #expect(abs(point.y - expected.y) <= 1e-6)
        #expect(abs(point.uncertaintyPixels - (expected.uncertaintyPixels ?? 0)) <= 1e-6)
        #expect(abs(point.opacity - (expected.opacity ?? 0)) <= 1e-6)
        #expect(point.provenance == expected.provenance)
        #expect(point.source == expected.source)
    }
}

@Test func decodesFrozenAssociatedTrajectory() throws {
    let document = try decode("ml/output/top-iron.annotation-free.track.json")
    #expect(document.launchFrame == 141)
    #expect(document.terminationFrame == 235)
    #expect(document.trajectory.count == 95)
    #expect(document.video.timestampsSeconds.count == document.video.frameCount)
}

private func expectSwiftAssociationMatchesFrozenPythonTrack(
    candidatePaths: [String],
    trajectoryPath: String,
    expectedCandidateCount: Int? = nil
) throws {
    let candidateDocuments = try candidatePaths.map(decodeCandidates)
    let referenceVideo = try #require(candidateDocuments.first?.video)
    let python = try decode(trajectoryPath)
    let combined = try CandidateMerger.combine(
        documents: candidateDocuments,
        frameIndexes: 1..<(referenceVideo.frameCount - 1),
        mergeRadius: 8
    )
    if let expectedCandidateCount {
        #expect(combined.reduce(0) { $0 + $1.count } == expectedCandidateCount)
    }

    let swift = try TrajectoryAssociator.associateFullClip(
        candidatesByFrame: combined,
        timestampsSeconds: referenceVideo.timestampsSeconds,
        address: (610.1333333333333, 1674.6666666666667)
    )
    #expect(swift.launchFrame == python.launchFrame)
    #expect(swift.terminationFrame == python.terminationFrame)
    #expect(swift.trajectory.count == python.trajectory.count)
    for (actual, expected) in zip(swift.trajectory, python.trajectory) {
        let expectedTimestamp = expected.timestampSeconds
            ?? python.video.timestampsSeconds[expected.frameIndex]
        let expectedProvenance = expected.provenance
            ?? ((expected.estimated ?? false) ? .estimatedGap : .observed)
        #expect(actual.frameIndex == expected.frameIndex)
        #expect(actual.timestampSeconds == expectedTimestamp)
        #expect(actual.x == expected.x)
        #expect(actual.y == expected.y)
        #expect(actual.confidence == expected.confidence)
        #expect(actual.sources == (expected.sources ?? []))
        #expect(actual.sourceRanks == (expected.sourceRanks ?? [:]))
        #expect(actual.provenance == expectedProvenance)
    }
}

@Test func swiftMotionOnlyAssociationMatchesFrozenPythonTrack() throws {
    try expectSwiftAssociationMatchesFrozenPythonTrack(
        candidatePaths: ["ml/output/top-iron.annotation-free.motion-candidates.json"],
        trajectoryPath: "ml/output/top-iron.annotation-free.motion-track.json",
        expectedCandidateCount: 29_091
    )
}

@Test func swiftCombinedAssociationMatchesFrozenPythonTrack() throws {
    try expectSwiftAssociationMatchesFrozenPythonTrack(
        candidatePaths: [
            "ml/output/top-iron.annotation-free.motion-candidates.json",
            "ml/output/top-iron.annotation-free.tiled-candidates.json",
        ],
        trajectoryPath: "ml/output/top-iron.annotation-free.track.json"
    )
}

@Test func automaticAddressProposalFindsFrozenLaunchesWithoutATap() throws {
    let cases: [(name: String, launch: Int, x: Double, y: Double)] = [
        ("left-3wood", 131, 790, 1518),
        ("curve-3wood", 80, 720, 1516),
        ("straight-wedge", 247, 689.0667, 1649.0667),
        ("top-iron", 141, 610.1333, 1674.6667),
    ]
    for value in cases {
        let motion = try decodeCandidates(
            "ml/output/\(value.name).annotation-free.motion-candidates.json"
        )
        let tiled = try decodeCandidates(
            "ml/output/\(value.name).annotation-free.tiled-candidates.json"
        )
        let combined = try CandidateMerger.combine(
            documents: [motion, tiled],
            frameIndexes: 1..<(motion.video.frameCount - 1),
            mergeRadius: 8
        )
        let decision = AutomaticShotSetup.proposeAddress(
            candidatesByFrame: combined,
            frameSize: (motion.video.width, motion.video.height)
        )
        guard case let .accepted(proposal) = decision else {
            Issue.record("Automatic address unexpectedly required a tap for \(value.name)")
            continue
        }
        #expect(proposal.proposedLaunchFrame == value.launch)
        #expect(hypot(proposal.x - value.x, proposal.y - value.y) <= 40)
        #expect(proposal.confidence >= 0.55)
    }
}

@Test func automaticProfileSelectionMatchesAcceptedReviewProfiles() throws {
    let cases: [(path: String, profile: ShotProfile, x: Double, y: Double)] = [
        ("ml/output/left-3wood.annotation-free.track.json", .wood, 790, 1518),
        ("ml/output/curve-3wood.annotation-free.track.json", .wood, 720, 1516),
        ("ml/output/straight-wedge.annotation-free.track.json", .wedge, 689.0667, 1649.0667),
    ]
    for value in cases {
        let document = try decode(value.path)
        let decision = AutomaticShotSetup.selectProfile(
            document: document,
            address: (value.x, value.y)
        )
        guard case let .accepted(selection) = decision else {
            Issue.record("Automatic profile selection failed for \(value.path)")
            continue
        }
        #expect(selection.profile == value.profile)
        #expect(selection.continuation.profileSource == "automatic_template_fit")
        let completed = document.appendingPhysicsContinuation(selection.continuation)
        #expect(completed.continuationMode == "physics_template")
        #expect(completed.continuation?.shotProfile?.value == value.profile)
        #expect(completed.continuation?.shotProfile?.source == "automatic_template_fit")
        #expect(completed.continuation?.endpoint?.isMeasuredLanding == false)
        #expect(completed.trajectory.count > document.trajectory.count)
    }
}

@Test(arguments: [
    ("ml/output/left-3wood.physics-template.track.json", 158, 132),
    ("ml/output/curve-3wood.physics-template.track.json", 117, 138),
    ("ml/output/straight-wedge.physics-template.track.json", 305, 152),
])
func decodesFrozenPhysicsTemplate(
    relativePath: String,
    observationEndFrame: Int,
    inferredCount: Int
) throws {
    let document = try decode(relativePath)
    #expect(document.trajectorySchemaVersion == 2)
    #expect(document.observationEndFrame == observationEndFrame)
    #expect(document.continuationMode == "physics_template")
    #expect(document.continuation?.mode == "physics_template")
    #expect(document.continuation?.qualityStatus == "accepted")
    #expect(document.continuation?.shotProfile != nil)
    #expect(document.continuation?.endpoint?.isMeasuredLanding == false)
    #expect(document.trajectory.filter { $0.provenance == .inferred }.count == inferredCount)
}

@Test(arguments: [
    (
        "ml/output/left-3wood.annotation-free.track.json",
        "ml/output/left-3wood.physics-template.track.json",
        ShotProfile.wood,
        790.0,
        1518.0
    ),
    (
        "ml/output/curve-3wood.annotation-free.track.json",
        "ml/output/curve-3wood.physics-template.track.json",
        ShotProfile.wood,
        720.0,
        1516.0
    ),
    (
        "ml/output/straight-wedge.annotation-free.track.json",
        "ml/output/straight-wedge.physics-template.track.json",
        ShotProfile.wedge,
        689.0667,
        1649.0667
    ),
])
func swiftPhysicsTemplateMatchesFrozenPython(
    sourcePath: String,
    referencePath: String,
    profile: ShotProfile,
    addressX: Double,
    addressY: Double
) throws {
    try expectPhysicsTemplateMatchesFrozenPython(
        sourcePath: sourcePath,
        referencePath: referencePath,
        profile: profile,
        address: (addressX, addressY)
    )
}

@Test(.disabled("Host sandbox cannot decode the media fixture; run in the iOS/device integration scheme"))
func sequentialReaderYieldsMonotonicPresentationTimestamps() async throws {
    // The host test uses the software-decodable regression render. Source
    // iPhone HEVC Main 10 decoding is an on-device integration gate because
    // this sandbox does not expose the required media decoder.
    let reader = try await SequentialVideoFrameReader(
        url: repositoryRoot().appending(path: "ml/output/top-iron.annotation-free.mp4")
    )
    let trajectory = try decode("ml/output/top-iron.annotation-free.track.json")
    #expect(reader.width == trajectory.video.width)
    #expect(reader.height == trajectory.video.height)
    var previousTimestamp = -Double.infinity
    for expectedIndex in 0..<8 {
        let decoded = try reader.nextFrame()
        let frame = try #require(decoded)
        #expect(frame.frameIndex == expectedIndex)
        #expect(frame.presentationTimeSeconds > previousTimestamp)
        previousTimestamp = frame.presentationTimeSeconds
    }
    reader.cancel()
}

@Test func exporterRefusesToOverwriteAnExistingFile() async throws {
    let output = FileManager.default.temporaryDirectory
        .appending(path: "shot-tracer-existing-\(UUID().uuidString).mp4")
    try Data().write(to: output)
    defer { try? FileManager.default.removeItem(at: output) }
    let trajectory = try decode("ml/output/top-iron.annotation-free.track.json")
    do {
        _ = try await TrajectoryVideoExporter().export(
            sourceURL: repositoryRoot().appending(path: "does-not-need-to-exist.mov"),
            outputURL: output,
            trajectory: trajectory
        )
        Issue.record("Exporter unexpectedly overwrote an existing file")
    } catch let error as TrajectoryVideoExportError {
        #expect(error == .outputAlreadyExists)
    }
}

@Test(.disabled("Requires the device media integration scheme and bundled fixture"))
func exporterRendersSequentialVideoWithExactFrameCount() async throws {
    let source = repositoryRoot().appending(path: "ml/output/top-iron.annotation-free.mp4")
    let output = FileManager.default.temporaryDirectory
        .appending(path: "shot-tracer-export-\(UUID().uuidString).mp4")
    defer { try? FileManager.default.removeItem(at: output) }
    let trajectory = try decode("ml/output/top-iron.annotation-free.track.json")
    let result = try await TrajectoryVideoExporter().export(
        sourceURL: source,
        outputURL: output,
        trajectory: trajectory,
        configuration: .init(codec: .h264, preservesAudio: false)
    )
    #expect(result.renderedFrameCount == trajectory.video.frameCount)
    #expect(FileManager.default.fileExists(atPath: output.path))
}
