import Foundation
import ShotTracerCore

public struct ShotProcessingRequest: Sendable {
    public let sourceURL: URL
    public let manualAddress: CGPoint
    public let profileOverride: ShotProfile?

    public init(sourceURL: URL, manualAddress: CGPoint, profileOverride: ShotProfile?) {
        self.sourceURL = sourceURL
        self.manualAddress = manualAddress
        self.profileOverride = profileOverride
    }
}

public struct ShotProcessingOutput: Sendable {
    public let videoURL: URL
    public let selectedProfile: ShotProfile?

    public init(
        videoURL: URL,
        selectedProfile: ShotProfile?
    ) {
        self.videoURL = videoURL
        self.selectedProfile = selectedProfile
    }
}

public protocol ShotProcessingClient: Sendable {
    func process(
        _ request: ShotProcessingRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ShotProcessingOutput
}

public struct ClosureShotProcessingClient: ShotProcessingClient {
    public typealias Handler = @Sendable (
        ShotProcessingRequest,
        @escaping @Sendable (Double) -> Void
    ) async throws -> ShotProcessingOutput

    private let handler: Handler

    public init(handler: @escaping Handler) {
        self.handler = handler
    }

    public func process(
        _ request: ShotProcessingRequest,
        progress: @escaping @Sendable (Double) -> Void
    ) async throws -> ShotProcessingOutput {
        try await handler(request, progress)
    }
}
