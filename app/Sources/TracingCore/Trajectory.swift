import Foundation

/// How a trajectory point was produced. Kept per point so failures can be
/// diagnosed later: the wiki's data-collection page requires storing both what
/// the model said and what the human corrected it to.
public enum PointSource: String, Codable, Sendable {
    /// Tracker/detector found the ball in this frame.
    case detected
    /// No reliable detection; position is a Kalman (or other) prediction.
    case predicted
    /// Placed or moved by the user in the correction UI.
    case manual
}

/// A single per-frame observation of the ball.
public struct TrackedPoint: Codable, Equatable, Sendable {
    public var frameIndex: Int
    /// Presentation timestamp in seconds. iPhone high-fps capture can be
    /// variable frame rate, so time must come from the PTS, never from
    /// `frameIndex / nominalFPS`.
    public var timestamp: Double
    /// Ball center in pixels, in the source video's coordinate space.
    public var x: Double
    public var y: Double
    /// False when the ball is known to be out of frame or occluded; the tracer
    /// still needs the interpolated position to draw through the gap.
    public var isVisible: Bool
    /// Confidence in [0, 1]. Manual points are 1.
    public var confidence: Double
    public var source: PointSource

    public init(
        frameIndex: Int,
        timestamp: Double,
        x: Double,
        y: Double,
        isVisible: Bool = true,
        confidence: Double = 1,
        source: PointSource
    ) {
        self.frameIndex = frameIndex
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.isVisible = isVisible
        self.confidence = confidence
        self.source = source
    }
}

/// The full tracked flight of one shot. This is the harness's JSON sidecar
/// format and, later, the app's editing model.
public struct Trajectory: Codable, Equatable, Sendable {
    public var impactFrameIndex: Int?
    /// Ordered by frameIndex.
    public var points: [TrackedPoint]

    public init(impactFrameIndex: Int? = nil, points: [TrackedPoint] = []) {
        self.impactFrameIndex = impactFrameIndex
        self.points = points
    }
}
