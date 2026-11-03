import Foundation

extension Duration {
    public var secondsDouble: Double {
        let (seconds, attoseconds) = components
        return Double(seconds) + Double(attoseconds) / 1e18
    }
}

/// Accumulates named wall-clock stage durations for on-device performance
/// profiling. Additive/merge-friendly so each pipeline stage can report its
/// own slice and the orchestrator can assemble one end-to-end summary.
public struct StageTimings: Sendable {
    public struct Entry: Sendable {
        public let name: String
        public let seconds: Double
    }

    public private(set) var entries: [Entry] = []

    public init() {}

    public mutating func record(_ name: String, seconds: Double) {
        entries.append(Entry(name: name, seconds: seconds))
    }

    public mutating func merge(_ other: StageTimings) {
        entries.append(contentsOf: other.entries)
    }

    public var totalSeconds: Double {
        entries.reduce(0) { $0 + $1.seconds }
    }

    public var summary: String {
        let stageText = entries
            .map { String(format: "%@=%.3fs", $0.name, $0.seconds) }
            .joined(separator: " ")
        return String(format: "%@ total=%.3fs", stageText, totalSeconds)
    }
}
