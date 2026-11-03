import Foundation

public struct RawCandidate: Codable, Sendable {
    public let frameIndex: Int
    public let x: Double
    public let y: Double
    public let confidence: Double
    public let source: String?
    public let rank: Int?

    enum CodingKeys: String, CodingKey {
        case x, y, confidence, source, rank
        case frameIndex = "frame_index"
    }
}

public struct CandidateDocument: Codable, Sendable {
    public let detector: String
    public let video: VideoMetadata
    public let candidates: [RawCandidate]

    public static func decode(from data: Data) throws -> CandidateDocument {
        try JSONDecoder().decode(CandidateDocument.self, from: data)
    }
}

public struct AssociationCandidate: Sendable {
    public let frameIndex: Int
    public let x: Double
    public let y: Double
    public let confidence: Double
    public let sources: [String]
    public let sourceRanks: [String: Int]

    public init(
        frameIndex: Int,
        x: Double,
        y: Double,
        confidence: Double,
        sources: [String],
        sourceRanks: [String: Int]
    ) {
        self.frameIndex = frameIndex
        self.x = x
        self.y = y
        self.confidence = confidence
        self.sources = sources
        self.sourceRanks = sourceRanks
    }

    var quality: Double {
        var value = 0.18 * max(0, min(1, confidence))
        if sources.contains("overlapping_tiled_yolo") {
            value += 0.22 * max(0, min(1, confidence))
        }
        if sources.contains("three_frame_motion") {
            value += 0.08
        }
        if sources.contains("apple_vision_trajectory") {
            value += 0.12 * max(0, min(1, confidence))
        }
        if sources == ["median_background_motion"] {
            value -= 0.25
        }
        if sources.count > 1 {
            value += 0.08
        }
        let bestRank = sourceRanks.values.min() ?? 100
        value += 0.08 * max(0, 1 - Double(bestRank - 1) / 100)
        return value
    }
}

public enum CandidateMergeError: Error, Equatable {
    case invalidMergeRadius
    case mismatchedVideoMetadata
}

public enum CandidateMerger {
    public static func combine(
        documents: [CandidateDocument],
        frameIndexes: Range<Int>,
        mergeRadius: Double,
        allowBackgroundOnly: Bool = true
    ) throws -> [[AssociationCandidate]] {
        guard mergeRadius >= 0 else { throw CandidateMergeError.invalidMergeRadius }
        guard let reference = documents.first?.video else {
            return Array(repeating: [], count: max(0, frameIndexes.upperBound))
        }
        guard documents.allSatisfy({
            $0.video.frameCount == reference.frameCount
                && $0.video.timestampsSeconds == reference.timestampsSeconds
        }) else {
            throw CandidateMergeError.mismatchedVideoMetadata
        }

        struct RankedRaw {
            let raw: RawCandidate
            let source: String
            let rank: Int
            let insertionOrder: Int
        }
        var rawByFrame = Array(repeating: [RankedRaw](), count: reference.frameCount)
        var insertionOrder = 0
        for document in documents {
            var perFrameRank: [Int: Int] = [:]
            for raw in document.candidates where frameIndexes.contains(raw.frameIndex) {
                let nextRank = (perFrameRank[raw.frameIndex] ?? 0) + 1
                perFrameRank[raw.frameIndex] = nextRank
                let source = raw.source ?? document.detector
                if source == "median_background_motion" && !allowBackgroundOnly {
                    continue
                }
                rawByFrame[raw.frameIndex].append(RankedRaw(
                    raw: raw,
                    source: source,
                    rank: raw.rank ?? nextRank,
                    insertionOrder: insertionOrder
                ))
                insertionOrder += 1
            }
        }

        var combined = Array(repeating: [AssociationCandidate](), count: reference.frameCount)
        for frameIndex in frameIndexes {
            let ordered = rawByFrame[frameIndex].sorted {
                if $0.raw.confidence != $1.raw.confidence {
                    return $0.raw.confidence > $1.raw.confidence
                }
                return $0.insertionOrder < $1.insertionOrder
            }
            var kept: [AssociationCandidate] = []
            for value in ordered {
                if let existingIndex = kept.firstIndex(where: {
                    hypot($0.x - value.raw.x, $0.y - value.raw.y) <= mergeRadius
                }) {
                    let existing = kept[existingIndex]
                    var sources = Set(existing.sources)
                    sources.insert(value.source)
                    var ranks = existing.sourceRanks
                    ranks[value.source] = min(ranks[value.source] ?? value.rank, value.rank)
                    kept[existingIndex] = AssociationCandidate(
                        frameIndex: frameIndex,
                        x: existing.x,
                        y: existing.y,
                        confidence: max(existing.confidence, value.raw.confidence),
                        sources: sources.sorted(),
                        sourceRanks: ranks
                    )
                } else {
                    kept.append(AssociationCandidate(
                        frameIndex: frameIndex,
                        x: value.raw.x,
                        y: value.raw.y,
                        confidence: value.raw.confidence,
                        sources: [value.source],
                        sourceRanks: [value.source: value.rank]
                    ))
                }
            }
            combined[frameIndex] = kept
        }
        return combined
    }
}
