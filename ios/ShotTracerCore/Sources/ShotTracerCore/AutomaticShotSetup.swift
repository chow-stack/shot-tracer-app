import Foundation

public enum AutomaticSetupFallbackReason: String, Sendable, Equatable {
    case noLaunchHypothesis
    case weakLaunchHypothesis
    case associationRejected
    case noPhysicsProfileFit
}

public struct AutomaticAddressProposal: Sendable, Equatable {
    public let x: Double
    public let y: Double
    public let proposedLaunchFrame: Int
    public let confidence: Double
}

public struct AutomaticShotInitialization: Sendable {
    public let address: AutomaticAddressProposal
    public let association: AssociationResult
}

public enum AutomaticAddressDecision: Sendable {
    case accepted(AutomaticAddressProposal)
    case manualRequired(AutomaticSetupFallbackReason)
}

public enum AutomaticInitializationDecision: Sendable {
    case accepted(AutomaticShotInitialization)
    case manualRequired(AutomaticSetupFallbackReason)
}

public struct AutomaticProfileSelection: Sendable {
    public let profile: ShotProfile
    public let confidence: Double
    public let continuation: PhysicsTemplateContinuationResult
}

public enum AutomaticProfileDecision: Sendable {
    case accepted(AutomaticProfileSelection)
    case manualRequired(AutomaticSetupFallbackReason)
}

/// A deliberately small pre-initializer. It proposes an address only from a
/// strong detector-supported upward launch in the lower frame, then requires
/// the frozen association engine to confirm a sustained trajectory.
public enum AutomaticShotSetup {
    private struct LaunchHypothesis {
        let score: Double
        let sequence: Int
        let start: AssociationCandidate
    }

    public static func proposeAddress(
        candidatesByFrame: [[AssociationCandidate]],
        frameSize: (width: Int, height: Int)
    ) -> AutomaticAddressDecision {
        guard frameSize.width > 0, frameSize.height > 0,
              candidatesByFrame.count >= 4 else {
            return .manualRequired(.noLaunchHypothesis)
        }
        let minimumY = 0.65 * Double(frameSize.height)
        var hypotheses: [LaunchHypothesis] = []
        var sequence = 0
        for frameIndex in 1..<(candidatesByFrame.count - 2) {
            for start in candidatesByFrame[frameIndex]
            where start.y >= minimumY
                && hasDetectorSupport(start) {
                for following in candidatesByFrame[frameIndex + 1] {
                    let dx = following.x - start.x
                    let dy = following.y - start.y
                    let displacement = hypot(dx, dy)
                    guard displacement >= 40, displacement <= 320,
                          dy < -35, abs(dx) <= 170 else { continue }
                    let score = start.quality + following.quality
                        - 0.002 * abs(dx)
                        + start.y / Double(frameSize.height)
                        + 0.30
                    hypotheses.append(LaunchHypothesis(
                        score: score, sequence: sequence, start: start
                    ))
                    sequence += 1
                }
            }
        }
        guard let best = hypotheses.max(by: hypothesisIsLess) else {
            return .manualRequired(.noLaunchHypothesis)
        }
        let nearBest = hypotheses.filter { $0.score >= best.score - 0.15 }
        guard let selected = nearBest.min(by: {
            if $0.start.frameIndex != $1.start.frameIndex {
                return $0.start.frameIndex < $1.start.frameIndex
            }
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.sequence < $1.sequence
        }), selected.score >= 1.60 else {
            return .manualRequired(.weakLaunchHypothesis)
        }
        let confidence = min(0.98, max(0.55, (selected.score - 1.35) / 0.90))
        return .accepted(AutomaticAddressProposal(
            x: selected.start.x,
            y: selected.start.y,
            proposedLaunchFrame: selected.start.frameIndex,
            confidence: confidence
        ))
    }

    public static func initialize(
        candidatesByFrame: [[AssociationCandidate]],
        timestampsSeconds: [Double],
        frameSize: (width: Int, height: Int),
        associationConfiguration: AssociationConfiguration = .init()
    ) -> AutomaticInitializationDecision {
        let proposal: AutomaticAddressProposal
        switch proposeAddress(candidatesByFrame: candidatesByFrame, frameSize: frameSize) {
        case let .accepted(value): proposal = value
        case let .manualRequired(reason): return .manualRequired(reason)
        }
        do {
            let association = try TrajectoryAssociator.associateFullClip(
                candidatesByFrame: candidatesByFrame,
                timestampsSeconds: timestampsSeconds,
                address: (proposal.x, proposal.y),
                configuration: associationConfiguration
            )
            guard abs(association.launchFrame - proposal.proposedLaunchFrame) <= 3,
                  association.trajectory.count >= 12,
                  association.finalScore > 0 else {
                return .manualRequired(.associationRejected)
            }
            return .accepted(AutomaticShotInitialization(
                address: proposal, association: association
            ))
        } catch {
            return .manualRequired(.associationRejected)
        }
    }

    public static func selectProfile(
        document: TrajectoryDocument,
        address: (x: Double, y: Double),
        durationSeconds: Double = 6,
        horizonY: Double? = nil
    ) -> AutomaticProfileDecision {
        let observedTimes = document.trajectory.compactMap { point -> Double? in
            let provenance = point.provenance
                ?? ((point.estimated ?? false) ? .estimatedGap : .observed)
            guard provenance == .observed,
                  document.video.timestampsSeconds.indices.contains(point.frameIndex) else {
                return nil
            }
            return point.timestampSeconds ?? document.video.timestampsSeconds[point.frameIndex]
        }.sorted()
        let observedSpan = max(0, (observedTimes.last ?? 0) - (observedTimes.first ?? 0))
        var values: [(ShotProfile, PhysicsTemplateContinuationResult, Double)] = []
        for profile in ShotProfile.allCases {
            if let continuation = try? PhysicsTemplateContinuation.generate(
                from: document,
                durationSeconds: durationSeconds,
                address: address,
                profile: profile,
                profileSource: "automatic_template_fit",
                horizonY: horizonY,
                horizonSource: horizonY == nil ? "not_provided" : "explicit_user_input"
            ) {
                // This preserves the reviewed fixture behavior, but the app
                // only exposes the continuation when the resulting profile
                // margin is strong. Weak family guesses remain observed-only.
                let profilePrior: Double
                switch profile {
                case .wood: profilePrior = observedSpan < 0.55 ? -2 : 0
                case .iron: profilePrior = 0
                case .wedge: profilePrior = observedSpan >= 0.80 ? -5 : 0
                }
                values.append((
                    profile,
                    continuation,
                    continuation.fit.selectionScore + profilePrior
                ))
            }
        }
        values.sort { $0.2 < $1.2 }
        guard let best = values.first else {
            return .manualRequired(.noPhysicsProfileFit)
        }
        let margin = values.count > 1
            ? values[1].2 - best.2 : 10
        let confidence = min(0.98, max(0.50, 0.55 + margin / 20))
        return .accepted(AutomaticProfileSelection(
            profile: best.0, confidence: confidence, continuation: best.1
        ))
    }

    private static func hypothesisIsLess(
        _ first: LaunchHypothesis, _ second: LaunchHypothesis
    ) -> Bool {
        if first.score != second.score { return first.score < second.score }
        return first.sequence > second.sequence
    }

    private static func hasDetectorSupport(_ candidate: AssociationCandidate) -> Bool {
        candidate.sources.contains("overlapping_tiled_yolo")
            || candidate.sources.contains("apple_vision_trajectory")
            || candidate.sources.contains("three_frame_motion")
    }
}
