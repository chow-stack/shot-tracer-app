import Foundation

public struct PhysicsObservation: Sendable, Equatable {
    public let frameIndex: Int
    public let timestampSeconds: Double
    public let x: Double
    public let y: Double

    public init(frameIndex: Int, timestampSeconds: Double, x: Double, y: Double) {
        self.frameIndex = frameIndex
        self.timestampSeconds = timestampSeconds
        self.x = x
        self.y = y
    }
}

public struct PhysicsContinuationPoint: Sendable, Equatable {
    public let frameIndex: Int
    public let timestampSeconds: Double
    public let x: Double
    public let y: Double
    public let uncertaintyPixels: Double
    public let opacity: Double
    public let provenance: PointProvenance
    public let source: String
}

public struct PhysicsProjectionParameters: Sendable, Equatable {
    public let timeScale: Double
    public let timeOffsetSeconds: Double
    public let cameraDistanceMeters: Double
    public let horizontalLateralScale: Double
    public let horizontalDepthScale: Double
    public let verticalHeightScale: Double
    public let verticalDepthScale: Double
    public let anchorTemplateTimeSeconds: Double
    public let anchorX: Double
    public let anchorY: Double
}

public struct PhysicsTemplateFitSummary: Sendable, Equatable {
    public let selectedTemplateID: String
    public let closeAlternativeCount: Int
    public let visibleRMSErrorPixels: Double
    public let joinVelocityErrorPixelsPerSecond: Double
    public let addressErrorPixels: Double
    public let stabilityEndpointShiftPixels: Double
    public let endpointX: Double
    public let endpointY: Double
    public let selectionScore: Double
    public let projection: PhysicsProjectionParameters
}

public struct PhysicsTemplateContinuationResult: Sendable, Equatable {
    public let profile: ShotProfile
    public let profileSource: String
    public let horizonY: Double
    public let horizonSource: String
    public let points: [PhysicsContinuationPoint]
    public let fit: PhysicsTemplateFitSummary
}

public enum PhysicsContinuationError: Error, Equatable {
    case insufficientObservations
    case invalidObservedPoint
    case observedTimesNotIncreasing
    case observedSpanTooShort
    case invalidFrameSize
    case invalidAddress
    case invalidHorizon
    case invalidDuration
    case mismatchedFutureFrames
    case noFutureFrames
    case futureTimesNotIncreasing
    case noTemplateFit
    case unstableFit
    case excessiveFitError
    case endpointOutsideFrame
    case templateAlreadyAtGround
    case continuationRejected
}

private struct Vector2: Sendable, Equatable {
    var x: Double
    var y: Double

    var magnitude: Double { hypot(x, y) }

    static func + (left: Self, right: Self) -> Self {
        Self(x: left.x + right.x, y: left.y + right.y)
    }

    static func - (left: Self, right: Self) -> Self {
        Self(x: left.x - right.x, y: left.y - right.y)
    }

    static func * (left: Self, right: Double) -> Self {
        Self(x: left.x * right, y: left.y * right)
    }
}

private struct FeaturePair {
    var first: [Vector2]
    var second: [Vector2]
}

private struct TemplateFit {
    let template: PhysicsTemplate
    let projection: PhysicsProjectionParameters
    let visibleRMSErrorPixels: Double
    let joinVelocityErrorPixelsPerSecond: Double
    let addressErrorPixels: Double
    let stabilityEndpointShiftPixels: Double
    let score: Double
    let sequence: Int
}

public enum PhysicsTemplateContinuation {
    public static let projectionModelVersion = "anchored-perspective-linear-v1"
    public static let uncertaintyModelVersion = "fit-growth-v1"
    public static let templateBankVersion = PhysicsTemplateBank.version
    public static let aerodynamicModelVersion = PhysicsSimulator.aerodynamicModelVersion

    private static let timeScales = [0.65, 0.85, 1.05, 1.3, 1.6, 2.0]
    private static let timeOffsetsSeconds = [0.0, 0.05, 0.10, 0.15]
    private static let cameraDistancesMeters = [4.0, 8.0, 15.0, 30.0, 60.0]
    private static let verticalHeightScales = [-600.0, -1_000.0, -1_600.0, -2_500.0]

    public static func generate(
        from document: TrajectoryDocument,
        durationSeconds: Double = 6,
        address: (x: Double, y: Double),
        profile: ShotProfile,
        profileSource: String = "explicit_user_input",
        horizonY: Double? = nil,
        horizonSource: String = "not_provided"
    ) throws -> PhysicsTemplateContinuationResult {
        guard durationSeconds.isFinite, durationSeconds > 0, durationSeconds <= 8 else {
            throw PhysicsContinuationError.invalidDuration
        }
        let observations = document.trajectory.compactMap { point -> PhysicsObservation? in
            let provenance = point.provenance
                ?? ((point.estimated ?? false) ? .estimatedGap : .observed)
            guard provenance == .observed,
                  document.video.timestampsSeconds.indices.contains(point.frameIndex) else {
                return nil
            }
            return PhysicsObservation(
                frameIndex: point.frameIndex,
                timestampSeconds: point.timestampSeconds
                    ?? document.video.timestampsSeconds[point.frameIndex],
                x: point.x,
                y: point.y
            )
        }
        guard let last = observations.max(by: {
            $0.timestampSeconds < $1.timestampSeconds
        }) else { throw PhysicsContinuationError.insufficientObservations }
        let endTime = last.timestampSeconds + durationSeconds
        let firstFutureFrame = last.frameIndex + 1
        guard firstFutureFrame < document.video.frameCount else {
            throw PhysicsContinuationError.noFutureFrames
        }
        let frameIndexes = Array(firstFutureFrame..<document.video.frameCount).filter {
            document.video.timestampsSeconds.indices.contains($0)
                && document.video.timestampsSeconds[$0] <= endTime
        }
        guard !frameIndexes.isEmpty else { throw PhysicsContinuationError.noFutureFrames }
        return try generate(
            observed: observations,
            frameIndexes: frameIndexes,
            timestampsSeconds: frameIndexes.map { document.video.timestampsSeconds[$0] },
            frameSize: (document.video.width, document.video.height),
            address: address,
            profile: profile,
            profileSource: profileSource,
            horizonY: horizonY,
            horizonSource: horizonSource
        )
    }

    public static func generate(
        observed: [PhysicsObservation],
        frameIndexes: [Int],
        timestampsSeconds: [Double],
        frameSize: (width: Int, height: Int),
        address: (x: Double, y: Double),
        profile: ShotProfile,
        profileSource: String = "explicit_user_input",
        horizonY: Double? = nil,
        horizonSource: String = "not_provided"
    ) throws -> PhysicsTemplateContinuationResult {
        guard frameIndexes.count == timestampsSeconds.count, !frameIndexes.isEmpty else {
            throw PhysicsContinuationError.mismatchedFutureFrames
        }
        let ordered = try validatedObservations(observed)
        guard frameSize.width > 0, frameSize.height > 0 else {
            throw PhysicsContinuationError.invalidFrameSize
        }
        guard address.x.isFinite, address.y.isFinite else {
            throw PhysicsContinuationError.invalidAddress
        }
        if let horizonY,
           (!horizonY.isFinite || horizonY < 0 || horizonY > Double(frameSize.height)) {
            throw PhysicsContinuationError.invalidHorizon
        }
        let effectiveHorizonY = horizonY ?? 0.32 * Double(frameSize.height)
        let effectiveHorizonSource = horizonY == nil
            ? "shared_default_frame_fraction" : horizonSource
        let fitResult = try fit(
            observed: ordered,
            frameSize: frameSize,
            address: address,
            profile: profile,
            horizonY: effectiveHorizonY
        )
        let selected = fitResult.fit
        let startTime = ordered[ordered.count - 1].timestampSeconds
        let elapsed = timestampsSeconds.map { $0 - startTime }
        guard elapsed[0] > 0,
              zip(elapsed, elapsed.dropFirst()).allSatisfy({ $0 < $1 }) else {
            throw PhysicsContinuationError.futureTimesNotIncreasing
        }
        let parameters = selected.projection
        let allTemplateTimes = elapsed.map {
            parameters.anchorTemplateTimeSeconds + parameters.timeScale * $0
        }
        let validCount = allTemplateTimes.prefix {
            $0 <= selected.template.simulation.flightSeconds + 1e-9
        }.count
        guard validCount > 0 else { throw PhysicsContinuationError.templateAlreadyAtGround }
        let templateTimes = Array(allTemplateTimes.prefix(validCount))
        guard let positions = samplePositions(selected.template, at: templateTimes),
              let anchorPosition = samplePositions(
                selected.template, at: [parameters.anchorTemplateTimeSeconds]
              )?.first else {
            throw PhysicsContinuationError.templateAlreadyAtGround
        }
        var projected = try project(
            positions: positions, anchorPosition: anchorPosition, parameters: parameters
        )
        let terminalVelocity = terminalVelocity(ordered)
        let epsilon = min(
            0.01,
            0.25 * (selected.template.simulation.flightSeconds
                - parameters.anchorTemplateTimeSeconds)
        )
        guard let derivativePositions = samplePositions(selected.template, at: [
            parameters.anchorTemplateTimeSeconds,
            parameters.anchorTemplateTimeSeconds + epsilon,
        ]) else { throw PhysicsContinuationError.templateAlreadyAtGround }
        let derivativePixels = try project(
            positions: derivativePositions,
            anchorPosition: anchorPosition,
            parameters: parameters
        )
        let templateVelocity = (derivativePixels[1] - derivativePixels[0])
            * (parameters.timeScale / epsilon)
        let velocityDelta = terminalVelocity - templateVelocity
        let validElapsed = Array(elapsed.prefix(validCount))
        for index in projected.indices {
            let seconds = validElapsed[index]
            projected[index] = projected[index]
                + velocityDelta * (seconds * exp(-seconds / 0.35))
        }

        let width = Double(frameSize.width)
        let height = Double(frameSize.height)
        let diagonal = hypot(width, height)
        let baseUncertainty = max(5, selected.visibleRMSErrorPixels)
        var points: [PhysicsContinuationPoint] = []
        for index in 0..<validCount {
            let pixel = projected[index]
            let seconds = validElapsed[index]
            let uncertainty = baseUncertainty + 10 * pow(seconds, 1.35)
                + 0.10 * selected.stabilityEndpointShiftPixels
            if uncertainty > 0.14 * diagonal { break }
            if pixel.x < -0.12 * width || pixel.x > 1.12 * width
                || pixel.y < -0.12 * height || pixel.y > 1.08 * height { break }
            let opacity = min(
                0.78,
                max(0.16, 0.82 * (1 - uncertainty / (0.16 * diagonal)))
            )
            points.append(PhysicsContinuationPoint(
                frameIndex: frameIndexes[index],
                timestampSeconds: timestampsSeconds[index],
                x: pixel.x,
                y: pixel.y,
                uncertaintyPixels: uncertainty,
                opacity: opacity,
                provenance: .inferred,
                source: "physics_template_projection"
            ))
        }
        guard !points.isEmpty else { throw PhysicsContinuationError.continuationRejected }
        let endpoint = try endpoint(selected.template, parameters: parameters)
        return PhysicsTemplateContinuationResult(
            profile: profile,
            profileSource: profileSource,
            horizonY: effectiveHorizonY,
            horizonSource: effectiveHorizonSource,
            points: points,
            fit: PhysicsTemplateFitSummary(
                selectedTemplateID: selected.template.specification.identifier,
                closeAlternativeCount: fitResult.closeCount,
                visibleRMSErrorPixels: selected.visibleRMSErrorPixels,
                joinVelocityErrorPixelsPerSecond: selected.joinVelocityErrorPixelsPerSecond,
                addressErrorPixels: selected.addressErrorPixels,
                stabilityEndpointShiftPixels: selected.stabilityEndpointShiftPixels,
                endpointX: endpoint.x,
                endpointY: endpoint.y,
                selectionScore: selected.score,
                projection: parameters
            )
        )
    }

    private static func validatedObservations(
        _ values: [PhysicsObservation]
    ) throws -> [PhysicsObservation] {
        guard values.count >= 8 else { throw PhysicsContinuationError.insufficientObservations }
        let ordered = values.sorted { $0.timestampSeconds < $1.timestampSeconds }
        guard ordered.allSatisfy({
            $0.timestampSeconds.isFinite && $0.x.isFinite && $0.y.isFinite
        }) else { throw PhysicsContinuationError.invalidObservedPoint }
        guard zip(ordered, ordered.dropFirst()).allSatisfy({
            $0.timestampSeconds < $1.timestampSeconds
        }) else { throw PhysicsContinuationError.observedTimesNotIncreasing }
        guard ordered.last!.timestampSeconds - ordered.first!.timestampSeconds >= 0.10 else {
            throw PhysicsContinuationError.observedSpanTooShort
        }
        return ordered
    }

    private static func fit(
        observed: [PhysicsObservation],
        frameSize: (width: Int, height: Int),
        address: (x: Double, y: Double),
        profile: ShotProfile,
        horizonY: Double
    ) throws -> (fit: TemplateFit, closeCount: Int) {
        var candidates: [TemplateFit] = []
        var sequence = 0
        for template in PhysicsTemplateBank.templates(profile: profile) {
            for timeScale in timeScales {
                for timeOffset in timeOffsetsSeconds {
                    for cameraDistance in cameraDistancesMeters {
                        for verticalHeightScale in verticalHeightScales {
                            if let mapping = try fixedMapping(
                                template: template,
                                observed: observed,
                                timeScale: timeScale,
                                timeOffset: timeOffset,
                                cameraDistance: cameraDistance,
                                address: address,
                                horizonY: horizonY,
                                verticalHeightScale: verticalHeightScale
                            ) {
                                let endpoint = try endpoint(
                                    template, parameters: mapping.parameters
                                )
                                let horizonPenalty = 0.12 * abs(endpoint.y - horizonY)
                                let score = mapping.rms + 0.08 * mapping.addressError
                                    + 0.025 * mapping.velocityError + horizonPenalty
                                candidates.append(TemplateFit(
                                    template: template,
                                    projection: mapping.parameters,
                                    visibleRMSErrorPixels: mapping.rms,
                                    joinVelocityErrorPixelsPerSecond: mapping.velocityError,
                                    addressErrorPixels: mapping.addressError,
                                    stabilityEndpointShiftPixels: .infinity,
                                    score: score,
                                    sequence: sequence
                                ))
                                sequence += 1
                            }
                        }
                    }
                }
            }
        }
        guard !candidates.isEmpty else { throw PhysicsContinuationError.noTemplateFit }
        candidates.sort {
            $0.score == $1.score ? $0.sequence < $1.sequence : $0.score < $1.score
        }
        let drop = max(1, Int((0.15 * Double(observed.count)).rounded(.toNearestOrEven)))
        let shortened = Array(observed.dropFirst(drop))
        var stable: [TemplateFit] = []
        for candidate in candidates.prefix(12) {
            let parameters = candidate.projection
            let adjustedOffset = parameters.timeOffsetSeconds + parameters.timeScale
                * (shortened[0].timestampSeconds - observed[0].timestampSeconds)
            guard let refit = try fixedMapping(
                template: candidate.template,
                observed: shortened,
                timeScale: parameters.timeScale,
                timeOffset: adjustedOffset,
                cameraDistance: parameters.cameraDistanceMeters,
                address: address,
                horizonY: horizonY,
                verticalHeightScale: parameters.verticalHeightScale
            ) else { continue }
            let originalEndpoint = try endpoint(candidate.template, parameters: parameters)
            let refitEndpoint = try endpoint(
                candidate.template, parameters: refit.parameters
            )
            let shift = (originalEndpoint - refitEndpoint).magnitude
            stable.append(TemplateFit(
                template: candidate.template,
                projection: parameters,
                visibleRMSErrorPixels: candidate.visibleRMSErrorPixels,
                joinVelocityErrorPixelsPerSecond: candidate.joinVelocityErrorPixelsPerSecond,
                addressErrorPixels: candidate.addressErrorPixels,
                stabilityEndpointShiftPixels: shift,
                score: candidate.score + 0.08 * shift,
                sequence: candidate.sequence
            ))
        }
        guard !stable.isEmpty else { throw PhysicsContinuationError.unstableFit }
        stable.sort {
            $0.score == $1.score ? $0.sequence < $1.sequence : $0.score < $1.score
        }
        let best = stable[0]
        guard best.visibleRMSErrorPixels <= 60 else {
            throw PhysicsContinuationError.excessiveFitError
        }
        guard best.stabilityEndpointShiftPixels <= 120 else {
            throw PhysicsContinuationError.unstableFit
        }
        let selectedEndpoint = try endpoint(best.template, parameters: best.projection)
        let width = Double(frameSize.width)
        let height = Double(frameSize.height)
        guard selectedEndpoint.x >= -0.1 * width, selectedEndpoint.x <= 1.1 * width,
              selectedEndpoint.y >= -0.1 * height, selectedEndpoint.y <= 1.05 * height else {
            throw PhysicsContinuationError.endpointOutsideFrame
        }
        let closeThreshold = best.score * 1.10 + 2
        return (best, stable.filter { $0.score <= closeThreshold }.count)
    }

    private static func fixedMapping(
        template: PhysicsTemplate,
        observed: [PhysicsObservation],
        timeScale: Double,
        timeOffset: Double,
        cameraDistance: Double,
        address: (x: Double, y: Double),
        horizonY: Double,
        verticalHeightScale: Double
    ) throws -> (parameters: PhysicsProjectionParameters, rms: Double,
                  addressError: Double, velocityError: Double)? {
        let firstTime = observed[0].timestampSeconds
        let templateTimes = observed.map {
            timeOffset + timeScale * ($0.timestampSeconds - firstTime)
        }
        guard let positions = samplePositions(template, at: templateTimes) else { return nil }
        let anchorPosition = positions[positions.count - 1]
        let featureValues = try features(positions, cameraDistance: cameraDistance)
        let anchorHorizontal = featureValues.first[featureValues.first.count - 1]
        let anchorVertical = featureValues.second[featureValues.second.count - 1]
        var design: [Vector2] = featureValues.first.map { $0 - anchorHorizontal }
        let launchPosition = template.simulation.states[0].positionMeters
        let launchFeatures = try features([launchPosition], cameraDistance: cameraDistance)
        design.append((launchFeatures.first[0] - anchorHorizontal) * 0.35)
        let finalPixel = Vector2(x: observed.last!.x, y: observed.last!.y)
        var xValues = observed.map { $0.x - finalPixel.x }
        xValues.append(0.35 * (address.x - finalPixel.x))
        guard let horizontalCoefficients = solveRidge(design: design, values: xValues) else {
            return nil
        }
        let parameters = PhysicsProjectionParameters(
            timeScale: timeScale,
            timeOffsetSeconds: timeOffset,
            cameraDistanceMeters: cameraDistance,
            horizontalLateralScale: horizontalCoefficients.x,
            horizontalDepthScale: horizontalCoefficients.y,
            verticalHeightScale: verticalHeightScale,
            verticalDepthScale: -(address.y - horizonY),
            anchorTemplateTimeSeconds: templateTimes.last!,
            anchorX: finalPixel.x,
            anchorY: finalPixel.y
        )
        let projected = try project(
            positions: positions, anchorPosition: anchorPosition, parameters: parameters
        )
        let squaredResiduals = zip(projected, observed).map {
            let delta = $0 - Vector2(x: $1.x, y: $1.y)
            return delta.x * delta.x + delta.y * delta.y
        }
        let rms = sqrt(squaredResiduals.reduce(0, +) / Double(squaredResiduals.count))
        let launchProjected = try project(
            positions: [launchPosition], anchorPosition: anchorPosition, parameters: parameters
        )[0]
        let addressError = (launchProjected - Vector2(x: address.x, y: address.y)).magnitude
        let epsilon = min(
            0.01, 0.25 * (template.simulation.flightSeconds - templateTimes.last!)
        )
        guard epsilon > 1e-6,
              let derivativePositions = samplePositions(
                template, at: [templateTimes.last!, templateTimes.last! + epsilon]
              ) else { return nil }
        let derivativePixels = try project(
            positions: derivativePositions,
            anchorPosition: anchorPosition,
            parameters: parameters
        )
        let predictedVelocity = (derivativePixels[1] - derivativePixels[0])
            * (timeScale / epsilon)
        let velocityError = (predictedVelocity - terminalVelocity(observed)).magnitude
        _ = anchorVertical
        return (parameters, rms, addressError, velocityError)
    }

    private static func samplePositions(
        _ template: PhysicsTemplate, at requestedTimes: [Double]
    ) -> [Vector3]? {
        guard let first = requestedTimes.first, let last = requestedTimes.last,
              first >= 0, last <= template.simulation.flightSeconds else { return nil }
        let sourceTimes = template.simulation.timesSeconds
        let states = template.simulation.states
        return requestedTimes.map { time in
            if time == sourceTimes[sourceTimes.count - 1] {
                return states[states.count - 1].positionMeters
            }
            var low = 0
            var high = sourceTimes.count - 1
            while low + 1 < high {
                let middle = (low + high) / 2
                if sourceTimes[middle] <= time { low = middle } else { high = middle }
            }
            let leftTime = sourceTimes[low]
            let rightTime = sourceTimes[high]
            let ratio = (time - leftTime) / (rightTime - leftTime)
            let left = states[low].positionMeters
            let right = states[high].positionMeters
            return left + (right - left) * ratio
        }
    }

    private static func features(
        _ positions: [Vector3], cameraDistance: Double
    ) throws -> FeaturePair {
        var horizontal: [Vector2] = []
        var vertical: [Vector2] = []
        for position in positions {
            let denominator = cameraDistance + position.x
            guard denominator > 1e-6 else { throw PhysicsContinuationError.noTemplateFit }
            horizontal.append(Vector2(
                x: position.y / denominator, y: position.x / denominator
            ))
            vertical.append(Vector2(
                x: position.z / denominator, y: position.x / denominator
            ))
        }
        return FeaturePair(first: horizontal, second: vertical)
    }

    private static func project(
        positions: [Vector3],
        anchorPosition: Vector3,
        parameters: PhysicsProjectionParameters
    ) throws -> [Vector2] {
        let valueFeatures = try features(
            positions, cameraDistance: parameters.cameraDistanceMeters
        )
        let anchorFeatures = try features(
            [anchorPosition], cameraDistance: parameters.cameraDistanceMeters
        )
        return positions.indices.map { index in
            let horizontal = valueFeatures.first[index] - anchorFeatures.first[0]
            let vertical = valueFeatures.second[index] - anchorFeatures.second[0]
            return Vector2(
                x: parameters.anchorX
                    + horizontal.x * parameters.horizontalLateralScale
                    + horizontal.y * parameters.horizontalDepthScale,
                y: parameters.anchorY
                    + vertical.x * parameters.verticalHeightScale
                    + vertical.y * parameters.verticalDepthScale
            )
        }
    }

    private static func endpoint(
        _ template: PhysicsTemplate, parameters: PhysicsProjectionParameters
    ) throws -> Vector2 {
        guard let anchor = samplePositions(
            template, at: [parameters.anchorTemplateTimeSeconds]
        )?.first else { throw PhysicsContinuationError.noTemplateFit }
        return try project(
            positions: [template.simulation.states.last!.positionMeters],
            anchorPosition: anchor,
            parameters: parameters
        )[0]
    }

    private static func solveRidge(design: [Vector2], values: [Double]) -> Vector2? {
        var a00 = 1e-8
        var a01 = 0.0
        var a11 = 1e-8
        var b0 = 0.0
        var b1 = 0.0
        for (row, value) in zip(design, values) {
            a00 += row.x * row.x
            a01 += row.x * row.y
            a11 += row.y * row.y
            b0 += row.x * value
            b1 += row.y * value
        }
        let determinant = a00 * a11 - a01 * a01
        guard determinant.isFinite, abs(determinant) > 1e-20 else { return nil }
        return Vector2(
            x: (b0 * a11 - a01 * b1) / determinant,
            y: (a00 * b1 - a01 * b0) / determinant
        )
    }

    private static func terminalVelocity(_ observed: [PhysicsObservation]) -> Vector2 {
        let count = min(7, observed.count)
        let tail = observed.suffix(count)
        let values = zip(tail, tail.dropFirst()).map { first, second in
            let deltaTime = second.timestampSeconds - first.timestampSeconds
            return Vector2(
                x: (second.x - first.x) / deltaTime,
                y: (second.y - first.y) / deltaTime
            )
        }
        return Vector2(
            x: median(values.map(\.x)),
            y: median(values.map(\.y))
        )
    }

    private static func median(_ values: [Double]) -> Double {
        let ordered = values.sorted()
        let middle = ordered.count / 2
        if ordered.count.isMultiple(of: 2) {
            return (ordered[middle - 1] + ordered[middle]) / 2
        }
        return ordered[middle]
    }
}
