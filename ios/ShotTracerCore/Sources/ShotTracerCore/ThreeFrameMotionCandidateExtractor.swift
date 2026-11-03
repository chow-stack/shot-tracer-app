import Accelerate
import CoreVideo
import Foundation

enum ThreeFrameMotionCandidateError: Error {
    case unsupportedPixelFormat(OSType)
    case unavailablePixelBufferMemory
    case scaleFailed(vImage_Error)
}

/// Streaming three-frame motion response used as a second, detector-independent
/// candidate source. Work is bounded to two difference buffers, two thresholds,
/// 100 candidates per frame, and one sequential pass through the video.
struct ThreeFrameMotionCandidateExtractor {
    private let sourceWidth: Int
    private let sourceHeight: Int
    private let analysisWidth: Int
    private let analysisHeight: Int
    private var previousPixels: [UInt8]?
    private var previousChange: [UInt8]?
    private var previousFrameIndex: Int?
    private let clock = ContinuousClock()
    /// Cumulative time in vImage downscale plus frame-difference/change work,
    /// for stage-level device profiling. Excludes component extraction.
    public private(set) var vImageDownscaleChangeSeconds: Double = 0
    /// Cumulative time in threshold/flood-fill connected-component extraction,
    /// for stage-level device profiling.
    public private(set) var connectedComponentsSeconds: Double = 0

    init(width: Int, height: Int) {
        sourceWidth = width
        sourceHeight = height
        // Half-resolution analysis preserves the launch-sized ball while
        // bounding CPU and memory on 60 fps iPhone video.
        analysisWidth = max(1, (width + 1) / 2)
        analysisHeight = max(1, (height + 1) / 2)
    }

    mutating func consume(
        pixelBuffer: CVPixelBuffer,
        frameIndex: Int
    ) throws -> (frameIndex: Int, candidates: [AssociationCandidate])? {
        let scaleStart = clock.now
        let currentPixels = try scaledBGRA(pixelBuffer)
        vImageDownscaleChangeSeconds += (clock.now - scaleStart).secondsDouble
        guard let previousPixels else {
            self.previousPixels = currentPixels
            previousFrameIndex = frameIndex
            return nil
        }
        let changeStart = clock.now
        let currentChange = Self.frameChange(
            previousPixels,
            currentPixels,
            pixelCount: analysisWidth * analysisHeight
        )
        vImageDownscaleChangeSeconds += (clock.now - changeStart).secondsDouble
        defer {
            self.previousPixels = currentPixels
            previousChange = currentChange
            previousFrameIndex = frameIndex
        }
        guard let previousChange, let middleFrame = previousFrameIndex else {
            return nil
        }
        let combineStart = clock.now
        var response = [UInt8](repeating: 0, count: currentChange.count)
        response.withUnsafeMutableBufferPointer { output in
            previousChange.withUnsafeBufferPointer { first in
                currentChange.withUnsafeBufferPointer { second in
                    guard let outputBase = output.baseAddress,
                          let firstBase = first.baseAddress,
                          let secondBase = second.baseAddress else { return }
                    for index in 0..<output.count {
                        outputBase[index] = min(firstBase[index], secondBase[index])
                    }
                }
            }
        }
        vImageDownscaleChangeSeconds += (clock.now - combineStart).secondsDouble
        let scaleX = Double(sourceWidth) / Double(analysisWidth)
        let scaleY = Double(sourceHeight) / Double(analysisHeight)
        let componentsStart = clock.now
        let candidates = Self.extractCandidates(
            response: response,
            width: analysisWidth,
            height: analysisHeight,
            frameIndex: middleFrame,
            scaleX: scaleX,
            scaleY: scaleY
        )
        connectedComponentsSeconds += (clock.now - componentsStart).secondsDouble
        return (middleFrame, candidates)
    }

    static func extractCandidates(
        response: [UInt8],
        width: Int,
        height: Int,
        frameIndex: Int,
        scaleX: Double = 1,
        scaleY: Double = 1
    ) -> [AssociationCandidate] {
        guard width > 0, height > 0, response.count == width * height else { return [] }
        struct Component {
            let x: Double
            let y: Double
            let confidence: Double
        }
        var components: [Component] = []
        // Downsampling reduces response amplitude, so use two inclusive
        // thresholds instead of the three full-resolution Python thresholds.
        for threshold: UInt8 in [8, 18] {
            var mask = response.map { $0 >= threshold ? UInt8(1) : UInt8(0) }
            var queue: [Int] = []
            queue.reserveCapacity(320)
            for seed in mask.indices where mask[seed] != 0 {
                queue.removeAll(keepingCapacity: true)
                queue.append(seed)
                mask[seed] = 0
                var cursor = 0
                var area = 0
                var minX = width
                var maxX = 0
                var minY = height
                var maxY = 0
                var weightTotal = 0.0
                var weightedX = 0.0
                var weightedY = 0.0
                var values: [UInt8] = []
                values.reserveCapacity(64)
                while cursor < queue.count {
                    let index = queue[cursor]
                    cursor += 1
                    let y = index / width
                    let x = index - y * width
                    let value = response[index]
                    area += 1
                    minX = min(minX, x)
                    maxX = max(maxX, x)
                    minY = min(minY, y)
                    maxY = max(maxY, y)
                    let weight = Double(value)
                    weightTotal += weight
                    weightedX += Double(x) * weight
                    weightedY += Double(y) * weight
                    if values.count <= 300 { values.append(value) }

                    let y0 = max(0, y - 1)
                    let y1 = min(height - 1, y + 1)
                    let x0 = max(0, x - 1)
                    let x1 = min(width - 1, x + 1)
                    for neighborY in y0...y1 {
                        for neighborX in x0...x1 {
                            let neighbor = neighborY * width + neighborX
                            if mask[neighbor] != 0 {
                                mask[neighbor] = 0
                                queue.append(neighbor)
                            }
                        }
                    }
                }
                let componentWidth = maxX - minX + 1
                let componentHeight = maxY - minY + 1
                let aspect = Double(max(componentWidth, componentHeight))
                    / Double(max(1, min(componentWidth, componentHeight)))
                guard area >= 1, area <= 300,
                      componentWidth <= 55, componentHeight <= 55,
                      aspect <= 20, weightTotal > 0 else { continue }
                values.sort()
                let percentileIndex = min(
                    values.count - 1,
                    Int((0.90 * Double(values.count - 1)).rounded())
                )
                let peak90 = Double(values[percentileIndex]) / 255
                let mean = weightTotal / Double(area) / 255
                let sizeScore = min(1, sqrt(Double(area) / 2))
                    * exp(-Double(area) / 300)
                let confidence = min(1, 0.15 + 0.55 * peak90 + 0.25 * mean
                    + 0.20 * sizeScore)
                components.append(Component(
                    x: weightedX / weightTotal,
                    y: weightedY / weightTotal,
                    confidence: confidence
                ))
            }
        }

        let ordered = components.sorted { $0.confidence > $1.confidence }
        var merged: [Component] = []
        for component in ordered {
            guard !merged.contains(where: {
                hypot($0.x - component.x, $0.y - component.y) <= 4
            }) else { continue }
            merged.append(component)
            if merged.count == 100 { break }
        }
        return merged.enumerated().map { offset, component in
            AssociationCandidate(
                frameIndex: frameIndex,
                x: (component.x + 0.5) * scaleX,
                y: (component.y + 0.5) * scaleY,
                confidence: component.confidence,
                sources: ["three_frame_motion"],
                sourceRanks: ["three_frame_motion": offset + 1]
            )
        }
    }

    private func scaledBGRA(_ pixelBuffer: CVPixelBuffer) throws -> [UInt8] {
        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        guard format == kCVPixelFormatType_32BGRA else {
            throw ThreeFrameMotionCandidateError.unsupportedPixelFormat(format)
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let sourceData = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw ThreeFrameMotionCandidateError.unavailablePixelBufferMemory
        }
        var output = [UInt8](
            repeating: 0,
            count: analysisWidth * analysisHeight * 4
        )
        let error = output.withUnsafeMutableBytes { destinationBytes -> vImage_Error in
            var source = vImage_Buffer(
                data: sourceData,
                height: vImagePixelCount(sourceHeight),
                width: vImagePixelCount(sourceWidth),
                rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
            )
            var destination = vImage_Buffer(
                data: destinationBytes.baseAddress,
                height: vImagePixelCount(analysisHeight),
                width: vImagePixelCount(analysisWidth),
                rowBytes: analysisWidth * 4
            )
            return vImageScale_ARGB8888(&source, &destination, nil, vImage_Flags(kvImageNoFlags))
        }
        guard error == kvImageNoError else {
            throw ThreeFrameMotionCandidateError.scaleFailed(error)
        }
        return output
    }

    private static func frameChange(
        _ first: [UInt8],
        _ second: [UInt8],
        pixelCount: Int
    ) -> [UInt8] {
        var output = [UInt8](repeating: 0, count: pixelCount)
        first.withUnsafeBufferPointer { firstBuffer in
            second.withUnsafeBufferPointer { secondBuffer in
                output.withUnsafeMutableBufferPointer { destination in
                    guard let firstBase = firstBuffer.baseAddress,
                          let secondBase = secondBuffer.baseAddress,
                          let destinationBase = destination.baseAddress else { return }
                    for pixel in 0..<pixelCount {
                        let offset = pixel * 4
                        let blue = abs(Int(firstBase[offset]) - Int(secondBase[offset]))
                        let green = abs(Int(firstBase[offset + 1]) - Int(secondBase[offset + 1]))
                        let red = abs(Int(firstBase[offset + 2]) - Int(secondBase[offset + 2]))
                        destinationBase[pixel] = UInt8(max(blue, green, red))
                    }
                }
            }
        }
        return output
    }
}
