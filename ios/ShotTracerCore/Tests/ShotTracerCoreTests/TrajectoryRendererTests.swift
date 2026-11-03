import CoreVideo
import Foundation
import Testing
@testable import ShotTracerCore

private func point(
    frame: Int,
    x: Double,
    y: Double,
    provenance: PointProvenance,
    opacity: Double? = nil
) -> TrajectoryPoint {
    TrajectoryPoint(
        frameIndex: frame,
        timestampSeconds: Double(frame) / 60,
        x: x,
        y: y,
        confidence: nil,
        uncertaintyPixels: nil,
        opacity: opacity,
        provenance: provenance,
        estimated: provenance != .observed,
        source: nil,
        sources: nil,
        sourceRanks: nil
    )
}

private func blackPixelBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
    var value: CVPixelBuffer?
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        width,
        height,
        kCVPixelFormatType_32BGRA,
        nil,
        &value
    )
    #expect(status == kCVReturnSuccess)
    let buffer = try #require(value)
    CVPixelBufferLockBaseAddress(buffer, [])
    if let bytes = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) {
        let byteCount = CVPixelBufferGetBytesPerRow(buffer) * height
        bytes.initialize(repeating: 0, count: byteCount)
        for y in 0..<height {
            for x in 0..<width {
                bytes[y * CVPixelBufferGetBytesPerRow(buffer) + x * 4 + 3] = 255
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(buffer, [])
    return buffer
}

private func pixel(_ buffer: CVPixelBuffer, x: Int, y: Int) -> (b: UInt8, g: UInt8, r: UInt8) {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let offset = y * CVPixelBufferGetBytesPerRow(buffer) + x * 4
    return (bytes[offset], bytes[offset + 1], bytes[offset + 2])
}

@Test func rendererUsesTopLeftCoordinatesAndDistinctProvenanceStyles() throws {
    let buffer = try blackPixelBuffer(width: 128, height: 160)
    let renderer = TrajectoryRenderer(trajectory: [
        point(frame: 0, x: 20, y: 20, provenance: .observed),
        point(frame: 1, x: 50, y: 20, provenance: .observed),
        point(frame: 2, x: 80, y: 40, provenance: .estimatedGap),
        point(frame: 3, x: 100, y: 100, provenance: .inferred, opacity: 0.5),
    ])
    try renderer.render(into: buffer, frameIndex: 3)

    let observed = pixel(buffer, x: 35, y: 20)
    #expect(observed.b > observed.r)
    #expect(observed.g > observed.r)

    let estimatedRegion = (55...78).flatMap { x in
        (20...40).map { y in pixel(buffer, x: x, y: y) }
    }
    #expect(estimatedRegion.contains { $0.r > $0.b && $0.g > $0.b })

    let inferredRegion = (82...98).flatMap { x in
        (48...90).map { y in pixel(buffer, x: x, y: y) }
    }
    #expect(inferredRegion.contains { $0.r > 0 && $0.b > 0 })

    let head = pixel(buffer, x: 100, y: 100)
    #expect(head.r > 0 && head.g > 0 && head.b > 0)
    let verticallyFlippedHead = pixel(buffer, x: 100, y: 60)
    #expect(verticallyFlippedHead.r == 0)
    #expect(verticallyFlippedHead.g == 0)
    #expect(verticallyFlippedHead.b == 0)
}

@Test func rendererDoesNotDrawTrajectoryBeforeItsFirstFrame() throws {
    let buffer = try blackPixelBuffer(width: 64, height: 64)
    let renderer = TrajectoryRenderer(trajectory: [
        point(frame: 10, x: 20, y: 20, provenance: .observed),
    ])
    try renderer.render(into: buffer, frameIndex: 9)
    let sample = pixel(buffer, x: 20, y: 20)
    #expect(sample.r == 0 && sample.g == 0 && sample.b == 0)
}
