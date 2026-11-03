import Testing
@testable import ShotTracerCore

@Test func threeFrameMotionComponentsUseWeightedCentersAndMergeThresholds() throws {
    let width = 20
    let height = 16
    var response = [UInt8](repeating: 0, count: width * height)
    response[7 * width + 8] = 90
    response[7 * width + 9] = 180
    response[8 * width + 9] = 240

    let candidates = ThreeFrameMotionCandidateExtractor.extractCandidates(
        response: response,
        width: width,
        height: height,
        frameIndex: 12,
        scaleX: 2,
        scaleY: 2
    )
    let candidate = try #require(candidates.first)
    #expect(candidates.count == 1)
    #expect(candidate.frameIndex == 12)
    #expect(candidate.sources == ["three_frame_motion"])
    #expect(candidate.x > 18 && candidate.x < 20)
    #expect(candidate.y > 15 && candidate.y < 18)
    #expect(candidate.confidence > 0.5)
}

@Test func threeFrameMotionRejectsLargeMovingRegions() {
    let width = 80
    let height = 80
    var response = [UInt8](repeating: 0, count: width * height)
    for y in 10..<40 {
        for x in 10..<40 {
            response[y * width + x] = 255
        }
    }
    let candidates = ThreeFrameMotionCandidateExtractor.extractCandidates(
        response: response,
        width: width,
        height: height,
        frameIndex: 3
    )
    #expect(candidates.isEmpty)
}
