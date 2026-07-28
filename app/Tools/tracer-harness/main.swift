import Foundation
import TracingCore

// Phase 0 harness (see vault: phases, ADR 0006). Planned flow:
//   1. Decode the clip with AVAssetReader (frame-accurate PTS).
//   2. Run a tracker (frame differencing and optical flow, compared — ADR 0004).
//   3. Write an annotated video and a Trajectory JSON sidecar next to the input.

let arguments = CommandLine.arguments.dropFirst()

guard let inputPath = arguments.first else {
    FileHandle.standardError.write(Data("usage: tracer-harness <video-file>\n".utf8))
    exit(64)
}

guard FileManager.default.fileExists(atPath: inputPath) else {
    FileHandle.standardError.write(Data("tracer-harness: no such file: \(inputPath)\n".utf8))
    exit(66)
}

print("tracer-harness: tracking not implemented yet (Phase 0 in progress)")
print("input: \(inputPath)")
exit(1)
