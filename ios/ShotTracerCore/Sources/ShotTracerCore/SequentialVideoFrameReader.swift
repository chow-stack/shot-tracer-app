import AVFoundation
import CoreVideo
import Foundation

public enum SequentialVideoFrameReaderError: Error {
    case noVideoTrack
    case couldNotCreateVideoComposition(String?)
    case couldNotCreateReader
    case couldNotStartReader(String?)
    case decodeFailed(String?)
}

public struct DecodedVideoFrame {
    public let frameIndex: Int
    public let presentationTime: CMTime
    public let presentationTimeSeconds: Double
    public let sampleBuffer: CMSampleBuffer
    public let pixelBuffer: CVPixelBuffer
}

/// Sequential-only AVAssetReader wrapper. The Python reference showed that
/// random seeking in iPhone MOV files can return frames several indexes away
/// while reporting success, so the production contract never exposes seeking.
public final class SequentialVideoFrameReader {
    private let reader: AVAssetReader
    private let output: AVAssetReaderOutput
    private var nextFrameIndex = 0
    public let width: Int
    public let height: Int

    public init(url: URL, copiesSampleData: Bool = false) async throws {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw SequentialVideoFrameReaderError.noVideoTrack
        }
        _ = try await asset.load(.duration)
        let videoComposition: AVVideoComposition = try await withCheckedThrowingContinuation {
            continuation in
            AVVideoComposition.videoComposition(withPropertiesOf: asset) { composition, error in
                if let composition {
                    continuation.resume(returning: composition)
                } else {
                    continuation.resume(throwing:
                        SequentialVideoFrameReaderError.couldNotCreateVideoComposition(
                            error?.localizedDescription
                        )
                    )
                }
            }
        }
        width = Int(videoComposition.renderSize.width.rounded())
        height = Int(videoComposition.renderSize.height.rounded())
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw SequentialVideoFrameReaderError.couldNotCreateReader
        }
        let videoOutput = AVAssetReaderVideoCompositionOutput(
            videoTracks: [track],
            videoSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
        )
        videoOutput.videoComposition = videoComposition
        videoOutput.alwaysCopiesSampleData = copiesSampleData
        output = videoOutput
        guard reader.canAdd(output) else {
            throw SequentialVideoFrameReaderError.couldNotCreateReader
        }
        reader.add(output)
        guard reader.startReading() else {
            throw SequentialVideoFrameReaderError.couldNotStartReader(reader.error?.localizedDescription)
        }
    }

    public func nextFrame() throws -> DecodedVideoFrame? {
        guard let sample = output.copyNextSampleBuffer() else {
            if reader.status == .failed {
                throw SequentialVideoFrameReaderError.decodeFailed(reader.error?.localizedDescription)
            }
            return nil
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
            throw SequentialVideoFrameReaderError.decodeFailed("Decoded sample has no pixel buffer")
        }
        let presentationTime = CMSampleBufferGetPresentationTimeStamp(sample)
        let frame = DecodedVideoFrame(
            frameIndex: nextFrameIndex,
            presentationTime: presentationTime,
            presentationTimeSeconds: presentationTime.seconds,
            sampleBuffer: sample,
            pixelBuffer: pixelBuffer
        )
        nextFrameIndex += 1
        return frame
    }

    public func cancel() {
        reader.cancelReading()
    }
}
