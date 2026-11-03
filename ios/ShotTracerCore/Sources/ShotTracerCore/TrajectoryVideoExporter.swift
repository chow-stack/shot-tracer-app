import AVFoundation
import Foundation

public enum TrajectoryVideoExportError: LocalizedError, Equatable {
    case outputAlreadyExists
    case noAudioFormat
    case couldNotCreateReader(String?)
    case couldNotCreateWriter(String?)
    case couldNotConfigureVideo
    case couldNotConfigureAudio
    case couldNotStartWriting(String?)
    case couldNotAppendVideo(String?)
    case couldNotAppendAudio(String?)
    case exportFailed(String?)
    case frameDimensionsDoNotMatch
    case noVideoFrames
    case writerStalled

    public var errorDescription: String? {
        switch self {
        case .outputAlreadyExists:
            "The output video already exists."
        case .noAudioFormat:
            "The video's audio format could not be read."
        case let .couldNotCreateReader(message):
            message ?? "The selected video could not be opened."
        case let .couldNotCreateWriter(message):
            message ?? "The output video could not be created."
        case .couldNotConfigureVideo:
            "The video encoder could not be configured."
        case .couldNotConfigureAudio:
            "The audio encoder could not be configured."
        case let .couldNotStartWriting(message):
            message ?? "Video export could not start."
        case let .couldNotAppendVideo(message):
            message ?? "A video frame could not be exported."
        case let .couldNotAppendAudio(message):
            message ?? "The video's audio could not be exported."
        case let .exportFailed(message):
            message ?? "Video export failed."
        case .frameDimensionsDoNotMatch:
            "The rendered video dimensions do not match the source."
        case .noVideoFrames:
            "The selected video has no readable frames."
        case .writerStalled:
            "Video export stopped responding, likely because this recording's format is too slow to decode and re-encode on this device. All video formats are accepted, but a shorter clip or a standard (non-ProRes) recording will process more reliably."
        }
    }
}

public enum VideoExportCodec: String, Sendable {
    case hevc
    case h264

    var avCodec: AVVideoCodecType {
        switch self {
        case .hevc: .hevc
        case .h264: .h264
        }
    }
}

public struct TrajectoryVideoExportConfiguration: Sendable {
    public var codec: VideoExportCodec
    public var averageBitRate: Int?
    public var preservesAudio: Bool

    public init(
        codec: VideoExportCodec = .hevc,
        averageBitRate: Int? = nil,
        preservesAudio: Bool = true
    ) {
        self.codec = codec
        self.averageBitRate = averageBitRate
        self.preservesAudio = preservesAudio
    }
}

public struct TrajectoryVideoExportResult: Sendable, Equatable {
    public let outputURL: URL
    public let renderedFrameCount: Int
    public let durationSeconds: Double
    public let includedAudio: Bool
}

/// Serializes AVAssetReader/AVAssetWriter state in one actor. Export always
/// reads frames forward, preserves their presentation timestamps, applies the
/// source presentation transform, and responds cooperatively to cancellation.
public actor TrajectoryVideoExporter {
    public init() {}

    public func export(
        sourceURL: URL,
        outputURL: URL,
        trajectory: TrajectoryDocument,
        style: TrajectoryRenderingStyle = TrajectoryRenderingStyle(),
        configuration: TrajectoryVideoExportConfiguration = .init(),
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> TrajectoryVideoExportResult {
        guard !FileManager.default.fileExists(atPath: outputURL.path) else {
            throw TrajectoryVideoExportError.outputAlreadyExists
        }

        let asset = AVURLAsset(url: sourceURL)
        let duration = try await asset.load(.duration)
        let durationSeconds = max(0, duration.seconds)
        let frameReader = try await SequentialVideoFrameReader(
            url: sourceURL, copiesSampleData: true
        )
        guard frameReader.width == trajectory.video.width,
              frameReader.height == trajectory.video.height else {
            frameReader.cancel()
            throw TrajectoryVideoExportError.frameDimensionsDoNotMatch
        }
        let renderer = TrajectoryRenderer(trajectory: trajectory.trajectory, style: style)

        let writer: AVAssetWriter
        do {
            writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        } catch {
            frameReader.cancel()
            throw TrajectoryVideoExportError.couldNotCreateWriter(error.localizedDescription)
        }
        let videoInput = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: videoSettings(
                width: frameReader.width,
                height: frameReader.height,
                configuration: configuration
            )
        )
        videoInput.expectsMediaDataInRealTime = false
        guard writer.canAdd(videoInput) else {
            frameReader.cancel()
            throw TrajectoryVideoExportError.couldNotConfigureVideo
        }
        writer.add(videoInput)
        let pixelBufferAdaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: nil
        )

        let audioPipeline = try await configuration.preservesAudio
            ? makeAudioPipeline(asset: asset, writer: writer) : nil
        guard writer.startWriting() else {
            frameReader.cancel()
            audioPipeline?.reader.cancelReading()
            throw TrajectoryVideoExportError.couldNotStartWriting(
                writer.error?.localizedDescription
            )
        }
        writer.startSession(atSourceTime: .zero)

        var renderedFrameCount = 0
        do {
            while let frame = try frameReader.nextFrame() {
                try Task.checkCancellation()
                if let audioPipeline {
                    try await appendAudio(
                        audioPipeline,
                        through: frame.presentationTime,
                        writer: writer
                    )
                }
                try await waitUntilReady(videoInput, writer: writer)
                try renderer.render(into: frame.pixelBuffer, frameIndex: frame.frameIndex)
                guard pixelBufferAdaptor.append(
                    frame.pixelBuffer, withPresentationTime: frame.presentationTime
                ) else {
                    throw TrajectoryVideoExportError.couldNotAppendVideo(
                        writer.error?.localizedDescription
                    )
                }
                renderedFrameCount += 1
                if durationSeconds > 0 {
                    progress(min(0.98, max(0, frame.presentationTimeSeconds / durationSeconds)))
                }
            }
            guard renderedFrameCount > 0 else {
                throw TrajectoryVideoExportError.noVideoFrames
            }
            videoInput.markAsFinished()
            if let audioPipeline {
                try await appendRemainingAudio(audioPipeline, writer: writer)
            }
            try Task.checkCancellation()
            try await finish(writer)
            guard writer.status == .completed else {
                throw TrajectoryVideoExportError.exportFailed(writer.error?.localizedDescription)
            }
            progress(1)
            return TrajectoryVideoExportResult(
                outputURL: outputURL,
                renderedFrameCount: renderedFrameCount,
                durationSeconds: durationSeconds,
                includedAudio: audioPipeline != nil
            )
        } catch {
            frameReader.cancel()
            audioPipeline?.reader.cancelReading()
            if writer.status != .completed && writer.status != .failed {
                writer.cancelWriting()
            }
            throw error
        }
    }

    private final class AudioPipeline {
        let reader: AVAssetReader
        let output: AVAssetReaderTrackOutput
        let input: AVAssetWriterInput
        var pendingSample: CMSampleBuffer?

        init(
            reader: AVAssetReader,
            output: AVAssetReaderTrackOutput,
            input: AVAssetWriterInput
        ) {
            self.reader = reader
            self.output = output
            self.input = input
        }
    }

    private func makeAudioPipeline(
        asset: AVAsset, writer: AVAssetWriter
    ) async throws -> AudioPipeline? {
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else {
            return nil
        }
        let formatDescriptions = try await track.load(.formatDescriptions)
        guard let sourceFormat = formatDescriptions.first else {
            throw TrajectoryVideoExportError.noAudioFormat
        }
        let reader: AVAssetReader
        do {
            reader = try AVAssetReader(asset: asset)
        } catch {
            throw TrajectoryVideoExportError.couldNotCreateReader(error.localizedDescription)
        }
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw TrajectoryVideoExportError.couldNotConfigureAudio
        }
        reader.add(output)
        let input = AVAssetWriterInput(
            mediaType: .audio,
            outputSettings: nil,
            sourceFormatHint: sourceFormat
        )
        input.expectsMediaDataInRealTime = false
        guard writer.canAdd(input) else {
            throw TrajectoryVideoExportError.couldNotConfigureAudio
        }
        writer.add(input)
        guard reader.startReading() else {
            throw TrajectoryVideoExportError.couldNotCreateReader(
                reader.error?.localizedDescription
            )
        }
        return AudioPipeline(reader: reader, output: output, input: input)
    }

    private func appendAudio(
        _ pipeline: AudioPipeline,
        through presentationTime: CMTime,
        writer: AVAssetWriter
    ) async throws {
        while true {
            try Task.checkCancellation()
            if pipeline.pendingSample == nil {
                pipeline.pendingSample = pipeline.output.copyNextSampleBuffer()
            }
            guard let sample = pipeline.pendingSample else {
                try checkAudioReader(pipeline)
                return
            }
            let sampleTime = CMSampleBufferGetPresentationTimeStamp(sample)
            guard CMTimeCompare(sampleTime, presentationTime) <= 0 else { return }
            try await waitUntilReady(pipeline.input, writer: writer)
            guard pipeline.input.append(sample) else {
                throw TrajectoryVideoExportError.couldNotAppendAudio(
                    writer.error?.localizedDescription
                )
            }
            pipeline.pendingSample = nil
        }
    }

    private func appendRemainingAudio(
        _ pipeline: AudioPipeline, writer: AVAssetWriter
    ) async throws {
        while true {
            try Task.checkCancellation()
            if pipeline.pendingSample == nil {
                pipeline.pendingSample = pipeline.output.copyNextSampleBuffer()
            }
            guard let sample = pipeline.pendingSample else { break }
            try await waitUntilReady(pipeline.input, writer: writer)
            guard pipeline.input.append(sample) else {
                throw TrajectoryVideoExportError.couldNotAppendAudio(
                    writer.error?.localizedDescription
                )
            }
            pipeline.pendingSample = nil
        }
        try checkAudioReader(pipeline)
        pipeline.input.markAsFinished()
    }

    private func checkAudioReader(_ pipeline: AudioPipeline) throws {
        if pipeline.reader.status == .failed {
            throw TrajectoryVideoExportError.couldNotAppendAudio(
                pipeline.reader.error?.localizedDescription
            )
        }
    }

    private func waitUntilReady(
        _ input: AVAssetWriterInput, writer: AVAssetWriter
    ) async throws {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: .seconds(30))
        while !input.isReadyForMoreMediaData {
            try Task.checkCancellation()
            switch writer.status {
            case .writing:
                break
            case .failed:
                throw TrajectoryVideoExportError.exportFailed(
                    writer.error?.localizedDescription
                )
            case .cancelled:
                throw CancellationError()
            case .completed:
                throw TrajectoryVideoExportError.exportFailed(
                    "The video writer finished before all media was appended."
                )
            case .unknown:
                throw TrajectoryVideoExportError.exportFailed(
                    "The video writer is not active."
                )
            @unknown default:
                throw TrajectoryVideoExportError.exportFailed(
                    writer.error?.localizedDescription
                )
            }
            guard clock.now < deadline else {
                throw TrajectoryVideoExportError.writerStalled
            }
            try await Task.sleep(for: .milliseconds(2))
        }
    }

    private func finish(_ writer: AVAssetWriter) async throws {
        await withCheckedContinuation { continuation in
            writer.finishWriting { continuation.resume() }
        }
        if writer.status == .failed {
            throw TrajectoryVideoExportError.exportFailed(writer.error?.localizedDescription)
        }
    }

    private func videoSettings(
        width: Int,
        height: Int,
        configuration: TrajectoryVideoExportConfiguration
    ) -> [String: Any] {
        var compression: [String: Any] = [:]
        if let averageBitRate = configuration.averageBitRate {
            compression[AVVideoAverageBitRateKey] = averageBitRate
        }
        var settings: [String: Any] = [
            AVVideoCodecKey: configuration.codec.avCodec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ]
        if !compression.isEmpty {
            settings[AVVideoCompressionPropertiesKey] = compression
        }
        return settings
    }
}
