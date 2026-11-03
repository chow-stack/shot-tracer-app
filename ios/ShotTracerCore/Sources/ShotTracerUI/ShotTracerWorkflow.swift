import AVFoundation
import CoreTransferable
import CoreGraphics
import Foundation
import Observation
import PhotosUI
import ShotTracerCore
import SwiftUI
import UniformTypeIdentifiers

public enum ShotTracerWorkflowPhase: Sendable, Equatable {
    case empty
    case importing
    case ready
    case processing
    case preview
    case failed
}

public enum ShotTracerUIError: LocalizedError {
    case videoImportFailed

    public var errorDescription: String? {
        switch self {
        case .videoImportFailed: "The selected video could not be imported."
        }
    }
}

struct ImportedMovie: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(importedContentType: .movie) { received in
            let extensionName = received.file.pathExtension.isEmpty
                ? "mov" : received.file.pathExtension
            let destination = FileManager.default.temporaryDirectory
                .appending(path: "shot-tracer-import-\(UUID().uuidString)")
                .appendingPathExtension(extensionName)
            try FileManager.default.copyItem(at: received.file, to: destination)
            return ImportedMovie(url: destination)
        }
    }
}

@MainActor
@Observable
public final class ShotTracerWorkflow {
    public private(set) var phase: ShotTracerWorkflowPhase = .empty
    public private(set) var sourceURL: URL?
    public private(set) var previewImage: CGImage?
    public private(set) var output: ShotProcessingOutput?
    public private(set) var progress = 0.0
    public private(set) var errorMessage: String?
    public var manualAddress: CGPoint?
    public var profileOverride: ShotProfile?

    @ObservationIgnored private let processor: any ShotProcessingClient
    @ObservationIgnored private var processingTask: Task<Void, Never>?

    public init(processor: any ShotProcessingClient) {
        self.processor = processor
    }

    public func importVideo(from item: PhotosPickerItem) async {
        cancelProcessing()
        phase = .importing
        errorMessage = nil
        do {
            guard let movie = try await item.loadTransferable(type: ImportedMovie.self) else {
                throw ShotTracerUIError.videoImportFailed
            }
            let image = try await thumbnail(for: movie.url)
            sourceURL = movie.url
            previewImage = image
            manualAddress = nil
            profileOverride = nil
            output = nil
            progress = 0
            phase = .ready
        } catch {
            fail(error)
        }
    }

    /// Every video requires an explicit address-ball tap before processing;
    /// there is no automatic-address fallback in the production workflow.
    public var canStartProcessing: Bool {
        sourceURL != nil && manualAddress != nil && (phase == .ready || phase == .failed)
    }

    public func startProcessing() {
        guard let sourceURL, let manualAddress,
              phase == .ready || phase == .failed else { return }
        processingTask?.cancel()
        phase = .processing
        progress = 0
        errorMessage = nil
        let request = ShotProcessingRequest(
            sourceURL: sourceURL,
            manualAddress: manualAddress,
            profileOverride: profileOverride
        )
        processingTask = Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await processor.process(request) { [weak self] value in
                    Task { @MainActor in
                        self?.progress = min(1, max(0, value))
                    }
                }
                try Task.checkCancellation()
                output = result
                progress = 1
                phase = .preview
            } catch is CancellationError {
                phase = .ready
                progress = 0
            } catch {
                fail(error)
            }
        }
    }

    public func cancelProcessing() {
        processingTask?.cancel()
        processingTask = nil
        if phase == .processing {
            phase = sourceURL == nil ? .empty : .ready
            progress = 0
        }
    }

    public func reset() {
        cancelProcessing()
        if let sourceURL { try? FileManager.default.removeItem(at: sourceURL) }
        if let output { try? FileManager.default.removeItem(at: output.videoURL) }
        sourceURL = nil
        previewImage = nil
        output = nil
        manualAddress = nil
        profileOverride = nil
        progress = 0
        errorMessage = nil
        phase = .empty
    }

    /// No `maximumSize` cap: the returned image's pixel dimensions must match
    /// the full decode-resolution coordinate space that `VisionShotDetector`
    /// and the exporter use, or a downscaled preview silently shifts the tap.
    /// Zero time tolerance: without it, `AVAssetImageGenerator` is free to
    /// snap to the nearest sync sample instead of frame zero, which on some
    /// clips lands well after address with the ball already in flight.
    private func thumbnail(for url: URL) async throws -> CGImage {
        let generator = AVAssetImageGenerator(asset: AVURLAsset(url: url))
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        return try await generator.image(at: .zero).image
    }

    private func fail(_ error: Error) {
        errorMessage = error.localizedDescription
        phase = .failed
    }
}
