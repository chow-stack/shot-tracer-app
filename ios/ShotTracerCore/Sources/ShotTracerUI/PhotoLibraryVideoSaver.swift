import Foundation
import Photos

enum PhotoLibraryVideoSaveError: LocalizedError {
    case permissionDenied
    case missingVideo

    var errorDescription: String? {
        switch self {
        case .permissionDenied:
            "Photos access was denied. Allow Shot Tracer to add photos in Settings and try again."
        case .missingVideo:
            "The finished video is no longer available."
        }
    }
}

enum PhotoLibraryVideoSaver {
    static func save(_ url: URL) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PhotoLibraryVideoSaveError.missingVideo
        }
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else {
            throw PhotoLibraryVideoSaveError.permissionDenied
        }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAssetFromVideo(atFileURL: url)
        }
    }
}
