import AVKit
import PhotosUI
import ShotTracerCore
import SwiftUI

public struct ShotTracerAppView: View {
    private enum PhotoSaveState: Equatable {
        case idle
        case saving
        case saved
        case failed(String)
    }

    @State private var workflow: ShotTracerWorkflow
    @State private var selectedItem: PhotosPickerItem?
    @State private var photoSaveState = PhotoSaveState.idle

    public init(processor: any ShotProcessingClient) {
        _workflow = State(initialValue: ShotTracerWorkflow(processor: processor))
    }

    public var body: some View {
        NavigationStack {
            Group {
                switch workflow.phase {
                case .empty, .importing:
                    importView
                case .ready, .failed:
                    setupView
                case .processing:
                    processingView
                case .preview:
                    previewView
                }
            }
            .navigationTitle("Shot Tracer")
        }
        .onChange(of: selectedItem) {
            let item = selectedItem
            guard let item else { return }
            Task { await workflow.importVideo(from: item) }
        }
    }

    private var importView: some View {
        ContentUnavailableView {
            Label("Trace a Golf Shot", systemImage: "figure.golf")
        } description: {
            Text("Choose a stable, rear-view video. You'll tap the ball at address before it processes.")
        } actions: {
            PhotosPicker(selection: $selectedItem, matching: .videos) {
                Label("Choose Video", systemImage: "video.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            if workflow.phase == .importing {
                ProgressView("Importing…")
            }
        }
    }

    private var setupView: some View {
        ScrollView {
            VStack(spacing: 20) {
                if let image = workflow.previewImage {
                    AddressCorrectionView(
                        image: image,
                        isEditable: true,
                        address: $workflow.manualAddress
                    )
                    .frame(height: 440)
                    .clipShape(RoundedRectangle(cornerRadius: 18))
                }

                VStack(alignment: .leading, spacing: 10) {
                    Label("Tap the ball at address", systemImage: "hand.tap")
                        .font(.headline)
                    Text(workflow.manualAddress == nil
                         ? "Tap the ball's position in the frame above before tracing. This anchors tracking to the correct pixels."
                         : "Ball position set. Tap again anywhere in the frame to move it.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                VStack(alignment: .leading, spacing: 10) {
                    Text("Flight profile")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                    Picker("Flight profile", selection: $workflow.profileOverride) {
                        Text("Automatic").tag(ShotProfile?.none)
                        ForEach(ShotProfile.allCases, id: \.self) { profile in
                            Text(profile.rawValue.capitalized).tag(Optional(profile))
                        }
                    }
                    .pickerStyle(.segmented)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if let message = workflow.errorMessage {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Button(action: workflow.startProcessing) {
                    Label("Trace Shot", systemImage: "wand.and.stars")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(!workflow.canStartProcessing)

                Button("Choose Another Video", role: .cancel) {
                    selectedItem = nil
                    photoSaveState = .idle
                    workflow.reset()
                }
            }
            .padding()
        }
    }

    private var processingView: some View {
        VStack(spacing: 24) {
            ProgressView(value: workflow.progress) {
                Text("Tracing shot…")
            } currentValueLabel: {
                Text(workflow.progress, format: .percent.precision(.fractionLength(0)))
            }
            .progressViewStyle(.linear)
            .padding(.horizontal, 30)

            Text("Keep Shot Tracer open while it analyzes and exports the video.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)

            Button("Cancel", role: .cancel, action: workflow.cancelProcessing)
                .buttonStyle(.bordered)
        }
        .padding()
    }

    @ViewBuilder
    private var previewView: some View {
        if let output = workflow.output {
            VStack(spacing: 18) {
                AutoplayVideoPlayer(url: output.videoURL)
                    .clipShape(RoundedRectangle(cornerRadius: 18))
                    .padding(.horizontal)

                HStack {
                    Label("Manual ball position", systemImage: "hand.tap")
                    Spacer()
                    Text(output.selectedProfile?.rawValue.capitalized ?? "Observed flight")
                        .foregroundStyle(.secondary)
                }
                .font(.subheadline)
                .padding(.horizontal)

                Button {
                    Task { await saveToPhotos(output.videoURL) }
                } label: {
                    switch photoSaveState {
                    case .saving:
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    case .saved:
                        Label("Saved to Photos", systemImage: "checkmark.circle.fill")
                            .frame(maxWidth: .infinity)
                    case .idle, .failed:
                        Label("Save to Photos", systemImage: "square.and.arrow.down")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .disabled(photoSaveState == .saving || photoSaveState == .saved)
                .padding(.horizontal)

                ShareLink(item: output.videoURL) {
                    Label("Share", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.large)
                .padding(.horizontal)

                if case let .failed(message) = photoSaveState {
                    Label(message, systemImage: "exclamationmark.triangle.fill")
                        .font(.subheadline)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                Button("Trace Another Shot") {
                    selectedItem = nil
                    photoSaveState = .idle
                    workflow.reset()
                }
            }
            .padding(.vertical)
        }
    }

    @MainActor
    private func saveToPhotos(_ url: URL) async {
        photoSaveState = .saving
        do {
            try await PhotoLibraryVideoSaver.save(url)
            photoSaveState = .saved
        } catch {
            photoSaveState = .failed(error.localizedDescription)
        }
    }
}
