import CoreGraphics
import SwiftUI

struct AddressCorrectionView: View {
    let image: CGImage
    let isEditable: Bool
    @Binding var address: CGPoint?

    var body: some View {
        GeometryReader { geometry in
            let imageSize = CGSize(width: image.width, height: image.height)
            let fitted = aspectFitRect(content: imageSize, in: geometry.size)
            ZStack(alignment: .topLeading) {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                if let address {
                    Circle()
                        .stroke(.yellow, lineWidth: 3)
                        .background(Circle().fill(.black.opacity(0.2)))
                        .frame(width: 28, height: 28)
                        .position(
                            x: fitted.minX + address.x / imageSize.width * fitted.width,
                            y: fitted.minY + address.y / imageSize.height * fitted.height
                        )
                }
            }
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0).onEnded { gesture in
                    guard fitted.contains(gesture.location) else { return }
                    address = CGPoint(
                        x: (gesture.location.x - fitted.minX) / fitted.width * imageSize.width,
                        y: (gesture.location.y - fitted.minY) / fitted.height * imageSize.height
                    )
                },
                including: isEditable ? .all : .none
            )
        }
        .accessibilityLabel(isEditable ? "Address ball position" : "Selected video preview")
        .accessibilityHint(isEditable ? "Tap the golf ball before the shot starts" : "")
    }

    private func aspectFitRect(content: CGSize, in container: CGSize) -> CGRect {
        let scale = min(container.width / content.width, container.height / content.height)
        let size = CGSize(width: content.width * scale, height: content.height * scale)
        return CGRect(
            x: (container.width - size.width) / 2,
            y: (container.height - size.height) / 2,
            width: size.width,
            height: size.height
        )
    }
}
