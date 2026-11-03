import ShotTracerUI
import SwiftUI

@main
struct ShotTracerApp: App {
    private let processor = AppShotProcessor()

    var body: some Scene {
        WindowGroup {
            ShotTracerAppView(processor: processor)
        }
    }
}
