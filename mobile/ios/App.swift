import SwiftUI

@main
struct DeepfakeMeetingShieldApp: App {
    var body: some Scene {
        WindowGroup {
            DeepfakeDetectorView()
                .preferredColorScheme(.dark)
        }
    }
}
