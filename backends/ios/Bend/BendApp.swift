import SwiftUI

@main
struct BendApp: App {
    /// The engines start with the app (chat loads in the background when its model is present); the web view polls them.
    /// R15: the OS bridges register here, before launch ends (BGTaskScheduler's rule). ios/Bend/OSBridge.swift.
    init() {
        #if BEND_LLB
        EngineManager.console(String(format: "BEND boot app init at %.0f ms since process start", Boot.sinceStartMs()))
        #endif
        EngineManager.shared.start()
        #if BEND_FETCH
        BendApp.wireModels()
        #endif
        OSBridge.shared.launch()
        // PHONEVOICE: voice mode's port (ios/Bend/Voice): on device and in the simulator alike (the engines need not be on).
        // BENDSDK: under -hostEngines (simulator) the Mac harness serves voice too; its port is the one the bridge names.
        if !EngineManager.hostEngines { VoiceServer.shared.start() }
    }

    #if BEND_CAMERA
    /// PHONECAM: the camera surface over the web app (Camera/CameraView.swift).
    @State private var camera = CameraPresenter.shared
    #endif
    #if BEND_LLB
    /// PHONEUX: the boot overlay until the page has painted (BootOverlay.swift).
    @State private var boot = BootModel.shared
    #endif
    #if BEND_FETCH
    /// IOSFETCH: the first-run setup card (Models/ModelSetupView.swift), over the web app while a model is missing.
    @State private var models = ModelStore.shared
    /// VOICESHOP: Speech settings (the voice picker + More voices), a sheet over the web app.
    @State private var voices = VoicePicker.shared
    #endif

    var body: some Scene {
        WindowGroup {
            ZStack {
                WebAppView()
                    // The web app pads itself with env(safe-area-inset-*), so the view runs edge to edge.
                    .ignoresSafeArea()
                #if BEND_LLB
                if !boot.pageUp {
                    BootOverlay(model: boot).transition(.opacity)
                }
                #endif
                #if BEND_FETCH
                if models.isPresented {
                    ModelSetupView(store: models) { withAnimation(.easeOut(duration: 0.2)) { models.isPresented = false } }
                        .transition(.opacity)
                }
                #endif
            }
                #if BEND_LLB
                .animation(.easeOut(duration: 0.2), value: boot.pageUp)
                #endif
                .background(Color.bendBackground)
                .preferredColorScheme(.dark)
                // bend://shortcut?status=… is a Shortcuts x-callback: coming back is all it's for.
                // PHONECAM: bend://camera opens the camera surface. IOSFETCH: bend://models opens the setup card.
                .onOpenURL { url in
                    #if BEND_FETCH
                    if url.host == "models" { models.present(); return }
                    if url.host == "voices" { voices.present(shop: url.path == "/shop"); return }  // VOICESHOP: bend://voices, bend://voices/shop
                    #endif
                    #if BEND_CAMERA
                    if url.host == "camera" {
                        camera.open(source: url.query?.contains("source=photo") == true ? .photo : .camera)
                        return
                    }
                    #endif
                    OSBridge.say("back from \(url.absoluteString)")
                }
                #if BEND_CAMERA
                .fullScreenCover(isPresented: $camera.isPresented) { CameraView() }
                #endif
                #if BEND_FETCH
                .sheet(isPresented: $voices.isPresented) {
                    VoiceSettingsView(picker: voices, shop: VoiceShop.shared) { voices.isPresented = false }
                }
                #endif
        }
    }
}

#if BEND_FETCH
extension BendApp {
    /// IOSFETCH: the download's console lines and its hand-off to the engine (docs/MODELS.md).
    static func wireModels() {
        ModelStore.log = { EngineManager.console("BEND models \($0)") }
        // VOICESHOP: the shop's lines on the console; an installed pack shows in the picker at once.
        VoiceShop.log = { EngineManager.console("BEND voices \($0)") }
        VoiceShop.shared.onInstalled = { id in VoicePicker.shared.refresh(); VoiceShop.log("installed \(id) → the picker lists it") }
        let lib = VoiceLibrary.shared.packs(refresh: true)
        VoiceShop.log("launch catalog=\(VoiceCatalog.url.absoluteString)\(VoiceCatalog.overridden ? " (debug override)" : "") choice=\(VoiceLibrary.shared.choice) "
            + "voices=\(lib.count) [\(lib.map { "\($0.id)(\($0.origin.rawValue))" }.joined(separator: " "))]")
        let store = ModelStore.shared
        store.onInstalled = { EngineManager.shared.modelsArrived($0) }
        ModelStore.log("launch host=\(ModelManifest.baseURL.absoluteString)\(ModelManifest.overridden ? " (debug override)" : "") "
            + store.items.map { "\($0.id.rawValue)=\(store.present($0) ? "present" : "missing")" }.joined(separator: " ")
            + " setup=\(store.isPresented ? "shown" : "not shown") fit=\"\(store.fit().line)\"")
        let center = NotificationCenter.default
        center.addObserver(forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { ModelStore.shared.away() }
        }
        center.addObserver(forName: UIApplication.willEnterForegroundNotification, object: nil, queue: .main) { _ in
            MainActor.assumeIsolated { ModelStore.shared.back() }
        }
    }
}
#endif

extension Color {
    /// The web app's --background (oklch 19% 0 0), so launch, overscroll and the page never flash.
    static let bendBackground = Color(red: 20 / 255, green: 20 / 255, blue: 20 / 255)
}
