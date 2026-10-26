import Foundation

// The Mac harness (scripts/engine-mac.sh): Bend's native engine layer (ios/Bend/Engines/ControlServer.swift +
// Engines/LLB/*.swift, the same files the app compiles) linked against LLBEngine's macOS slice, serving the same two
// loopback ports. Lets the web transport, curl and the -engineCheck flags run against our engine on the Mac.
// Env: BEND_MODELS_DIR (the qwen35-4b-q4km/ and kev-4b-q4km/ dirs), BEND_CACHE_DIR, BEND_CHECK_DIR.
// PHONEVOICE: voice mode's port too (ios/Bend/Voice, 127.0.0.1:8932), so scripts/voice-measure.sh runs the voice loop
// (SpeechAnalyzer → the warm engine → AVSpeechSynthesizer) through the same Swift the app compiles.
MainActor.assumeIsolated { EngineManager.shared.start() }
VoiceServer.shared.start()
RunLoop.main.run()
