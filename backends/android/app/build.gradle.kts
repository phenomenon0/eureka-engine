import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// APKSIGN: release signing comes from a props file outside every repo (docs/RELEASE.md). Without it, assembleRelease
// still builds, but leaves app-release-unsigned.apk, which will not install.
val signing = File((findProperty("bendSigning") as String?) ?: "${System.getProperty("user.home")}/.dsh/keys/bend-release.properties")
    .takeIf { it.isFile }?.let { f -> Properties().apply { f.inputStream().use(::load) } }

android {
    namespace = "dev.bend.app"
    compileSdk = 35
    defaultConfig {
        applicationId = "dev.bend.app"
        minSdk = 34
        targetSdk = 35
        versionCode = 8
        versionName = "0.8-models"
        ndk { abiFilters += "arm64-v8a" }
    }
    signingConfigs {
        if (signing != null) create("release") {
            storeFile = file(signing.getProperty("storeFile"))
            keyAlias = signing.getProperty("keyAlias")
            storePassword = file(signing.getProperty("passwordFile")).readText().trim()
            keyPassword = storePassword
            enableV2Signing = true
            enableV3Signing = true // v3 = key rotation support; AGP leaves it off by default
        }
    }
    buildTypes {
        // PHONEFAST: the debug build installs beside the release-signed dev.bend.app (its key's password is not on this box,
        // and uninstalling 0.6 would delete its model files), as dev.bend.app.gpu with its own files dir.
        debug { applicationIdSuffix = ".gpu"; versionNameSuffix = "-debug" }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.findByName("release")
        }
    }
    buildFeatures { buildConfig = true }
    // The engine is an executable named libbendengine.so: it must be extracted to nativeLibraryDir to be exec'd.
    packaging { jniLibs { useLegacyPackaging = true } }
    // KOKORODROID: the Kokoro model is mmapped straight out of the APK (Kokoro.kt), so it must be stored, not deflated.
    androidResources { noCompress += listOf("onnx", "bin") } // ANDROIDVOICE: the whisper model too (Whisper.kt openFd)
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    // KOKORODROID: onnxruntime-android 1.24.2 (MIT), fetched by android/voice/fetch-voice.sh (pinned sha256), not from a repo:
    // the build stays --offline.
    implementation(files("libs/onnxruntime-android-1.24.2.aar"))
}
