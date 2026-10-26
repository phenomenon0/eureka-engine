// ANDROIDVOICE: the app's ears. whisper.cpp (ggml, CPU) behind two JNI calls for Whisper.kt: load a model once (kept for
// the process's life), turn 16 kHz mono float PCM into text. Built by build.sh into jniLibs/arm64-v8a/libbendwhisper.so.
#include <jni.h>
#include <string.h>
#include <stdlib.h>
#include <android/log.h>
#include "whisper.h"

#define TAG "BendWhisper"

JNIEXPORT jlong JNICALL Java_dev_bend_app_Whisper_load(JNIEnv *env, jclass cls, jstring path) {
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(p, cp);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return (jlong) ctx;
}

JNIEXPORT jstring JNICALL Java_dev_bend_app_Whisper_transcribe(JNIEnv *env, jclass cls, jlong ctx, jfloatArray pcm, jint threads) {
    struct whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.n_threads = threads;
    wp.language = "en";
    wp.translate = false;
    wp.no_context = true;
    wp.single_segment = true;   // one utterance, one segment
    wp.no_timestamps = true;
    wp.suppress_blank = true;
    wp.suppress_nst = true;     // no "[BLANK_AUDIO]" / "(music)" tokens
    wp.print_progress = wp.print_realtime = wp.print_special = wp.print_timestamps = false;
    jsize n = (*env)->GetArrayLength(env, pcm);
    jfloat *a = (*env)->GetFloatArrayElements(env, pcm, NULL);
    int rc = whisper_full((struct whisper_context *) ctx, wp, a, n);
    (*env)->ReleaseFloatArrayElements(env, pcm, a, JNI_ABORT);
    if (rc != 0) { __android_log_print(ANDROID_LOG_ERROR, TAG, "whisper_full rc=%d", rc); return NULL; }
    size_t len = 1;
    int segs = whisper_full_n_segments((struct whisper_context *) ctx);
    // whisper's own "was that speech?": noise it would otherwise caption as "you" / "Thank you." is dropped (0.6 = OpenAI's).
    if (segs > 0 && whisper_full_get_segment_no_speech_prob((struct whisper_context *) ctx, 0) > 0.6f) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "no speech p=%.2f: %s", whisper_full_get_segment_no_speech_prob((struct whisper_context *) ctx, 0), whisper_full_get_segment_text((struct whisper_context *) ctx, 0));
        segs = 0;
    }
    for (int i = 0; i < segs; i++) len += strlen(whisper_full_get_segment_text((struct whisper_context *) ctx, i));
    char *out = calloc(len, 1);
    for (int i = 0; i < segs; i++) strcat(out, whisper_full_get_segment_text((struct whisper_context *) ctx, i));
    jstring s = (*env)->NewStringUTF(env, out);
    free(out);
    return s;
}
