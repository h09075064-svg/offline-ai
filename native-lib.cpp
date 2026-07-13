// JNI bridge between LlamaBridge.kt and llama.cpp.
//
// Everything in this file runs on-device. There are no sockets, no HTTP
// clients, nothing that reaches off the phone. Inference is 100% local.

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>

#include "llama.h"

#define LOG_TAG "OfflineAI-Native"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
    std::mutex g_mutex;
    llama_model* g_model = nullptr;
    llama_context* g_ctx = nullptr;
    int g_n_threads = 4;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_offlineai_app_llm_LlamaBridge_nativeLoadModel(
        JNIEnv* env, jobject /*thiz*/, jstring modelPath, jint nThreads, jint contextSize) {

    std::lock_guard<std::mutex> lock(g_mutex);

    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    LOGI("Loading model from %s (threads=%d, ctx=%d)", path, nThreads, contextSize);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    // n_gpu_layers = 0 keeps this on CPU for Phase 1. Phase 2 wires this to
    // HardwareProfiler so capable devices offload layers to the GPU/NPU.
    model_params.n_gpu_layers = 0;

    g_model = llama_load_model_from_file(path, model_params);
    env->ReleaseStringUTFChars(modelPath, path);

    if (!g_model) {
        LOGE("Failed to load model");
        return JNI_FALSE;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize;
    ctx_params.n_threads = nThreads;
    ctx_params.n_threads_batch = nThreads;
    g_n_threads = nThreads;

    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_free_model(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_offlineai_app_llm_LlamaBridge_nativeGenerate(
        JNIEnv* env, jobject /*thiz*/, jstring prompt, jint maxTokens, jobject callback) {

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_ctx || !g_model) {
        LOGE("generate() called before a model was loaded");
        return;
    }

    // Resolve the Kotlin TokenCallback.onToken(String) method once.
    jclass cbClass = env->GetObjectClass(callback);
    jmethodID onToken = env->GetMethodID(cbClass, "onToken", "(Ljava/lang/String;)V");

    const char* promptChars = env->GetStringUTFChars(prompt, nullptr);
    std::string promptStr(promptChars);
    env->ReleaseStringUTFChars(prompt, promptChars);

    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    // Tokenize prompt.
    std::vector<llama_token> tokens(promptStr.size() + 8);
    int n_tokens = llama_tokenize(vocab, promptStr.c_str(), (int32_t) promptStr.size(),
                                   tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, promptStr.c_str(), (int32_t) promptStr.size(),
                                   tokens.data(), (int32_t) tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("Initial decode failed");
        return;
    }

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    for (int i = 0; i < maxTokens; i++) {
        llama_token new_token = llama_sampler_sample(sampler, g_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            jstring jpiece = env->NewStringUTF(piece.c_str());
            env->CallVoidMethod(callback, onToken, jpiece);
            env->DeleteLocalRef(jpiece);
        }

        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Decode failed mid-generation");
            break;
        }
    }

    llama_sampler_free(sampler);
}

extern "C" JNIEXPORT void JNICALL
Java_com_offlineai_app_llm_LlamaBridge_nativeUnload(JNIEnv* /*env*/, jobject /*thiz*/) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ctx) { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_free_model(g_model); g_model = nullptr; }
    llama_backend_free();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_offlineai_app_llm_LlamaBridge_nativeIsLoaded(JNIEnv* /*env*/, jobject /*thiz*/) {
    return (g_ctx != nullptr && g_model != nullptr) ? JNI_TRUE : JNI_FALSE;
}
