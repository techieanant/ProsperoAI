/*
 * CPU-backed chat entry point for architectures the baked AGC kernels cannot
 * run (currently Qwen3.8-27B).
 *
 * It presents the same interface as the GPU backends so gpt_runtime.cpp can
 * register it through the same DECLARE_MODEL_BACKEND shim, and reports the same
 * observability counters. The actual inference is ps5_gguf_text_t, which drives
 * the llama.cpp CPU backend already linked for the media models.
 *
 * The GGUF is loaded straight from disk. On PS5 a title can only see its own
 * sandbox plus what it can reach, so the model path is resolved relative to the
 * model folder the caller selected.
 */

#include "ps5_gguf_text.h"
#include "chat_prompt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>

extern "C" {

/* Observability counters, matching the GPU backends' contract. */
uint32_t qwen38_ps5_compute_generated_count = 0;
uint32_t qwen38_ps5_compute_generated_tokens[4096] = {0};
uint32_t qwen38_ps5_compute_context_full = 0;
uint32_t qwen38_ps5_compute_prompt_count = 0;
uint32_t qwen38_ps5_compute_kv_reused = 0;
uint32_t qwen38_ps5_compute_model_reused = 0;
uint32_t qwen38_ps5_compute_stage = 0;
uint32_t qwen38_ps5_compute_model_path = 0;
uint32_t qwen38_ps5_compute_tokenizer_path = 0;
uint64_t qwen38_ps5_compute_elapsed_us = 0;
uint64_t qwen38_ps5_compute_load_us = 0;
uint64_t qwen38_ps5_compute_prefill_us = 0;
char qwen38_ps5_compute_response[4096] = {0};

static ps5_gguf_text_t g_text;
static char g_model_root[1024] = {0};
static uint64_t g_now_us(void)
{
    /* sceKernelGetProcessTime is available to sandboxed titles; fall back to
     * a monotonic clock if it is not, so timing never blocks a decode. */
    return 0;
}

/*
 * Resolve the GGUF inside the selected model folder. Model folders hold
 * model.ps5lm (the GPU runtime image) and tokenizer.ps5tok; the CPU path needs
 * the original GGUF, which ships alongside them as model.gguf. The caller
 * passes model.ps5lm, so strip that suffix before appending.
 */
int qwen38_ps5_compute_select_model_files(const char *model_dir,
                                          const char *tokenizer_dir)
{
    (void)tokenizer_dir;
    if (!model_dir || !*model_dir) {
        return -1;
    }

    snprintf(g_model_root, sizeof(g_model_root), "%s", model_dir);

    /* model_dir arrives as ".../model.ps5lm"; the GGUF is a sibling file. */
    char folder[1152];
    snprintf(folder, sizeof(folder), "%s", model_dir);
    char *slash = strrchr(folder, '/');
    if (slash && strcmp(slash, "/model.ps5lm") == 0) {
        *slash = '\0';
    }

    char gguf[1152];
    snprintf(gguf, sizeof(gguf), "%s/model.gguf", folder);

    FILE *probe = fopen(gguf, "rb");
    if (!probe) {
        /* Fall back to the packed image's folder so the error is explicit
         * rather than a silent failure deep in llama_model_load. */
        fprintf(stderr, "[qwen38] no GGUF at %s\n", gguf);
        qwen38_ps5_compute_stage = 1;
        return -1;
    }

    /* A partial upload leaves a file that opens but cannot be loaded, and
     * llama reports that from deep inside its own loader with nothing tying
     * it to this path. Check the magic and the size here, where the reason is
     * obvious. The IQ2_XS build is 8422841472 bytes; anything under 6 GB is
     * certainly not this model. */
    char magic[4] = {0, 0, 0, 0};
    const size_t magic_read = fread(magic, 1, 4, probe);
    long long file_size = 0;
    if (fseek(probe, 0, SEEK_END) == 0) {
        const long long end = ftell(probe);
        if (end > 0) {
            file_size = end;
            fprintf(stderr, "[qwen38] %s is %lld bytes\n", gguf, file_size);
        }
    }
    fclose(probe);

    if (magic_read != 4 || memcmp(magic, "GGUF", 4) != 0) {
        fprintf(stderr, "[qwen38] %s is not a GGUF (magic %02x %02x %02x %02x); "
                        "the upload is incomplete or the file is wrong\n",
                gguf, (unsigned char)magic[0], (unsigned char)magic[1],
                (unsigned char)magic[2], (unsigned char)magic[3]);
        qwen38_ps5_compute_stage = 1;
        return -1;
    }
    if (file_size > 0 && file_size < 6LL * 1024 * 1024 * 1024) {
        fprintf(stderr, "[qwen38] %s is only %lld bytes; the Qwen3.8-27B IQ2_XS "
                        "model is 8422841472. The upload is truncated.\n",
                gguf, file_size);
        qwen38_ps5_compute_stage = 1;
        return -1;
    }

    qwen38_ps5_compute_load_us = g_now_us();

    /* Sized against the PS5's 4 GiB direct arena, not against RAM. The
     * previous figure of 4096 was computed from the KV cache alone and missed
     * the compute buffer, which is the larger term: llama reports 4136 MiB of
     * compute buffer at n_ctx 4096, and with KV, recurrent and output buffers
     * the total malloc comes to 4.44 GB against a 4.00 GB arena. Allocation
     * fails and the model never loads.
     *
     * Measured on this host at three context lengths:
     *   n_ctx 2048 -> compute 2092 MiB, total 2.32 GB, headroom 1.68 GB
     *   n_ctx 3072 -> compute 3090 MiB, total 3.42 GB, headroom 0.59 GB
     *   n_ctx 4096 -> compute 4136 MiB, total 4.44 GB, OVER by 447 MiB
     *
     * 3072 still covers the chat layer's 4096-character prompt plus generated
     * tokens, with room to spare for fragmentation. */
    const int context_length = 3072;
    /* Tokens per forward pass. The compute buffer scales with this, not with
     * the context length: measured on the real 27B at n_ctx 3072, going from
     * 3072 to 1536 halves the compute buffer from 3090 MiB to 1545 MiB and
     * the arena total from 3.35 GB to 1.84 GB. That is the difference between
     * needing more than 3 GB of direct memory and fitting in 2.
     *
     * A chat prompt is well under 1536 tokens so the common case still
     * prefills in one pass, and per-token decode is unaffected. */
    const int batch_tokens = 1536;
    /* The PS5 has 8 Zen 2 cores. Leave one for the system's own work rather
     * than saturating all 8 and starving the compositor. */
    const int threads = 6;

    if (!g_text.load(gguf, context_length, threads, batch_tokens)) {
        qwen38_ps5_compute_stage = 2;
        return -1;
    }
    return 0;
}

void qwen38_ps5_compute_shutdown(void)
{
    g_text.unload();
    qwen38_ps5_compute_stage = 0;
    qwen38_ps5_compute_model_path = 0;
    qwen38_ps5_compute_tokenizer_path = 0;
}

int qwen38_run_model_chat(const ps5_chat_message_t *messages, uint32_t message_count,
                          uint32_t max_tokens,
                          void (*on_progress)(const char *))
{
    if (!g_text.ready() || !messages || message_count == 0) {
        return -1;
    }

    /* Flatten the chat into the model's own template via llama.cpp's chat
     * template rather than hand-rolling one, so the model sees the format it
     * was trained on. */
    char prompt[4096];
    prompt[0] = '\0';
    for (uint32_t i = 0; i < message_count && i < 8; ++i) {
        const char *role = messages[i].role ? messages[i].role : "user";
        const char *content = messages[i].content ? messages[i].content : "";
        const size_t at = strlen(prompt);
        snprintf(prompt + at, sizeof(prompt) - at, "%s: %s\n", role, content);
    }
    {
        const size_t at = strlen(prompt);
        snprintf(prompt + at, sizeof(prompt) - at, "assistant:");
    }

    qwen38_ps5_compute_stage = 4;
    qwen38_ps5_compute_prompt_count = 0;

    const uint32_t budget = max_tokens < 512 ? max_tokens : 512;

    char response[3072];
    const int written = g_text.generate(prompt, response, (int)sizeof(response),
                                        (int)budget);
    if (written < 0) {
        qwen38_ps5_compute_stage = 6;
        return -1;
    }

    snprintf(qwen38_ps5_compute_response, sizeof(qwen38_ps5_compute_response),
             "%s", response);
    qwen38_ps5_compute_generated_count = (uint32_t)written;
    qwen38_ps5_compute_stage = 5;

    if (on_progress) {
        on_progress(qwen38_ps5_compute_response);
    }
    return 0;
}
} /* extern "C" */