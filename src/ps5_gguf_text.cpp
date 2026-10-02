/*
 * PS5 GGUF-backed text inference for models the AGC kernels cannot run.
 *
 * Why this exists
 * ---------------
 * ProsperoAI's text backends drive 16 precompiled AGC compute kernels whose
 * shape constants are compiled into the ISA. Verified while adding 27B
 * support: `4096` — the Qwen3.5-9B embedding width — appears as an immediate
 * inside `real_layer_agc.inc`'s `.shader_text`. Those kernels cannot be
 * retargeted to Qwen3.8-27B's 5120 width, and the repository ships no shader
 * source, only the compiled packages.
 *
 * So for architectures whose shape does not match a baked kernel, fall back to
 * the CPU backend. llama.cpp is already vendored and statically linked
 * (libllama.a, libggml-cpu.a) for the image/audio models; this reuses it for
 * text. Every GGUF then works, at CPU speed, with no new GPU kernels.
 *
 * This is the only route to running a large model on this hardware today.
 */

#include "ps5_gguf_text.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* llama.cpp is vendored as prebuilt archives; declare the small subset we use
 * rather than pulling in the full header, which the media paths do not include
 * either. Signatures follow vendor/include/llama.h in llama.cpp. */
struct llama_model;
struct llama_context;
struct llama_vocab;

/* llama.cpp's public header, vendored at vendor/include/llama.h.
 *
 * This file previously declared the llama structs and prototypes by hand.
 * Those declarations were wrong: llama_model_params has many more fields
 * than the 14 written here, in a different order, so writing n_gpu_layers or
 * vocab_only through them would have written to the wrong offsets and
 * corrupted the caller's copy. Using the real header removes that whole
 * class of bug — the compiler now checks field names and types.
 */
#include "llama.h"

ps5_gguf_text_t::ps5_gguf_text_t()
    : model_(nullptr), ctx_(nullptr), vocab_(nullptr)
{
}

ps5_gguf_text_t::~ps5_gguf_text_t()
{
    unload();
}

bool ps5_gguf_text_t::load(const char *gguf_path, int context_length,
                           int threads)
{
    unload();

    llama_backend_init();

    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; /* CPU path; the AGC kernels cannot run this model */
    mp.vocab_only = false;

    model_ = llama_model_load_from_file(gguf_path, mp);
    if (!model_) {
        fprintf(stderr, "[gguf_text] model load failed: %s\n", gguf_path);
        return false;
    }
    vocab_ = llama_model_get_vocab(model_);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)context_length;
    cp.n_batch = (uint32_t)context_length;
    cp.n_ubatch = (uint32_t)context_length;
    cp.offload_kqv = false;
    cp.n_threads = threads > 0 ? threads : 4;
    cp.n_threads_batch = cp.n_threads;
    /* The shipped ggml predates ggml_flash_attn_ext_set_n_kv_max, which
     * llama-graph.cpp references whenever flash attention is enabled. Plain
     * attention is correct here and keeps the link clean. */
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    ctx_ = llama_new_context_with_model(model_, cp);
    if (!ctx_) {
        fprintf(stderr, "[gguf_text] context creation failed\n");
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }
    printf("[gguf_text] loaded %s ctx=%d threads=%d vocab=%d\n", gguf_path,
           context_length, cp.n_threads, llama_vocab_n_tokens(vocab_));
    return true;
}

void ps5_gguf_text_t::unload()
{
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
}

bool ps5_gguf_text_t::ready() const
{
    return model_ != nullptr && ctx_ != nullptr;
}

/*
 * Greedy decode: tokenize, run the context, pick the highest-logit token,
 * repeat. Greedy keeps this path minimal and deterministic; sampling hooks can
 * be layered on later without touching the load path.
 */
int ps5_gguf_text_t::generate(const char *prompt, char *out, int out_capacity,
                              int max_tokens)
{
    if (!ready() || !prompt || !out || out_capacity <= 1) {
        return -1;
    }
    out[0] = '\0';

    /* llama_tokenize returns a negative value when the buffer is too small, and
     * the chat layer builds prompts up to 4096 characters — comfortably over a
     * fixed 512 tokens. Grow until it fits rather than failing every long
     * prompt. */
    int32_t capacity = 512;
    int32_t *tokens = (int32_t *)calloc((size_t)capacity, sizeof(int32_t));
    if (!tokens) {
        return -1;
    }

    const int32_t text_len = (int32_t)strlen(prompt);
    int32_t n_prompt = llama_tokenize(vocab_, prompt, text_len, tokens, capacity,
                                      true, true);
    while (n_prompt < 0 && capacity < 8192) {
        const int32_t grown = capacity * 2;
        int32_t *bigger =
            (int32_t *)realloc(tokens, (size_t)grown * sizeof(int32_t));
        if (!bigger) {
            free(tokens);
            return -1;
        }
        tokens = bigger;
        capacity = grown;
        n_prompt = llama_tokenize(vocab_, prompt, text_len, tokens, capacity,
                                  true, true);
    }
    if (n_prompt <= 0) {
        free(tokens);
        fprintf(stderr, "[gguf_text] tokenize failed (%d)\n", n_prompt);
        return -1;
    }

    /* llama_decode asserts n_tokens <= n_batch. A prompt longer than the
     * batch aborts the process rather than returning an error, which on the
     * PS5 is a hard crash with no diagnostic. Trim to the last token that
     * fits so an over-long prompt degrades to a truncated one. */
    int32_t batch_limit = (int32_t)llama_n_batch(ctx_);
    if (batch_limit <= 0) {
        batch_limit = n_prompt;
    }
    const int32_t trimmed = n_prompt > batch_limit ? batch_limit : n_prompt;
    if (trimmed != n_prompt) {
        fprintf(stderr, "[gguf_text] prompt %d tokens exceeds batch %d; "
                        "truncating\n", n_prompt, batch_limit);
        n_prompt = trimmed;
    }

    /* Drop any previous prompt. Without this the second turn's prompt is
     * appended to the first one's cache, and the model answers as though
     * it were still in the previous conversation. */
    llama_memory_clear(llama_get_memory(ctx_), true);

    struct llama_batch batch = llama_batch_get_one(tokens, n_prompt);
    if (llama_decode(ctx_, batch) != 0) {
        free(tokens);
        fprintf(stderr, "[gguf_text] prefill decode failed\n");
        return -1;
    }

    const float *logits = llama_get_logits_ith(ctx_, -1);
    if (!logits) {
        free(tokens);
        return -1;
    }

    const int32_t vocab = llama_vocab_n_tokens(vocab_);
    int written = 0;
    int32_t next = 0;
    int produced = 0;

    for (int step = 0; step < max_tokens && produced < out_capacity - 1; ++step) {
        next = 0;
        float best = logits[0];
        for (int32_t v = 1; v < vocab; ++v) {
            if (logits[v] > best) {
                best = logits[v];
                next = v;
            }
        }
        if (llama_vocab_is_eog(vocab_, next)) {
            break;
        }
        const char *piece = llama_vocab_get_text(vocab_, next);
        if (!piece) {
            break;
        }
        /* This vocabulary uses byte-level BPE sentinels, all multi-byte UTF-8
         * and all needing to be decoded rather than copied:
         *   c4 a0     U+0120  a space
         *   c4 8a     U+010A  a newline
         *   e2 96 81  U+2581  a space, GPT-2 style, unused by this vocab
         * The first guess here assumed the GPT-2 mapping, which is backwards
         * for this vocabulary and turned every space into a newline. Measured
         * by dumping pieces: token 695 is exactly c4 a0 c4 8a, a space
         * followed by a newline. */
        for (const unsigned char *p = (const unsigned char *)piece; *p;) {
            char decoded;
            int width;
            if (p[0] == 0xC4 && p[1] == 0xA0) {
                decoded = ' ';
                width = 2;
            } else if (p[0] == 0xC4 && p[1] == 0x8A) {
                decoded = '\n';
                width = 2;
            } else if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) {
                decoded = '\n';
                width = 3;
            } else {
                decoded = (char)*p;
                width = 1;
            }
            p += (unsigned)width;
            if (written + 1 >= out_capacity) {
                break;
            }
            out[written++] = decoded;
        }
        out[written] = '\0';
        produced = written;

        int32_t step_token = next;
        struct llama_batch step_batch = llama_batch_get_one(&step_token, 1);
        if (llama_decode(ctx_, step_batch) != 0) {
            break;
        }
        logits = llama_get_logits_ith(ctx_, -1);
        if (!logits) {
            break;
        }
    }

    free(tokens);
    return written;
}