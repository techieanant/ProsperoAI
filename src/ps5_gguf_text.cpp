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

struct llama_model_params {
    int32_t n_gpu_layers;
    int32_t n_ctx;
    int32_t n_batch;
    int32_t n_ubatch;
    int32_t n_seq_max;
    bool flash_attn;
    bool no_perf;
    void *kv_overrides;
    void *tensor_split;
    bool use_mmap;
    bool use_mlock;
    bool check_tensors;
    bool vocab_only;
    bool embeddings;
};

struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    bool offload_kqv;
    bool no_perf;
    bool flash_attn;
    void *kv_overrides;
};

struct llama_batch {
    int32_t n_tokens;
    int32_t *token;
    int32_t *embd;
    int32_t *pos;
    int32_t *n_seq_id;
    int32_t **seq_id;
    int32_t *logits;
};

extern "C" {
void llama_backend_init(void);
void llama_backend_free(void);
const struct llama_model_params *llama_model_default_params(void);
const struct llama_context_params *llama_context_default_params(void);
struct llama_model *llama_model_load_from_file(const char *path,
                                              struct llama_model_params);
void llama_model_free(struct llama_model *);
struct llama_context *llama_new_context_with_model(
    struct llama_model *, struct llama_context_params);
void llama_free(struct llama_context *);
const struct llama_vocab *llama_model_get_vocab(const struct llama_model *);
int32_t llama_tokenize(const struct llama_vocab *, const char *, int32_t,
                       int32_t *, int32_t, bool, bool);
struct llama_batch llama_batch_get_one(int32_t *tokens, int32_t n_tokens);
int32_t llama_decode(struct llama_context *, struct llama_batch);
float *llama_get_logits_ith(struct llama_context *, int32_t);
int32_t llama_vocab_n_tokens(const struct llama_vocab *);
int32_t llama_vocab_is_eog(const struct llama_vocab *, int32_t);
const char *llama_vocab_get_text(const struct llama_vocab *, int32_t);

struct llama_chat_message
{
    const char *role;
    const char *content;
};
const char *llama_chat_builtin_templates(const char *name);
int32_t llama_chat_apply_template(const char *tmpl,
                                  const struct llama_chat_message *chat,
                                  size_t n_msg, bool add_ass, char *buf,
                                  int32_t length);
}

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

    struct llama_model_params mp = *llama_model_default_params();
    mp.n_gpu_layers = 0; /* CPU path; the AGC kernels cannot run this model */
    mp.use_mmap = true;
    mp.vocab_only = false;
    /* The shipped ggml predates ggml_flash_attn_ext_set_n_kv_max, which
     * llama-graph.cpp references whenever flash attention is enabled. Plain
     * attention is correct here and keeps the link clean. */
    mp.flash_attn = false;

    model_ = llama_model_load_from_file(gguf_path, mp);
    if (!model_) {
        fprintf(stderr, "[gguf_text] model load failed: %s\n", gguf_path);
        return false;
    }
    vocab_ = llama_model_get_vocab(model_);

    struct llama_context_params cp = *llama_context_default_params();
    cp.n_ctx = (uint32_t)context_length;
    cp.n_batch = (uint32_t)context_length;
    cp.n_ubatch = (uint32_t)context_length;
    cp.offload_kqv = false;
    cp.n_threads = threads > 0 ? threads : 4;
    cp.n_threads_batch = cp.n_threads;

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

    const int prompt_tokens = 512;
    int32_t *tokens = (int32_t *)calloc(prompt_tokens, sizeof(int32_t));
    if (!tokens) {
        return -1;
    }
    const int32_t n_prompt = llama_tokenize(
        vocab_, prompt, (int32_t)strlen(prompt), tokens, prompt_tokens, true,
        true);
    if (n_prompt <= 0) {
        free(tokens);
        fprintf(stderr, "[gguf_text] tokenize failed (%d)\n", n_prompt);
        return -1;
    }

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
        const int length = (int)strlen(piece);
        if (written + length < out_capacity) {
            memcpy(out + written, piece, length);
            written += length;
            out[written] = '\0';
            produced += length;
        } else {
            break;
        }

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