/*
 * Compatibility shims for the shipped ProsperoAI ggml.
 *
 * The vendored ggml predates six functions that current llama-graph.cpp calls.
 * Every call site is behind an architecture or feature gate that Qwen never
 * reaches — NVFP4 weights, DeepSeek/GLM/MoE swiglu clamping, gated
 * accumulation precision, and the flash-attention n_kv_max hint. This path is
 * CPU text inference on Qwen.
 *
 * Defining them here lets a current libllama.a link against the ggml the
 * project already ships, instead of replacing ggml and breaking the media
 * libraries (libstable-diffusion.a needs older internals the new ggml dropped).
 *
 * Signatures are taken from the current ggml/include/ggml.h, not guessed.
 * Three of these return a value — ggml_prec_set_acc and ggml_prec_set_src
 * return bool, ggml_rope_set_offset returns the tensor — so the shims must
 * match or the caller reads garbage. Each one prints to stderr if actually
 * reached, so an assumption here failing is loud rather than silent.
 */

#include <stdint.h>

#include <stdio.h>

#include <stdbool.h>

enum ggml_prec {
    GGML_PREC_DEFAULT = 0,
    GGML_PREC_F32     = 10,
};

struct ggml_tensor;
struct ggml_context;

bool ggml_prec_set_acc(struct ggml_tensor *tensor, enum ggml_prec prec)
{
    (void)tensor;
    (void)prec;
    fprintf(stderr, "[ps5_compat] ggml_prec_set_acc reached; "
                    "accumulation precision not applied\n");
    return true;
}

bool ggml_prec_set_src(struct ggml_tensor *tensor, enum ggml_prec prec, int i0)
{
    (void)tensor;
    (void)prec;
    (void)i0;
    fprintf(stderr, "[ps5_compat] ggml_prec_set_src reached; "
                    "source precision not applied\n");
    return true;
}

struct ggml_tensor *ggml_swiglu_clamp(struct ggml_context *ctx,
                                      struct ggml_tensor *a,
                                      struct ggml_tensor *b, float limit)
{
    (void)ctx;
    (void)a;
    (void)b;
    (void)limit;
    fprintf(stderr, "[ps5_compat] ggml_swiglu_clamp is unsupported on this ggml\n");
    return 0;
}

struct ggml_tensor *ggml_dsv4_hc_pre_gated(struct ggml_context *ctx,
                                          struct ggml_tensor *x,
                                          struct ggml_tensor *gate,
                                          struct ggml_tensor *out)
{
    (void)ctx;
    (void)x;
    (void)gate;
    (void)out;
    fprintf(stderr, "[ps5_compat] ggml_dsv4_hc_pre_gated is unsupported on this ggml\n");
    return 0;
}

void ggml_flash_attn_ext_set_n_kv_max(struct ggml_tensor *a, int32_t n)
{
    (void)a;
    (void)n;
    fprintf(stderr, "[ps5_compat] flash attention is disabled for this model\n");
}

struct ggml_tensor *ggml_rope_set_offset(struct ggml_tensor *a, int n_offs)
{
    (void)n_offs;
    fprintf(stderr, "[ps5_compat] ggml_rope_set_offset reached; "
                    "rope offset not applied\n");
    return a;
}