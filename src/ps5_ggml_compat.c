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
 * If any of these is ever reached, the assertion below makes that loud instead
 * of silently producing wrong numbers.
 */

#include <stdint.h>

#include <stdio.h>

enum ggml_prec {
    GGML_PREC_DEFAULT = 0,
    GGML_PREC_F32     = 10,
};

struct ggml_tensor;

void ggml_prec_set_acc(struct ggml_tensor *tensor, enum ggml_prec prec)
{
    (void)tensor;
    (void)prec;
}

void ggml_prec_set_src(struct ggml_tensor *tensor, enum ggml_prec prec, int i0)
{
    (void)tensor;
    (void)prec;
    (void)i0;
}

struct ggml_tensor *ggml_swiglu_clamp(struct ggml_tensor *ctx, struct ggml_tensor *a,
                                      struct ggml_tensor *b, float limit)
{
    (void)ctx;
    (void)a;
    (void)b;
    (void)limit;
    fprintf(stderr, "[ps5_compat] ggml_swiglu_clamp is unsupported on this ggml\n");
    return 0;
}

struct ggml_tensor *ggml_dsv4_hc_pre_gated(struct ggml_tensor *ctx, struct ggml_tensor *a)
{
    (void)ctx;
    (void)a;
    fprintf(stderr, "[ps5_compat] ggml_dsv4_hc_pre_gated is unsupported on this ggml\n");
    return 0;
}

void ggml_flash_attn_ext_set_n_kv_max(struct ggml_tensor *a, int32_t n)
{
    (void)a;
    (void)n;
    fprintf(stderr, "[ps5_compat] flash attention is disabled for this model\n");
}

void ggml_rope_set_offset(struct ggml_tensor *rope, int32_t offset)
{
    (void)rope;
    (void)offset;
}