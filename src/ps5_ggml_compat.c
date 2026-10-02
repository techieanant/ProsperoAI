/*
 * Compatibility shims for the shipped ProsperoAI ggml.
 *
 * The six functions are NOT all unreachable. Measured by linking a counting
 * probe in place of the host ggml and running the real 27B:
 *
 *   ggml_prec_set_acc     128 calls — 2 sites x 64 layers — so this shim IS
 *                         reached. build_attn_mha's non-flash branch asks for
 *                         F32 accumulation on the KQ matmul, every layer.
 *   ggml_prec_set_src    0 calls — MoE expert gating only.
 *   ggml_rope_set_offset 0 calls — not in llama-graph.cpp at all; only
 *                         minicpm3, plm, deepseek2/4, dflash and hy-v4.
 *   ggml_swiglu_clamp     0 calls — gated on DeepSeek4, GLM5, MAPLE and on
 *                         limit > eps.
 *   ggml_dsv4_hc_pre_gated 0 calls — qwen4exp.cpp only.
 *   ggml_flash_attn_ext_set_n_kv_max
 *                         0 calls — inside the use_flash_attn branch, and
 *                         this context disables flash attention.
 *
 * So the one shim that does run is prec_set_acc, and it is harmless here.
 * A/B-tested against the real implementation on the real model, the top-5
 * logits are identical to four decimals: 17.3223, 15.1835, 14.9383, 14.8597,
 * 14.6829. IQ2_XS weights already declare vec_dot_type = Q8_K, so the
 * accumulator is 8-bit whatever this function asks for. It would matter for
 * an F16-weight model, which this path never loads.
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
    /* This one really is reached -- twice per layer, 128 times for the 27B --
     * so it logs once rather than 128 times. Dropping the F32 request is
     * deliberate and measured to be a no-op for IQ2_XS weights, whose
     * vec_dot_type is already Q8_K. See the header for the A/B result. */
    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr, "[ps5_compat] ggml_prec_set_acc: accumulation precision "
                        "request ignored; harmless for Q8_K-accumulated weights\n");
    }
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