/* Exercises src/ps5_ggml_compat.c.
 *
 * The six shims ship inside eboot.bin and stand in for ggml functions the
 * project's older ggml predates. Three of them return a value that
 * llama-graph.cpp consumes, so a wrong return type is silent corruption
 * rather than a link error -- and they had shipped uncalled.
 *
 * Build and run:
 *   clang -std=c11 -O2 -c tests/test_ggml_compat.c -o t.o
 *   clang -std=c11 -O2 -c src/ps5_ggml_compat.c -o c.o
 *   clang t.o c.o -o t && ./t
 */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

enum ggml_prec { GGML_PREC_DEFAULT = 0, GGML_PREC_F32 = 10 };
struct ggml_tensor { int dummy; };
struct ggml_context { int dummy; };

bool ggml_prec_set_acc(struct ggml_tensor *, enum ggml_prec);
bool ggml_prec_set_src(struct ggml_tensor *, enum ggml_prec, int);
struct ggml_tensor *ggml_swiglu_clamp(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, float);
struct ggml_tensor *ggml_dsv4_hc_pre_gated(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *);
void ggml_flash_attn_ext_set_n_kv_max(struct ggml_tensor *, int32_t);
struct ggml_tensor *ggml_rope_set_offset(struct ggml_tensor *, int);

static int failures = 0;
static void check(const char *what, int ok) {
    printf("  %-34s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

int main(void) {
    struct ggml_tensor t = {0};
    struct ggml_tensor *tp = &t;
    struct ggml_context c = {0};

    /* Each prints to stderr; the point is the return value and that the
     * caller's memory survives. */
    check("ggml_prec_set_acc returns true",
          ggml_prec_set_acc(tp, GGML_PREC_F32) == true);
    check("ggml_prec_set_src returns true",
          ggml_prec_set_src(tp, GGML_PREC_F32, 0) == true);
    check("ggml_rope_set_offset returns its tensor",
          ggml_rope_set_offset(tp, 7) == tp);
    check("ggml_swiglu_clamp returns null",
          ggml_swiglu_clamp(&c, tp, tp, 1.0f) == NULL);
    check("ggml_dsv4_hc_pre_gated returns null",
          ggml_dsv4_hc_pre_gated(&c, tp, tp, tp) == NULL);

    ggml_flash_attn_ext_set_n_kv_max(tp, 128);

    check("caller struct intact", t.dummy == 0);

    printf("\n%s (%d failures)\n", failures ? "RESULT: SHIM TEST FAILED" : "RESULT: SHIMS BEHAVE", failures);
    return failures != 0;
}
