/*
 * GGUF-backed text inference for models the baked AGC kernels cannot run.
 *
 * See ps5_gguf_text.cpp for why this exists. In short: the text backends drive
 * precompiled GPU kernels whose shape constants are compiled into the ISA, so
 * they only work for the exact model shape they were built for. Anything else
 * runs on the CPU backend, reusing the llama.cpp already linked for media.
 */

#pragma once

#include <stdint.h>

struct llama_model;
struct llama_context;
struct llama_vocab;

struct ps5_gguf_text_t
{
    struct llama_model *model_;
    struct llama_context *ctx_;
    const struct llama_vocab *vocab_;

    ps5_gguf_text_t();
    ~ps5_gguf_text_t();

    /* Load a GGUF and create a context. Returns false on failure. */
    bool load(const char *gguf_path, int context_length, int threads);

    void unload();
    bool ready() const;

    /* Greedy decode. Returns bytes written to out (excluding NUL), or -1. */
    int generate(const char *prompt, char *out, int out_capacity,
                 int max_tokens);
};