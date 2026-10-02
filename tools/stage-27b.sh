#!/usr/bin/env bash
# Stages the Qwen3.8-27B IQ2_XS model into dist/PPSA99004/.
#
# The build does not stage models: models/ is populated by the user on the
# console, and an 8 GB copy is not something a build step should trigger. This
# script exists so re-staging is one command instead of a sequence that can be
# got subtly wrong -- which it was, twice.
#
# Usage: tools/stage-27b.sh [--verify]
#   (no args)  copy the model, tokenizer and manifest into the bundle
#   --verify   check an existing bundle by size and sha256, copy nothing
#
# Inputs:  ~/Downloads/q38/Qwen3.8-27B-IQ2_XS.gguf
#          ~/Downloads/q38/pack/bundle/tokenizer.ps5tok
# Outputs: dist/PPSA99004/models/qwen3.8-27b-iq2_xs/{model.gguf,tokenizer.ps5tok,model.json}

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app="$root/dist/PPSA99004"
dir="$app/models/qwen3.8-27b-iq2_xs"
src="${MODEL_27B:-$HOME/Downloads/q38/Qwen3.8-27B-IQ2_XS.gguf}"
tok="${TOKENIZER_27B:-$HOME/Downloads/q38/pack/bundle/tokenizer.ps5tok}"

verify_only=0
[[ ${1:-} == --verify ]] && verify_only=1

report() { printf '%s\n' "$@"; }

if (( verify_only )); then
    missing=0
    for f in "$app/eboot.bin" "$app/sce_sys/param.json" "$app/sce_module/libc.prx" \
             "$dir/model.gguf" "$dir/tokenizer.ps5tok" "$dir/model.json"; do
        if [[ -f $f ]]; then
            report "  OK       ${f#"$app"/}  $(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")"
        else
            report "  MISSING  ${f#"$app"/}"
            missing=1
        fi
    done
    if [[ -f $dir/model.gguf && -f $src ]]; then
        a=$(shasum -a 256 "$dir/model.gguf" | cut -d' ' -f1)
        b=$(shasum -a 256 "$src" | cut -d' ' -f1)
        if [[ $a == "$b" ]]; then
            report "  model.gguf sha256 matches source"
        else
            report "  model.gguf sha256 MISMATCH"
            missing=1
        fi
    fi
    (( missing == 0 )) && report "  bundle complete" || report "  bundle INCOMPLETE"
    exit $missing
fi

[[ -f $src ]] || { echo "missing source model: $src" >&2; exit 1; }
[[ -f $tok ]] || { echo "missing tokenizer: $tok" >&2; exit 1; }

mkdir -p "$dir"
report "staging 27B into $dir"
cp "$src" "$dir/model.gguf"
cp "$tok" "$dir/tokenizer.ps5tok"
cat > "$dir/model.json" <<'JSON'
{
  "name": "Qwen3.8 27B IQ2_XS",
  "purpose": "text-to-text"
}
JSON
report "staged. run with --verify to confirm."