#!/usr/bin/env bash
# ps5-debug.sh — collect logs from a jailbroken PS5 over FTP after a crash.
#
# Purpose : gather whatever the console recorded, over the read-only ftpsrv.
# Inputs  : env PS5_HOST (default 10.0.0.199), env PS5_FTP (default 2121)
# Outputs : a timestamped directory of logs under ./ps5-logs-<stamp>/
#
# Usage: ./ps5-debug.sh            # probe, collect what exists, print a summary
#        ./ps5-debug.sh --watch    # poll until the console answers, then collect
#
# ftpsrv is READ-ONLY (verified: MKD returns 550), so this never writes to the
# console. It only reads.

set -uo pipefail

PS5_HOST="${PS5_HOST:-10.0.0.199}"
PS5_PORT="${PS5_FTP:-2121}"
BASE="ftp://${PS5_HOST}:${PS5_PORT}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="./ps5-logs-${STAMP}"
mkdir -p "$OUT"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }

probe() {
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/" 2>/dev/null)
    if nc -z -w 4 "$PS5_HOST" "$PS5_PORT" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

if [ "${1:-}" = "--watch" ]; then
    bold "Waiting for $PS5_HOST:$PS5_PORT ..."
    until probe; do
        printf '.'
        sleep 10
    done
    printf '\n'
    bold "Console answered."
fi

if ! probe; then
    warn "Console is not reachable at ${PS5_HOST}:${PS5_PORT}."
    warn "A hard crash usually needs a power cycle: hold the power button ~10s."
    warn "Once it boots and you have re-run the jailbreak, re-run this script."
    exit 1
fi

bold "Console reachable. Collecting."

# Where a homebrew crash normally leaves evidence.
PATHS=(
    "data/logs"
    "data/pldmgr"
    "data/homebrew/PPSA99004/models"
    "user/home"
    "system_ex/app"
    "user/app"
)

for p in "${PATHS[@]}"; do
    name=$(printf '%s' "$p" | tr '/' '_')
    curl -s -m 30 "$BASE/$p/" -o "$OUT/listing${name}.txt" 2>/dev/null
    if [ -s "$OUT/listing${name}.txt" ]; then
        printf '  %-34s %s entries\n' "$p" "$(wc -l < "$OUT/listing${name}.txt" | tr -d ' ')"
    fi
done

# Pull any file that looks like a log or a crash dump.
bold "Fetching log/dump files ..."
for p in "${PATHS[@]}"; do
    while read -r entry; do
        case "$entry" in
            *log*|*dump*|*crash*|*klog*|*.log|*.dmp|*.txt) ;;
            *) continue ;;
        esac
        name=$(printf '%s' "$p/$entry" | tr '/' '_')
        curl -s -m 60 "$BASE/$p/$entry" -o "$OUT/$name" 2>/dev/null
        [ -s "$OUT/$name" ] && printf '  fetched %s/%s\n' "$p" "$entry"
    done < "$OUT/listing$(printf '%s' "$p" | tr '/' '_').txt" 2>/dev/null
done

bold "Summary of $OUT"
for f in "$OUT"/listing*; do
    [ -s "$f" ] || continue
    printf '\n--- %s ---\n' "$(basename "$f")"
    head -25 "$f"
done

# Surface anything that looks like a crash signature.
bold "Runtime indicators"
# The CPU text path fails at load rather than crashing, so a silent title that
# never shows a model is the expected symptom. These patterns name that case,
# along with the compat shims, which is the other silent failure mode.
patterns='crash|fatal|abort|segfault|exception|assert|trap|panic|direct arena|direct memory|ps5_compat|gguf_text|prosperoai'
found=0
for f in "$OUT"/*; do
    case "$f" in *.txt) continue ;; esac
    [ -s "$f" ] || continue
    if grep -aiE "$patterns" "$f" >/dev/null 2>&1; then
        printf '\n=== %s ===\n' "$(basename "$f")"
        grep -aiE -B2 -A2 "$patterns" "$f" | head -40
        found=1
    fi
done
[ "$found" -eq 0 ] && warn "No crash or runtime indicators in the files retrieved."
printf '\nLogs in %s\n' "$OUT"
