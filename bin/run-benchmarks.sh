#!/usr/bin/env bash
set -euo pipefail

# ── Config ───────────────────────────────────────────────
PORT="${LLAMA_BENCH_PORT:-53085}"
HOST="${LLAMA_BENCH_HOST:-127.0.0.1}"
THREADS="${LLAMA_BENCH_THREADS:-24}"
CTX="${LLAMA_BENCH_CTX:-8192}"          # 5-shot MMLU prompts are ~3K tokens; 8K gives headroom
NGL="${LLAMA_BENCH_NGL:-50}"
LOG_DIR="$HOME/.ollama/logs"
RESULTS_DIR="$HOME/.ollama/benchmark-results"
ALIAS_FILE="$(dirname "$0")/model-aliases.json"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

CONDA_PY="/opt/anaconda3/envs/llama_cpp/bin/python3"
LLAMA_SERVER="$HOME/.ollama/llama.cpp/build/bin/llama-server"
TOKENIZER_PATH="${LLAMA_BENCH_TOKENIZER:-/Users/bobo/.ollama/models/source/DeepSeek-V4-Flash}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ALLOW_CODE_EVAL="1"

# ── Helpers ──────────────────────────────────────────────
wait_for_server() {
    local max_wait=120
    local waited=0
    while ! curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if [[ $waited -ge $max_wait ]]; then
            echo "ERROR: server did not become ready within ${max_wait}s" >&2
            return 1
        fi
    done
    echo "Server ready on port $PORT"
}

stop_server() {
    local pid
    pid=$(cat "$LOG_DIR/llama-server-bench.pid" 2>/dev/null || true)
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        rm -f "$LOG_DIR/llama-server-bench.pid"
    fi
    # Also kill anything on our port
    for p in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true); do
        kill "$p" 2>/dev/null || true
    done
    sleep 2
}

start_server() {
    local model="$1"
    local log="$LOG_DIR/llama-server-bench.log"
    echo "Starting server with model: $(basename "$model")"
    nohup "$LLAMA_SERVER" \
        -m "$model" \
        --port "$PORT" \
        --host "$HOST" \
        -t "$THREADS" \
        -c "$CTX" \
        -ngl "$NGL" \
        --no-warmup \
        > "$log" 2>&1 < /dev/null &
    echo $! > "$LOG_DIR/llama-server-bench.pid"
    wait_for_server
}

MODEL_ARGS="base_url=http://${HOST}:${PORT}/v1/completions,model=local-model,tokenizer=${TOKENIZER_PATH},tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=1,max_retries=3,max_length=${CTX}"

run_mmlu() {
    local model_label="$1"
    local out="$RESULTS_DIR/mmlu_${model_label}"
    echo "=== Running MMLU (5-shot) on ${model_label} ==="
    $CONDA_PY -m lm_eval \
        --model local-completions \
        --model_args "$MODEL_ARGS" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 1 \
        --output_path "$out" \
        --log_samples \
        2>&1 | tee "$RESULTS_DIR/mmlu_${model_label}.log"
    echo "MMLU ${model_label} done -> $out"
}

run_humaneval() {
    local model_label="$1"
    local out="$RESULTS_DIR/humaneval_${model_label}"
    echo "=== Running HumanEval (0-shot) on ${model_label} ==="
    $CONDA_PY -m lm_eval \
        --model local-completions \
        --model_args "$MODEL_ARGS" \
        --tasks humaneval \
        --num_fewshot 0 \
        --batch_size 1 \
        --output_path "$out" \
        --log_samples \
        --confirm_run_unsafe_code \
        2>&1 | tee "$RESULTS_DIR/humaneval_${model_label}.log"
    echo "HumanEval ${model_label} done -> $out"
}

# ── Cleanup trap ─────────────────────────────────────────
cleanup() { stop_server; }
trap cleanup EXIT

# ── Usage ────────────────────────────────────────────────
usage() {
    cat <<USAGE
Usage: $(basename "$0") [model ...]

  model    Model alias (from model-aliases.json) or full path to GGUF file.
           If not specified, defaults to all models in model-aliases.json.

Examples:
  $(basename "$0")
  $(basename "$0") deepseek-v4-flash-fp4-fp8-native
  $(basename "$0") /path/to/model.gguf
  $(basename "$0") model-a model-b

USAGE
}

# ── Alias resolution ─────────────────────────────────────
resolve_model() {
    local arg="$1"
    # If it's an existing file path, use it directly
    if [[ -f "$arg" ]]; then
        echo "$arg"
        return
    fi
    # Try alias file
    if [[ -f "$ALIAS_FILE" ]]; then
        local resolved
        resolved=$(python3 - "$arg" "$ALIAS_FILE" <<'PY3'
import json, sys, os
model = sys.argv[1]
alias_file = sys.argv[2]
if os.path.exists(model):
    print(model)
    sys.exit(0)
with open(alias_file) as f:
    aliases = json.load(f)
resolved = aliases.get(model, "")
if resolved and os.path.exists(resolved):
    print(resolved)
else:
    print("")
PY3
        )
        if [[ -n "$resolved" ]]; then
            echo "$resolved"
            return
        fi
    fi
    echo ""
}

# Derive a safe label from a name or path
derive_label() {
    local name="$1"
    # If it looks like an alias (no slashes), use it directly
    if [[ "$name" != *"/"* ]]; then
        echo "$name"
        return
    fi
    # Otherwise use the basename without extension
    local base
    base=$(basename "$name" .gguf)
    base=$(basename "$base" .GGUF)
    # Sanitize: replace non-alnum with dash
    echo "$base" | tr -c '[:alnum:]-' '-' | sed 's/--*/-/g' | sed 's/^-//;s/-$//'
}

# ── Main ─────────────────────────────────────────────────

# Collect models from args, or fall back to all aliases
MODEL_ENTRIES=()
if [[ $# -gt 0 ]]; then
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        usage; exit 0
    fi
    for arg in "$@"; do
        path=$(resolve_model "$arg")
        if [[ -z "$path" ]]; then
            echo "ERROR: model not found: $arg (not a file, not an alias)" >&2
            if [[ -f "$ALIAS_FILE" ]]; then
                echo "Known aliases:" >&2
                python3 - "$ALIAS_FILE" <<'PY4' >&2
import json, sys
with open(sys.argv[1]) as f:
    aliases = json.load(f)
for k in sorted(aliases):
    if not k.startswith("$"):
        print(f"  {k}")
PY4
            fi
            exit 1
        fi
        label=$(derive_label "$arg")
        echo "[model] $label -> $path"
        MODEL_ENTRIES+=("$label|$path")
    done
elif [[ -f "$ALIAS_FILE" ]]; then
    # No args: use all aliases as defaults
    while IFS= read -r key && IFS= read -r val; do
        if [[ -n "$val" && -f "$val" ]]; then
            echo "[model] $key -> $val"
            MODEL_ENTRIES+=("$key|$val")
        fi
    done < <(python3 - "$ALIAS_FILE" <<'PY5'
import json, sys, os
with open(sys.argv[1]) as f:
    aliases = json.load(f)
for k, v in sorted(aliases.items()):
    if not k.startswith("$"):
        print(k)
        print(v)
PY5
    )
else
    echo "ERROR: no models specified and no $ALIAS_FILE found" >&2
    echo "Pass model names/aliases as arguments, or create $ALIAS_FILE" >&2
    exit 1
fi

if [[ ${#MODEL_ENTRIES[@]} -eq 0 ]]; then
    echo "ERROR: no valid models found to benchmark" >&2
    exit 1
fi

echo "Benchmarks start: $(date)"
echo "Results dir: $RESULTS_DIR"
echo "Models to benchmark: ${#MODEL_ENTRIES[@]}"
echo "---"

for entry in "${MODEL_ENTRIES[@]}"; do
    model_label="${entry%%|*}"
    model_path="${entry##*|}"

    start_server "$model_path"

    run_mmlu "$model_label"
    run_humaneval "$model_label"

    stop_server
    echo "---"
done

echo "All benchmarks done: $(date)"
echo "Results in: $RESULTS_DIR"
ls -la "$RESULTS_DIR/"
