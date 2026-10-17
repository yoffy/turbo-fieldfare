#!/usr/bin/env bash
# Qwen 3.8 efficiency measurement pass.
#
#   Scripts/measure_q38.sh
#
# One model process at a time: the script aborts when another instance
# (server, app, decode service, CLI, tests) is already running, and never
# terminates anything by itself.
#
# What it collects into one log (scratch/q38_measure_<timestamp>.log):
#   1. host / revision / memory pressure and the precheck
#   2. install geometry: expert stride, per-expert sub-tensors, and the
#      derived per-token expert traffic plus slot-memory estimates
#   3. regression run with the phases footer (expert cache hit rate)
#   4. decode slot-count curve, two passes (compare the second pass)
#   5. decode read-ahead advice modes
#   6. prefill baseline on a long prompt
#
# Overrides:
#   Q38_MODEL=scratch/qwen38.gturbo   Q38_PROMPT="Say the word apple."
#   Q38_DECODE_TOKENS=64              Q38_SLOTS="8 16 32"
#   Q38_RDADVISE="off bounded adaptive"
#   Q38_PREFILL_WORDS=28              Q38_TOP_K=10
#   Q38_SKIP_BUILD=1                  Q38_LOG=/some/path.log
#
# Requires macOS 26+ / Swift 6.2+ and a completed scratch/qwen38.gturbo.

set -uo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.." || exit 1

model="${Q38_MODEL:-scratch/qwen38.gturbo}"
prompt="${Q38_PROMPT:-Say the word apple.}"
decode_tokens="${Q38_DECODE_TOKENS:-64}"
slot_counts="${Q38_SLOTS:-8 16 32}"
rdadvise_modes="${Q38_RDADVISE:-off bounded adaptive}"
prefill_words="${Q38_PREFILL_WORDS:-28}"
top_k="${Q38_TOP_K:-10}"
skip_build="${Q38_SKIP_BUILD:-0}"
binary=".build/release/TurboFieldfareCLI"
tail_lines=18

# The phases footer is diagnostic output on stderr; the counters only exist
# when it is enabled.
export TURBO_FIELDFARE_PHASES=1

# Processes that must not run while this script measures. Keep in sync with
# AGENTS.md ("Test rules").
model_process_names=(
  TurboFieldfareServer
  TurboFieldfareMac
  TurboFieldfareDecodeService
  TurboFieldfareCLI
  TurboFieldfarePackageTests
  swiftpm-testing-helper
  mlx_lm
  mlx-lm
)

# True when a model process is already running. `pgrep -f` matches whole
# command lines, so this script's own launcher can match by containing the
# pattern text; the current PID, its parent, and anything whose command line
# mentions this script are ignored.
check_no_model_process() {
  local pattern matches=() pid command_line
  pattern="$(IFS='|'; printf '%s' "${model_process_names[*]}")"
  for pid in $(pgrep -f "$pattern" || true); do
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command_line" == *measure_q38.sh* ]] && continue
    matches+=("$pid ${command_line:-unknown}")
  done
  if (( ${#matches[@]} > 0 )); then
    printf '%s\n' "${matches[@]}"
    return 1
  fi
  return 0
}

main() {
  echo "== measure_q38 $(date)"
  echo "== host"
  sw_vers 2>/dev/null || true
  echo "cpu: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "memory_bytes: $(sysctl -n hw.memsize 2>/dev/null || echo unknown)"

  echo "== revision"
  git rev-parse --short HEAD 2>/dev/null || echo "(not a git checkout)"
  git status --short 2>/dev/null || true

  echo "== precheck: no other model process"
  if ! check_no_model_process; then
    echo "!! another model process is running; stop it and rerun (nothing was terminated)."
    return 1
  fi
  echo "(clear)"

  if [[ ! -e "$model" ]]; then
    echo "!! model not found: $model (set Q38_MODEL=...)"
    return 1
  fi

  echo "== memory pressure"
  memory_pressure -Q 2>/dev/null | tail -n 3 || echo "(memory_pressure unavailable)"

  if [[ "$skip_build" != "1" ]]; then
    echo "== build"
    if ! swift build -c release --target TurboFieldfareServer; then
      echo "!! build failed"
      return 1
    fi
  fi
  if [[ ! -x "$binary" ]]; then
    echo "!! missing executable: $binary (build it or unset Q38_SKIP_BUILD)"
    return 1
  fi

  echo "== install geometry and slot-memory estimates"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$model" "$slot_counts" "$top_k" <<'PY' || echo "!! geometry dump failed"
import json, os, sys

model, slot_counts, top_k = sys.argv[1], [int(s) for s in sys.argv[2].split()], int(sys.argv[3])
root = os.path.join(model, "packed_experts")
layout = json.load(open(os.path.join(root, "layout.json")))
stride = int(layout["expertStride"])
layers = int(layout["numLayers"])
per_layer = int(layout["expertsPerLayer"])
print(f"expertStride {stride} bytes, layers {layers}, experts/layer {per_layer}")

first = layout["layers"][0]
path = os.path.join(root, first["file"])
print(f"layer file {first['file']}: {os.path.getsize(path)} bytes")
sub_tensors = first["experts"][0].get("tensors",
                                      first["experts"][0].get("subTensors", {}))
for name, entry in sorted(sub_tensors.items()):
    print(f"  {name}: {entry}")

gib = 1 << 30
per_token = top_k * layers * stride
print(f"cold expert traffic per token (top-{top_k}): "
      f"{per_token / 1e9:.2f} GB ({per_token / gib:.2f} GiB)")
for slots in slot_counts:
    total = slots * layers * stride
    print(f"slot memory at {slots:>2} slots/layer: "
          f"{total / 1e9:.2f} GB ({total / gib:.2f} GiB)")
PY
  else
    echo "(python3 unavailable; skipping the geometry dump)"
  fi

  # Runs the CLI and shows the tail (generated text + phases footer). A failed
  # run is reported and counted instead of aborting the whole pass.
  local failed_runs=0
  run_bench() {
    local label="$1"
    shift
    echo "-- $label"
    if ! "$binary" "$@" 2>&1 | tail -n "$tail_lines"; then
      echo "!! run failed: $label"
      failed_runs=$((failed_runs + 1))
    fi
  }

  echo "== regression run: same prompt as before, phases footer"
  run_bench "prompt='$prompt' max-new=16" \
    --model "$model" --prompt "$prompt" --max-new 16

  echo "== decode: slot-count curve (two passes; compare the second)"
  local pass slots mode
  for pass in 1 2; do
    for slots in $slot_counts; do
      run_bench "pass=$pass slots=$slots max-new=$decode_tokens" \
        --model "$model" --prompt "$prompt" \
        --max-new "$decode_tokens" --expert-cache-slots "$slots"
    done
  done

  echo "== decode: read-ahead advice modes"
  for mode in $rdadvise_modes; do
    run_bench "rdadvise=$mode max-new=$decode_tokens" \
      --model "$model" --prompt "$prompt" \
      --max-new "$decode_tokens" --rdadvise "$mode"
  done

  echo "== prefill: about $((prefill_words * 9)) words"
  local long_prompt=""
  for ((i = 0; i < prefill_words; i++)); do
    long_prompt+="the quick brown fox jumps over the lazy dog "
  done
  run_bench "long prompt max-new=8" \
    --model "$model" --prompt "$long_prompt" --max-new 8

  echo "== done"
  if (( failed_runs > 0 )); then
    echo "!! $failed_runs run(s) failed; inspect the log above"
    return 1
  fi
  return 0
}

mkdir -p "$(dirname -- "${Q38_LOG:-scratch/measure.log}")"
log="${Q38_LOG:-scratch/q38_measure_$(date +%Y%m%d_%H%M%S).log}"
main 2>&1 | tee "$log"
status="${PIPESTATUS[0]}"
echo "log: $log" | tee -a "$log"
exit "$status"
