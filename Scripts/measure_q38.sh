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
#   3. control run: the per-process fixed cost. The CLI verifies every
#      routed-expert layer file by SHA-256 on first touch (48 x 1.5 GiB
#      = 75.7 GB at ~2 GB/s), and that cost lands inside the reported
#      prefill. Long-prompt steps subtract this control.
#   4. regression run with the phases footer (expert cache hit rate)
#   5. decode slot-count curve, two passes (compare the second)
#   6. decode read-ahead advice modes
#   7. long-prompt prefill: slot sweep with the fixed cost subtracted, plus
#      the expert read volume per slot count
#
# Overrides:
#   Q38_MODEL=scratch/qwen38.gturbo   Q38_PROMPT="Say the word apple."
#   Q38_DECODE_TOKENS=64              Q38_SLOTS="16 24 32"
#   Q38_PREFILL_SLOTS="16 24 32"      Q38_PREFILL_WORDS=28
#   Q38_PREFILL_NEW=8                 Q38_RDADVISE="off bounded adaptive"
#   Q38_SEED=0 (Q38_SEED=off disables --seed)   Q38_TOP_K=10
#   Q38_SKIP_BUILD=1                  Q38_LOG=/some/path.log
#
# Slot counts below the routed top-k cannot be planned: one token routes
# top-k experts through a per-layer cache, so 8 is not usable on Qwen 3.8.
#
# Requires macOS 26+ / Swift 6.2+ and a completed scratch/qwen38.gturbo.

set -uo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_directory/.." || exit 1

model="${Q38_MODEL:-scratch/qwen38.gturbo}"
prompt="${Q38_PROMPT:-Say the word apple.}"
decode_tokens="${Q38_DECODE_TOKENS:-64}"
slot_counts="${Q38_SLOTS:-16 24 32}"
prefill_slots="${Q38_PREFILL_SLOTS:-16 24 32}"
rdadvise_modes="${Q38_RDADVISE:-off bounded adaptive}"
prefill_words="${Q38_PREFILL_WORDS:-28}"
prefill_new="${Q38_PREFILL_NEW:-8}"
top_k="${Q38_TOP_K:-10}"
seed="${Q38_SEED:-0}"
skip_build="${Q38_SKIP_BUILD:-0}"
binary=".build/release/TurboFieldfareCLI"
tail_lines=18

# A fixed seed makes the sampled token sequence reproducible, so the slot
# curve compares the same workload instead of sampling noise.
seed_args=()
if [[ "$seed" != "off" ]]; then
  seed_args=(--seed "$seed")
fi

# Parsed from the CLI footer / phases footer of the last run.
last_prefill_tok=""
last_prefill_s=""
last_new_tok=""
last_decode_s=""
last_tps=""
last_hit_pct=""
last_read_mb=""
# Prefill wall of the control run: the per-process cost that is not prefill.
control_prefill_s=""

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

clear_run_footer() {
  last_prefill_tok=""
  last_prefill_s=""
  last_new_tok=""
  last_decode_s=""
  last_tps=""
  last_hit_pct=""
  last_read_mb=""
}

# Pulls the timing footer and the expert-cache line out of a captured run.
parse_run_footer() {  # <captured-output-file>
  clear_run_footer
  local out_file="$1" footer=""
  footer="$(grep -oE '\[stop=[^]]*\]' "$out_file" | tail -1 || true)"
  [[ -n "$footer" ]] || return 0
  last_prefill_tok="$(grep -oE 'prefill=[0-9]+' <<<"$footer" | head -1 | tr -dc '0-9' || true)"
  last_prefill_s="$(grep -oE 'tok/[0-9.]+s' <<<"$footer" | head -1 | tr -dc '0-9.' || true)"
  last_new_tok="$(grep -oE 'new=[0-9]+' <<<"$footer" | head -1 | tr -dc '0-9' || true)"
  last_decode_s="$(grep -oE 'decode=[0-9.]+s' <<<"$footer" | head -1 | tr -dc '0-9.' || true)"
  last_tps="$(grep -oE 'tok/s=[0-9.]+' <<<"$footer" | head -1 | tr -dc '0-9.' || true)"
  last_hit_pct="$(grep -oE '\([0-9.]+%\)' "$out_file" | tail -1 | tr -dc '0-9.' || true)"
  last_read_mb="$(grep -oE '[0-9]+ MB read' "$out_file" | tail -1 | tr -dc '0-9' || true)"
}

print_run_summary() {
  [[ -n "$last_prefill_s" ]] || return 0
  printf '  summary: prefill %ss (%s tok) | decode %ss x %s tok = %s tok/s | cache %s%% hit, %s MB read\n' \
    "$last_prefill_s" "$last_prefill_tok" "$last_decode_s" "$last_new_tok" \
    "$last_tps" "$last_hit_pct" "$last_read_mb"
}

# Long-prompt prefill with the control's fixed cost removed.
fixed_adjusted_prefill() {
  awk -v wall="$last_prefill_s" -v fixed="$control_prefill_s" -v tok="$last_prefill_tok" 'BEGIN {
    adjusted = wall - fixed
    if (adjusted < 0) adjusted = 0
    printf "%.2fs over %s tok (%.1f ms/tok, %.1f tok/s)", \
      adjusted, tok, (tok > 0 ? adjusted * 1000 / tok : 0), (adjusted > 0 ? tok / adjusted : 0)
  }'
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
    if ! swift build -c release; then
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
    local out_file status=0
    out_file="$(mktemp "${TMPDIR:-/tmp}/q38measure.XXXXXX")" || {
      echo "!! mktemp failed"
      failed_runs=$((failed_runs + 1))
      return
    }
    "$binary" "$@" ${seed_args[@]+"${seed_args[@]}"} >"$out_file" 2>&1 || status=$?
    tail -n "$tail_lines" "$out_file"
    parse_run_footer "$out_file"
    rm -f "$out_file"
    if (( status != 0 )); then
      echo "!! run failed: $label (exit $status)"
      failed_runs=$((failed_runs + 1))
      clear_run_footer
      return
    fi
    print_run_summary
  }

  echo "== control: per-process fixed cost (expert layer SHA-256 verification)"
  run_bench "prompt='hello' max-new=1" \
    --model "$model" --prompt "hello" --max-new 1
  control_prefill_s="$last_prefill_s"
  if [[ -n "$control_prefill_s" ]]; then
    echo "   fixed cost ≈ ${control_prefill_s}s (48 layer files, 75.7 GB); the long-prompt steps below subtract it"
  else
    echo "!! control run produced no prefill footer; long-prompt adjustments skipped"
  fi

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

  echo "== prefill: about $((prefill_words * 9)) words, slot sweep"
  local long_prompt=""
  for ((i = 0; i < prefill_words; i++)); do
    long_prompt+="the quick brown fox jumps over the lazy dog "
  done
  local sweep_file
  sweep_file="$(mktemp "${TMPDIR:-/tmp}/q38sweep.XXXXXX")" || {
    echo "!! mktemp failed"
    return 1
  }
  for slots in $prefill_slots; do
    run_bench "long prompt slots=$slots max-new=$prefill_new" \
      --model "$model" --prompt "$long_prompt" --max-new "$prefill_new" \
      --expert-cache-slots "$slots"
    if [[ -n "$last_prefill_s" ]]; then
      printf '   fixed-cost adjusted prefill: %s\n' "$(fixed_adjusted_prefill)"
      printf '%s|%s|%s|%s\n' \
        "$slots" "$last_read_mb" "$last_prefill_s" "$last_prefill_tok" >>"$sweep_file"
    fi
  done
  echo "== long-prompt slot sweep (expert reads)"
  printf 'slots  read(MB)   prefill(s)  prefill(ms/tok)  fixed-adjusted(s)\n'
  while IFS='|' read -r sweep_slots sweep_read sweep_wall sweep_tok; do
    [[ -n "$sweep_slots" ]] || continue
    awk -v slots="$sweep_slots" -v read="$sweep_read" -v wall="$sweep_wall" \
        -v tok="$sweep_tok" -v fixed="$control_prefill_s" 'BEGIN {
      adjusted = wall - fixed
      if (adjusted < 0) adjusted = 0
      printf "%-6s %-10s %-11.2f %-15.1f %.2f\n", \
        slots, read, wall, (tok > 0 ? adjusted * 1000 / tok : 0), adjusted
    }'
  done <"$sweep_file"
  rm -f "$sweep_file"

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
