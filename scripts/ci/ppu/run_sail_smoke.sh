#!/usr/bin/env bash
set -euo pipefail

: "${SAIL_TESTED_SHA:?}"
: "${SAIL_BUILD_IMAGE:?}"
: "${SAIL_MANIFEST_SHA256:?}"
: "${SAIL_SMOKE_RUN_TOKEN:?}"
: "${SGLANG_PPU_EXPECTED_DEVICE_NAME:?}"
: "${SGLANG_PPU_SMOKE_MODEL:?}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export SAIL_ARTIFACT_DIR="$repo/wheel-python3.12-sail"
[[ "$SGLANG_PPU_EXPECTED_DEVICE_NAME" == ZW810E || "$SGLANG_PPU_EXPECTED_DEVICE_NAME" == ZW-M890P ]]
[[ -d "$SGLANG_PPU_SMOKE_MODEL" && -f "$SGLANG_PPU_SMOKE_MODEL/config.json" ]]
export SAIL_VALIDATION_SHA="${SAIL_VALIDATION_SHA:-$SAIL_TESTED_SHA}"
[[ "$SAIL_VALIDATION_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ $(git -c "safe.directory=$repo" -C "$repo" rev-parse HEAD) == "$SAIL_VALIDATION_SHA" ]] || {
  echo "Checkout HEAD differs from SAIL_VALIDATION_SHA" >&2
  exit 1
}
# Normalize the scheduler-selected PPU for the vendor runtime.
if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then
  : "${NVIDIA_VISIBLE_DEVICES:?}"
  [[ "$NVIDIA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
    echo "Expected one numeric scheduler-selected PPU" >&2
    exit 1
  }
  export CUDA_VISIBLE_DEVICES="$NVIDIA_VISIBLE_DEVICES"
fi
[[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || {
  echo "Expected exactly one CUDA-visible PPU" >&2
  exit 1
}
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1 SGLANG_TEST_MAX_RETRY=0 HF_HUB_OFFLINE=0 SGLANG_IS_IN_CI=1
work=$(mktemp -d "${TMPDIR:-/tmp}/sglang-ppu-smoke.XXXXXXXX")
export SAIL_SMOKE_RESULT_DIR="$work/results"
mkdir "$SAIL_SMOKE_RESULT_DIR"
runner_pid=""
cleanup() {
  code=$?
  trap - EXIT TERM INT
  set +e
  if [[ -n "$runner_pid" ]] && kill -0 -- "-$runner_pid" 2>/dev/null; then
    kill -TERM -- "-$runner_pid" 2>/dev/null
    for ((i=0; i<30; i++)); do
      kill -0 -- "-$runner_pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$runner_pid" 2>/dev/null
  fi
  if (( code != 0 )); then
    tail -n 200 "$SAIL_SMOKE_RESULT_DIR"/server.*.log 2>/dev/null
    cat "$SAIL_ARTIFACT_DIR/manifest.json"
    timeout --kill-after=5s 60s python3 -m pip list
    timeout --kill-after=5s 30s ppu-smi
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
python3 "$repo/scripts/ci/ppu/sail_wheel_manifest.py" verify "$SAIL_ARTIFACT_DIR" \
  --expected-sha "$SAIL_TESTED_SHA" --build-image "$SAIL_BUILD_IMAGE"
printf '%s  %s\n' "$SAIL_MANIFEST_SHA256" "$SAIL_ARTIFACT_DIR/manifest.json" | sha256sum --check --strict
setsid timeout --kill-after=30s 900s python3 "$repo/scripts/ci/ppu/prepare_sail_env.py" replay \
  --venv "$work/venv" --artifact "$SAIL_ARTIFACT_DIR" &
runner_pid=$!
wait "$runner_pid"
runner_pid=""
export PATH="$work/venv/bin:$PATH"
python3 "$repo/scripts/ci/ppu/sail_smoke_checks.py" registration --source "$repo"
cd "$repo/test"
setsid timeout --kill-after=30s 2100s python3 run_suite.py \
  --hw ppu --suite stage-a-test-1-gpu-ppu --timeout-from-est-time &
runner_pid=$!
wait "$runner_pid"
python3 "$repo/scripts/ci/ppu/sail_smoke_checks.py" result \
  --artifact "$SAIL_ARTIFACT_DIR" --result "$SAIL_SMOKE_RESULT_DIR/result.json"
