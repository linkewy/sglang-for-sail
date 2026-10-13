#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || -z "$1" ]]; then
  echo "Usage: SAIL_TESTED_SHA=<commit> SAIL_BUILD_IMAGE=<image@sha256:digest> bash $0 <new-output-directory>" >&2
  exit 2
fi
: "${SAIL_TESTED_SHA:?Exact tested commit is required}"
: "${SAIL_BUILD_IMAGE:?Immutable build image is required}"
[[ "$SAIL_TESTED_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$SAIL_BUILD_IMAGE" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]]
repo=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
[[ $(git -C "$repo" rev-parse HEAD) == "$SAIL_TESTED_SHA" ]]
[[ -z $(git -C "$repo" status --porcelain) ]] || { echo "Build requires a clean committed checkout" >&2; exit 1; }
[[ $(git -C "$repo" rev-parse --is-shallow-repository) == false ]] || { echo "Full history and tags are required" >&2; exit 1; }
[[ ! -e "$1" ]] || { echo "Output directory must not exist" >&2; exit 1; }
output=$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$1")
helper="$repo/scripts/ci/ppu/prepare_sail_env.py"
manifest="$repo/scripts/ci/ppu/sail_wheel_manifest.py"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV SETUPTOOLS_SCM_PRETEND_VERSION SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG
unset SGLANG_BUILD_RUST_EXTS SGL_KERNEL_SKIP_SAILIFY SGL_KERNEL_SKIP_THIRD_PARTY SGL_KERNEL_THIRD_PARTY_DIR SGL_KERNEL_ACTLIZE_DIR
export PYTHONNOUSERSITE=1 PYTORCH_SAIL_ARCH="ppu_10;ppu_15" SGL_KERNEL_USE_SAILIFY=1
export SGL_KERNEL_COMPILE_THREADS="${SGL_KERNEL_COMPILE_THREADS:-16}"
case "$SGL_KERNEL_COMPILE_THREADS" in
  [1-9]|1[0-6]) ;;
  *) echo "Compile threads must be between 1 and 16" >&2; exit 2 ;;
esac
export MAX_JOBS="$SGL_KERNEL_COMPILE_THREADS"
python3 - <<'PY'
import platform
import sys
from torch.utils.cpp_extension import _get_cuda_arch_flags
if (sys.version_info[:2] != (3, 12) or sys.implementation.name != "cpython"
        or sys.platform != "linux" or platform.machine() != "x86_64"):
    raise RuntimeError("CPython 3.12 on linux/x86_64 is required")
flags = _get_cuda_arch_flags()
print("SAIL architecture flags:", flags)
if "--gpu-architecture=ppu_10,ppu_15" not in flags or any("gencode" in f for f in flags):
    raise RuntimeError("Vendor PyTorch hgcc integration is required")
PY
command -v hgcc
hgcc --version
command -v rustup
work=$(mktemp -d "${TMPDIR:-/tmp}/sglang-sail-build.XXXXXXXX")
cleanup() {
  code=$?
  trap - EXIT
  python3 -c 'import shutil,sys; shutil.rmtree(sys.argv[1])' "$work"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir "$work/source" "$work/artifact"
git -C "$repo" archive "$SAIL_TESTED_SHA" | tar -xf - -C "$work/source"
python3 "$helper" check-staging --source "$work/source"
python3 "$helper" tools --venv "$work/tools"
export PATH="$work/tools/bin:$PATH"
python3 "$helper" snapshot > "$work/protected-before.json"
sailify_revision=f401ecae76d7b4991d44920bde9edaac7e655fd8
git init -q "$work/sailify"
git -C "$work/sailify" fetch --depth=1 https://github.com/t-head/sailify.git "$sailify_revision"
git -C "$work/sailify" checkout --detach FETCH_HEAD
[[ $(git -C "$work/sailify" rev-parse HEAD) == "$sailify_revision" ]]
python3 -m pip install --no-deps --no-build-isolation "$work/sailify"
python3 "$helper" snapshot > "$work/protected-after.json"
python3 - "$work/protected-before.json" "$work/protected-after.json" <<'PY'
import json
import sys
before, after = (json.load(open(p)) for p in sys.argv[1:])
if before["protected"] != after["protected"]:
    raise RuntimeError("sailify installation changed the vendor runtime")
PY

export RUSTUP_TOOLCHAIN=1.92.0
rustup toolchain install 1.92.0 --profile minimal
rustc --version
cargo metadata --locked --no-deps --format-version 1 --manifest-path "$work/source/rust/Cargo.toml" > "$work/rust-metadata.json"
cargo metadata --locked --no-deps --format-version 1 --manifest-path "$work/source/rust/sglang-radix-tree/Cargo.toml" > "$work/radix-metadata.json"
version=$(python3 - "$repo" <<'PY'
import pathlib
import sys
import tomllib
from setuptools_scm import get_version
root = pathlib.Path(sys.argv[1])
config = tomllib.loads((root / "python/pyproject_other.toml").read_text())["tool"]["setuptools_scm"]
version = get_version(root=str(root), git_describe_command=config["git_describe_command"])
if version == "0.0.0.dev0":
    raise RuntimeError("SCM fallback version is forbidden")
print(version)
PY
)
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="$version"
cp "$work/source/python/pyproject_other.toml" "$work/source/python/pyproject.toml"
(
  cd "$work/source/python"
  python3 -m build --wheel --no-isolation --outdir "$work/artifact"
)
(
  cd "$work/source/python/sglang/kernels/aot"
  cp pyproject_sail.toml pyproject.toml
  python3 setup_sail.py bdist_wheel --dist-dir "$work/artifact"
)
# Use the image's Python interpreter to validate in a separate environment,
# without inheriting the build-tool venv.
base_python=$("$work/tools/bin/python3" -c 'import sys; print(sys._base_executable)')
"$base_python" "$helper" install-only --venv "$work/verify" --artifact "$work/artifact"
"$base_python" "$helper" snapshot > "$work/image.json"
"$base_python" "$manifest" create "$work/artifact" --stage build --snapshot "$work/image.json" \
  --expected-sha "$SAIL_TESTED_SHA" --build-image "$SAIL_BUILD_IMAGE"
mkdir -p "$(dirname "$output")"
mv "$work/artifact" "$output"
"$base_python" "$manifest" verify "$output" --stage build --expected-sha "$SAIL_TESTED_SHA" --build-image "$SAIL_BUILD_IMAGE"
