"""Convert CUDA sources and transitive headers in a cache mirror for hgcc.

sailify's recursive API rewrites inputs in place, so pass only mirrored copies,
never source-tree or installed third-party package directories. The mirror
preserves the relative layout of absolute paths and is keyed by a content digest
of the include closure, search paths, and converter/SDK versions.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import uuid
from typing import Dict, List, Optional, Set, Tuple

from sglang.kernels.jit.utils.common import cache_once

_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]+[<"]([^>"]+)[>"]', re.MULTILINE)
_SOURCE_EXTENSIONS = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".inl")
_KEY_HEX_LEN = 16


@cache_once
def sailify_version() -> str:
    """Get the installed sailify version without blocking caching if unknown."""
    try:
        from importlib.metadata import version as dist_version

        return dist_version("sailify")
    except Exception:
        pass
    try:
        from sailify.version import __version__

        return str(__version__)
    except Exception:
        return "unknown"


@cache_once
def _ppu_sdk_version() -> str:
    """Include the SDK version in the mirror digest for version-dependent headers."""
    try:
        from sailify.ppu_sdk_version import (
            detect_ppu_sdk_version,
            format_version_string,
        )

        return format_version_string(detect_ppu_sdk_version())
    except Exception:
        return "unknown"


def _sailify_root() -> pathlib.Path:
    from sglang.srt.environ import envs

    configured = envs.SGLANG_JIT_CACHE_DIR.get() or "~/.cache/sglang/jit"
    return pathlib.Path(configured).expanduser() / "sailified"


def _mirror_path(root: pathlib.Path, source: str) -> pathlib.Path:
    path = pathlib.Path(source)
    return root / path.relative_to(path.anchor)


def _discover_closure(
    cuda_files: List[str], include_dirs: List[str]
) -> Tuple[Set[str], Dict[str, Optional[str]]]:
    """Discover the closure read-only and record each header's include root."""
    visited: Set[str] = set()
    bases: Dict[str, Optional[str]] = {}
    queue: List[str] = []
    for filepath in cuda_files:
        filepath = os.path.abspath(filepath)
        if filepath not in visited:
            visited.add(filepath)
            bases[filepath] = None
            queue.append(filepath)

    while queue:
        filepath = queue.pop(0)
        try:
            source = pathlib.Path(filepath).read_text(errors="replace")
        except OSError:
            continue
        file_dir = os.path.dirname(filepath)
        for match in _INCLUDE_RE.finditer(source):
            inc = match.group(1)
            candidates = [(os.path.join(file_dir, inc), None)] + [
                (os.path.join(inc_dir, inc), os.path.abspath(inc_dir))
                for inc_dir in include_dirs
            ]
            for candidate, base in candidates:
                candidate = os.path.abspath(candidate)
                if candidate in visited:
                    break
                if os.path.isfile(candidate):
                    if os.path.splitext(candidate)[1] in _SOURCE_EXTENSIONS:
                        visited.add(candidate)
                        bases[candidate] = base
                        queue.append(candidate)
                    break
    return visited, bases


def _mirror_digest(closure: Set[str], include_dirs: List[str]) -> str:
    # Avoid circular imports: cache's environment fingerprint calls sailify_version.
    from sglang.kernels.jit.utils.compile import cache

    digest = hashlib.sha256()
    for part in (sailify_version(), _ppu_sdk_version(), tuple(include_dirs)):
        digest.update(repr(part).encode())
        digest.update(b"\0")
    for path in sorted(closure):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update((cache._file_digest(pathlib.Path(path)) or "").encode())
        digest.update(b"\0")
    return digest.hexdigest()[:_KEY_HEX_LEN]


def convert_cuda_sources(
    cuda_files: List[str], include_dirs: List[str]
) -> Tuple[List[str], List[str]]:
    """Return converted source paths and device flags for mirror/compat headers."""
    try:
        from sailify.sailify_python import sailify_extra_files_recursive
    except ImportError as error:
        raise RuntimeError(
            "JIT builds on PPU require sailify. "
            "Install it with `pip install -e /path/to/sailify`."
        ) from error

    closure, bases = _discover_closure(cuda_files, include_dirs)
    outdir = _sailify_root() / _mirror_digest(closure, include_dirs)
    converted = [str(_mirror_path(outdir, os.path.abspath(f))) for f in cuda_files]
    compat_header = outdir / ".ppu_compat" / "compatible_wrapper.h"
    if not (
        outdir.is_dir()
        and compat_header.is_file()
        and all(_mirror_path(outdir, path).is_file() for path in closure)
    ):
        tmp = outdir.parent / f"{outdir.name}.tmp-{uuid.uuid4().hex}"
        try:
            mirror: Dict[str, str] = {}
            for source in sorted(closure):
                target = _mirror_path(tmp, source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                mirror[source] = str(target)
            results = sailify_extra_files_recursive(
                output_directory=str(tmp),
                extra_files=[mirror[source] for source in sorted(closure)],
                header_include_dirs=[],
            )
            failures = [
                f"{path}: {result.status}"
                for path, result in results.items()
                if result.status != "ok"
            ]
            if failures:
                raise RuntimeError("\n".join(failures))
            if not (tmp / ".ppu_compat" / "compatible_wrapper.h").is_file():
                raise RuntimeError("sailify did not generate compatible_wrapper.h")
            outdir.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(tmp, outdir)
            except OSError:
                # Reuse a complete mirror published first by an identical conversion.
                if not (
                    compat_header.is_file()
                    and all(_mirror_path(outdir, path).is_file() for path in closure)
                ):
                    raise
        except Exception as error:
            raise RuntimeError(
                f"sailify failed to convert CUDA sources for hgcc: {error}"
            ) from error
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    mirrored_include_dirs = [
        f"-I{_mirror_path(outdir, base)}"
        for base in dict.fromkeys(bases.values())
        if base is not None
    ]
    flags = (
        [f"-I{outdir / '.ppu_compat'}"]
        + mirrored_include_dirs
        + [
            "-include",
            "compatible_wrapper.h",
        ]
    )
    return converted, flags
