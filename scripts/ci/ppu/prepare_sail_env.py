import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import urlopen

import tomllib
from sail_wheel_manifest import (
    PROJECTS,
    basename,
    canonical,
    inspect_wheel,
    read_json,
    require,
    sha256,
)

PROTECTED = {"torch", "triton", "torch-for-sail", "triton-for-sail"}
SELF = Path(__file__).resolve()


def clean_env():
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def run(command, **kwargs):
    return subprocess.run(
        [str(x) for x in command], check=True, env=clean_env(), **kwargs
    )


def snapshot():
    installed = {}
    for dist in metadata.distributions():
        name = canonical(dist.metadata["Name"])
        if name not in installed:
            installed[name] = {
                "version": dist.version,
                "path": str(Path(dist.locate_file("")).resolve()),
            }
    providers = metadata.packages_distributions()
    modules = {}
    for name in ("torch", "triton"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        matches = []
        for provider in providers.get(name, []):
            dist = metadata.distribution(provider)
            if any(
                Path(dist.locate_file(f)).resolve() == path for f in dist.files or []
            ):
                matches.append(canonical(provider))
        require(len(matches) == 1, f"Ambiguous provider for {name}")
        modules[name] = {
            "path": str(path),
            "distribution": matches[0],
            "version": installed[matches[0]]["version"],
        }
    protected_names = PROTECTED | {item["distribution"] for item in modules.values()}
    return {
        "installed": installed,
        "protected": {
            "modules": modules,
            "distributions": {
                k: v for k, v in installed.items() if k in protected_names
            },
        },
    }


def snapshot_using(python):
    return json.loads(
        run([python, "-s", SELF, "snapshot"], capture_output=True, text=True).stdout
    )


def compare_protected(before, after):
    require(before == after, "Vendor torch/triton distribution or import path changed")


def safe_url(url):
    parts = urlsplit(url)
    require(
        parts.scheme == "https"
        and parts.hostname
        and not parts.username
        and not parts.password
        and not parts.query
        and not parts.fragment,
        "Dependency URL must be credential-free HTTPS",
    )
    filename = basename(unquote(parts.path.rsplit("/", 1)[-1]))
    require(filename.endswith(".whl"), "Only binary dependency wheels are allowed")
    return filename


def validate_plan(report, protected, projects):
    forbidden = PROTECTED | set(protected["distributions"])
    entries = []
    seen = set()
    for item in report["install"]:
        name = canonical(item["metadata"]["name"])
        require(name not in seen, "Duplicate dependency in installation plan")
        seen.add(name)
        require(
            name not in forbidden and not name.startswith("nvidia-"),
            f"Forbidden runtime replacement: {name}",
        )
        info = item["download_info"]
        digest = info.get("archive_info", {}).get("hashes", {}).get("sha256", "")
        if name in PROJECTS:
            require(
                name in projects
                and info["url"] == projects[name]["url"]
                and digest == projects[name]["sha256"],
                "Project wheel must come from this artifact",
            )
            continue
        filename = safe_url(info["url"])
        require(re.fullmatch(r"[0-9a-f]{64}", digest), "Missing dependency SHA256")
        entries.append(
            {
                "name": name,
                "version": item["metadata"]["version"],
                "url": info["url"],
                "filename": filename,
                "sha256": digest,
            }
        )
    return entries


def validate_locked(entries, protected):
    forbidden = PROTECTED | PROJECTS | set(protected["distributions"])
    require(isinstance(entries, list), "Locked dependencies must be a list")
    names, filenames = set(), set()
    for record in entries:
        name = record["name"]
        require(
            re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) and name == canonical(name),
            "Locked dependency name must be canonical",
        )
        require(
            name not in forbidden and not name.startswith("nvidia-"),
            "Forbidden locked dependency",
        )
        require(name not in names, "Duplicate locked dependency")
        names.add(name)
        filename = safe_url(record["url"])
        require(
            filename == record["filename"] and filename not in filenames,
            "Invalid or duplicate dependency filename",
        )
        filenames.add(filename)
        require(
            re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Invalid dependency SHA256"
        )
        pieces = filename.removesuffix(".whl").split("-")
        require(
            len(pieces) in (5, 6)
            and canonical(pieces[0]) == name
            and pieces[1] == record["version"],
            "Locked dependency filename mismatch",
        )


def verify_download(path, record):
    require(sha256(path) == record["sha256"], "Dependency hash mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate dependency ZIP entry")
        infos = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(infos) == 1, "Ambiguous dependency metadata")
        info = BytesParser().parsebytes(archive.read(infos[0]))
        require(
            canonical(info.get("Name", "")) == record["name"]
            and info.get("Version") == record["version"],
            "Dependency metadata mismatch",
        )
        for member in archive.infolist():
            parts = member.filename.rstrip("/").split("/")
            require(
                all(p not in ("", ".", "..") for p in parts)
                and "\\" not in member.filename
                and not stat.S_ISLNK(member.external_attr >> 16),
                "Unsafe dependency ZIP path",
            )
            if (
                parts[0].endswith(".data")
                and len(parts) > 2
                and parts[1] in ("purelib", "platlib")
            ):
                parts = parts[2:]
            require(
                parts[0].split(".", 1)[0] not in ("torch", "triton"),
                "Dependency wheel overlaps protected modules",
            )
        pieces = path.name.removesuffix(".whl").split("-")
        require(
            canonical(pieces[0]) == record["name"] and pieces[1] == record["version"],
            "Dependency filename mismatch",
        )


def install_locked(python, entries, cache, protected):
    validate_locked(entries, protected)
    files = []
    for record in entries:
        path = cache / basename(record["filename"])
        with urlopen(record["url"], timeout=120) as response:
            safe_url(response.url)
            with path.open("xb") as dest:
                shutil.copyfileobj(response, dest)
        verify_download(path, record)
        files.append(path)
    if files:
        run([python, "-m", "pip", "install", "--no-index", "--no-deps", *files])


def resolve_install(python, requirements, before, projects, work):
    constraints = work / "protected.txt"
    constraints.write_text(
        "".join(
            f"{name}=={item['version']}\n"
            for name, item in before["protected"]["distributions"].items()
        )
    )
    report = work / "pip-report.json"
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--report",
            report,
            "--only-binary=:all:",
            "--constraint",
            constraints,
            *requirements,
        ]
    )
    entries = validate_plan(read_json(report), before["protected"], projects)
    install_locked(python, entries, work, before["protected"])
    compare_protected(before["protected"], snapshot_using(python)["protected"])
    return entries


def require_venv_path(path, prefix):
    path, prefix = path.resolve(), prefix.resolve()
    require(
        path.is_relative_to(prefix) and "site-packages" in path.parts,
        "Module/distribution was not installed into the fresh venv",
    )
    return path


def check_installed(artifact):
    records = [inspect_wheel(path) for path in sorted(artifact.glob("*.whl"))]
    require(
        len(records) == 2 and {r["name"] for r in records} == PROJECTS,
        "Invalid project wheel set",
    )
    prefix = Path(sys.prefix)
    for record in records:
        dist = metadata.distribution(record["name"])
        require(dist.version == record["version"], "Installed project version mismatch")
        require_venv_path(Path(dist.locate_file("")), prefix)
        module_name = "sglang" if record["name"] == "sglang" else "sgl_kernel"
        spec = importlib.util.find_spec(module_name)
        require(
            spec is not None and spec.origin is not None, "Missing installed package"
        )
        require_venv_path(Path(spec.origin), prefix)
        for relative, digest in record["native_files"].items():
            path = require_venv_path(Path(dist.locate_file(relative)), prefix)
            require(
                sha256(path) == digest, "Installed native extension differs from wheel"
            )
            if record["name"] == "sglang" and "multimodal" in relative:
                module_name = Path(relative).name.split(".")[0]
                extension = importlib.util.spec_from_file_location(module_name, path)
                extension.loader.exec_module(importlib.util.module_from_spec(extension))
    imported = importlib.import_module("sglang")
    require_venv_path(Path(imported.__file__), prefix)


def prepare(mode, prefix, artifact=None):
    require(
        sys.version_info[:2] == (3, 12) and sys.implementation.name == "cpython",
        "CPython 3.12 required",
    )
    require(not prefix.exists(), "Venv must be a new installation prefix")
    base = snapshot_using(sys.executable)
    run([sys.executable, "-m", "venv", "--system-site-packages", prefix])
    python = prefix / "bin/python3"
    before = snapshot_using(python)
    compare_protected(base["protected"], before["protected"])
    with tempfile.TemporaryDirectory(prefix="sail-deps-") as temp:
        work = Path(temp)
        if mode == "tools":
            resolve_install(
                python,
                [
                    "build",
                    "setuptools>=61",
                    "setuptools-rust>=1.10",
                    "setuptools-scm>=8",
                    "wheel",
                    "ninja",
                    "PyYAML",
                ],
                before,
                {},
                work,
            )
            return
        artifact = artifact.resolve()
        wheels = [inspect_wheel(path) for path in sorted(artifact.glob("*.whl"))]
        require(
            len(wheels) == 2 and {w["name"] for w in wheels} == PROJECTS,
            "Invalid project wheels",
        )
        projects = {
            w["name"]: {
                "url": (artifact / w["filename"]).as_uri(),
                "sha256": w["sha256"],
            }
            for w in wheels
        }
        lock_path = artifact / "runtime-deps.lock.json"
        if mode == "resolve":
            sglang = next(w for w in wheels if w["name"] == "sglang")
            entries = resolve_install(
                python,
                [
                    f"{artifact / sglang['filename']}[runtime_common]",
                    "sgl-eval==0.1.2",
                    # Resolve the image's protobuf consumers with the new runtime.
                    "google-api-core",
                    "googleapis-common-protos",
                    "pytest",
                    "PyYAML",
                ],
                before,
                projects,
                work,
            )
            added = {r["name"] for r in entries} | PROJECTS
            inherited = {
                name: item["version"]
                for name, item in before["installed"].items()
                if name not in added
            }
            lock = {
                "schema_version": 1,
                "protected": before["protected"],
                "inherited": inherited,
                "wheels": entries,
            }
        else:
            lock = read_json(lock_path)
            require(lock["schema_version"] == 1, "Unknown dependency lock schema")
            compare_protected(lock["protected"], before["protected"])
            for name, version in lock["inherited"].items():
                require(
                    before["installed"].get(name, {}).get("version") == version,
                    f"Inherited dependency differs: {name}",
                )
            install_locked(python, lock["wheels"], work, before["protected"])
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                *[artifact / w["filename"] for w in wheels],
            ]
        )
        compare_protected(before["protected"], snapshot_using(python)["protected"])
        run([python, "-m", "pip", "check"])
        run([python, "-s", SELF, "check-installed", "--artifact", artifact], cwd=work)
        if mode == "resolve":
            lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def validate_staging(root):
    for relative in (
        "python/setup.py",
        "python/pyproject_other.toml",
        "rust/Cargo.toml",
        "rust/Cargo.lock",
        "rust/rust-toolchain.toml",
        "rust/sglang-radix-tree/Cargo.toml",
        "rust/sglang-radix-tree/Cargo.lock",
        "scripts/release/get_version_tag.py",
        "python/sglang/srt/rust_extensions/torch_build.py",
        "rust/sglang-mm/src/lib.rs",
    ):
        require((root / relative).is_file(), f"Missing staging input: {relative}")
    toolchain = tomllib.loads((root / "rust/rust-toolchain.toml").read_text())
    require(toolchain["toolchain"]["channel"] == "1.92", "Unexpected Rust toolchain")
    workspace = tomllib.loads((root / "rust/Cargo.toml").read_text())["workspace"]
    for member in workspace["members"]:
        require(
            (root / "rust" / member / "Cargo.toml").is_file(),
            f"Missing Rust member: {member}",
        )
    require(not list(root.rglob("*.so")), "Staging includes stale native binaries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "snapshot",
            "tools",
            "resolve",
            "replay",
            "check-installed",
            "check-staging",
        ],
    )
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(snapshot()))
    elif args.command == "check-installed":
        check_installed(args.artifact)
    elif args.command == "check-staging":
        validate_staging(args.source)
    else:
        require(args.venv is not None, "--venv is required")
        prepare(args.command, args.venv.resolve(), args.artifact)


if __name__ == "__main__":
    main()
