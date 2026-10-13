import argparse
import ensurepip
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import tomllib
from sail_wheel_manifest import (
    PROJECTS,
    basename,
    canonical,
    create as create_artifact,
    inspect_wheel,
    read_json,
    require,
    sha256,
    verify as verify_artifact,
)

PROTECTED = {"torch", "triton", "torch-for-sail", "triton-for-sail"}
CI_CLEANUP_IMAGE = (
    "download.t-head.cn/docker_release/pytorch@sha256:"
    "a33a67832f7ec442c481421e94255518dd67e464406f8e610c15d2b594b21f80"
)
CI_IMAGE_EXCLUSIONS = {"cloud-tpu-client": "0.10"}
CPU_TORCHVISION = {
    "name": "torchvision",
    "version": "0.28.0+cpu",
    "url": "https://download-r2.pytorch.org/whl/cpu/torchvision-0.28.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl",
    "filename": "torchvision-0.28.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
    "sha256": "b545d46f4d2f9d30381281cf22874bfe1d32a8a7b0ee8396fccde89f30c6a9d9",
}
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
    requirements = {}
    for dist in metadata.distributions():
        name = canonical(dist.metadata["Name"])
        requirements.setdefault(name, []).extend(dist.requires or [])
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
        "requirements": requirements,
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


def ci_image_exclusions():
    enabled = os.environ.get("SAIL_ALLOW_IMAGE_CLEANUP", "0")
    require(enabled in ("0", "1"), "Invalid image cleanup opt-in")
    if enabled == "0":
        return {}
    require(
        os.environ.get("SAIL_BUILD_IMAGE") == CI_CLEANUP_IMAGE,
        "Unapproved cleanup image",
    )
    require(
        sys.platform == "linux" and sys.prefix == sys.base_prefix,
        "Image cleanup requires the base Linux interpreter",
    )
    require(
        Path("/.dockerenv").is_file()
        or bool(os.environ.get("KUBERNETES_SERVICE_HOST")),
        "Image cleanup requires an explicitly disposable container",
    )
    return dict(CI_IMAGE_EXCLUSIONS)


def image_cleanup_plan(before):
    protected = PROTECTED | set(before["protected"]["distributions"])
    require(
        not protected.intersection(CI_IMAGE_EXCLUSIONS),
        "Cannot clean up a protected provider",
    )
    require(
        set(before["installed"]) <= set(before["requirements"]),
        "Incomplete reverse dependency snapshot",
    )
    for consumer, requirements in before["requirements"].items():
        if consumer in CI_IMAGE_EXCLUSIONS:
            continue
        for requirement in requirements:
            match = re.match(
                r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?=[\[(@;<>=!~]|$)", requirement
            )
            require(match is not None, f"Unrecognized requirement from {consumer}")
            require(
                canonical(match[1]) not in CI_IMAGE_EXCLUSIONS,
                f"Cleanup blocked by reverse dependency: {consumer}: {requirement}",
            )
    plan = {}
    for name, version in CI_IMAGE_EXCLUSIONS.items():
        if name in before["installed"]:
            require(
                before["installed"][name]["version"] == version,
                f"Unexpected cleanup version: {name}",
            )
            plan[name] = version
    return plan


def validate_image_exclusions(expected, actual, installed):
    require(expected == actual, "Image exclusion policy mismatch")
    require(
        not set(actual).intersection(installed),
        "Excluded package is still installed or planned",
    )


def normalize_ci_image(before):
    policy = ci_image_exclusions()
    if not policy:
        return before, policy
    plan = image_cleanup_plan(before)
    print("CI image cleanup plan:", json.dumps(plan, sort_keys=True), flush=True)
    if plan:
        run([sys.executable, "-m", "pip", "uninstall", "--yes", *sorted(plan)])
        after = snapshot_using(sys.executable)
        compare_protected(before["protected"], after["protected"])
        for field in ("installed", "requirements"):
            expected = {k: v for k, v in before[field].items() if k not in plan}
            require(
                after[field] == expected, f"Image cleanup changed unrelated {field}"
            )
    else:
        after = before
    validate_image_exclusions(policy, policy, after["installed"])
    print("CI image exclusions:", json.dumps(policy, sort_keys=True), flush=True)
    return after, policy


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
    validate_locked(entries, protected)
    return entries


def validate_locked(entries, protected):
    forbidden = PROTECTED | PROJECTS | set(protected["distributions"])
    require(isinstance(entries, list), "Locked dependencies must be a list")
    names, filenames = set(), set()
    for record in entries:
        name = record["name"]
        if name == "torchvision":
            require(record == CPU_TORCHVISION, "Expected pinned CPU torchvision wheel")
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
    for index, record in enumerate(entries, 1):
        path = cache / basename(record["filename"])
        print(
            f"Downloading dependency {index}/{len(entries)}: "
            f"{record['name']} {record['url']}",
            flush=True,
        )
        request = Request(record["url"], headers={"User-Agent": "sglang-sail-ci/1.0"})
        try:
            with urlopen(request, timeout=120) as response:
                safe_url(response.url)
                with path.open("xb") as dest:
                    shutil.copyfileobj(response, dest)
        except HTTPError as exc:
            raise RuntimeError(
                f"Dependency {record['name']} download failed: HTTP {exc.code}; "
                f"URL: {record['url']}"
            ) from None
        verify_download(path, record)
        files.append(path)
    if files:
        run([python, "-m", "pip", "install", "--no-index", "--no-deps", *files])


def resolve_install(python, requirements, before, projects, work, exclusions=None):
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
    policy = exclusions or {}
    validate_image_exclusions(policy, policy, {r["name"] for r in entries})
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


def check_torchvision():
    require(
        metadata.version("torchvision") == CPU_TORCHVISION["version"],
        "Expected pinned CPU torchvision version",
    )
    torch = importlib.import_module("torch")
    vision = importlib.import_module("torchvision")
    require_venv_path(Path(vision.__file__), Path(sys.prefix))
    boxes = torch.tensor(
        [[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 6, 6]], dtype=torch.float32, device="cpu"
    )
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="cpu")
    keep = vision.ops.nms(boxes, scores, 0.5)
    require(keep.tolist() == [0, 2], "CPU torchvision NMS returned incorrect indices")
    print("CPU torchvision native NMS check passed", flush=True)


def check_test_entrypoints():
    root = SELF.parents[3]
    for relative in (
        "test/run_suite.py",
        "test/registered/ppu/test_sail_qwen3_smoke.py",
    ):
        print(f"Checking test entrypoint imports: {relative}", flush=True)
        runpy.run_path(str(root / relative), run_name="sail_ci_import_check")
    print("Test entrypoint import checks passed", flush=True)


def check_installed(artifact, static_only=False):
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
            if (
                not static_only
                and record["name"] == "sglang"
                and "multimodal" in relative
            ):
                module_name = Path(relative).name.split(".")[0]
                extension = importlib.util.spec_from_file_location(module_name, path)
                extension.loader.exec_module(importlib.util.module_from_spec(extension))
    if not static_only:
        check_torchvision()
        imported = importlib.import_module("sglang")
        require_venv_path(Path(imported.__file__), prefix)
        check_test_entrypoints()


def prepare(mode, prefix, artifact=None):
    require(
        sys.version_info[:2] == (3, 12) and sys.implementation.name == "cpython",
        "CPython 3.12 required",
    )
    require(
        mode in ("tools", "install-only", "resolve", "replay"),
        "Unknown environment mode",
    )
    require(not prefix.exists(), "Venv must be a new installation prefix")
    base = snapshot_using(sys.executable)
    policy = {}
    if mode in ("resolve", "replay"):
        artifact = artifact.resolve()
        document = verify_artifact(
            artifact,
            os.environ.get("SAIL_TESTED_SHA", ""),
            os.environ.get("SAIL_BUILD_IMAGE", ""),
            stage="build" if mode == "resolve" else "runtime",
        )
        validation_sha = os.environ.get("SAIL_VALIDATION_SHA", document["git_sha"])
        require(re.fullmatch(r"[0-9a-f]{40}", validation_sha), "Invalid validation SHA")
        if mode == "replay":
            require(
                document.get("validation_sha", document["git_sha"]) == validation_sha,
                "Validation SHA differs from the runtime artifact",
            )
        manifest_digest = sha256(artifact / "manifest.json")
        if mode == "resolve":
            require(
                manifest_digest == os.environ.get("SAIL_BUILD_MANIFEST_SHA256"),
                "Build manifest hash mismatch",
            )
            compare_protected(document["vendor_runtime"], base["protected"]["modules"])
        policy = ci_image_exclusions()
        if mode == "replay":
            lock = read_json(artifact / "runtime-deps.lock.json")
            require(lock["schema_version"] == 1, "Unknown dependency lock schema")
            compare_protected(lock["protected"], base["protected"])
            validate_locked(lock["wheels"], base["protected"])
            validate_image_exclusions(
                lock.get("image_exclusions", {}),
                policy,
                set(lock["inherited"]) | {r["name"] for r in lock["wheels"]},
            )
            for name, version in lock["inherited"].items():
                expected = (
                    ensurepip.version()
                    if name == "pip"
                    else base["installed"].get(name, {}).get("version")
                )
                require(expected == version, f"Inherited dependency differs: {name}")
        base, actual_policy = normalize_ci_image(base)
        validate_image_exclusions(policy, actual_policy, base["installed"])
    run([sys.executable, "-m", "venv", "--system-site-packages", prefix])
    python = prefix / "bin/python3"
    before = snapshot_using(python)
    compare_protected(base["protected"], before["protected"])
    if mode == "replay":
        for name, version in lock["inherited"].items():
            require(
                before["installed"].get(name, {}).get("version") == version,
                f"Inherited dependency differs: {name}",
            )
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
                    f"torchvision @ {CPU_TORCHVISION['url']}#sha256={CPU_TORCHVISION['sha256']}",
                    # Resolve the full consumer chain without legacy 1.x fallbacks.
                    "google-api-core>=2",
                    "googleapis-common-protos",
                    "google-api-python-client>=2",
                    "pytest",
                    "PyYAML",
                    "tabulate",
                ],
                before,
                projects,
                work,
                exclusions=policy,
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
                "image_exclusions": policy,
            }
        elif mode == "replay":
            install_locked(python, lock["wheels"], work, before["protected"])
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--force-reinstall",
                *[artifact / w["filename"] for w in wheels],
            ]
        )
        after = snapshot_using(python)
        compare_protected(before["protected"], after["protected"])
        if mode == "install-only":
            run(
                [
                    python,
                    "-s",
                    SELF,
                    "check-installed",
                    "--static-only",
                    "--artifact",
                    artifact,
                ],
                cwd=work,
            )
            return
        validate_image_exclusions(policy, policy, after["installed"])
        run([python, "-m", "pip", "check"])
        run([python, "-s", SELF, "check-installed", "--artifact", artifact], cwd=work)
        if mode == "resolve":
            require(
                sha256(artifact / "manifest.json") == manifest_digest,
                "Build manifest changed during validation",
            )
            require(
                [inspect_wheel(artifact / w["filename"]) for w in document["wheels"]]
                == document["wheels"],
                "Built wheels changed during runtime validation",
            )
            lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
            create_artifact(
                artifact,
                dict(
                    document,
                    build_manifest_sha256=manifest_digest,
                    validation_sha=validation_sha,
                    build_run_id=os.environ.get("SAIL_BUILD_RUN_ID", ""),
                ),
            )


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
            "install-only",
            "resolve",
            "replay",
            "check-installed",
            "check-staging",
        ],
    )
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot":
        print(json.dumps(snapshot()))
    elif args.command == "check-installed":
        check_installed(args.artifact, static_only=args.static_only)
    elif args.command == "check-staging":
        validate_staging(args.source)
    else:
        require(args.venv is not None, "--venv is required")
        prepare(args.command, args.venv.resolve(), args.artifact)


if __name__ == "__main__":
    main()
