import argparse
import hashlib
import json
import re
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ARCHES = ["ppu_10", "ppu_15"]
TAG = "cp312-cp312-linux_x86_64"
SAILIFY_REVISION = "f401ecae76d7b4991d44920bde9edaac7e655fd8"
PROJECTS = {"sglang", "sglang-kernel"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def regular(path):
    require(
        path.exists() and stat.S_ISREG(path.lstat().st_mode),
        f"Not a regular file: {path.name}",
    )
    return path


def sha256(path):
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def basename(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", value),
        "Unsafe artifact filename",
    )
    return value


def read_json(path):
    try:
        return json.loads(regular(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON: {path.name}") from exc


def inspect_wheel(path):
    path = regular(Path(path))
    basename(path.name)
    pieces = path.name.removesuffix(".whl").split("-")
    require(path.suffix == ".whl" and len(pieces) in (5, 6), "Invalid wheel filename")
    name, version = canonical(pieces[0]), pieces[1]
    require(name in PROJECTS, "Unexpected project wheel")
    require("-".join(pieces[-3:]) == TAG, "Wheel must use cp312-cp312-linux_x86_64")
    native = {}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "Duplicate ZIP entry")
            for member in names:
                p = PurePosixPath(member)
                require(
                    not p.is_absolute() and ".." not in p.parts and "\\" not in member,
                    "Unsafe wheel member",
                )
                mode = archive.getinfo(member).external_attr >> 16
                require(not stat.S_ISLNK(mode), "Wheel contains symlink")
            metadata_paths = [n for n in names if n.endswith(".dist-info/METADATA")]
            wheel_paths = [n for n in names if n.endswith(".dist-info/WHEEL")]
            require(
                len(metadata_paths) == len(wheel_paths) == 1, "Ambiguous wheel metadata"
            )
            require(
                metadata_paths[0].rsplit("/", 1)[0] == wheel_paths[0].rsplit("/", 1)[0],
                "Different dist-info directories",
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            tags = (
                BytesParser()
                .parsebytes(archive.read(wheel_paths[0]))
                .get_all("Tag", [])
            )
            require(
                canonical(metadata.get("Name", "")) == name, "Package name mismatch"
            )
            require(metadata.get("Version") == version, "Package version mismatch")
            require(tags == [TAG], "WHEEL tags mismatch")
            for member in names:
                if member.endswith(".so"):
                    with archive.open(member) as stream:
                        header = stream.read(64)
                    require(
                        header[:6] == b"\x7fELF\x02\x01"
                        and header[18:20] == b"\x3e\x00",
                        "Native extension is not x86_64 ELF",
                    )
                    with archive.open(member) as stream:
                        native[member] = hashlib.file_digest(
                            stream, "sha256"
                        ).hexdigest()
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid wheel ZIP") from exc
    if name == "sglang-kernel":
        require(
            any(
                re.fullmatch(r"sgl_kernel/common_ops\.cpython-312[^/]*\.so", n)
                for n in native
            ),
            "Missing CPython 3.12 common_ops",
        )
    else:
        require(version != "0.0.0.dev0", "SCM fallback version is forbidden")
        require(
            any("multimodal" in n and n.startswith("sglang/") for n in native),
            "Missing multimodal Rust extension",
        )
    return {
        "filename": path.name,
        "name": name,
        "version": version,
        "sha256": sha256(path),
        "tag": TAG,
        "native_files": native,
    }


def validate_identity(identity):
    require(
        re.fullmatch(r"[0-9a-f]{40}", identity.get("git_sha", "")),
        "Expected immutable git SHA",
    )
    image = identity.get("build_image", "")
    require(
        re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image),
        "Expected immutable image digest",
    )
    require(
        identity.get("sailify_revision") == SAILIFY_REVISION,
        "Unexpected sailify revision",
    )
    require(identity.get("rust_toolchain") == "1.92", "Unexpected Rust toolchain")
    for module in ("torch", "triton"):
        record = identity.get("vendor_runtime", {}).get(module, {})
        require(
            bool(record.get("distribution")) and bool(record.get("version")),
            f"Missing vendor runtime: {module}",
        )


def checksums(root, files):
    return "".join(f"{sha256(root / name)}  {name}\n" for name in sorted(files))


def create(root, identity, stage="runtime"):
    root = Path(root)
    require(stage in ("build", "runtime"), "Unknown artifact stage")
    validate_identity(identity)
    wheels = [inspect_wheel(path) for path in sorted(root.glob("*.whl"))]
    require(
        len(wheels) == 2 and {w["name"] for w in wheels} == PROJECTS,
        "Exactly one wheel per project is required",
    )
    lock = root / "runtime-deps.lock.json"
    if stage == "runtime":
        require(
            read_json(lock).get("schema_version") == 1, "Unknown dependency lock schema"
        )
    else:
        require(not lock.exists(), "Build artifact must not include a runtime lock")
    document = dict(
        identity,
        schema_version=1,
        stage=stage,
        python_tag="cp312",
        abi_tag="cp312",
        platform_tag="linux_x86_64",
        pytorch_sail_arch=ARCHES,
        build_image_digest=identity["build_image"].split("@", 1)[1],
        wheels=wheels,
    )
    files = [w["filename"] for w in wheels] + ["manifest.json"]
    if stage == "runtime":
        document["runtime_deps_sha256"] = sha256(lock)
        files.append(lock.name)
    (root / "manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
    )
    (root / "SHA256SUMS").write_text(checksums(root, files))
    verify(root, identity["git_sha"], identity["build_image"], stage=stage)
    return document


def verify(root, expected_sha, expected_image, stage="runtime"):
    root = Path(root)
    require(stage in ("build", "runtime"), "Unknown artifact stage")
    document = read_json(root / "manifest.json")
    try:
        validate_identity(document)
        require(document["stage"] == stage, "Artifact stage mismatch")
        require(
            type(document["schema_version"]) is int and document["schema_version"] == 1,
            "Unknown manifest schema",
        )
        require(document["git_sha"] == expected_sha, "Git SHA mismatch")
        require(document["build_image"] == expected_image, "Image mismatch")
        require(
            document["build_image_digest"] == expected_image.split("@", 1)[1],
            "Digest mismatch",
        )
        require(
            (document["python_tag"], document["abi_tag"], document["platform_tag"])
            == ("cp312", "cp312", "linux_x86_64"),
            "Python/ABI/platform mismatch",
        )
        require(document["pytorch_sail_arch"] == ARCHES, "SAIL architecture mismatch")
        wheels = document["wheels"]
        require(
            len(wheels) == 2 and {w["name"] for w in wheels} == PROJECTS,
            "Missing or duplicate project wheel",
        )
        files = [basename(w["filename"]) for w in wheels]
        require(len(files) == len(set(files)), "Duplicate wheel filename")
        expected_files = set(files) | {"manifest.json", "SHA256SUMS"}
        if stage == "runtime":
            expected_files.add("runtime-deps.lock.json")
        else:
            require(
                "runtime_deps_sha256" not in document,
                "Build artifact claims a runtime lock",
            )
        require(
            {p.name for p in root.iterdir()} == expected_files,
            "Artifact file set mismatch",
        )
        for record in wheels:
            require(
                inspect_wheel(root / record["filename"]) == record,
                "Wheel metadata/hash mismatch",
            )
        if stage == "runtime":
            require(
                sha256(root / "runtime-deps.lock.json")
                == document["runtime_deps_sha256"],
                "Dependency lock hash mismatch",
            )
            require(
                read_json(root / "runtime-deps.lock.json").get("schema_version") == 1,
                "Unknown dependency lock schema",
            )
        require(
            regular(root / "SHA256SUMS").read_text()
            == checksums(root, expected_files - {"SHA256SUMS"}),
            "SHA256SUMS mismatch",
        )
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("Malformed manifest") from exc
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create", "verify"])
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--build-image", required=True)
    parser.add_argument("--stage", choices=["build", "runtime"], default="runtime")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        if args.stage == "build":
            require(
                args.snapshot is not None, "Build creation requires an image snapshot"
            )
            lock = read_json(args.snapshot)
        else:
            lock = read_json(args.artifact_dir / "runtime-deps.lock.json")
        identity = {
            "git_sha": args.expected_sha,
            "build_image": args.build_image,
            "sailify_revision": SAILIFY_REVISION,
            "rust_toolchain": "1.92",
            "vendor_runtime": lock["protected"]["modules"],
        }
        create(args.artifact_dir, identity, stage=args.stage)
    else:
        verify(args.artifact_dir, args.expected_sha, args.build_image, stage=args.stage)
    print(f"Artifact {args.command}: OK")


if __name__ == "__main__":
    main()
