import argparse
import hashlib
import json
import re
import sys
from importlib import metadata
from pathlib import Path

BUILD_TARGETS = {"sglang", "sgl-kernel", "sglang-kernel"}


def snapshot():
    packages = {}
    for dist in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name not in packages:
            packages[name] = {
                "version": dist.version,
                "record": hashlib.sha256(
                    (dist.read_text("RECORD") or "").encode()
                ).hexdigest(),
            }
    return packages


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("before", type=Path)
    check.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        args.output.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")
        return 0

    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    changed = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in before.keys() | after.keys()
        if name not in BUILD_TARGETS
        and not (name == "sailify" and name not in before)
        and before.get(name) != after.get(name)
    }
    if changed:
        print("Unexpected dependency changes:", file=sys.stderr)
        print(json.dumps(changed, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print("All preinstalled dependencies other than build targets are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
