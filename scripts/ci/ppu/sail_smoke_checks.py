import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from sail_wheel_manifest import read_json, require, sha256, verify

SUITE = "stage-a-test-1-gpu-ppu"
TEST_FILE = "test/registered/ppu/test_sail_qwen3_smoke.py"


def identity():
    return {
        "run_token": os.environ["SAIL_SMOKE_RUN_TOKEN"],
        "git_sha": os.environ["SAIL_TESTED_SHA"],
        "manifest_sha256": os.environ["SAIL_MANIFEST_SHA256"],
        "expected_device": os.environ["SGLANG_PPU_EXPECTED_DEVICE_NAME"],
    }


def validate_registration(entries, has_main):
    require(
        has_main and len(entries) == 1,
        "Exactly one executable PPU registration is required",
    )
    entry = entries[0]
    require(
        entry.backend.name == "PPU"
        and entry.effective_suite == SUITE
        and entry.disabled is None
        and not entry.nightly
        and entry.est_time == 300,
        "Unexpected or disabled PPU registration",
    )


def validate_result(result, expected, manifest):
    try:
        require(result["schema_version"] == 1, "Unknown smoke result schema")
        require(
            all(result[k] == v for k, v in expected.items()),
            "Stale or foreign smoke result",
        )
        require(
            result["tests_run"] == 2
            and all(result[k] == 0 for k in ("failures", "errors", "skipped")),
            "Both smoke cases must execute without skips or failures",
        )
        require(
            sorted(result["successful_tests"]) == ["test_generate", "test_kernel_ops"],
            "Missing successful smoke case",
        )
        require(expected["expected_device"] in result["device"], "Wrong PPU board")
        require(result["cleanup_ok"] is True, "Server cleanup did not complete")
        http = result["http"]
        require(
            http["status"] == 200
            and http["text_nonempty"] is True
            and type(http["completion_tokens"]) is int
            and http["completion_tokens"] > 0,
            "Missing successful HTTP generation",
        )
        kernel = next(w for w in manifest["wheels"] if w["name"] == "sglang-kernel")
        native = result["native"]
        require(
            native["relative_path"].startswith("sgl_kernel/common_ops.cpython-312")
            and kernel["native_files"][native["relative_path"]] == native["sha256"],
            "Native extension does not match artifact",
        )
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError("Incomplete smoke result") from exc


def pod_evidence(pods, token, image, selector):
    matches = []
    for pod in pods:
        for container in pod.get("spec", {}).get("containers", []):
            env = {e["name"]: e.get("value") for e in container.get("env", [])}
            if (
                container["name"] == "worker"
                and env.get("SAIL_SMOKE_RUN_TOKEN") == token
            ):
                matches.append((pod, container))
    require(len(matches) <= 1, "Ambiguous worker Pod identity")
    if not matches:
        return None
    pod, container = matches[0]
    require(container["image"] == image, "Worker image differs from build image")
    key, value = selector.split("=", 1)
    require(
        pod["spec"].get("nodeSelector", {}).get(key) == value,
        "Worker selector mismatch",
    )
    statuses = [
        s
        for s in pod.get("status", {}).get("containerStatuses", [])
        if s["name"] == "worker"
    ]
    if not statuses or not statuses[0].get("imageID"):
        return None
    image_id = statuses[0]["imageID"]
    require(
        image_id.rsplit("@", 1)[-1].removeprefix("containerd://")
        == image.split("@", 1)[1],
        "Worker actual imageID digest mismatch",
    )
    meta = pod["metadata"]
    return dict(
        pod=meta["name"],
        uid=meta["uid"],
        namespace=meta["namespace"],
        job=meta["labels"]["ppu-job"],
        image=image,
        image_id=image_id,
        run_token=token,
    )


def observe_pod(output, namespace, selector):
    deadline = time.monotonic() + 80 * 60
    failures = 0
    while time.monotonic() < deadline:
        try:
            response = subprocess.run(
                [
                    "kubectl",
                    "get",
                    "pods",
                    "-n",
                    namespace,
                    "-l",
                    "app.kubernetes.io/managed-by=ppu-distributed-action",
                    "-o",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failures += 1
            detail = exc.stderr or str(exc)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            print(
                f"Pod query failed ({failures}/3): {detail[-4096:].strip()}",
                file=sys.stderr,
                flush=True,
            )
            if failures >= 3:
                raise RuntimeError(
                    "Pod observer stopped after 3 consecutive query failures"
                ) from exc
            time.sleep(5)
            continue
        failures = 0
        evidence = pod_evidence(
            json.loads(response.stdout)["items"],
            os.environ["SAIL_SMOKE_RUN_TOKEN"],
            os.environ["SAIL_BUILD_IMAGE"],
            selector,
        )
        if evidence:
            with output.open("x") as stream:
                json.dump(evidence, stream, sort_keys=True)
            return
        time.sleep(5)
    raise TimeoutError("No worker imageID evidence within scheduler budget")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["registration", "result", "observe", "receipt"]
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--selector")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.command == "registration":
        from sglang.test.ci.ci_register import ut_parse_one_file

        validate_registration(*ut_parse_one_file(str(args.source / TEST_FILE)))
    elif args.command == "observe":
        observe_pod(args.output, args.namespace, args.selector)
    else:
        artifact = verify(
            args.artifact, os.environ["SAIL_TESTED_SHA"], os.environ["SAIL_BUILD_IMAGE"]
        )
        require(
            sha256(args.artifact / "manifest.json")
            == os.environ["SAIL_MANIFEST_SHA256"],
            "Artifact differs from the single build output",
        )
        if args.command == "result":
            result = read_json(args.result)
            validate_result(result, identity(), artifact)
            print(json.dumps(result, sort_keys=True))
        else:
            evidence = read_json(args.result)
            require(
                evidence["job"] == args.job
                and evidence["run_token"] == os.environ["SAIL_SMOKE_RUN_TOKEN"],
                "Observer evidence belongs to another scheduler job",
            )
            require(
                evidence["image"] == os.environ["SAIL_BUILD_IMAGE"],
                "Observer image mismatch",
            )
            subprocess.run(
                [
                    "kubectl",
                    "wait",
                    "--for=delete",
                    f"pod/{evidence['pod']}",
                    "-n",
                    args.namespace,
                    "--timeout=90s",
                ],
                check=True,
                timeout=100,
            )
            with args.output.open("x") as stream:
                json.dump(
                    dict(identity(), build_image=evidence["image"], pod=evidence),
                    stream,
                    sort_keys=True,
                )


if __name__ == "__main__":
    main()
