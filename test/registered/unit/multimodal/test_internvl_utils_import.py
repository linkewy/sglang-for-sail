import builtins
import importlib.util
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
# Load only the CI registration helper before the server runtime is installed.
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SOURCE = ROOT / "python/sglang/srt/multimodal/internvl_utils.py"


def load_source():
    spec = importlib.util.spec_from_file_location("internvl_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_missing_dependency(error_type):
    original = builtins.__import__
    failure = error_type("incompatible torchvision test fixture")

    def import_without_torchvision(name, *args, **kwargs):
        if name.startswith("torchvision"):
            raise failure
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_without_torchvision):
        module = load_source()
        from PIL import Image

        image = Image.new("RGB", (32, 32))
        calls = [
            lambda: module.build_transform(
                16, mean=module.IMAGENET_MEAN, std=module.IMAGENET_STD
            ),
            lambda: module.dynamic_resize_image(image, 4, 0.5, 4, 64),
            lambda: module.resize_image_to_pixels(image, 16, 16),
            lambda: module.video_to_pixel_values(image, 4, 0.5, 16, True),
        ]
        for call in calls:
            try:
                call()
            except RuntimeError as exc:
                assert "requires a torchvision build compatible" in str(exc), str(exc)
                assert exc.__cause__ is failure
            else:
                raise AssertionError("Image preprocessing accepted missing torchvision")


class TestInternVLDependency(unittest.TestCase):
    def test_missing_or_broken_torchvision(self):
        for error in ("ImportError", "RuntimeError"):
            with self.subTest(error=error):
                result = subprocess.run(
                    [sys.executable, __file__, "--missing", error],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestInternVLImages(unittest.TestCase):
    def test_image_and_video_numerics(self):
        import torch
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F

        module = load_source()
        image = Image.new("RGB", (32, 24), (80, 120, 200))
        image.putpixel((2, 3), (250, 0, 120))

        def reference(width, height):
            return F.normalize(
                F.to_tensor(image.resize((width, height), Image.BICUBIC)),
                module.IMAGENET_MEAN,
                module.IMAGENET_STD,
            ).unsqueeze(0)

        transform = module.build_transform(
            16, mean=module.IMAGENET_MEAN, std=module.IMAGENET_STD
        )
        expected = F.normalize(
            F.to_tensor(F.resize(image, [16, 16], InterpolationMode.BICUBIC)),
            module.IMAGENET_MEAN,
            module.IMAGENET_STD,
        )
        torch.testing.assert_close(transform(image), expected)
        resized = module.resize_image_to_pixels(image, 16, 8)
        self.assertEqual(resized.dtype, torch.float32)
        torch.testing.assert_close(resized, reference(16, 8))
        dynamic, tokens = module.dynamic_resize_image(image, 4, 0.5, 4, 64)
        width, height, expected_tokens = module.compute_dynamic_image_size(
            32, 24, 4, 0.5, 4, 64
        )
        self.assertEqual(tokens, expected_tokens)
        torch.testing.assert_close(dynamic, reference(width, height))
        for maintain_aspect in (True, False):
            video, tokens = module.video_to_pixel_values(
                image, 4, 0.5, 16, maintain_aspect
            )
            width, height, expected_tokens = (
                module.get_video_target_size_and_feature_size(
                    32, 24, 16, maintain_aspect, 4, 0.5
                )
            )
            self.assertEqual(tokens, expected_tokens)
            torch.testing.assert_close(video, reference(width, height))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--missing":
        check_missing_dependency(getattr(builtins, sys.argv[2]))
    else:
        unittest.main()
