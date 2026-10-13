import ast
import importlib.util
import runpy
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
# Load only the CI registration helper before the server runtime is installed.
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=20, suite="base-a-test-cpu")
KERNEL = ROOT / "python/sglang/kernels/aot/python/sgl_kernel"


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scenario(kind, backend, dtype="float16", compiling=False):
    import torch

    sail = backend == "sail"
    available = backend != "cpu"
    with (
        patch.object(torch.cuda, "is_available", return_value=available),
        patch.object(
            torch.cuda,
            "get_device_name",
            return_value="ZW810E" if sail else "NVIDIA H100",
        ),
        patch.object(torch.compiler, "is_dynamo_compiling", return_value=compiling),
    ):
        if kind == "dual":
            runtime = types.ModuleType("sglang.srt.runtime_context")
            runtime.get_forward = Mock()
            runtime.get_parallel = Mock()
            utils = types.ModuleType("sglang.srt.utils")
            utils.is_cuda = lambda: available
            utils.is_sail = lambda: sail
            native_name = "sglang.kernels.ops.gemm.cutedsl_dual_gemm"
            native = types.ModuleType(native_name)
            native.MAX_DUAL_GEMM_DECODE_TOKENS = 8
            native.DualGemmQuantMode = types.SimpleNamespace(UNQUANT="unquant")
            native.can_use_dual_gemm = Mock(return_value=True)
            unquant = types.ModuleType("sglang.srt.layers.quantization.unquant")
            unquant.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})
            sys.modules.update(
                {
                    runtime.__name__: runtime,
                    utils.__name__: utils,
                    native_name: None if sail else native,
                    unquant.__name__: unquant,
                }
            )
            module = load_source(
                "dual_under_test", ROOT / "python/sglang/srt/layers/dual_gemm.py"
            )
            obj = module.DualGemm.__new__(module.DualGemm)
            layer = types.SimpleNamespace(
                quant_method=unquant.UnquantizedLinearMethod(),
                params_dtype=torch.float16,
                output_partition_sizes=[128],
            )
            mode = obj._select_mode(layer, 128)
            assert mode == (None if sail or not available else "unquant"), mode
            return

        package = types.ModuleType("sgl_kernel")
        package.__path__ = [str(KERNEL)]
        sys.modules["sgl_kernel"] = package
        utils = load_source("sgl_kernel.utils", KERNEL / "utils.py")
        if kind == "detect":
            assert hasattr(utils, "is_sail"), "SAIL detection helper is missing"
            assert utils.is_sail() == sail
            assert utils.is_sail() == sail
            assert utils.is_sail.cache_info().maxsize == 1
            assert utils.is_sail.cache_info().hits == 1
            return

        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__path__ = []
        sys.modules["flashinfer"] = flashinfer
        for part in ("norm", "sampling"):
            mod = types.ModuleType(f"flashinfer.{part}")
            setattr(flashinfer, part, mod)
            sys.modules[mod.__name__] = None if backend == "rocm" else mod
        names = (
            "rmsnorm",
            "fused_add_rmsnorm",
            "gemma_rmsnorm",
            "gemma_fused_add_rmsnorm",
        )
        sampling_names = ("top_k_renorm_probs", "top_p_renorm_probs")
        ops = types.SimpleNamespace()
        for name in names + sampling_names:
            setattr(ops, name, types.SimpleNamespace(default=Mock()))
            target = flashinfer.norm if name in names else flashinfer.sampling
            setattr(target, name, Mock())
        with patch.object(torch.ops, "sgl_kernel", ops):
            if kind == "norm":
                module = load_source(
                    "sgl_kernel.elementwise", KERNEL / "elementwise.py"
                )
                x = torch.ones(2, 16, dtype=getattr(torch, dtype))
                w = torch.ones(16, dtype=x.dtype)
                for name in names:
                    args = (x, x.clone(), w) if "fused" in name else (x, w)
                    getattr(module, name)(*args, enable_pdl=False)
                    internal = (
                        sail or backend == "rocm" or dtype == "float32" or compiling
                    )
                    assert getattr(ops, name).default.call_count == int(internal), name
                    assert getattr(flashinfer.norm, name).call_count == int(
                        not internal
                    ), name
            else:
                module = load_source("sgl_kernel.sampling", KERNEL / "sampling.py")
                x = torch.ones(2, 16)
                if backend == "musa":
                    x = Mock(spec=torch.Tensor)
                    x.device = types.SimpleNamespace(type="musa")
                    x.float.return_value = torch.ones(2, 16)
                for name, threshold in zip(sampling_names, (4, 0.8)):
                    getattr(module, name)(x, threshold)
                    internal = sail or backend in ("musa", "rocm")
                    assert getattr(ops, name).default.call_count == int(internal), name
                    assert getattr(flashinfer.sampling, name).call_count == int(
                        not internal
                    ), name


class TestSailDispatch(unittest.TestCase):
    def check_scenario(self, *args):
        result = subprocess.run(
            [sys.executable, __file__, "--scenario", *args],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_detection(self):
        for backend in ("sail", "cuda", "cpu"):
            with self.subTest(backend=backend):
                self.check_scenario("detect", backend)

    def test_dual_gemm(self):
        for backend in ("sail", "cuda", "cpu"):
            with self.subTest(backend=backend):
                self.check_scenario("dual", backend)

    def test_fp8_utils_only_imports_flashinfer_on_nvidia_cuda(self):
        source = ROOT / "python/sglang/srt/layers/quantization/fp8_utils.py"
        tree = ast.parse(source.read_text())
        dispatch = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom) and child.module == "flashinfer"
                for child in ast.walk(node)
            )
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module == "sglang.kernels.ops.gemm"
                and any(alias.name == "fp8_scaled_mm" for alias in child.names)
                for child in ast.walk(node)
            )
        )
        executable = compile(
            ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[])),
            str(source),
            "exec",
        )
        gemm = types.ModuleType("sglang.kernels.ops.gemm")
        gemm.fp8_scaled_mm = Mock()
        blockwise = types.ModuleType("sglang.kernels.ops.gemm.fp8_blockwise_gemm")
        blockwise.fp8_blockwise_scaled_mm = Mock()
        patch_torch = types.ModuleType("sglang.srt.utils.patch_torch")
        patch_torch.register_fake_if_exists = lambda _name: lambda function: function
        modules = {
            gemm.__name__: gemm,
            blockwise.__name__: blockwise,
            patch_torch.__name__: patch_torch,
        }
        for sail in (True, False):
            with self.subTest(sail=sail), patch.dict(sys.modules, modules):
                namespace = {"_is_cuda": True, "_is_sail": sail}
                if sail:
                    exec(executable, namespace)
                    self.assertIs(namespace["fp8_scaled_mm"], gemm.fp8_scaled_mm)
                    self.assertIs(
                        namespace["fp8_blockwise_scaled_mm"],
                        blockwise.fp8_blockwise_scaled_mm,
                    )
                else:
                    with self.assertRaisesRegex(ModuleNotFoundError, "flashinfer"):
                        exec(executable, namespace)

    def test_sampler_only_imports_flashinfer_on_nvidia_cuda(self):
        source = ROOT / "python/sglang/srt/layers/sampler.py"
        tree = ast.parse(source.read_text())
        dispatch = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module == "flashinfer.sampling"
                for child in ast.walk(node)
            )
            and any(
                isinstance(child, ast.ImportFrom) and child.module == "sgl_kernel"
                for child in ast.walk(node)
            )
        )
        executable = compile(
            ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[])),
            str(source),
            "exec",
        )
        kernel = types.ModuleType("sgl_kernel")
        kernel.top_k_renorm_prob = Mock()
        kernel.top_p_renorm_prob = Mock()
        modules = {
            kernel.__name__: kernel,
            "flashinfer": None,
            "flashinfer.sampling": None,
        }
        for sail in (True, False):
            with self.subTest(sail=sail), patch.dict(sys.modules, modules):
                namespace = {
                    "is_cuda": lambda: True,
                    "is_sail": lambda: sail,
                }
                if sail:
                    exec(executable, namespace)
                    self.assertIs(
                        namespace["top_k_renorm_prob"], kernel.top_k_renorm_prob
                    )
                    self.assertIs(
                        namespace["top_p_renorm_prob"], kernel.top_p_renorm_prob
                    )
                else:
                    with self.assertRaisesRegex(ModuleNotFoundError, "flashinfer"):
                        exec(executable, namespace)

    def test_dflash_only_imports_flashinfer_on_nvidia_cuda(self):
        source = ROOT / "python/sglang/srt/speculative/dflash_utils.py"
        tree = ast.parse(source.read_text())
        dispatch = next(
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module == "flashinfer.sampling"
                for child in ast.walk(node)
            )
            and any(
                isinstance(child, ast.ImportFrom)
                and child.module == "sglang.kernels.ops.speculative.sampling"
                for child in ast.walk(node)
            )
        )
        executable = compile(
            ast.fix_missing_locations(ast.Module(body=[dispatch], type_ignores=[])),
            str(source),
            "exec",
        )
        flashinfer = types.ModuleType("flashinfer.sampling")
        flashinfer.top_k_renorm_probs = Mock()
        flashinfer.top_p_renorm_probs = Mock()
        speculative = types.ModuleType("sglang.kernels.ops.speculative.sampling")
        speculative.tree_speculative_sampling_target_only = Mock()
        modules = {
            flashinfer.__name__: flashinfer,
            speculative.__name__: speculative,
        }
        for sail in (True, False):
            with self.subTest(sail=sail), patch.dict(sys.modules, modules):
                namespace = {
                    "_DFLASH_SAMPLING_VERIFY_AVAILABLE": False,
                    "is_cuda": lambda: True,
                    "is_sail": lambda: sail,
                    "is_musa": lambda: False,
                    "is_hip": lambda: False,
                }
                exec(executable, namespace)
                if sail:
                    self.assertIsNone(namespace["top_k_renorm_prob"])
                    self.assertIsNone(namespace["top_p_renorm_prob"])
                    self.assertIsNone(
                        namespace["tree_speculative_sampling_target_only"]
                    )
                    self.assertFalse(namespace["_DFLASH_SAMPLING_VERIFY_AVAILABLE"])
                else:
                    self.assertIs(
                        namespace["top_k_renorm_prob"],
                        flashinfer.top_k_renorm_probs,
                    )
                    self.assertIs(
                        namespace["top_p_renorm_prob"],
                        flashinfer.top_p_renorm_probs,
                    )
                    self.assertIs(
                        namespace["tree_speculative_sampling_target_only"],
                        speculative.tree_speculative_sampling_target_only,
                    )
                    self.assertTrue(namespace["_DFLASH_SAMPLING_VERIFY_AVAILABLE"])

    def test_norm_dispatch(self):
        for backend, dtype, compiling in (
            ("sail", "float16", "0"),
            ("cuda", "float16", "0"),
            ("cuda", "float32", "0"),
            ("cuda", "bfloat16", "1"),
            ("rocm", "float16", "0"),
        ):
            with self.subTest(backend=backend, dtype=dtype, compiling=compiling):
                self.check_scenario("norm", backend, dtype, compiling)

    def test_sampling_dispatch(self):
        for backend in ("sail", "cuda", "rocm", "musa"):
            with self.subTest(backend=backend):
                self.check_scenario("sampling", backend)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scenario":
        scenario(
            sys.argv[2],
            sys.argv[3],
            sys.argv[4] if len(sys.argv) > 4 else "float16",
            len(sys.argv) > 5 and sys.argv[5] == "1",
        )
    else:
        unittest.main()
