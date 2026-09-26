#!/usr/bin/env python3
"""Static oracle for the add-only A61 B12X KDA-prefill candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import pathlib
import tempfile
import types


KDA_POST_SHA256 = "6601dffb13ed76caebf4b4156cb1f0cfdaea1369e4924c6a4d433149b3e9d9d4"
ARG_UTILS_POST_SHA256 = "01e07be29378ac38f0b6e8c52ee8dff061db6683ca91bbf525d90325ab12b56b"
B12X_UTILS_POST_SHA256 = "b8a94398c7fba6aa498e56238b1e06632748029f9ce73eb71a2f3cd71d966565"
KERNELS_SHA256 = "1580ca67469b145901ce61c24552caf1714f972c04484d15ae4208361338c54b"
DEFAULT_SITE = pathlib.Path("/usr/local/lib/python3.12/dist-packages")
DEFAULT_OVERLAY = pathlib.Path("/opt/b12x-kda/overlay")
DEFAULT_SUMS = pathlib.Path("/opt/b12x-kda/overlay.SHA256SUMS")


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"missing method {class_name}.{method_name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _verify_overlay(overlay: pathlib.Path, sums_path: pathlib.Path) -> None:
    expected: dict[pathlib.PurePosixPath, str] = {}
    for line in sums_path.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = pathlib.PurePosixPath(relative)
        assert path.parts[:2] in (("b12x", "policy"), ("b12x", "sequence")), path
        assert path not in expected, path
        expected[path] = digest
    actual = {
        pathlib.PurePosixPath(path.relative_to(overlay).as_posix())
        for path in overlay.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert len(expected) == 22, len(expected)
    assert actual == set(expected), (sorted(actual - set(expected)), sorted(set(expected) - actual))
    for relative, digest in expected.items():
        assert _sha(overlay / pathlib.Path(relative)) == digest, relative

    kernels = overlay / "b12x/sequence/kda_prefill/_cute_kernels.py"
    assert _sha(kernels) == KERNELS_SHA256
    source = kernels.read_text()
    required_order = (
        source.index("enqueue_prepare(0)"),
        source.index("main.wait_event(prepared[window])"),
        source.index("run_recurrence(binding, window=window)"),
        source.index("consumed[window].record(main)"),
        source.index("enqueue_prepare(window + 1)"),
    )
    assert required_order == tuple(sorted(required_order)), required_order
    assert "torch.cuda.Stream(device=device, priority=-1)" in source
    assert "side.wait_event(consumed[window - 2])" in source


def _verify_backend_resolution(tree: ast.Module) -> None:
    class _Api:
        supported = True

        @classmethod
        def is_supported(cls, _device: object) -> bool:
            return cls.supported

    class _Platform:
        cuda = True

        @classmethod
        def is_cuda(cls) -> bool:
            return cls.cuda

        @staticmethod
        def current_device() -> str:
            return "cuda:0"

        @staticmethod
        def get_device_capability() -> types.SimpleNamespace:
            return types.SimpleNamespace(major=12)

    bf16, fp32 = object(), object()
    fake_torch = types.SimpleNamespace(
        bfloat16=bf16,
        float32=fp32,
        dtype=object,
        device=lambda value: value,
    )
    functions = [
        _function(tree, "_is_flashkda_supported"),
        _function(tree, "_is_b12x_kda_prefill_supported"),
        _function(tree, "_resolve_kda_prefill_backend"),
    ]
    namespace = {
        "torch": fake_torch,
        "current_platform": _Platform,
        "get_b12x_kda_prefill": lambda: _Api,
    }
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, "<b12x-kda-backend-test>", "exec"), namespace)
    resolve = namespace["_resolve_kda_prefill_backend"]
    assert resolve("auto", 128, bf16, -5.0, fp32) == "triton"
    assert resolve("triton", 128, bf16, -5.0, fp32) == "triton"
    assert resolve("flashkda", 128, bf16, -5.0, fp32) == "flashkda"
    assert resolve("b12x", 128, bf16, -5.0, fp32) == "b12x"
    _Api.supported = False
    try:
        resolve("b12x", 128, bf16, -5.0, fp32)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsupported explicit B12X selection did not fail closed")
    # Auto must never silently select the experimental backend.
    assert resolve("auto", 128, bf16, -5.0, fp32) == "triton"


def _verify_vllm(site: pathlib.Path) -> None:
    kda = site / "vllm/models/glm5next/nvidia/kda.py"
    arg_utils = site / "vllm/engine/arg_utils.py"
    b12x_utils = site / "vllm/utils/b12x.py"
    assert _sha(kda) == KDA_POST_SHA256
    assert _sha(arg_utils) == ARG_UTILS_POST_SHA256
    assert _sha(b12x_utils) == B12X_UTILS_POST_SHA256

    source = kda.read_text()
    tree = ast.parse(source, filename=str(kda))
    _verify_backend_resolution(tree)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextLinearAttention"
    )
    methods = {
        node.name: node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "get_kv_cache_spec" not in methods, "candidate must not change cache geometry"
    bind_source = ast.unparse(_method(tree, "Glm5NextLinearAttention", "bind_kv_cache"))
    assert "super().bind_kv_cache(kv_cache)" in bind_source
    assert "self.kv_cache[1].shape[0]" in bind_source
    plan_source = ast.unparse(
        _method(tree, "Glm5NextLinearAttention", "_make_b12x_kda_prefill_plan")
    )
    assert "checkpoint_export=False" in plan_source
    assert "metadata_validation='trusted'" in plan_source
    assert "null_state_index=NULL_BLOCK_ID" in plan_source

    run_source = ast.unparse(
        _method(tree, "Glm5NextLinearAttention", "_run_b12x_kda_prefill")
    )
    for marker in (
        "recurrent_state=recurrent_state",
        "final_state_indices=state_indices",
        "checkpoint_state_indices=null_indices",
        "output=output",
        "api.run",
    ):
        assert marker in run_source, marker
    forward_source = ast.unparse(_method(tree, "Glm5NextLinearAttention", "_forward"))
    assert "current_workspace_manager().get_simultaneous" in forward_source
    assert "for spec in plan.scratch_specs()" in forward_source
    assert "*scratch_buffers, b12x_out = workspaces" in forward_source
    assert "if not use_b12x_prefill:\n            scatter_states" in forward_source
    assert "gather_initial_states" in forward_source
    assert "_flashkda_prefill" in forward_source

    args_source = arg_utils.read_text()
    assert 'Literal["auto", "triton", "flashkda", "b12x"]' in args_source
    assert 'choices=["auto", "triton", "flashkda", "b12x"]' in args_source
    utils_source = b12x_utils.read_text()
    assert '"b12x.sequence.kda_prefill"' in utils_source
    assert "def get_b12x_kda_prefill()" in utils_source
    helper = ast.unparse(ast.parse(utils_source))
    assert "current_workspace_manager().get_simultaneous" in helper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=pathlib.Path, default=DEFAULT_SITE)
    parser.add_argument("--overlay", type=pathlib.Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--sums", type=pathlib.Path, default=DEFAULT_SUMS)
    args = parser.parse_args()
    _verify_overlay(args.overlay, args.sums)
    _verify_vllm(args.site)
    print(
        "B12X_KDA_PREFILL_STATIC_OK "
        f"kda={KDA_POST_SHA256} overlay_kernels={KERNELS_SHA256}"
    )


if __name__ == "__main__":
    main()
