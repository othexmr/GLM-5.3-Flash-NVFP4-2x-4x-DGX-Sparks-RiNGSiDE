#!/usr/bin/env python3
"""Static and CPU-only unit oracle for the pinned GLM FlashKDA patch."""

from __future__ import annotations

import ast
import hashlib
import pathlib
import sys
import types
from typing import Any


PREIMAGE_SHA256 = "99cfc24678c0a99f2c813c20e3dcc30c8bfac4f55ec4299737689d5d51f472cd"
POSTIMAGE_SHA256 = "8ea6607c1eafb515e6effc96e101d227f3655b74f191863f484adbc22883c8c2"
DEFAULT_TARGET = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py"
)


def _definition(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"missing method {class_name}.{method_name}")


def _call_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call) and _call_name(item) == name
    ]


class _FakeTensor:
    def __init__(self, name: str, shape: tuple[int, ...] = ()) -> None:
        self.name = name
        self.shape = shape

    def contiguous(self) -> _FakeTensor:
        return _FakeTensor(f"{self.name}.contiguous", self.shape)

    def view(self, *shape: int) -> _FakeTensor:
        shape_text = ",".join(str(dim) for dim in shape)
        return _FakeTensor(f"{self.name}.view({shape_text})", tuple(shape))


class _FakeFlashKDA:
    def __init__(self) -> None:
        self.args: tuple[Any, ...] | None = None

    def fwd(self, *args: Any) -> None:
        self.args = args


def _exec_functions(
    definitions: list[ast.FunctionDef], namespace: dict[str, Any]
) -> dict[str, Any]:
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, "<glm-flashkda-test>", "exec"), namespace)
    return namespace


def _test_backend_resolution(tree: ast.Module) -> None:
    class Capability:
        major = 12

    class Platform:
        cuda = True

        @classmethod
        def is_cuda(cls) -> bool:
            return cls.cuda

        @staticmethod
        def get_device_capability() -> Capability:
            return Capability()

    fake_torch = types.SimpleNamespace(dtype=object, bfloat16=object())
    namespace = _exec_functions(
        [
            _definition(tree, "_is_flashkda_supported"),
            _definition(tree, "_resolve_kda_prefill_backend"),
        ],
        {"torch": fake_torch, "current_platform": Platform},
    )
    resolve = namespace["_resolve_kda_prefill_backend"]
    dtype = fake_torch.bfloat16
    assert resolve("auto", 128, dtype, -1.0) == "triton"
    assert resolve("triton", 128, dtype, -1.0) == "triton"
    assert resolve("flashkda", 128, dtype, -1.0) == "flashkda"
    try:
        resolve("invalid", 128, dtype, -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid backend did not fail closed")
    Platform.cuda = False
    try:
        resolve("flashkda", 128, dtype, -1.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsupported FlashKDA platform did not fail closed")


def _test_flashkda_abi(tree: ast.Module) -> None:
    fake_flashkda = _FakeFlashKDA()
    fake_torch = types.SimpleNamespace(
        Tensor=_FakeTensor,
        ops=types.SimpleNamespace(_flashkda_C=fake_flashkda),
    )
    saved_vllm = sys.modules.get("vllm")
    saved_extension = sys.modules.get("vllm._flashkda_C")
    package = types.ModuleType("vllm")
    package.__path__ = []
    sys.modules["vllm"] = package
    sys.modules["vllm._flashkda_C"] = types.ModuleType("vllm._flashkda_C")
    try:
        namespace = _exec_functions(
            [_definition(tree, "_flashkda_prefill")], {"torch": fake_torch}
        )
        tensors = {
            name: _FakeTensor(name, (1, 17, 4, 128))
            for name in ("q", "k", "v", "g")
        }
        tensors.update(
            {
                "beta": _FakeTensor("beta", (1, 17, 4)),
                "A_log": _FakeTensor("A_log", (1, 1, 4, 1)),
                "dt_bias": _FakeTensor("dt_bias", (512,)),
                "initial_state": _FakeTensor("initial_state", (2, 4, 128, 128)),
                "cu_seqlens": _FakeTensor("cu_seqlens", (3,)),
                "out": _FakeTensor("out", (1, 17, 4, 128)),
                "final_state": _FakeTensor("final_state", (2, 4, 128, 128)),
                "workspace": _FakeTensor("workspace", (4096,)),
            }
        )
        result = namespace["_flashkda_prefill"](
            **tensors, lower_bound=-1.0
        )
    finally:
        if saved_vllm is None:
            sys.modules.pop("vllm", None)
        else:
            sys.modules["vllm"] = saved_vllm
        if saved_extension is None:
            sys.modules.pop("vllm._flashkda_C", None)
        else:
            sys.modules["vllm._flashkda_C"] = saved_extension

    assert result == (tensors["out"], tensors["final_state"])
    args = fake_flashkda.args
    assert args is not None and len(args) == 16
    assert [args[index].name for index in range(4)] == [
        "q.contiguous",
        "k.contiguous",
        "v.contiguous",
        "g.contiguous",
    ]
    assert args[4] is tensors["beta"], "FlashKDA must receive raw beta"
    assert args[8].name == "A_log.view(-1).contiguous"
    assert args[9].name == "dt_bias.view(-1,128).contiguous"
    assert args[12] is tensors["final_state"]
    assert args[14:] == (None, None)


def _test_prefill_wiring(tree: ast.Module) -> None:
    init = _method(tree, "Glm5NextLinearAttention", "__init__")
    public_forward = _method(tree, "Glm5NextLinearAttention", "forward")
    forward = _method(tree, "Glm5NextLinearAttention", "_forward")

    assert len(_calls(init, "get_workspace_size")) == 1
    assert len(_calls(forward, "get_simultaneous")) == 1
    core_out_assignments = [
        item
        for item in ast.walk(public_forward)
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "core_attn_out"
            for target in item.targets
        )
        and isinstance(item.value, ast.Call)
        and _call_name(item.value) == "empty"
    ]
    assert len(core_out_assignments) == 1
    core_out_call = core_out_assignments[0].value
    assert isinstance(core_out_call, ast.Call)
    assert ast.unparse(core_out_call.args[0]) == (
        "(1, num_tokens, self.local_num_heads, self.head_dim)"
    )
    flash_out_assignments = [
        item
        for item in ast.walk(forward)
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "flashkda_out"
            for target in item.targets
        )
    ]
    assert len(flash_out_assignments) == 1
    # Kimi slices q.shape[1] after q is [1,T,H,D]. GLM keeps q_ns as
    # [T,H*D] until the call, so q_ns.shape[0] is the same token extent.
    assert ast.unparse(flash_out_assignments[0].value).endswith(
        "[:, :q_ns.shape[0]]"
    )
    flash_calls = _calls(forward, "_flashkda_prefill")
    assert len(flash_calls) == 1
    flash_keywords = {keyword.arg: keyword.value for keyword in flash_calls[0].keywords}
    assert isinstance(flash_keywords["g"], ast.Name)
    assert flash_keywords["g"].id == "g1_ns"
    assert isinstance(flash_keywords["beta"], ast.Name)
    assert flash_keywords["beta"].id == "beta_ns"
    assert isinstance(flash_keywords["out"], ast.Name)
    assert flash_keywords["out"].id == "flashkda_out"
    assert isinstance(flash_keywords["workspace"], ast.Name)
    assert flash_keywords["workspace"].id == "workspace"

    triton_calls = _calls(forward, "chunk_kda_with_fused_gate")
    assert len(triton_calls) == 1
    triton_keywords = {keyword.arg: keyword.value for keyword in triton_calls[0].keywords}
    assert isinstance(triton_keywords["raw_g"], ast.Name)
    assert triton_keywords["raw_g"].id == "g1_ns"
    assert len(_calls(triton_keywords["beta"], "_cast_sigmoid")) == 1
    # The patch must not replace either spec-verify or plain-decode recurrence.
    assert len(_calls(forward, "fused_recurrent_kda")) == 2
    assert len(_calls(forward, "scatter_states")) == 1


def main() -> None:
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    source_bytes = target.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    assert digest == POSTIMAGE_SHA256, (
        f"patched GLM KDA drift: {digest}, expected {POSTIMAGE_SHA256}"
    )
    tree = ast.parse(source_bytes, filename=str(target))
    _test_backend_resolution(tree)
    _test_flashkda_abi(tree)
    _test_prefill_wiring(tree)
    print(
        "GLM53_FLASHKDA_PREFILL_OK",
        f"preimage_sha256={PREIMAGE_SHA256}",
        f"postimage_sha256={POSTIMAGE_SHA256}",
    )


if __name__ == "__main__":
    main()
