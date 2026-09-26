#!/usr/bin/env python3
"""SM120/SM121 numeric and metadata gate for the A61 B12X KDA candidate.

This is intentionally an image-build gate, not a serving benchmark.  It checks
the exact h32 GLM geometry against A59's existing Triton implementation and
uses transactional plans to prove malformed slot metadata fails without state
mutation.  Production keeps the already-qualified cache layout and opts into
trusted metadata only after this gate passes.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable

import torch

from b12x.sequence import kda_prefill
from vllm.models.glm5next.nvidia.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
)


HEADS = 32
HEAD_DIM = 128
LOWER_BOUND = -5.0
MODEL_DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32


@dataclasses.dataclass
class Case:
    lengths: tuple[int, ...]
    seed: int
    strided_final: bool = False


def _bf16(
    generator: torch.Generator,
    device: torch.device,
    *shape: int,
    scale: float = 0.25,
) -> torch.Tensor:
    return (torch.randn(*shape, generator=generator) * scale).to(
        device=device, dtype=MODEL_DTYPE
    )


def _rmse_ratio(reference: torch.Tensor, actual: torch.Tensor) -> float:
    delta = (reference.float() - actual.float()).flatten()
    base = reference.float().flatten()
    return float(
        delta.square().mean().sqrt()
        / (base.square().mean().sqrt() + 1e-8)
    )


def _assert_close(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    limit: float,
) -> float:
    assert torch.isfinite(actual.float()).all(), f"{name}: non-finite output"
    ratio = _rmse_ratio(reference, actual)
    assert ratio <= limit, f"{name}: relative RMSE {ratio:.6g} > {limit}"
    return ratio


def _byte_range(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    return start, start + tensor.numel() * tensor.element_size()


def _assert_pairwise_disjoint(tensors: Iterable[torch.Tensor]) -> None:
    values = list(tensors)
    for left, lhs in enumerate(values):
        lhs_range = _byte_range(lhs)
        for rhs in values[left + 1 :]:
            rhs_range = _byte_range(rhs)
            assert lhs_range[1] <= rhs_range[0] or rhs_range[1] <= lhs_range[0], (
                lhs_range,
                rhs_range,
            )


def _inputs(case: Case, device: torch.device) -> dict[str, torch.Tensor | int]:
    lengths = case.lengths
    tokens = sum(lengths)
    seqs = len(lengths)
    capacity = max(1, tokens)
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    q = _bf16(generator, device, capacity, HEADS, HEAD_DIM)
    k = _bf16(generator, device, capacity, HEADS, HEAD_DIM)
    v = _bf16(generator, device, capacity, HEADS, HEAD_DIM)
    raw_g = _bf16(generator, device, capacity, HEADS, HEAD_DIM, scale=1.0)
    raw_beta = _bf16(generator, device, capacity, HEADS, scale=1.0)
    a_log = (torch.randn(HEADS, generator=generator) * 0.1).to(
        device=device, dtype=torch.float32
    )
    dt_bias = (torch.randn(HEADS, HEAD_DIM, generator=generator) * 0.1).to(
        device=device, dtype=torch.float32
    )
    slots = 2 * seqs + 3
    pool = (torch.randn(slots, HEADS, HEAD_DIM, HEAD_DIM, generator=generator) * 0.05).to(
        device=device, dtype=STATE_DTYPE
    )
    initial = torch.arange(seqs + 1, 2 * seqs + 1, dtype=torch.int32, device=device)
    if seqs:
        initial[0] = 0
    final_values = torch.arange(1, seqs + 1, dtype=torch.int32, device=device)
    if case.strided_final:
        final_storage = torch.full(
            (seqs, 3), -1, dtype=torch.int32, device=device
        )
        final_storage[:, 0] = final_values
        final = final_storage[:, 0]
        assert final.stride(0) == 3
    else:
        final = final_values
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    return {
        "tokens": tokens,
        "seqs": seqs,
        "q": q,
        "k": k,
        "v": v,
        "raw_g": raw_g,
        "raw_beta": raw_beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "pool": pool,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device=device),
        "initial": initial,
        "final": final,
    }


def _triton_reference(data: dict[str, torch.Tensor | int]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = int(data["tokens"])
    seqs = int(data["seqs"])
    if tokens == 0:
        return (
            torch.empty(
                (0, HEADS, HEAD_DIM),
                device=data["pool"].device,
                dtype=MODEL_DTYPE,
            ),
            torch.stack(
                [
                    torch.zeros_like(data["pool"][0])
                    if int(data["initial"][seq]) == 0
                    else data["pool"][int(data["initial"][seq])]
                    for seq in range(seqs)
                ]
            ),
        )
    initial = torch.stack(
        [
            torch.zeros_like(data["pool"][0])
            if int(data["initial"][seq]) == 0
            else data["pool"][int(data["initial"][seq])]
            for seq in range(seqs)
        ]
    )
    output, final = chunk_kda_with_fused_gate(
        q=data["q"][:tokens].unsqueeze(0),
        k=data["k"][:tokens].unsqueeze(0),
        # The pinned Triton implementation aliases its output to ``v``.  Keep
        # the oracle from mutating the source tensor that is subsequently
        # bound to the B12X candidate.
        v=data["v"][:tokens].unsqueeze(0).clone(),
        raw_g=data["raw_g"][:tokens].unsqueeze(0),
        beta=torch.sigmoid(data["raw_beta"][:tokens].float()).unsqueeze(0),
        A_log=data["A_log"].view(1, 1, HEADS, 1),
        g_bias=data["dt_bias"],
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=data["cu_seqlens"],
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )
    return output[0], final


def _torch_reference(data: dict[str, torch.Tensor | int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent packed recurrence for cases Triton's oracle cannot encode.

    The pinned Triton helper mislabels a live sequence that follows a zero-length
    sequence.  Keep that metadata case in the gate, but use B12X's pure-PyTorch
    token recurrence as its control.  This path is intentionally limited to a
    tiny case because it is a correctness oracle, not a performance kernel.
    """
    tokens = int(data["tokens"])
    seqs = int(data["seqs"])
    pool = data["pool"].clone()
    output = torch.zeros_like(data["q"])
    checkpoint_indices = torch.zeros(seqs, dtype=torch.int32, device=pool.device)
    checkpoint_offsets = torch.zeros(seqs, dtype=torch.int32, device=pool.device)
    kda_prefill.reference.prefill_kda(
        data["q"],
        data["k"],
        data["v"],
        data["raw_g"],
        data["raw_beta"],
        data["A_log"],
        data["dt_bias"],
        pool,
        data["cu_seqlens"],
        data["initial"],
        data["final"],
        checkpoint_indices,
        checkpoint_offsets,
        seqs,
        tokens,
        lower_bound=LOWER_BOUND,
        qk_l2norm=True,
        null_state_index=0,
        output=output,
    )
    final = torch.stack([pool[int(data["final"][seq])] for seq in range(seqs)])
    return output[:tokens], final


def _binding(
    data: dict[str, torch.Tensor | int],
    *,
    validation: str,
) -> tuple[object, list[torch.Tensor], torch.Tensor]:
    tokens = int(data["tokens"])
    seqs = int(data["seqs"])
    caps = kda_prefill.Caps(
        device=data["pool"].device,
        max_tokens=max(1, tokens),
        max_seqs=max(1, seqs),
        max_state_slots=int(data["pool"].shape[0]),
        heads=HEADS,
        head_dim=HEAD_DIM,
        model_dtype=MODEL_DTYPE,
        state_dtype=STATE_DTYPE,
        qk_l2norm=True,
        checkpoint_export=False,
        null_state_index=0,
        metadata_validation=validation,
    )
    plan = kda_prefill.plan(caps)
    scratch_buffers = [
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    ]
    output = torch.empty(
        (max(1, tokens), HEADS, HEAD_DIM),
        dtype=MODEL_DTYPE,
        device=data["pool"].device,
    )
    _assert_pairwise_disjoint([*scratch_buffers, output, data["pool"]])
    scratch: torch.Tensor | list[torch.Tensor]
    scratch = scratch_buffers[0] if len(scratch_buffers) == 1 else scratch_buffers
    binding = kda_prefill.bind(
        plan,
        scratch=scratch,
        q=data["q"],
        k=data["k"],
        v=data["v"],
        raw_g=data["raw_g"],
        raw_beta=data["raw_beta"],
        A_log=data["A_log"],
        dt_bias=data["dt_bias"],
        recurrent_state=data["pool"],
        cu_seqlens=data["cu_seqlens"],
        initial_state_indices=data["initial"],
        final_state_indices=data["final"],
        checkpoint_state_indices=torch.zeros(
            max(1, seqs), dtype=torch.int32, device=data["pool"].device
        )[:seqs],
        checkpoint_offsets=torch.zeros(
            max(1, seqs), dtype=torch.int32, device=data["pool"].device
        )[:seqs],
        num_seqs=torch.tensor([seqs], dtype=torch.int32, device=data["pool"].device),
        num_tokens=torch.tensor([tokens], dtype=torch.int32, device=data["pool"].device),
        output=output,
    )
    return binding, scratch_buffers, output


def _positive_case(
    case: Case,
    device: torch.device,
    *,
    reference: str = "triton",
) -> dict[str, float | int | str]:
    data = _inputs(case, device)
    original_pool = data["pool"].clone()
    read_only = {
        name: data[name].clone()
        for name in (
            "q",
            "k",
            "v",
            "raw_g",
            "raw_beta",
            "A_log",
            "dt_bias",
            "cu_seqlens",
            "initial",
            "final",
        )
    }
    if reference == "triton":
        expected_output, expected_final = _triton_reference(data)
    elif reference == "torch":
        expected_output, expected_final = _torch_reference(data)
    else:
        raise ValueError(f"unknown reference {reference!r}")
    torch.cuda.synchronize(device)
    for name, before in read_only.items():
        torch.testing.assert_close(data[name], before, rtol=0, atol=0)
    binding, _scratch, output = _binding(data, validation="transactional")
    kda_prefill.run(
        binding,
        lower_bound=LOWER_BOUND,
        max_live_tokens=int(data["tokens"]),
        max_live_seqs=int(data["seqs"]),
    )
    torch.cuda.synchronize(device)
    assert binding.error_code.item() == 0
    tokens = int(data["tokens"])
    out_ratio = 0.0
    if tokens:
        out_ratio = _assert_close(
            "output", expected_output, output[:tokens], 1.0e-2
        )
    state_ratios = []
    written: set[int] = set()
    for seq in range(int(data["seqs"])):
        slot = int(data["final"][seq])
        written.add(slot)
        state_ratios.append(
            _assert_close(
                f"state[{slot}]", expected_final[seq], data["pool"][slot], 5.0e-3
            )
        )
    untouched = [
        slot for slot in range(int(data["pool"].shape[0])) if slot not in written
    ]
    torch.testing.assert_close(
        data["pool"][untouched], original_pool[untouched], rtol=0, atol=0
    )
    for name, before in read_only.items():
        torch.testing.assert_close(data[name], before, rtol=0, atol=0)
    return {
        "lengths": ",".join(str(value) for value in case.lengths),
        "reference": reference,
        "windows": binding.plan.max_windows,
        "output_rmse": out_ratio,
        "state_rmse": max(state_ratios, default=0.0),
    }


def _invalid_case(
    name: str,
    mutate,
    device: torch.device,
) -> None:
    data = _inputs(Case((2048, 2048), seed=900), device)
    binding, _scratch, output = _binding(data, validation="transactional")
    assert binding.plan.max_windows >= 3, binding.plan.max_windows
    mutate(data)
    pool_before = data["pool"].clone()
    output.zero_()
    kda_prefill.run(
        binding,
        lower_bound=LOWER_BOUND,
        max_live_tokens=int(data["tokens"]),
        max_live_seqs=int(data["seqs"]),
    )
    torch.cuda.synchronize(device)
    assert binding.error_code.item() != 0, f"{name}: error bit not set"
    assert torch.isnan(output.float()).all(), f"{name}: output not poisoned"
    torch.testing.assert_close(data["pool"], pool_before, rtol=0, atol=0)


def _ordering_stress(device: torch.device) -> None:
    cases = (
        _inputs(Case((1, 1329, 2766), seed=1001), device),
        _inputs(Case((4096,), seed=1002), device),
    )
    bindings = [_binding(data, validation="transactional")[0] for data in cases]
    assert all(binding.plan.max_windows >= 3 for binding in bindings)
    for iteration in range(100):
        index = iteration & 1
        data = cases[index]
        binding = bindings[index]
        kda_prefill.run(
            binding,
            lower_bound=LOWER_BOUND,
            max_live_tokens=int(data["tokens"]),
            max_live_seqs=int(data["seqs"]),
        )
    torch.cuda.synchronize(device)
    assert all(binding.error_code.item() == 0 for binding in bindings)
    assert all(torch.isfinite(binding.output.float()).all() for binding in bindings)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    assert capability in ((12, 0), (12, 1)), capability
    assert kda_prefill.is_supported(device), "B12X KDA is unsupported on this GPU"

    results = [
        _positive_case(Case((0, 0), seed=100), device),
        # Tune the pinned Triton control at the production chunk before the
        # boundary ladder. Its autotune keys omit token count, so small-first
        # order can reuse an unsafe tiny-shape configuration at 14,336.
        _positive_case(Case((14336,), seed=107), device),
        _positive_case(
            Case((15, 17, 1329, 4096, 8879, 0), seed=108, strided_final=True),
            device,
        ),
        _positive_case(Case((0, 15), seed=109), device, reference="torch"),
        _positive_case(Case((1,), seed=101), device),
        _positive_case(Case((15,), seed=102), device),
        _positive_case(Case((16,), seed=103), device),
        _positive_case(Case((17,), seed=104), device),
        _positive_case(Case((1328,), seed=105), device),
        _positive_case(Case((1329,), seed=106), device),
    ]
    _invalid_case(
        "duplicate-final",
        lambda data: data["final"].__setitem__(1, int(data["final"][0])),
        device,
    )
    _invalid_case(
        "out-of-range-final",
        lambda data: data["final"].__setitem__(1, int(data["pool"].shape[0]) + 1),
        device,
    )
    _invalid_case(
        "null-final-multi-window",
        lambda data: data["final"].__setitem__(1, 0),
        device,
    )
    _ordering_stress(device)
    for result in results:
        print("B12X_KDA_NUMERIC", " ".join(f"{key}={value}" for key, value in result.items()))
    print(
        "B12X_KDA_PREFILL_GPU_GATE_OK "
        f"device={torch.cuda.get_device_name(device)!r} capability={capability} "
        "heads=32 head_dim=128 checkpoint_export=false repetitions=100"
    )


if __name__ == "__main__":
    main()
