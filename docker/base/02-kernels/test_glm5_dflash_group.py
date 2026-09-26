#!/usr/bin/env python3
"""Build-time oracle for the GLM-5.3 DFlash drafter KV-cache group in vllm/v1/core/kv_cache_utils.py (stage 02).

Runs in the image's vLLM right after the drafter-group patches. It builds the per-layer KV-cache specs of the two
served shapes (target block 2304 with 2 drafter KV heads, and 4608 with 4; each with 11 MLA, 11 indexer, 11 indexer
tail and 34 KDA layers, then 5 drafter layers, sliding window 2048) and checks the behaviour the drafter group must
show. P is the MLA page, Q the indexer page, M and I the MLA and indexer layer counts, D the drafter layer count, N
the block count, b the drafter bytes per token.

- Grouping: seven groups. The target groups equal the grouping without the drafter layers, and none is an EAGLE
  group. The drafter group comes last, holds the drafter layers in registration order and is the EAGLE group. Its
  per-layer specs are the incoming spec with two fields changed: block size P / b (1152 tokens) and no padding. Each
  page is exactly P. The drafter block and the target block divide one another.
- Layout: bytes per block stay M x P + I x Q. The one-request memory estimate adds the drafter group's window-bounded
  pages (16 at 14,336 in-flight tokens). In a KV-cache config of N blocks every tensor names one layer and has size
  N x (M x P + I x Q). Drafter layer i sits on the bytes of MLA layer i (offset i x P x N, block stride P, layer stride
  P x N), right after the KDA layers that share them; every other tensor is as without the drafter. The scheduler and
  hash block sizes are the target block and 1152.
- Preconditions: the grouping refuses unequal drafter specs, a block that is not a multiple of 64, blocks that do not
  divide one another, a page that is not a whole number of drafter tokens, and more drafter layers than MLA layers.
  A drafter window other than 2048 is grouped, but the list is no longer recognised as the GLM-5.3 layout.

Exits 0 and prints one line starting with GLM53_DFLASH_GROUP_OK when all of this holds; any failure raises.
"""

import dataclasses
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_bytes_per_block,
    _glm5_next_tensor_layout,
    _max_memory_usage_bytes_from_groups,
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

MAX_MODEL_LEN = 262144
IN_FLIGHT_TOKENS = 14336  # max_num_batched_tokens x max_concurrent_batches of both served plans
WINDOW = 2048
NUM_TARGET_LAYERS = 45
DRAFTER_LAYERS = [f"model.layers.{index}.self_attn.attn" for index in range(45, 50)]

# The served shapes and the values SPEC.md works out for them (0.2, 3.5, 4.3).
SHAPES = {
    "tp4": {
        "target_kv_block": 2304,
        "kda_shapes": ((10, 6144), (16, 128, 144)),
        "kda_dtypes": (torch.bfloat16, torch.float16),
        "drafter_heads": 2,
        "mla_page": 1179648,
        "indexer_page": 76032,
        "bytes_per_block": 13812480,
        "request_bytes_without_drafter": 1698935040,
        "request_bytes": 1919934720,
    },
    "tp2": {
        "target_kv_block": 4608,
        "kda_shapes": ((10, 12288), (32, 128, 128)),
        "kda_dtypes": (torch.bfloat16, torch.float32),
        "drafter_heads": 4,
        "mla_page": 2359296,
        "indexer_page": 152064,
        "bytes_per_block": 27624960,
        "request_bytes_without_drafter": 1823247360,
        "request_bytes": 2265246720,
    },
}
DRAFTER_BLOCK = 1152
DRAFTER_PAGES_PER_REQUEST = 16  # ceil((2048 - 1 + 14336) / 1152) + 1
DRAFTER_BLOCK_TABLE_ENTRIES = 228  # ceil(262144 / 1152)
NUM_BLOCKS = 3


def engine_config(target_kv_block: int, speculative: bool) -> SimpleNamespace:
    """The configuration fields the grouping and the three layout consumers read."""
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=SimpleNamespace(
            block_size=target_kv_block,
            mamba_cache_mode="align",
            enable_prefix_caching=True,
            prefix_match_unit=None,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=target_kv_block,
        ),
        speculative_config=(
            SimpleNamespace(method="dflash", use_eagle=lambda: True) if speculative else None
        ),
        kv_transfer_config=None,
        max_in_flight_tokens=IN_FLIGHT_TOKENS,
    )


def drafter_spec(heads: int, **changes) -> SlidingWindowSpec:
    fields = dict(
        block_size=16,
        num_kv_heads=heads,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
    )
    fields.update(changes)
    return SlidingWindowSpec(**fields)


def served_layers(shape: dict, drafter: SlidingWindowSpec | None) -> dict:
    """Per-layer specs of one rank, in construction order: for each decoder layer one KDA layer, or the indexer key
    cache, the indexer tail and the MLA attention; the drafter layers last."""
    block = shape["target_kv_block"]
    mla = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_e4m3",
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    indexer = MLAAttentionSpec(
        block_size=block,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
        storage_block_size=256,
    )
    tail = KpoolTailSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=4,
    )
    kda = MambaSpec(
        block_size=block,
        shapes=shape["kda_shapes"],
        dtypes=shape["kda_dtypes"],
        page_size_padded=shape["mla_page"],
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    layers = {}
    for index in range(NUM_TARGET_LAYERS):
        prefix = f"language_model.model.layers.{index}.self_attn"
        if index % 4 == 3:
            layers[f"{prefix}.indexer.k_cache"] = indexer
            layers[f"{prefix}.indexer.tail_cache"] = tail
            layers[f"{prefix}.attn"] = mla
        else:
            layers[prefix] = kda
    if drafter is not None:
        for name in DRAFTER_LAYERS:
            layers[name] = drafter
    return layers


def check_shape(label: str, shape: dict) -> None:
    block = shape["target_kv_block"]
    config = engine_config(block, speculative=True)
    target_config = engine_config(block, speculative=False)
    incoming = drafter_spec(shape["drafter_heads"])
    layers = served_layers(shape, incoming)
    target_layers = served_layers(shape, None)
    mla_names = [name for name, spec in layers.items()
                 if type(spec) is MLAAttentionSpec and spec.tokens_per_state == 1]
    indexer_names = [name for name, spec in layers.items()
                     if type(spec) is MLAAttentionSpec and spec.tokens_per_state > 1]
    mla_page = layers[mla_names[0]].page_size_bytes
    indexer_page = layers[indexer_names[0]].page_size_bytes
    assert (len(mla_names), len(indexer_names)) == (11, 11), label
    assert (mla_page, indexer_page) == (shape["mla_page"], shape["indexer_page"]), label

    # SPEC 4.1: bytes per token from the unpadded incoming page; the block that fills one MLA page exactly.
    assert incoming.unpadded_page_size_bytes % incoming.block_size == 0, label
    b = incoming.unpadded_page_size_bytes // incoming.block_size
    assert mla_page % b == 0 and mla_page // b == DRAFTER_BLOCK, (label, b)

    # SPEC 3.3 and 3.4: grouping.
    groups = get_kv_cache_groups(config, dict(layers))
    target_groups = get_kv_cache_groups(target_config, dict(target_layers))
    assert len(target_groups) == 6 and len(groups) == 7, (label, len(groups))
    assert groups[:-1] == target_groups, f"{label}: the target groups changed"
    assert not any(group.is_eagle_group for group in groups[:-1]), label
    drafter_group = groups[-1]
    assert drafter_group.layer_names == DRAFTER_LAYERS, label
    assert drafter_group.is_eagle_group is True, label
    assert drafter_group.enable_kv_transfer is True, label
    uniform = drafter_group.kv_cache_spec
    assert type(uniform) is UniformTypeKVCacheSpecs, label
    assert uniform.block_size == DRAFTER_BLOCK, (label, uniform.block_size)
    assert list(uniform.kv_cache_specs) == DRAFTER_LAYERS, label
    for name in DRAFTER_LAYERS:
        spec = uniform.kv_cache_specs[name]
        assert type(spec) is SlidingWindowSpec, (label, name)
        assert spec == dataclasses.replace(
            layers[name], block_size=DRAFTER_BLOCK, page_size_padded=None
        ), (label, name, spec)
        assert spec.page_size_padded is None and spec.page_size_bytes == mla_page, (label, name)
    assert uniform.page_size_bytes == len(DRAFTER_LAYERS) * mla_page, label
    assert uniform.max_num_blocks_per_req(config, MAX_MODEL_LEN) == DRAFTER_BLOCK_TABLE_ENTRIES, label
    mla_group_block = groups[0].kv_cache_spec.block_size
    assert mla_group_block == block, label
    assert max(mla_group_block, DRAFTER_BLOCK) % min(mla_group_block, DRAFTER_BLOCK) == 0, label

    # SPEC 3.5 and 3.6: the GLM-5.3 layout with the drafter group in the MLA slots.
    assert _glm5_next_tensor_layout(groups) is not None, f"{label}: layout not recognised"
    per_block = len(mla_names) * mla_page + len(indexer_names) * indexer_page
    assert per_block == shape["bytes_per_block"], label
    assert _get_kv_cache_bytes_per_block(groups) == per_block, label
    assert _get_kv_cache_bytes_per_block(target_groups) == per_block, label
    drafter_layer_spec = uniform.kv_cache_specs[DRAFTER_LAYERS[0]]
    drafter_pages = drafter_layer_spec.max_admission_blocks_per_request(IN_FLIGHT_TOKENS, MAX_MODEL_LEN)
    assert drafter_pages == DRAFTER_PAGES_PER_REQUEST, (label, drafter_pages)
    without = _max_memory_usage_bytes_from_groups(target_config, target_groups)
    with_drafter = _max_memory_usage_bytes_from_groups(config, groups)
    assert without == shape["request_bytes_without_drafter"], (label, without)
    assert with_drafter == without + drafter_pages * per_block == shape["request_bytes"], (label, with_drafter)

    available = per_block * NUM_BLOCKS + per_block - 1
    kv_config = get_kv_cache_config_from_groups(config, groups, available_memory=available)
    reference = get_kv_cache_config_from_groups(target_config, target_groups, available_memory=available)
    n = kv_config.num_blocks
    assert n == reference.num_blocks == NUM_BLOCKS, (label, n)
    tensors = kv_config.kv_cache_tensors
    assert all(len(t.layers) == 1 and t.size == n * per_block for t in tensors), label
    by_layer = {t.layers[0]: t for t in tensors}
    assert len(by_layer) == len(tensors) and set(by_layer) == set(layers), label
    for index, name in enumerate(DRAFTER_LAYERS):
        tensor, mla_tensor = by_layer[name], by_layer[mla_names[index]]
        assert tensor.offset == index * mla_page * n == mla_tensor.offset, (label, name)
        assert tensor.block_stride == mla_page and tensor.layer_stride == mla_page * n, (label, name)
    assert [t for t in tensors if t.layers[0] not in DRAFTER_LAYERS] == reference.kv_cache_tensors, label
    reference_names = [t.layers[0] for t in reference.kv_cache_tensors]
    slot_starts = [reference_names.index(name) for name in mla_names]
    indexer_start = min(reference_names.index(name) for name in indexer_names)
    assert slot_starts[0] == 0 and slot_starts == sorted(slot_starts) and slot_starts[-1] < indexer_start, label
    expected = []
    for index, start in enumerate(slot_starts):
        end = slot_starts[index + 1] if index + 1 < len(slot_starts) else indexer_start
        expected += reference_names[start:end]
        if index < len(DRAFTER_LAYERS):
            expected.append(DRAFTER_LAYERS[index])
    expected += reference_names[indexer_start:]
    assert [t.layers[0] for t in tensors] == expected, f"{label}: tensor order"
    assert resolve_kv_cache_block_sizes(kv_config, config) == (block, DRAFTER_BLOCK), label


def refused(config: SimpleNamespace, layers: dict) -> bool:
    try:
        get_kv_cache_groups(config, layers)
    except (AssertionError, ValueError):
        return True
    return False


def check_preconditions() -> int:
    """SPEC 4.2 and 3.6 on the TP4 shape."""
    shape = SHAPES["tp4"]
    config = engine_config(shape["target_kv_block"], speculative=True)
    heads = shape["drafter_heads"]

    unequal = served_layers(shape, drafter_spec(heads))
    unequal[DRAFTER_LAYERS[-1]] = drafter_spec(heads, sliding_window=1024)
    too_many = served_layers(shape, drafter_spec(heads))
    for index in range(50, 57):
        too_many[f"model.layers.{index}.self_attn.attn"] = drafter_spec(heads)
    cases = {
        "unequal drafter specs": unequal,
        "block 96, not a multiple of 64 (24 KV heads)": served_layers(shape, drafter_spec(24)),
        "blocks 1536 and 2304 do not divide one another (head size 96)":
            served_layers(shape, drafter_spec(heads, head_size=96, head_size_v=96)),
        "MLA page not a whole number of drafter tokens (10 KV heads)": served_layers(shape, drafter_spec(10)),
        "12 drafter layers for 11 MLA layers": too_many,
    }
    for label, layers in cases.items():
        assert refused(config, layers), f"accepted: {label}"

    # A window other than the served 2048: grouped as before, but the list leaves the GLM-5.3 layout.
    groups = get_kv_cache_groups(config, served_layers(shape, drafter_spec(heads, sliding_window=1024)))
    assert len(groups) == 7 and groups[-1].is_eagle_group, "window 1024: grouping"
    assert groups[-1].kv_cache_spec.block_size == DRAFTER_BLOCK, "window 1024: drafter block"
    assert _glm5_next_tensor_layout(groups) is None, "window 1024: still recognised as the GLM-5.3 layout"
    return len(cases)


def main() -> None:
    for label, shape in SHAPES.items():
        check_shape(label, shape)
    num_refused = check_preconditions()
    print(
        "GLM53_DFLASH_GROUP_OK",
        "shapes=" + ",".join(SHAPES),
        "groups=7",
        f"drafter_block={DRAFTER_BLOCK}",
        "bytes_per_block=" + ",".join(str(s["bytes_per_block"]) for s in SHAPES.values()),
        "request_bytes=" + ",".join(str(s["request_bytes"]) for s in SHAPES.values()),
        f"refused={num_refused}",
        "window_1024=generic_layout",
    )


if __name__ == "__main__":
    main()
