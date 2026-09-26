#!/usr/bin/env python3
"""Focused CPU oracle for A75's detached DFlash CacheConfig repair."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import CacheConfig, replace
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.worker import worker_base as worker_base_module
from vllm.v1.worker.worker_base import WorkerBase


class _FakeAttentionLayer(AttentionLayerBase):
    def __init__(self, cache_config: CacheConfig | None) -> None:
        self.impl = SimpleNamespace(cache_config=cache_config)

    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None


class _FakeAttentionLayerWithoutImpl(AttentionLayerBase):
    def get_attn_backend(self):
        raise NotImplementedError

    def get_kv_cache_spec(self, vllm_config):
        return None


def _worker(main_cache_config: CacheConfig, layers: list[object]) -> WorkerBase:
    worker = WorkerBase.__new__(WorkerBase)
    worker.vllm_config = SimpleNamespace(cache_config=main_cache_config)
    worker.compilation_config = SimpleNamespace(
        static_forward_context={str(i): layer for i, layer in enumerate(layers)}
    )
    return worker


class DFlashKVLayoutFanoutTests(unittest.TestCase):
    def test_real_replace_loses_layout_then_fanout_adopts_it(self) -> None:
        main = CacheConfig(block_size=64, cache_dtype="auto")
        draft = replace(main, cache_dtype="bfloat16")
        self.assertIsNot(draft, main)
        self.assertIsNone(draft.kv_cache_layout)

        _worker(main, [_FakeAttentionLayer(draft)]).set_kv_cache_layout("LBHNC")

        self.assertEqual(main.kv_cache_layout, "LBHNC")
        self.assertEqual(draft.kv_cache_layout, "LBHNC")
        self.assertEqual(draft.cache_dtype, "bfloat16")

    def test_duplicate_and_main_configs_are_recorded_once(self) -> None:
        main = CacheConfig(block_size=64, cache_dtype="auto")
        draft = replace(main, cache_dtype="bfloat16")
        calls: list[int] = []
        real_record = worker_base_module.record_kv_cache_layout

        def record_spy(cache_config, layout_name):
            calls.append(id(cache_config))
            real_record(cache_config, layout_name)

        worker = _worker(
            main,
            [
                _FakeAttentionLayer(main),
                _FakeAttentionLayer(draft),
                _FakeAttentionLayer(draft),
            ],
        )
        with patch.object(
            worker_base_module, "record_kv_cache_layout", side_effect=record_spy
        ):
            worker.set_kv_cache_layout("LBHNC")

        self.assertEqual(calls.count(id(main)), 1)
        self.assertEqual(calls.count(id(draft)), 1)

    def test_matching_layout_is_idempotent(self) -> None:
        main = CacheConfig(block_size=64, cache_dtype="auto")
        draft = replace(main, cache_dtype="bfloat16")
        record_kv_cache_layout(draft, "LBHNC")

        _worker(main, [_FakeAttentionLayer(draft)]).set_kv_cache_layout("LBHNC")

        self.assertEqual(main.kv_cache_layout, "LBHNC")
        self.assertEqual(draft.kv_cache_layout, "LBHNC")

    def test_conflicting_clone_layout_fails_closed(self) -> None:
        main = CacheConfig(block_size=64, cache_dtype="auto")
        draft = replace(main, cache_dtype="bfloat16")
        record_kv_cache_layout(draft, "LBNHC")

        with self.assertRaisesRegex(ValueError, "already resolved to LBNHC"):
            _worker(main, [_FakeAttentionLayer(draft)]).set_kv_cache_layout("LBHNC")

        self.assertEqual(draft.kv_cache_layout, "LBNHC")

    def test_non_attention_and_missing_impl_or_cache_config_are_ignored(self) -> None:
        main = CacheConfig(block_size=64, cache_dtype="auto")
        unrelated = CacheConfig(block_size=64, cache_dtype="bfloat16")
        no_cache = _FakeAttentionLayer(None)
        no_impl = _FakeAttentionLayerWithoutImpl()
        not_attention = SimpleNamespace(impl=SimpleNamespace(cache_config=unrelated))

        _worker(main, [no_cache, no_impl, not_attention]).set_kv_cache_layout("LBHNC")

        self.assertEqual(main.kv_cache_layout, "LBHNC")
        self.assertIsNone(unrelated.kv_cache_layout)


if __name__ == "__main__":
    unittest.main()
