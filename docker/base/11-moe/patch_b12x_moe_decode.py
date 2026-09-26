#!/usr/bin/env python3
"""Anchor-checked patch: GLM-5.3 decode fast path for the B12X NVFP4 dynamic MoE kernel.

What it changes (B12X 1.3.0, CuTe DSL sources under dist-packages/b12x/moe):

  b12x/moe/_shared/kernels/dynamic.py  (MoEDynamicKernelBackend)
    * a compile-time ``decode_fastpath`` mode, off by default, enabled on a built
      backend through ``configure_decode_fastpath()`` (the activation subclasses'
      constructors are not touched);
    * three small shared-memory tables (expert histogram, per-route physical row,
      scan partials; about 1.8 KiB at E=288, 64 routes, 96 threads);
    * when the mode is on, the resident-grid route/pack head that runs before any
      expert weight byte is streamed -- phase-0 grid barrier, global-atomic
      histogram, grid barrier, single-thread 288-expert prefix scan, grid barrier,
      atomic pair-claim route/pack, grid barrier, per-CTA-leader task publication,
      grid barrier (five resident-grid barriers, kernel lines 3006-3928 of the
      package copy) -- is replaced by: every CTA resolves the complete route table
      in shared memory with a deterministic pair-order rank (one 64-iteration loop
      on one thread plus a two-level scan over the experts), quantizes a static
      share of the routed rows with the unchanged per-block NVFP4 quantizer, CTA 0
      publishes the identical task list and token map, and the ONE remaining grid
      barrier (the existing pre-consume barrier) orders everything before the
      unchanged consumer loop.  The task slots, tile bases, valid-row counts,
      packed-activation bytes and scale bytes are the same values the original
      head produces; only the order of rows inside an expert's M tile changes
      (pair order instead of atomic-arrival order), which does not affect any
      per-row GEMM result.

  b12x/moe/fused_moe/_impl.py
    * ``B12X_GLM53_DECODE_FASTPATH=1`` (read once at import) selects the mode for
      NVFP4 silu launches with 0 < routed_rows <= B12X_GLM53_DECODE_FASTPATH_MAX_ROWS
      (default 64), tile M16, 128-aligned intermediate, materialized work queue,
      no direct routing, no deterministic output.  Prefill launches (routed rows
      above the limit) and every other recipe compile and run the original kernel;
      the flag is part of the dynamic-kernel cache key and compile spec.

Default (flag unset or 0): both files behave exactly as before; the added kernel
code is not traced (``cutlass.const_expr`` guards) and the added host code only
evaluates a Python boolean.

Every edit is anchored on a unique source fragment and refuses to run if the
anchor is missing or ambiguous; running twice refuses ("already patched").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_SITE = Path("/usr/local/lib/python3.12/dist-packages")
IMPL_REL = Path("b12x/moe/fused_moe/_impl.py")
DYN_REL = Path("b12x/moe/_shared/kernels/dynamic.py")

ENV_NAME = "B12X_GLM53_DECODE_FASTPATH"
ENV_MAX_ROWS = "B12X_GLM53_DECODE_FASTPATH_MAX_ROWS"


def replace_once(text: str, old: str, new: str, *, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"[{what}] expected exactly one anchor, found {count}: {old[:80]!r}")
    return text.replace(old, new, 1)


# --------------------------------------------------------------------------------------
# dynamic.py (kernel)
# --------------------------------------------------------------------------------------

DYN_CTOR_ANCHOR = (
    "        self.load_register_requirement = 32\n"
    "        self.mma_register_requirement = 232\n"
    "\n"
    "    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):\n"
)

DYN_CTOR_NEW = (
    "        self.load_register_requirement = 32\n"
    "        self.mma_register_requirement = 232\n"
    "        # GLM-5.3 decode fast path (patch_b12x_moe_decode.py).  Off unless\n"
    "        # configure_decode_fastpath() is called on the built backend; the\n"
    "        # placeholders keep the shared-memory struct shape valid when off.\n"
    "        self.decode_fastpath = False\n"
    "        self.fastpath_num_experts = 1\n"
    "        self.fastpath_max_pairs = 1\n"
    "        self.fastpath_scan_span = 1\n"
    "        self.fastpath_k_splits = 1\n"
    "\n"
    "    def configure_decode_fastpath(\n"
    "        self,\n"
    "        *,\n"
    "        num_experts: int,\n"
    "        max_pairs: int,\n"
    "        hidden_size: int,\n"
    "        k_splits: int = 2,\n"
    "    ) -> None:\n"
    "        \"\"\"Enable the decode route/pack fast path (routed rows <= max_pairs).\n"
    "\n"
    "        The fast path replaces the five-barrier resident-grid head (global\n"
    "        histogram, single-thread expert prefix scan, atomic pair claims,\n"
    "        leader task publication) by a per-CTA shared-memory route resolution\n"
    "        with a deterministic pair-order row rank, so a launch reaches its\n"
    "        first expert weight byte after one grid barrier.  The consumer loop,\n"
    "        the task layout, the packed activation bytes and the scale bytes are\n"
    "        unchanged.  NVFP4 materialized-queue grouped routing only.\n"
    "        \"\"\"\n"
    "        if self.quant_recipe != \"nvfp4\" or self.is_w4a8 or self.is_w6a8:\n"
    "            raise ValueError(\"decode fast path requires the nvfp4 recipe\")\n"
    "        if self.direct_routing or self.deterministic_output or self.swap_ab:\n"
    "            raise ValueError(\n"
    "                \"decode fast path requires grouped, non-deterministic, \"\n"
    "                \"non-swapped NVFP4 routing\"\n"
    "            )\n"
    "        if self.materialize_intermediate or self.w4a8_m1_materialized:\n"
    "            raise ValueError(\"decode fast path excludes materialized regimes\")\n"
    "        if self.work_source != _WORK_SOURCE_MATERIALIZED_QUEUE:\n"
    "            raise ValueError(\"decode fast path requires the materialized work queue\")\n"
    "        num_experts = int(num_experts)\n"
    "        max_pairs = int(max_pairs)\n"
    "        hidden_size = int(hidden_size)\n"
    "        if num_experts < 1 or not (1 <= max_pairs < 65536):\n"
    "            raise ValueError(\n"
    "                f\"decode fast path geometry out of range: E={num_experts}, \"\n"
    "                f\"max_pairs={max_pairs}\"\n"
    "            )\n"
    "        if hidden_size % 16 != 0:\n"
    "            raise ValueError(\"decode fast path requires hidden_size % 16 == 0\")\n"
    "        sf_blocks = hidden_size // 16\n"
    "        k_splits = int(k_splits)\n"
    "        if k_splits < 1 or sf_blocks % k_splits != 0:\n"
    "            k_splits = 1\n"
    "        self.decode_fastpath = True\n"
    "        self.fastpath_num_experts = num_experts\n"
    "        self.fastpath_max_pairs = max_pairs\n"
    "        self.fastpath_scan_span = (\n"
    "            num_experts + self.threads_per_cta - 1\n"
    "        ) // self.threads_per_cta\n"
    "        self.fastpath_k_splits = k_splits\n"
    "\n"
    "    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):\n"
)

DYN_STORAGE_ANCHOR = (
    "            reduce_scratch: cute.struct.MemRange[cutlass.Float32, 5]\n"
    "\n"
    "        storage = smem.allocate(Storage)\n"
)

DYN_STORAGE_NEW = (
    "            reduce_scratch: cute.struct.MemRange[cutlass.Float32, 5]\n"
    "            # GLM-5.3 decode fast path: per-CTA route table (placeholders when off).\n"
    "            fp_hist: cute.struct.MemRange[\n"
    "                cutlass.Int32,\n"
    "                self.fastpath_num_experts if self.decode_fastpath else 1,\n"
    "            ]\n"
    "            fp_pair_row: cute.struct.MemRange[\n"
    "                cutlass.Int32,\n"
    "                self.fastpath_max_pairs if self.decode_fastpath else 1,\n"
    "            ]\n"
    "            fp_partial: cute.struct.MemRange[\n"
    "                cutlass.Int32,\n"
    "                (self.threads_per_cta + 1) if self.decode_fastpath else 1,\n"
    "            ]\n"
    "\n"
    "        storage = smem.allocate(Storage)\n"
)

DYN_PHASE0_BARRIER_ANCHOR = (
    "        cute.arch.sync_threads()\n"
    "        self._resident_grid_barrier(\n"
    "            barrier_count,\n"
    "            barrier_epoch,\n"
    "            Int32(gdim_z),\n"
    "            is_cta_leader,\n"
    "        )\n"
    "\n"
    "        # General grouped execution compacts routes by expert."
)

DYN_PHASE0_BARRIER_NEW = (
    "        cute.arch.sync_threads()\n"
    "        # Decode fast path: nothing written in phase 0 is read by another CTA\n"
    "        # before the pre-consume barrier, so the first grid barrier is skipped.\n"
    "        if cutlass.const_expr(not self.decode_fastpath):\n"
    "            self._resident_grid_barrier(\n"
    "                barrier_count,\n"
    "                barrier_epoch,\n"
    "                Int32(gdim_z),\n"
    "                is_cta_leader,\n"
    "            )\n"
    "\n"
    "        # General grouped execution compacts routes by expert."
)

DYN_HIST_ANCHOR = (
    "        if cutlass.const_expr(not self.direct_routing):\n"
    "            hist_idx = flat_tid\n"
)
DYN_HIST_NEW = (
    "        if cutlass.const_expr(not self.direct_routing and not self.decode_fastpath):\n"
    "            hist_idx = flat_tid\n"
)

DYN_PRODUCE_ANCHOR = (
    "        produce_active = (\n"
    "            Int32(0) if cutlass.const_expr(self.w4a8_m1_materialized) else Int32(1)\n"
    "        )\n"
)
DYN_PRODUCE_NEW = (
    "        produce_active = (\n"
    "            Int32(0)\n"
    "            if cutlass.const_expr(self.w4a8_m1_materialized or self.decode_fastpath)\n"
    "            else Int32(1)\n"
    "        )\n"
)

DYN_FENCE_ANCHOR = (
    "        if cutlass.const_expr(not self.w4a8_m1_materialized):\n"
    "            cute.arch.sync_threads()\n"
    "            # Conservative publish fence before the last-producer CTA flushes\n"
)

# The fast-path block.  Inserted immediately before DYN_FENCE_ANCHOR, i.e. after
# the (runtime-skipped) producer loop and before the existing CTA fence, the
# (compile-time-skipped) barrier #4 / leader publish, and the retained barrier #5.
# Every dynamic-control-flow variable is pre-defined with a concrete type, as the
# surrounding kernel does, and every name is fp_-prefixed so no loop-carried
# value of the original head is touched.
DYN_FASTPATH_BLOCK = '''        if cutlass.const_expr(self.decode_fastpath):
            # ---- GLM-5.3 decode fast path (routed rows <= fastpath_max_pairs) ----
            # Every CTA resolves the whole route table in shared memory with a
            # deterministic pair-order rank; no global histogram atomics, no
            # single-thread global prefix scan, no pair-claim atomics and no grid
            # barrier before the packed activations are written.  CTA 0 alone
            # publishes the task list and the token map.  The retained
            # pre-consume grid barrier below orders everything.
            fp_tile_m = Int32(self.tile_shape_mnk[0])
            fp_num_experts = Int32(self.fastpath_num_experts)
            fp_threads = Int32(self.threads_per_cta)
            fp_span = Int32(self.fastpath_scan_span)
            fp_k_splits = Int32(self.fastpath_k_splits)
            fp_hist_addr = ctrl_base_addr + Int32(Storage._offsets["fp_hist"])
            fp_pair_row_addr = ctrl_base_addr + Int32(Storage._offsets["fp_pair_row"])
            fp_partial_addr = ctrl_base_addr + Int32(Storage._offsets["fp_partial"])
            fp_i = Int32(0)
            fp_p = Int32(0)
            fp_e = Int32(0)
            fp_j = Int32(0)
            fp_t = Int32(0)
            fp_u = Int32(0)
            fp_rank = Int32(0)
            fp_rows = Int32(0)
            fp_local = Int32(0)
            fp_acc = Int32(0)
            fp_v = Int32(0)
            fp_run = Int32(0)
            fp_word = Int32(0)
            fp_base = Int32(0)
            fp_row = Int32(0)
            fp_tile = Int32(0)
            fp_valid = Int32(0)
            fp_total = Int32(0)
            fp_split = Int32(0)
            fp_tok = Int32(0)
            fp_sf = Int32(0)
            fp_sf_end = Int32(0)
            fp_block_start = Int32(0)
            fp_k_tile = Int32(0)
            fp_inner_k = Int32(0)
            fp_sf_atom = Int32(0)
            fp_sf_row = Int32(0)
            fp_scale_offset = Int32(0)
            fp_gs = cutlass.Float32(0.0)
            fp_value = cutlass.Float32(0.0)
            fp_block_max = cutlass.Float32(0.0)
            fp_packed64 = Uint64(0)
            fp_scale_byte = Uint8(0)

            # (1) zero the shared-memory expert histogram and stage the route ids
            #     in the pair table (cooperative, one global load per route)
            fp_i = Int32(tidx)
            while fp_i < fp_num_experts:
                _st_shared_i32(fp_hist_addr + fp_i * Int32(4), Int32(0))
                fp_i += fp_threads
            fp_p = Int32(tidx)
            while fp_p < total_pairs:
                _st_shared_i32(
                    fp_pair_row_addr + fp_p * Int32(4), topk_ids[fp_p].to(Int32)
                )
                fp_p += fp_threads
            cute.arch.sync_threads()

            # (2) one thread, shared memory only: rows per expert and the
            #     pair-order rank of every active route (an inactive route,
            #     expert id < 0 or >= E, gets -1); the rank overwrites the id
            if Int32(tidx) == Int32(0):
                fp_p = Int32(0)
                while fp_p < total_pairs:
                    fp_e = _ld_shared_i32(fp_pair_row_addr + fp_p * Int32(4))
                    fp_rank = Int32(-1)
                    if fp_e >= Int32(0) and fp_e < fp_num_experts:
                        fp_rank = _ld_shared_i32(fp_hist_addr + fp_e * Int32(4))
                        _st_shared_i32(
                            fp_hist_addr + fp_e * Int32(4), fp_rank + Int32(1)
                        )
                    _st_shared_i32(fp_pair_row_addr + fp_p * Int32(4), fp_rank)
                    fp_p += Int32(1)
            cute.arch.sync_threads()

            # (3) exclusive prefix of ceil(rows / tile_m): thread t owns experts
            #     [t*span, (t+1)*span); thread 0 scans the per-thread partials
            fp_local = Int32(0)
            fp_j = Int32(0)
            while fp_j < fp_span:
                fp_e = Int32(tidx) * fp_span + fp_j
                if fp_e < fp_num_experts:
                    fp_rows = _ld_shared_i32(fp_hist_addr + fp_e * Int32(4))
                    fp_local += (fp_rows + fp_tile_m - Int32(1)) // fp_tile_m
                fp_j += Int32(1)
            _st_shared_i32(fp_partial_addr + Int32(tidx) * Int32(4), fp_local)
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                fp_acc = Int32(0)
                fp_t = Int32(0)
                while fp_t < fp_threads:
                    fp_v = _ld_shared_i32(fp_partial_addr + fp_t * Int32(4))
                    _st_shared_i32(fp_partial_addr + fp_t * Int32(4), fp_acc)
                    fp_acc += fp_v
                    fp_t += Int32(1)
                _st_shared_i32(fp_partial_addr + fp_threads * Int32(4), fp_acc)
            cute.arch.sync_threads()

            # (4) pack (rows | tile_base << 16) per expert in the histogram slot
            fp_run = _ld_shared_i32(fp_partial_addr + Int32(tidx) * Int32(4))
            fp_j = Int32(0)
            while fp_j < fp_span:
                fp_e = Int32(tidx) * fp_span + fp_j
                if fp_e < fp_num_experts:
                    fp_rows = _ld_shared_i32(fp_hist_addr + fp_e * Int32(4))
                    _st_shared_i32(
                        fp_hist_addr + fp_e * Int32(4),
                        fp_rows | (fp_run << Int32(16)),
                    )
                    fp_run += (fp_rows + fp_tile_m - Int32(1)) // fp_tile_m
                fp_j += Int32(1)
            cute.arch.sync_threads()

            # (5) physical row of every route: tile_base * tile_m + rank, split
            #     across M tiles when an expert holds more than tile_m rows
            fp_p = Int32(tidx)
            while fp_p < total_pairs:
                fp_rank = _ld_shared_i32(fp_pair_row_addr + fp_p * Int32(4))
                fp_row = Int32(-1)
                if fp_rank >= Int32(0):
                    fp_e = topk_ids[fp_p].to(Int32)
                    fp_word = _ld_shared_i32(fp_hist_addr + fp_e * Int32(4))
                    fp_base = fp_word >> Int32(16)
                    fp_row = (
                        fp_base + fp_rank // fp_tile_m
                    ) * fp_tile_m + fp_rank % fp_tile_m
                _st_shared_i32(fp_pair_row_addr + fp_p * Int32(4), fp_row)
                fp_p += fp_threads
            cute.arch.sync_threads()

            # (6) CTA 0: token map / weights, materialized task list, task tail
            #     (same slot layout as _publish_deferred_tasks in the original head)
            if Int32(bidz) == Int32(0):
                fp_p = Int32(tidx)
                while fp_p < total_pairs:
                    fp_row = _ld_shared_i32(fp_pair_row_addr + fp_p * Int32(4))
                    if fp_row >= Int32(0):
                        st_global_i32(
                            get_ptr_as_int64(token_map, fp_row), fp_p // num_topk
                        )
                        st_global_f32(
                            get_ptr_as_int64(token_weights, fp_row),
                            topk_weights[fp_p].to(cutlass.Float32),
                        )
                    fp_p += fp_threads
                fp_e = Int32(tidx)
                while fp_e < fp_num_experts:
                    fp_word = _ld_shared_i32(fp_hist_addr + fp_e * Int32(4))
                    fp_rows = fp_word & Int32(0xFFFF)
                    fp_tile = fp_word >> Int32(16)
                    while fp_rows > Int32(0):
                        fp_valid = fp_rows
                        if fp_valid > fp_tile_m:
                            fp_valid = fp_tile_m
                        self._publish_deferred_tasks(
                            task_expert,
                            task_valid_rows,
                            route_gate_tile_cnt,
                            task_slice_chunk,
                            fp_e,
                            fp_tile,
                            fp_valid,
                        )
                        fp_rows -= fp_tile_m
                        fp_tile += Int32(1)
                    fp_e += fp_threads
                if Int32(tidx) == Int32(0):
                    fp_total = _ld_shared_i32(fp_partial_addr + fp_threads * Int32(4))
                    st_global_i32(
                        get_ptr_as_int64(task_tail, Int32(0)),
                        fp_total * materialized_num_groups,
                    )

            # (7) quantize the routed rows: one (route, K-split) unit per warp,
            #     grid-strided over all resident warps; the per-block quantizer,
            #     per-expert input scale, packed layout and swizzled scale layout
            #     are exactly those of the original per-pair producer
            fp_blocks_per_split = sf_blocks_per_row // fp_k_splits
            fp_units = total_pairs * fp_k_splits
            fp_u = Int32(bidz) * num_cta_warps + warp_idx
            fp_ustride = Int32(gdim_z) * num_cta_warps
            while fp_u < fp_units:
                fp_p = fp_u // fp_k_splits
                fp_split = fp_u - fp_p * fp_k_splits
                fp_row = _ld_shared_i32(fp_pair_row_addr + fp_p * Int32(4))
                if fp_row >= Int32(0):
                    fp_e = topk_ids[fp_p].to(Int32)
                    fp_tok = fp_p // num_topk
                    fp_gs = input_global_scale[fp_e].to(cutlass.Float32)
                    fp_sf_end = (fp_split + Int32(1)) * fp_blocks_per_split
                    fp_sf = fp_split * fp_blocks_per_split + lane_id
                    while fp_sf < fp_sf_end:
                        fp_block_start = fp_sf * Int32(16)
                        fp_values = cute.make_rmem_tensor((16,), cutlass.Float32)
                        fp_block_max = cutlass.Float32(0.0)
                        for fp_elem in cutlass.range_constexpr(16):
                            fp_value = cutlass.Float32(
                                a_input[fp_tok, fp_block_start + Int32(fp_elem)]
                            )
                            fp_values[fp_elem] = fp_value
                            fp_block_max = fmax_f32(fp_block_max, fabs_f32(fp_value))
                        fp_packed64 = Uint64(0)
                        fp_scale_byte = Uint8(0)
                        if self.is_gated and self.fast_math:
                            fp_packed64, fp_scale_byte = quantize_block_fp4_fast(
                                fp_values, fp_block_max, fp_gs
                            )
                        else:
                            fp_packed64, fp_scale_byte = quantize_block_fp4(
                                fp_values, fp_block_max, fp_gs
                            )
                        st_global_u64(
                            get_ptr_as_int64(
                                packed_a_storage,
                                fp_row * output_bytes_per_row + fp_sf * Int32(8),
                            ),
                            fp_packed64,
                        )
                        fp_k_tile = fp_sf // Int32(4)
                        fp_inner_k = fp_sf % Int32(4)
                        fp_sf_atom = fp_row >> Int32(7)
                        fp_sf_row = fp_row & Int32(127)
                        fp_scale_offset = (
                            fp_sf_atom * num_k_tiles * Int32(32 * 4 * 4)
                            + fp_k_tile * Int32(32 * 4 * 4)
                            + (fp_sf_row % Int32(32)) * Int32(4 * 4)
                            + (fp_sf_row // Int32(32)) * Int32(4)
                            + fp_inner_k
                        )
                        scale_storage[fp_scale_offset] = fp_scale_byte
                        fp_sf += Int32(32)
                fp_u += fp_ustride

'''

DYN_BARRIER4_ANCHOR = (
    "            # physical tile, then consume a fully addressable work domain.\n"
    "            if cutlass.const_expr(not self.w4a8_m1_materialized):\n"
    "                self._resident_grid_barrier(\n"
    "                    barrier_count,\n"
    "                    barrier_epoch,\n"
    "                    Int32(gdim_z),\n"
    "                    is_cta_leader,\n"
    "                )\n"
    "\n"
    "            if is_cta_leader > Int32(0) and cutlass.const_expr(\n"
    "                not self.w4a8_m1_materialized\n"
    "            ):\n"
)
DYN_BARRIER4_NEW = (
    "            # physical tile, then consume a fully addressable work domain.\n"
    "            # Decode fast path: the task list is already published by CTA 0\n"
    "            # and every CTA owns the whole route table, so the rendezvous\n"
    "            # barrier and the leader publish are skipped; the pre-consume\n"
    "            # barrier below is the single grid barrier of the launch.\n"
    "            if cutlass.const_expr(\n"
    "                not self.w4a8_m1_materialized and not self.decode_fastpath\n"
    "            ):\n"
    "                self._resident_grid_barrier(\n"
    "                    barrier_count,\n"
    "                    barrier_epoch,\n"
    "                    Int32(gdim_z),\n"
    "                    is_cta_leader,\n"
    "                )\n"
    "\n"
    "            if is_cta_leader > Int32(0) and cutlass.const_expr(\n"
    "                not self.w4a8_m1_materialized and not self.decode_fastpath\n"
    "            ):\n"
)

DYN_TAIL_ANCHOR = (
    "            if flat_tid == Int32(0) and cutlass.const_expr(\n"
    "                not self.w4a8_m1_materialized\n"
    "            ):\n"
    "                materialized_tail = expert_tile_base[num_experts]\n"
)
DYN_TAIL_NEW = (
    "            if flat_tid == Int32(0) and cutlass.const_expr(\n"
    "                not self.w4a8_m1_materialized and not self.decode_fastpath\n"
    "            ):\n"
    "                materialized_tail = expert_tile_base[num_experts]\n"
)

# Retained as-is; checked for presence so a drifted source cannot silently lose
# the single remaining grid barrier of the fast path.
DYN_BARRIER5_ANCHOR = (
    "            if cutlass.const_expr(not self.w4a8_m1_materialized):\n"
    "                self._resident_grid_barrier(\n"
    "                    barrier_count,\n"
    "                    barrier_epoch,\n"
    "                    Int32(gdim_z),\n"
    "                    is_cta_leader,\n"
    "                )\n"
    "                if flat_tid == Int32(0):\n"
    "                    _st_global_release_i32(\n"
)


def patch_dynamic_source(source: str) -> str:
    if "decode_fastpath" in source:
        raise RuntimeError("dynamic.py already carries the decode fast path")
    source = replace_once(source, DYN_CTOR_ANCHOR, DYN_CTOR_NEW, what="ctor")
    source = replace_once(source, DYN_STORAGE_ANCHOR, DYN_STORAGE_NEW, what="storage")
    source = replace_once(
        source, DYN_PHASE0_BARRIER_ANCHOR, DYN_PHASE0_BARRIER_NEW, what="phase0-barrier"
    )
    source = replace_once(source, DYN_HIST_ANCHOR, DYN_HIST_NEW, what="histogram-guard")
    source = replace_once(source, DYN_PRODUCE_ANCHOR, DYN_PRODUCE_NEW, what="producer-loop")
    source = replace_once(
        source, DYN_FENCE_ANCHOR, DYN_FASTPATH_BLOCK + DYN_FENCE_ANCHOR, what="fastpath-block"
    )
    source = replace_once(source, DYN_BARRIER4_ANCHOR, DYN_BARRIER4_NEW, what="barrier4-publish")
    source = replace_once(source, DYN_TAIL_ANCHOR, DYN_TAIL_NEW, what="task-tail")
    if source.count(DYN_BARRIER5_ANCHOR) != 1:
        raise RuntimeError("[barrier5] the retained pre-consume grid barrier is not where expected")
    return source


# --------------------------------------------------------------------------------------
# _impl.py (host dispatch)
# --------------------------------------------------------------------------------------

IMPL_ENV_ANCHOR = '_DYNAMIC_WORK_SOURCE_ENV = "B12X_DYNAMIC_WORK_SOURCE"\n'
IMPL_ENV_NEW = (
    '_DYNAMIC_WORK_SOURCE_ENV = "B12X_DYNAMIC_WORK_SOURCE"\n'
    "# GLM-5.3 decode fast path (patch_b12x_moe_decode.py): the NVFP4 dynamic\n"
    "# kernel's resident-grid route/pack head is replaced by a per-CTA shared-\n"
    "# memory route resolution for routed_rows <= " + ENV_MAX_ROWS + "\n"
    "# (default 64).  Off unless " + ENV_NAME + "=1; both read once at import.\n"
    f'_GLM53_DECODE_FASTPATH_ENV = "{ENV_NAME}"\n'
    f'_GLM53_DECODE_FASTPATH_MAX_ROWS_ENV = "{ENV_MAX_ROWS}"\n'
    "_GLM53_DECODE_FASTPATH = (\n"
    '    os.environ.get(_GLM53_DECODE_FASTPATH_ENV, "0").strip() == "1"\n'
    ")\n"
    "_GLM53_DECODE_FASTPATH_MAX_ROWS = max(\n"
    "    0,\n"
    "    min(\n"
    "        256,\n"
    '        int(os.environ.get(_GLM53_DECODE_FASTPATH_MAX_ROWS_ENV, "64").strip() or "64"),\n'
    "    ),\n"
    ")\n"
)

IMPL_WORK_SOURCE_DEF_ANCHOR = "\n\ndef _dynamic_work_source() -> str:\n"
IMPL_DECISION_FN = '''

def _glm53_decode_fastpath_selected(
    *,
    quant_mode: str,
    activation: str,
    routed_rows: int,
    n: int,
    direct_routing: bool,
    deterministic_output: bool,
    w4a8_repacked: bool,
    selected_tile_m: int,
) -> bool:
    """Whether a dynamic launch takes the GLM-5.3 decode route/pack fast path.

    Structural predicate shared by kernel selection and the launch: NVFP4 silu,
    grouped (not direct) non-deterministic routing on the materialized queue,
    M16 tile, 128-aligned intermediate (no swap_ab), and a routed-row count
    inside the decode band.  Prefill launches fail the row bound and keep the
    original kernel.
    """

    if not _GLM53_DECODE_FASTPATH:
        return False
    return bool(
        _normalize_quant_mode(quant_mode) == "nvfp4"
        and activation == "silu"
        and not w4a8_repacked
        and not direct_routing
        and not deterministic_output
        and 0 < int(routed_rows) <= _GLM53_DECODE_FASTPATH_MAX_ROWS
        and int(selected_tile_m) == 16
        and int(n) % 128 == 0
        and _dynamic_work_source() == "materialized_queue"
    )
'''

IMPL_SIG_ANCHOR = (
    "    trellis_bits: int = 0,\n"
    "    trellis_coupled: bool = False,\n"
    "):\n"
    "    quant_mode = _normalize_quant_mode(quant_mode)\n"
    "    # w6a8_mx rides the nvfp4-shaped launch ABI"
)
IMPL_SIG_NEW = (
    "    trellis_bits: int = 0,\n"
    "    trellis_coupled: bool = False,\n"
    "    decode_fastpath: bool = False,\n"
    "):\n"
    "    quant_mode = _normalize_quant_mode(quant_mode)\n"
    "    # w6a8_mx rides the nvfp4-shaped launch ABI"
)

IMPL_KEY_ANCHOR = (
    "        int(trellis_bits),\n"
    "        bool(trellis_coupled),\n"
    "    )\n"
    "    last_kkey, last_kval = _LAST_KERNEL\n"
)
IMPL_KEY_NEW = (
    "        int(trellis_bits),\n"
    "        bool(trellis_coupled),\n"
    "        bool(decode_fastpath),\n"
    "    )\n"
    "    last_kkey, last_kval = _LAST_KERNEL\n"
)

IMPL_CTOR_ANCHOR = "    kernel = activation_spec.make_dynamic_kernel(**kernel_kwargs)\n"
IMPL_CTOR_NEW = (
    "    kernel = activation_spec.make_dynamic_kernel(**kernel_kwargs)\n"
    "    if decode_fastpath:\n"
    "        kernel.configure_decode_fastpath(\n"
    "            num_experts=E,\n"
    "            max_pairs=_GLM53_DECODE_FASTPATH_MAX_ROWS,\n"
    "            hidden_size=k,\n"
    "        )\n"
    "        logger.info(\n"
    '            "B12X GLM-5.3 decode fast path compiled: E=%d k=%d n=%d topk=%d "\n'
    '            "tile=%s max_pairs=%d k_splits=%d",\n'
    "            E,\n"
    "            k,\n"
    "            n,\n"
    "            num_topk,\n"
    "            mma_tiler_mn,\n"
    "            kernel.fastpath_max_pairs,\n"
    "            kernel.fastpath_k_splits,\n"
    "        )\n"
)

IMPL_PRECALL_ANCHOR = (
    "    # Do not enlarge this grid beyond the measured resident count: the fused\n"
    "    # kernel has a resident-grid barrier, so over-launching would deadlock.\n"
    "    compiled, mac = _get_dynamic_kernel(\n"
)
IMPL_PRECALL_NEW = (
    "    decode_fastpath = _glm53_decode_fastpath_selected(\n"
    "        quant_mode=quant_mode,\n"
    "        activation=activation,\n"
    "        routed_rows=routed_rows,\n"
    "        n=n,\n"
    "        direct_routing=direct_routing,\n"
    "        deterministic_output=deterministic_output,\n"
    "        w4a8_repacked=w4a8_repacked,\n"
    "        selected_tile_m=selected_tile_m,\n"
    "    )\n"
    "    # Do not enlarge this grid beyond the measured resident count: the fused\n"
    "    # kernel has a resident-grid barrier, so over-launching would deadlock.\n"
    "    compiled, mac = _get_dynamic_kernel(\n"
)

IMPL_CALL_ANCHOR = (
    "        trellis_bits=trellis_bits,\n"
    "        trellis_coupled=trellis_coupled,\n"
    "    )\n"
    "    if volatile_launch_state:\n"
)
IMPL_CALL_NEW = (
    "        trellis_bits=trellis_bits,\n"
    "        trellis_coupled=trellis_coupled,\n"
    "        decode_fastpath=decode_fastpath,\n"
    "    )\n"
    "    if volatile_launch_state:\n"
)


def patch_impl_source(source: str) -> str:
    if ENV_NAME in source:
        raise RuntimeError("_impl.py already carries the decode fast path switch")
    source = replace_once(source, IMPL_ENV_ANCHOR, IMPL_ENV_NEW, what="env-constants")
    source = replace_once(
        source,
        IMPL_WORK_SOURCE_DEF_ANCHOR,
        IMPL_DECISION_FN + IMPL_WORK_SOURCE_DEF_ANCHOR,
        what="decision-fn",
    )
    source = replace_once(source, IMPL_SIG_ANCHOR, IMPL_SIG_NEW, what="get-dynamic-kernel-signature")
    source = replace_once(source, IMPL_KEY_ANCHOR, IMPL_KEY_NEW, what="cache-key")
    source = replace_once(source, IMPL_CTOR_ANCHOR, IMPL_CTOR_NEW, what="kernel-configure")
    source = replace_once(source, IMPL_PRECALL_ANCHOR, IMPL_PRECALL_NEW, what="launch-decision")
    source = replace_once(source, IMPL_CALL_ANCHOR, IMPL_CALL_NEW, what="launch-kwarg")
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--site-packages",
        type=Path,
        default=DEFAULT_SITE,
        help="dist-packages root holding b12x/ (default: the image's python3.12 dist-packages)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify anchors and syntax without writing",
    )
    args = parser.parse_args(argv)
    impl_path = args.site_packages / IMPL_REL
    dyn_path = args.site_packages / DYN_REL
    impl_src = impl_path.read_text(encoding="utf-8")
    dyn_src = dyn_path.read_text(encoding="utf-8")
    impl_new = patch_impl_source(impl_src)
    dyn_new = patch_dynamic_source(dyn_src)
    compile(impl_new, str(impl_path), "exec")
    compile(dyn_new, str(dyn_path), "exec")
    if args.check_only:
        print("anchors and syntax ok (no files written)")
        return 0
    impl_path.write_text(impl_new, encoding="utf-8")
    dyn_path.write_text(dyn_new, encoding="utf-8")
    print(
        f"GLM-5.3 B12X decode fast path installed ({ENV_NAME}=1 to enable, "
        f"{ENV_MAX_ROWS} default 64): {impl_path}, {dyn_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
