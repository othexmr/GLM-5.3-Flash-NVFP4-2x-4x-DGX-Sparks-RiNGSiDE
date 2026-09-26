# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): rewritten as one launch of two direct-neighbour world-two exchanges
# (pair sum with rank^1, then with 3-rank), optionally in two pipelined chunks.
"""One CuTe launch for two direct-neighbour RDMA rounds (v4: optionally over two pipelined chunks).

Adapted from b12x.comm.roce._oneshot_cute (Apache-2.0, retained LICENSE).
The first logical world-two proxy exchanges original input with rank^1; the
second exchanges the canonical pair sum with 3-rank. Neither proxy ever sees
a physically opposite TP4 rank.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr

from ._cute_intrinsics import (
    atomic_add_relaxed_gpu_u32,
    f32_as_u32,
    fence_sc_gpu,
    fence_sc_sys,
    ld_global_v4_u32,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_u32,
    ld_relaxed_sys_v4_u32,
    pack_f32x2_to_bf16x2,
    pack_f32x2_to_f16x2,
    spin_until_eq_acquire_sys,
    st_global_v4_u32,
    st_release_gpu_u32,
    st_relaxed_sys_u32,
    u32_as_f32,
    unpack_bf16x2,
    unpack_f16x2,
)

PACK_BYTES = 16
_PACK_ELEMS = {"float32": 4, "float16": 8, "bfloat16": 8}
_PREPARED: set[tuple[object, ...]] = set()


class _LeanLaunch:
    def __init__(self, dtype_name: str, physical_rank: int, threads: int,
                 slots: int, flag_stride: int, chunks: int = 1) -> None:
        if dtype_name not in _PACK_ELEMS or physical_rank not in range(4):
            raise ValueError("unsupported dtype or rank")
        if chunks not in (1, 2, 4) or slots & (slots - 1) or slots < 2 * chunks:
            raise ValueError("chunks must be 1, 2 or 4 with at least 2 x chunks power-of-two slots")
        self.chunks = chunks
        self.dtype_name = dtype_name
        self.pack_elems = _PACK_ELEMS[dtype_name]
        self.physical_rank = physical_rank
        self.first_local_rank = physical_rank & 1
        self.second_local_rank = 0 if physical_rank in (0, 1) else 1
        self.threads = threads
        self.slots = slots
        self.flag_stride = flag_stride

    @cute.jit
    def _accumulate(self, acc: cute.Tensor, words,
                    initialize: cutlass.Constexpr[bool]) -> None:
        if cutlass.const_expr(self.dtype_name == "float32"):
            for j in cutlass.range_constexpr(4):
                v = u32_as_f32(words[j])
                if cutlass.const_expr(initialize):
                    acc[j] = v
                else:
                    acc[j] = acc[j] + v
        else:
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.dtype_name == "float16"):
                    lo, hi = unpack_f16x2(words[j])
                else:
                    lo, hi = unpack_bf16x2(words[j])
                if cutlass.const_expr(initialize):
                    acc[2*j] = lo
                    acc[2*j+1] = hi
                else:
                    acc[2*j] = acc[2*j] + lo
                    acc[2*j+1] = acc[2*j+1] + hi

    @cute.jit
    def _store(self, addr: Int64, acc: cute.Tensor) -> None:
        packed = cute.make_rmem_tensor((4,), cutlass.Uint32)
        if cutlass.const_expr(self.dtype_name == "float32"):
            for j in cutlass.range_constexpr(4):
                packed[j] = f32_as_u32(acc[j])
        else:
            for j in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.dtype_name == "float16"):
                    packed[j] = pack_f32x2_to_f16x2(acc[2*j], acc[2*j+1])
                else:
                    packed[j] = pack_f32x2_to_bf16x2(acc[2*j], acc[2*j+1])
        st_global_v4_u32(addr, packed[0], packed[1], packed[2], packed[3])

    @cute.jit
    def __call__(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer,
                 size_packs: Int32, nbytes: Int32,
                 recv1: Int64, flag1: Int64, send1: Int64, ctrl1: Int64,
                 recv2: Int64, flag2: Int64, send2: Int64, ctrl2: Int64,
                 slot_bytes: Int64, epoch_ptr: Int64,
                 stage1_ptr: Int64, stage2_ptr: Int64, tail_ptr: Int64,
                 poison_ptr: Int64, spin_limit: Uint32, grid_x: Int32,
                 stream: cuda.CUstream) -> None:
        self.kernel(
            input_ptr, output_ptr, size_packs, nbytes,
            recv1, flag1, send1, ctrl1, recv2, flag2, send2, ctrl2,
            slot_bytes, epoch_ptr, stage1_ptr, stage2_ptr, tail_ptr,
            poison_ptr, spin_limit,
        ).launch(grid=(grid_x, 1, 1), block=[self.threads, 1, 1],
                 cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, input_ptr: cute.Pointer, output_ptr: cute.Pointer,
               size_packs: Int32, nbytes: Int32,
               recv1: Int64, flag1: Int64, send1: Int64, ctrl1: Int64,
               recv2: Int64, flag2: Int64, send2: Int64, ctrl2: Int64,
               slot_bytes: Int64, epoch_ptr: Int64,
               stage1_ptr: Int64, stage2_ptr: Int64, tail_ptr: Int64,
               poison_ptr: Int64, spin_limit: Uint32) -> None:
        # v4 (2026-09-23): the message is cut into `chunks` contiguous pack ranges, each a logical op with its
        # own sequence (base + 1 + c), slot (seq & (slots - 1)) and staging counters. Round 1 stages and rings every
        # chunk first; round 2 of chunk c then waits only for chunk c's first-neighbour flag, so its exchange with
        # the second neighbour (the other cable) overlaps chunk c+1 still arriving on the first. chunks == 1 is the
        # v2 kernel with the slot taken modulo `slots`.
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        src = Int64(input_ptr.toint())
        dst = Int64(output_ptr.toint())
        base = ld_relaxed_gpu_u32(epoch_ptr)
        mask = Uint32(self.slots - 1)
        peer1 = 1 - self.first_local_rank
        peer2 = 1 - self.second_local_rank
        index = Int32(bidx) * Int32(self.threads) + Int32(tidx)
        stride = Int32(gdim) * Int32(self.threads)

        if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
            # Round 1: every chunk of the input -> the first neighbour. For each chunk the last staging block
            # publishes a level doorbell with that chunk's sequence and slot length.
            for c in cutlass.range_constexpr(self.chunks):
                seq = base + Uint32(c + 1)
                slot = Int64(seq & mask)
                lo = (size_packs * Int32(c)) // Int32(self.chunks)
                hi = (size_packs * Int32(c + 1)) // Int32(self.chunks)
                send1_slot = send1 + slot * slot_bytes
                pos = lo + index
                while pos < hi:
                    w = ld_global_v4_u32(src + Int64(pos) * Int64(PACK_BYTES))
                    st_global_v4_u32(send1_slot + Int64(pos - lo) * Int64(PACK_BYTES), w[0], w[1], w[2], w[3])
                    pos += stride
                cute.arch.sync_threads()
                if Int32(tidx) == Int32(0):
                    fence_sc_sys()
                    prior = atomic_add_relaxed_gpu_u32(stage1_ptr + Int64(4 * c), Uint32(1))
                    if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                        cbytes = Uint32(hi - lo) * Uint32(PACK_BYTES)
                        st_relaxed_sys_u32(ctrl1 + Int64(4), cbytes)
                        st_relaxed_sys_u32(ctrl1 + Int64(16) + slot * Int64(4), cbytes)
                        fence_sc_sys()
                        st_relaxed_sys_u32(ctrl1, seq)

            # Round 2, per chunk: (x0+x1) or (x2+x3) in canonical rank order, rounded to the input dtype, then the
            # second exchange.
            for c in cutlass.range_constexpr(self.chunks):
                seq = base + Uint32(c + 1)
                slot = Int64(seq & mask)
                lo = (size_packs * Int32(c)) // Int32(self.chunks)
                hi = (size_packs * Int32(c + 1)) // Int32(self.chunks)
                send2_slot = send2 + slot * slot_bytes
                if Int32(tidx) < Int32(2):
                    addr = flag1 + (
                        (Int64(peer1) * Int64(self.slots) + slot) * Int64(2)
                        + Int64(tidx)
                    ) * Int64(self.flag_stride)
                    if spin_until_eq_acquire_sys(addr, seq, spin_limit) != Uint32(0):
                        st_relaxed_sys_u32(ctrl1 + Int64(12), Uint32(peer1))
                        st_relaxed_sys_u32(ctrl1 + Int64(8), seq)
                        st_release_gpu_u32(poison_ptr, seq)
                cute.arch.sync_threads()
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    pos = lo + index
                    while pos < hi:
                        rel = Int64(pos - lo) * Int64(PACK_BYTES)
                        a = cute.make_rmem_tensor((self.pack_elems,), cutlass.Float32)
                        local = ld_global_v4_u32(src + Int64(pos) * Int64(PACK_BYTES))
                        remote = ld_relaxed_sys_v4_u32(
                            recv1 + (Int64(peer1) * Int64(self.slots) + slot) * slot_bytes + rel)
                        if cutlass.const_expr(self.first_local_rank == 0):
                            self._accumulate(a, local, True)
                            self._accumulate(a, remote, False)
                        else:
                            self._accumulate(a, remote, True)
                            self._accumulate(a, local, False)
                        self._store(send2_slot + rel, a)
                        pos += stride
                cute.arch.sync_threads()
                if Int32(tidx) == Int32(0):
                    fence_sc_sys()
                    prior = atomic_add_relaxed_gpu_u32(stage2_ptr + Int64(4 * c), Uint32(1))
                    if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                        if ld_relaxed_sys_u32(ctrl1 + Int64(8)) == Uint32(0):
                            cbytes = Uint32(hi - lo) * Uint32(PACK_BYTES)
                            st_relaxed_sys_u32(ctrl2 + Int64(4), cbytes)
                            st_relaxed_sys_u32(ctrl2 + Int64(16) + slot * Int64(4), cbytes)
                            fence_sc_sys()
                            st_relaxed_sys_u32(ctrl2, seq)

            # Round 3, per chunk: the canonical sum of the two pair sums, into the output.
            for c in cutlass.range_constexpr(self.chunks):
                seq = base + Uint32(c + 1)
                slot = Int64(seq & mask)
                lo = (size_packs * Int32(c)) // Int32(self.chunks)
                hi = (size_packs * Int32(c + 1)) // Int32(self.chunks)
                send2_slot = send2 + slot * slot_bytes
                if Int32(tidx) < Int32(2):
                    addr = flag2 + (
                        (Int64(peer2) * Int64(self.slots) + slot) * Int64(2)
                        + Int64(tidx)
                    ) * Int64(self.flag_stride)
                    if spin_until_eq_acquire_sys(addr, seq, spin_limit) != Uint32(0):
                        st_relaxed_sys_u32(ctrl2 + Int64(12), Uint32(peer2))
                        st_relaxed_sys_u32(ctrl2 + Int64(8), seq)
                        st_release_gpu_u32(poison_ptr, seq)
                cute.arch.sync_threads()
                if ld_relaxed_gpu_u32(poison_ptr) == Uint32(0):
                    pos = lo + index
                    while pos < hi:
                        rel = Int64(pos - lo) * Int64(PACK_BYTES)
                        a = cute.make_rmem_tensor((self.pack_elems,), cutlass.Float32)
                        local = ld_relaxed_sys_v4_u32(send2_slot + rel)
                        remote = ld_relaxed_sys_v4_u32(
                            recv2 + (Int64(peer2) * Int64(self.slots) + slot) * slot_bytes + rel)
                        if cutlass.const_expr(self.physical_rank < 2):
                            self._accumulate(a, local, True)
                            self._accumulate(a, remote, False)
                        else:
                            self._accumulate(a, remote, True)
                            self._accumulate(a, local, False)
                        self._store(dst + Int64(pos) * Int64(PACK_BYTES), a)
                        pos += stride

            fence_sc_gpu()
            cute.arch.sync_threads()
            if Int32(tidx) == Int32(0):
                prior = atomic_add_relaxed_gpu_u32(tail_ptr, Uint32(1))
                if (prior + Uint32(1)) % Uint32(gdim) == Uint32(0):
                    fence_sc_gpu()
                    if (ld_relaxed_sys_u32(ctrl1 + Int64(8)) == Uint32(0)
                            and ld_relaxed_sys_u32(ctrl2 + Int64(8)) == Uint32(0)):
                        st_release_gpu_u32(epoch_ptr, base + Uint32(self.chunks))


def _dummy():
    return make_ptr(cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16)


def _key(dtype_name: str, rank: int, threads: int, slots: int,
         flag_stride: int, chunks: int, device_index: int) -> tuple[object, ...]:
    return dtype_name, rank, threads, slots, flag_stride, chunks, device_index


def is_launcher_prepared(*key) -> bool:
    return _key(*key) in _PREPARED


@functools.cache
def get_launcher(dtype_name: str, rank: int, threads: int, slots: int,
                 flag_stride: int, chunks: int, device_index: int):
    key = _key(dtype_name, rank, threads, slots, flag_stride, chunks, device_index)
    launch = _LeanLaunch(dtype_name, rank, threads, slots, flag_stride, chunks)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    raw = b12x_compile(
        launch, _dummy(), _dummy(), 1, 16,
        *([16] * 8), 4096, 16, 16, 16, 16, 16, 1, 1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("comm.roce.lean_tp4_v4", 1, key[:-1]),
    )

    def run(input_address: int, output_address: int, size_packs: int, nbytes: int,
            recv1: int, flag1: int, send1: int, ctrl1: int,
            recv2: int, flag2: int, send2: int, ctrl2: int,
            slot_bytes: int, epoch: int, stage1: int, stage2: int, tail: int,
            poison: int, spin_limit: int, grid_x: int) -> None:
        raw(
            make_ptr(cutlass.Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            size_packs, nbytes, recv1, flag1, send1, ctrl1,
            recv2, flag2, send2, ctrl2, slot_bytes, epoch,
            stage1, stage2, tail, poison, spin_limit, grid_x, current_cuda_stream(),
        )

    _PREPARED.add(key)
    return run
