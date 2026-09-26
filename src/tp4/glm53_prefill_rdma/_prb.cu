// SPDX-License-Identifier: Apache-2.0
// Two-direction prefill ring collectives (PRB) for the four-node Spark ring (rdma-prefill lane, 2026-09-24).
// Derived from the one-way runtime _prc.cu (outputs/2026-09-23-rdma-prefill-collectives).
//
// Reduce-scatter and all-gather of [P, 4096] BF16 tensors over our own RDMA transport, both ways round the ring with
// one PCIe function ("PF") per direction: forward (r -> r+1) on lane 0 (rocep1s0f*, PCIe link A) of the port facing
// r+1, backward (r -> r-1) on lane 1 (roceP2p1s0f*, PCIe link B) of the port facing r-1 (the ceiling's ring_bi_split).
// The reduce-scatter keeps NCCL's association ((x[j+1] + x[j+2]) + x[j+3]) + x[j] on both chains, so its result is
// bit-identical to NCCL's; the all-gather only copies. protocol.py restates every offset, step and order in Python.
//
// Moving parts:
//   * One registered host region per rank (cudaHostAlloc, or 2 MiB-aligned anonymous memory with MADV_HUGEPAGE that the
//     GPU reaches through ATS; the host address is the device address either way):
//       recv[parity][dir][steps_cap][cr_max rows]  written by a neighbour's NIC (one slot per send step)
//       rflag[parity][dir][steps_cap]              u32 = call sequence, written by the neighbour's NIC after the data
//       stage[dir][nslot][cr_max rows]             written by our GPU kernel, read by our NIC
//       ready[dir][nslot]                          u64 = staged index + 1, GPU -> proxy
//       ctrl: send_done[dir] (u64, proxy -> GPU), poison (u32, GPU -> host), poison detail
//   * One GPU kernel per call: a grid of CTAs takes work units (item, part) in order from a device counter. Items are
//     the step-merged streams of protocol.streams(); every chunk is cut into `parts` pieces so several CTAs finish it
//     together (short per-chunk latency at the start and end of a call); the part that completes a staged chunk
//     publishes its ready word.
//   * One proxy thread per rank: per direction, strictly in call and step order, posts one RDMA WRITE per step followed
//     on the same QP by a 4-byte flag WRITE; staged steps wait for their ready word, relay steps (all-gather) post
//     straight from the receive buffer once the incoming flag is up. Signaled flag completions free staging slots.
//   * Receive areas alternate by call parity: before any rank writes call s+2 data, every rank has started call s+1's
//     kernel (each rank's call s+1 result needs data staged by every other rank's call s+1 kernel), so every rank has
//     finished reading its call s data, and every relay of call s has been received (argument in the README).
// Failure: a completion error stops the proxy (failed); a flag or slot wait past the timeout poisons the runtime. Both
// are reported by prb_health(), which the Python watchdog turns into a process exit. Nothing falls back to partial data.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <infiniband/verbs.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define PRB_ABI 2
#define PRB_WORLD 4
#define PRB_ROW_BYTES 8192u
#define PRB_ROW_PACKS 512u            // 16-byte packs per row
#define PRB_DIRS 2
#define PRB_FWD 0
#define PRB_BWD 1
#define PRB_FLAG_STRIDE 64u
#define PRB_NHCA 4
#define PRB_PORT 1
#define PRB_QCAP 8192                 // pending calls host -> proxy
#define PRB_ITEM_RING 1024            // posted, not yet completed steps per direction
#define PRB_KIND_RS 1
#define PRB_KIND_AG 2
#define PRB_THREADS 512
#define PRB_MEM_PINNED 0
#define PRB_MEM_HOSTTHP 1
#define PRB_SRC_RAW 0
#define PRB_SRC_RED 1
#define PRB_SRC_RELAY 2
enum { IT_STAGE_F = 0, IT_STAGE_B, IT_OUT_A, IT_OUT_B, IT_OUT_PREV, IT_OUT_NEXT, IT_OUT_FARF, IT_OUT_FARB };

static const char *HCA_NAMES[PRB_NHCA] = {"rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1"};

static int direct_path(int a, int b, int lane) {
    int lo = a < b ? a : b, hi = a < b ? b : a;
    int f1 = (lo == 0 && hi == 1) || (lo == 2 && hi == 3);
    int f0 = (lo == 1 && hi == 2) || (lo == 0 && hi == 3);
    if (!f0 && !f1) return -1;
    return (f1 ? 1 : 0) + 2 * lane;
}
static int send_hca(int r, int d) { return d == PRB_FWD ? direct_path(r, (r + 1) % 4, 0) : direct_path(r, (r + 3) % 4, 1); }
static int recv_hca(int r, int d) { return d == PRB_FWD ? direct_path(r, (r + 3) % 4, 0) : direct_path(r, (r + 1) % 4, 1); }

// ------------------------------------------------------------------------------------------------ geometry
struct Geo {
    uint32_t kind, rows, q, cr, nc, na, nb, nb1, nh;
    uint32_t steps[PRB_DIRS], staged[PRB_DIRS], items;
};

__host__ __device__ static inline uint32_t cdiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__host__ __device__ static inline void geo_make(Geo *g, uint32_t kind, uint32_t rows, uint32_t cr_max) {
    g->kind = kind; g->rows = rows; g->q = rows / PRB_WORLD;
    uint32_t nc = 4 * cdiv(cdiv(g->q, cr_max), 4);
    g->nc = nc < g->q ? nc : g->q;
    g->cr = cdiv(g->q, g->nc);
    g->na = g->nc / 4; g->nb = g->nc - g->na; g->nb1 = g->nb / 2; g->nh = g->nc / 2;
    if (kind == PRB_KIND_RS) {
        g->steps[PRB_FWD] = 3 * g->na + g->nb; g->steps[PRB_BWD] = 2 * g->nb;
        g->staged[PRB_FWD] = g->steps[PRB_FWD]; g->staged[PRB_BWD] = g->steps[PRB_BWD];
        g->items = 4 * g->nc;
    } else {
        g->steps[PRB_FWD] = g->nc + g->nh; g->steps[PRB_BWD] = 2 * g->nc - g->nh;
        g->staged[PRB_FWD] = g->nc; g->staged[PRB_BWD] = g->nc;
        g->items = 5 * g->nc;
    }
}

__host__ __device__ static inline void chunk_rows(const Geo &g, uint32_t k, uint32_t *c0, uint32_t *c1) {
    *c0 = (uint32_t)((uint64_t)k * g.q / g.nc);
    *c1 = (uint32_t)((uint64_t)(k + 1) * g.q / g.nc);
}

struct StepInfo { uint32_t chunk; int32_t src, blk_off, dep; };

// What a rank sends at step s of direction d (protocol.send_step).
__host__ __device__ static inline StepInfo send_step(const Geo &g, int d, uint32_t s) {
    StepInfo st;
    if (g.kind == PRB_KIND_RS) {
        if (d == PRB_FWD) {
            if (s < g.na) { st.chunk = s; st.src = PRB_SRC_RAW; st.blk_off = -1; st.dep = -1; return st; }
            s -= g.na;
            if (s < g.nb1) { st.chunk = g.na + s; st.src = PRB_SRC_RAW; st.blk_off = 1; st.dep = -1; return st; }
            s -= g.nb1;
            if (s < g.na) { st.chunk = s; st.src = PRB_SRC_RED; st.blk_off = -2; st.dep = (int32_t)s; return st; }
            s -= g.na;
            if (s < g.nb - g.nb1) { st.chunk = g.na + g.nb1 + s; st.src = PRB_SRC_RAW; st.blk_off = 1; st.dep = -1; return st; }
            s -= g.nb - g.nb1;
            st.chunk = s; st.src = PRB_SRC_RED; st.blk_off = 1; st.dep = (int32_t)(g.na + g.nb1 + s); return st;
        }
        if (s < g.nb) { st.chunk = g.na + s; st.src = PRB_SRC_RAW; st.blk_off = -2; st.dep = -1; return st; }
        s -= g.nb;
        st.chunk = g.na + s; st.src = PRB_SRC_RED; st.blk_off = -1; st.dep = (int32_t)s; return st;
    }
    st.blk_off = 0;
    if (d == PRB_FWD) {
        if (s < g.nc) { st.chunk = s; st.src = PRB_SRC_RAW; st.dep = -1; return st; }
        s -= g.nc;
        st.chunk = s; st.src = PRB_SRC_RELAY; st.dep = (int32_t)s; return st;
    }
    if (s < g.nc - g.nh) { st.chunk = g.nh + s; st.src = PRB_SRC_RAW; st.dep = -1; return st; }
    s -= g.nc - g.nh;
    if (s < g.nh) { st.chunk = s; st.src = PRB_SRC_RAW; st.dep = -1; return st; }
    s -= g.nh;
    st.chunk = g.nh + s; st.src = PRB_SRC_RELAY; st.dep = (int32_t)s; return st;
}

__host__ __device__ static inline uint32_t fb_step(const Geo &g, uint32_t c) { return c < g.nb1 ? g.na + c : 2 * g.na + c; }
__host__ __device__ static inline uint32_t out_next_chunk(const Geo &g, uint32_t s) {
    return s < g.nc - g.nh ? g.nh + s : s - (g.nc - g.nh);
}

struct Streams { uint32_t n, kind[6], start[6], count[6]; };

__host__ __device__ static inline void make_streams(const Geo &g, Streams *S) {
    if (g.kind == PRB_KIND_RS) {
        const uint32_t k[4] = {IT_STAGE_F, IT_STAGE_B, IT_OUT_B, IT_OUT_A};
        const uint32_t st[4] = {0, 0, g.nb + 1, 2 * g.na + g.nb + 1};
        const uint32_t ct[4] = {g.steps[PRB_FWD], g.steps[PRB_BWD], g.nb, g.na};
        S->n = 4;
        for (int i = 0; i < 4; i++) { S->kind[i] = k[i]; S->start[i] = st[i]; S->count[i] = ct[i]; }
    } else {
        const uint32_t k[6] = {IT_STAGE_F, IT_STAGE_B, IT_OUT_PREV, IT_OUT_NEXT, IT_OUT_FARF, IT_OUT_FARB};
        const uint32_t st[6] = {0, 0, 1, 1, g.nc + 1, g.nc + 1};
        const uint32_t ct[6] = {g.nc, g.nc, g.nc, g.nc, g.nh, g.nc - g.nh};
        S->n = 6;
        for (int i = 0; i < 6; i++) { S->kind[i] = k[i]; S->start[i] = st[i]; S->count[i] = ct[i]; }
    }
}

__host__ __device__ static inline uint32_t items_before(const Streams &S, uint32_t t) {
    uint32_t n = 0;
    for (uint32_t k = 0; k < S.n; k++)
        if (t > S.start[k]) n += (t - S.start[k] < S.count[k]) ? t - S.start[k] : S.count[k];
    return n;
}

// Item i of the step-merged work list -> (item kind, index within its stream) (protocol.item_at).
__host__ __device__ static inline void item_at(const Streams &S, uint32_t i, uint32_t *kind, uint32_t *j, uint32_t *step) {
    uint32_t lo = 0, hi = 0;
    for (uint32_t k = 0; k < S.n; k++) if (S.start[k] + S.count[k] > hi) hi = S.start[k] + S.count[k];
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) / 2;
        if (items_before(S, mid) <= i) lo = mid; else hi = mid;
    }
    uint32_t r = i - items_before(S, lo);
    *kind = 0xFFFFFFFFu; *j = 0; *step = lo;
    for (uint32_t k = 0; k < S.n; k++) {
        if (S.start[k] <= lo && lo < S.start[k] + S.count[k]) {
            if (r == 0) { *kind = S.kind[k]; *j = lo - S.start[k]; return; }
            r--;
        }
    }
}

// ------------------------------------------------------------------------------------------------ layout
struct prb_layout {
    uint64_t max_rows, cr_max, nslot, nc_max, steps_cap, chunk_bytes, dir_recv_bytes;
    uint64_t rflag_off, stage_off, ready_off, ctrl_off, total;
};

static int make_layout(prb_layout *L, uint64_t max_rows, uint64_t cr_max, uint64_t nslot) {
    if (max_rows == 0 || max_rows % PRB_WORLD || max_rows > (1u << 16) || cr_max < 1 || cr_max > 1024 || nslot < 2 ||
        nslot > 4096)
        return -1;
    L->max_rows = max_rows; L->cr_max = cr_max; L->nslot = nslot;
    uint64_t q = max_rows / PRB_WORLD;
    L->nc_max = 4 * ((((q + cr_max - 1) / cr_max) + 3) / 4);
    L->steps_cap = (3 * L->nc_max) / 2 + 2;
    L->chunk_bytes = cr_max * PRB_ROW_BYTES;
    L->dir_recv_bytes = L->steps_cap * L->chunk_bytes;
    L->rflag_off = 2ull * PRB_DIRS * L->dir_recv_bytes;
    L->stage_off = L->rflag_off + 2ull * PRB_DIRS * L->steps_cap * PRB_FLAG_STRIDE;
    L->ready_off = L->stage_off + PRB_DIRS * nslot * L->chunk_bytes;
    L->ctrl_off = L->ready_off + PRB_DIRS * nslot * PRB_FLAG_STRIDE;
    uint64_t raw = L->ctrl_off + 4096;
    L->total = (raw + (2ull << 20) - 1) / (2ull << 20) * (2ull << 20);
    return 0;
}

__host__ __device__ static inline uint64_t recv_off(uint64_t dir_recv_bytes, uint32_t parity, uint32_t d) {
    return (uint64_t)(parity * PRB_DIRS + d) * dir_recv_bytes;
}
__host__ __device__ static inline uint64_t rflag_at(uint64_t rflag_off, uint64_t steps_cap, uint32_t parity, uint32_t d, uint32_t s) {
    return rflag_off + ((uint64_t)(parity * PRB_DIRS + d) * steps_cap + s) * PRB_FLAG_STRIDE;
}

// ------------------------------------------------------------------------------------------------ device side
struct KArgs {
    const uint4 *in;
    uint4 *out;
    uint8_t *region;
    unsigned long long *work;
    unsigned long long *cnt;                 // part counters [dir][nslot], device memory
    uint64_t rflag_off, stage_off, ready_off, ctrl_off, dir_recv_bytes, chunk_bytes, steps_cap;
    uint64_t stage_base[PRB_DIRS], work_base, timeout_ns;
    uint32_t nslot, parts, rank, seq, parity;
    Geo g;
};

__device__ __forceinline__ uint64_t gtimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ uint32_t ld_acq_u32(const void *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long ld_acq_u64(const void *p) {
    unsigned long long v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_rel_u32(void *p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_rel_u64(void *p, unsigned long long v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_sys_v4(const uint4 *p) {
    uint4 v;
    // no "memory" clobber: ordering after the flag comes from thread 0's acquire + __syncthreads, and leaving the
    // compiler free to batch these loads keeps several 16-byte loads in flight per thread
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 add_bf16x8(uint4 a, uint4 b) {
    // element-wise FP32 add of two BF16 values, one round-to-nearest-even back to BF16: a correctly rounded BF16 add
    // (NCCL's __hadd2 on bf16x2); addition is commutative, so operand order is moot
    uint4 z;
    const __nv_bfloat162 *x = reinterpret_cast<const __nv_bfloat162 *>(&a);
    const __nv_bfloat162 *y = reinterpret_cast<const __nv_bfloat162 *>(&b);
    __nv_bfloat162 *o = reinterpret_cast<__nv_bfloat162 *>(&z);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float2 fx = __bfloat1622float2(x[j]), fy = __bfloat1622float2(y[j]);
        o[j] = __floats2bfloat162_rn(fx.x + fy.x, fx.y + fy.y);
    }
    return z;
}

// Wait helpers: thread 0 spins. On timeout the runtime is poisoned (seq, where) and false is returned; every CTA stops
// taking work at its next fetch.
__device__ bool wait_eq_u32(const KArgs &a, const void *flag, uint32_t want, uint32_t where) {
    uint64_t t_end = gtimer() + a.timeout_ns;
    for (;;) {
        if (ld_acq_u32(flag) == want) return true;
        if (ld_acq_u32(a.region + a.ctrl_off + 128) != 0) return false;
        if (gtimer() > t_end) {
            st_rel_u32(a.region + a.ctrl_off + 132, where);
            st_rel_u32(a.region + a.ctrl_off + 128, a.seq);
            return false;
        }
        __nanosleep(64);
    }
}
__device__ bool wait_ge_u64(const KArgs &a, const void *ctr, unsigned long long want, uint32_t where) {
    uint64_t t_end = gtimer() + a.timeout_ns;
    for (;;) {
        if (ld_acq_u64(ctr) >= want) return true;
        if (ld_acq_u32(a.region + a.ctrl_off + 128) != 0) return false;
        if (gtimer() > t_end) {
            st_rel_u32(a.region + a.ctrl_off + 132, where);
            st_rel_u32(a.region + a.ctrl_off + 128, a.seq);
            return false;
        }
        __nanosleep(64);
    }
}

__device__ __forceinline__ const uint4 *recv_slot(const KArgs &a, uint32_t d, uint32_t s) {
    return reinterpret_cast<const uint4 *>(a.region + recv_off(a.dir_recv_bytes, a.parity, d) +
                                           (uint64_t)s * a.g.cr * PRB_ROW_BYTES);
}

__global__ void __launch_bounds__(PRB_THREADS) prb_kernel(const KArgs a) {
    __shared__ unsigned long long s_unit;
    __shared__ uint32_t s_kind, s_j, s_ok;
    const uint32_t tid = threadIdx.x;
    const Geo &g = a.g;
    const unsigned long long n_units = (unsigned long long)g.items * a.parts;
    uint8_t *const ctrl = a.region + a.ctrl_off;
    for (;;) {
        if (tid == 0) {
            unsigned long long u = atomicAdd(a.work, 1ull) - a.work_base;
            if (ld_acq_u32(ctrl + 128) != 0) u = n_units;
            s_unit = u;
            s_ok = 1;
            if (u < n_units) {
                Streams S;
                make_streams(g, &S);
                uint32_t kind, j, step;
                item_at(S, (uint32_t)(u / a.parts), &kind, &j, &step);
                s_kind = kind; s_j = j;
                if (kind > IT_OUT_FARB) {                        // merge arithmetic broken: never guess
                    st_rel_u32(ctrl + 132, 0x500u);
                    st_rel_u32(ctrl + 128, a.seq);
                    s_ok = 0;
                }
                // incoming chunks this item consumes
                uint32_t dd[2], ss[2], nd = 0;
                if (kind == IT_STAGE_F || kind == IT_STAGE_B) {
                    const int d = kind == IT_STAGE_F ? PRB_FWD : PRB_BWD;
                    StepInfo st = send_step(g, d, j);
                    if (st.src == PRB_SRC_RED) { dd[nd] = d; ss[nd++] = (uint32_t)st.dep; }
                } else if (kind == IT_OUT_A) { dd[nd] = PRB_FWD; ss[nd++] = 2 * g.na + g.nb + j; }
                else if (kind == IT_OUT_B) { dd[nd] = PRB_BWD; ss[nd++] = g.nb + j; dd[nd] = PRB_FWD; ss[nd++] = fb_step(g, j); }
                else if (kind == IT_OUT_PREV) { dd[nd] = PRB_FWD; ss[nd++] = j; }
                else if (kind == IT_OUT_NEXT) { dd[nd] = PRB_BWD; ss[nd++] = j; }
                else if (kind == IT_OUT_FARF) { dd[nd] = PRB_FWD; ss[nd++] = g.nc + j; }
                else if (kind == IT_OUT_FARB) { dd[nd] = PRB_BWD; ss[nd++] = g.nc + j; }
                for (uint32_t k = 0; s_ok && k < nd; k++) {
                    const uint8_t *f = a.region + rflag_at(a.rflag_off, a.steps_cap, a.parity, dd[k], ss[k]);
                    if (!wait_eq_u32(a, f, a.seq, 0x100u | (kind << 4) | k)) { s_ok = 0; break; }
                }
                if (s_ok && (kind == IT_STAGE_F || kind == IT_STAGE_B)) {
                    const int d = kind == IT_STAGE_F ? PRB_FWD : PRB_BWD;
                    const uint64_t gi = a.stage_base[d] + j;
                    if (gi >= a.nslot &&
                        !wait_ge_u64(a, ctrl + d * PRB_FLAG_STRIDE, (unsigned long long)(gi - a.nslot + 1), 0x300u | d))
                        s_ok = 0;
                }
            }
        }
        __syncthreads();
        const unsigned long long u = s_unit;
        if (u >= n_units) break;
        if (!s_ok) break;
        const uint32_t kind = s_kind, j = s_j, part = (uint32_t)(u % a.parts);
        const uint32_t rank = a.rank, q = g.q;
        uint32_t chunk, c0, c1;
        int d = -1;
        StepInfo st;
        st.src = PRB_SRC_RAW; st.blk_off = 0; st.dep = -1; st.chunk = 0;
        if (kind == IT_STAGE_F || kind == IT_STAGE_B) {
            d = kind == IT_STAGE_F ? PRB_FWD : PRB_BWD;
            st = send_step(g, d, j);
            chunk = st.chunk;
        } else if (kind == IT_OUT_A || kind == IT_OUT_PREV || kind == IT_OUT_FARF) chunk = j;
        else if (kind == IT_OUT_B) chunk = g.na + j;
        else if (kind == IT_OUT_NEXT) chunk = out_next_chunk(g, j);
        else chunk = g.nh + j;                                  // IT_OUT_FARB
        chunk_rows(g, chunk, &c0, &c1);
        const uint32_t packs = (c1 - c0) * PRB_ROW_PACKS;
        const uint32_t per = (packs + a.parts - 1) / a.parts;
        const uint32_t p0 = min(part * per, packs), p1 = min(p0 + per, packs);
        uint64_t gi = 0;
        if (d >= 0) {
            gi = a.stage_base[d] + j;
            uint4 *dst = reinterpret_cast<uint4 *>(a.region + a.stage_off + ((uint64_t)d * a.nslot + gi % a.nslot) * a.chunk_bytes);
            const uint32_t blk = g.kind == PRB_KIND_RS ? (uint32_t)((int)rank + 4 + st.blk_off) & 3u : 0u;
            const uint4 *src = a.in + ((uint64_t)blk * (g.kind == PRB_KIND_RS ? q : 0u) + c0) * PRB_ROW_PACKS;
            if (st.src == PRB_SRC_RED) {
                const uint4 *recv = recv_slot(a, (uint32_t)d, (uint32_t)st.dep);
#pragma unroll 4
                for (uint32_t i = p0 + tid; i < p1; i += PRB_THREADS) dst[i] = add_bf16x8(ld_sys_v4(recv + i), __ldg(src + i));
            } else if (g.kind == PRB_KIND_AG && d == PRB_FWD) {
                uint4 *own = a.out + ((uint64_t)rank * q + c0) * PRB_ROW_PACKS;
#pragma unroll 4
                for (uint32_t i = p0 + tid; i < p1; i += PRB_THREADS) {
                    uint4 v = __ldg(src + i);
                    dst[i] = v;
                    own[i] = v;
                }
            } else {
#pragma unroll 4
                for (uint32_t i = p0 + tid; i < p1; i += PRB_THREADS) dst[i] = __ldg(src + i);
            }
        } else if (kind == IT_OUT_A) {
            const uint4 *recv = recv_slot(a, PRB_FWD, 2 * g.na + g.nb + j);
            const uint4 *mine = a.in + ((uint64_t)rank * q + c0) * PRB_ROW_PACKS;
            uint4 *dst = a.out + (uint64_t)c0 * PRB_ROW_PACKS;
#pragma unroll 4
            for (uint32_t i = p0 + tid; i < p1; i += PRB_THREADS) dst[i] = add_bf16x8(ld_sys_v4(recv + i), __ldg(mine + i));
        } else if (kind == IT_OUT_B) {
            const uint4 *bb2 = recv_slot(a, PRB_BWD, g.nb + j);
            const uint4 *fb = recv_slot(a, PRB_FWD, fb_step(g, j));
            const uint4 *mine = a.in + ((uint64_t)rank * q + c0) * PRB_ROW_PACKS;
            uint4 *dst = a.out + (uint64_t)c0 * PRB_ROW_PACKS;
            // ((x[j+1] + x[j+2]) + x[j+3]) + x[j]: bb2 carries x[j+1] + x[j+2], fb carries x[j+3]
#pragma unroll 4
            for (uint32_t i = p0 + tid; i < p1; i += PRB_THREADS)
                dst[i] = add_bf16x8(add_bf16x8(ld_sys_v4(bb2 + i), ld_sys_v4(fb + i)), __ldg(mine + i));
        } else {
            uint32_t dd, ss, blk;
            if (kind == IT_OUT_PREV) { dd = PRB_FWD; ss = j; blk = (rank + 3) & 3u; }
            else if (kind == IT_OUT_NEXT) { dd = PRB_BWD; ss = j; blk = (rank + 1) & 3u; }
            else if (kind == IT_OUT_FARF) { dd = PRB_FWD; ss = g.nc + j; blk = (rank + 2) & 3u; }
            else { dd = PRB_BWD; ss = g.nc + j; blk = (rank + 2) & 3u; }
            const uint4 *recv = recv_slot(a, dd, ss);
            uint4 *dst = a.out + ((uint64_t)blk * q + c0) * PRB_ROW_PACKS;
#pragma unroll 4
            for (uint32_t i = p0 + tid; i < p1; i += PRB_THREADS) dst[i] = ld_sys_v4(recv + i);
        }
        __syncthreads();
        if (d >= 0 && tid == 0) {
            // every part fences its writes, the part that completes the chunk publishes it for the proxy
            __threadfence_system();
            const uint64_t slot = gi % a.nslot;
            unsigned long long old = atomicAdd(a.cnt + (uint64_t)d * a.nslot + slot, 1ull);
            if ((old + 1) % a.parts == 0) {
                __threadfence_system();
                st_rel_u64(a.region + a.ready_off + ((uint64_t)d * a.nslot + slot) * PRB_FLAG_STRIDE,
                           (unsigned long long)(gi + 1));
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ host side
struct prb_call {
    uint32_t seq, parity;
    Geo g;
    uint64_t stage_base[PRB_DIRS];
};

struct prb_ctx {
    int rank, next, prev, gid_index, cpu, grid, mem_kind;
    uint32_t parts;
    prb_layout L;
    uint8_t *region;
    void *map_base;
    size_t map_len;
    unsigned long long *d_work, *d_cnt;
    uint64_t timeout_ns;
    // verbs: one QP per HCA; qp_send[d] on send_hca(d), qp_recv[d] (passive) on recv_hca(d)
    struct ibv_context *ctx[PRB_NHCA];
    struct ibv_pd *pd[PRB_NHCA];
    struct ibv_mr *mr[PRB_NHCA];
    struct ibv_cq *cq[PRB_NHCA];
    union ibv_gid gid[PRB_NHCA];
    enum ibv_mtu mtu[PRB_NHCA];
    struct ibv_qp *qp_send[PRB_DIRS], *qp_recv[PRB_DIRS];
    int h_send[PRB_DIRS], h_recv[PRB_DIRS];
    uint64_t peer_region[PRB_DIRS];
    uint32_t peer_rkey[PRB_DIRS];
    // host sequencing (caller thread)
    pthread_mutex_t mu;
    uint32_t seq;
    uint64_t stage_next[PRB_DIRS], work_next;
    // host -> proxy queue: tail written by the caller, one head per direction written by the proxy
    prb_call queue[PRB_QCAP];
    uint64_t q_tail, q_head[PRB_DIRS];
    // proxy
    pthread_t thread;
    int running, stopping, failed, started;
    char err[512];
    int64_t ring[PRB_DIRS][PRB_ITEM_RING];
    uint64_t posted[PRB_DIRS], done[PRB_DIRS];
    uint64_t stat_calls, stat_bytes[PRB_DIRS], stat_idle_naps, stat_relays;
};

struct prb_blob {
    uint32_t abi, rank;
    uint64_t region;
    uint32_t rkey[PRB_NHCA];
    uint8_t gid[PRB_NHCA][16];
    uint32_t mtu[PRB_NHCA];
    uint32_t qpn_send[PRB_DIRS], qpn_recv[PRB_DIRS];
    uint64_t layout_total, max_rows, cr_max, nslot, parts;
};

static void set_err(prb_ctx *c, const char *fmt, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), fmt, what, e ? strerror(e) : "failed");
}

extern "C" int prb_abi(void) { return PRB_ABI; }
extern "C" uint64_t prb_blob_bytes(void) { return sizeof(prb_blob); }

extern "C" int prb_layout_query(uint64_t max_rows, uint64_t cr_max, uint64_t nslot, uint64_t *out) {
    prb_layout L;
    if (make_layout(&L, max_rows, cr_max, nslot)) return -1;
    uint64_t v[12] = {L.max_rows, L.cr_max, L.nslot, L.nc_max, L.steps_cap, L.chunk_bytes, L.dir_recv_bytes,
                      L.rflag_off, L.stage_off, L.ready_off, L.ctrl_off, L.total};
    memcpy(out, v, sizeof(v));
    return 0;
}

// The host build of the device arithmetic, for cross-checks against protocol.py:
// out = q, cr, nc, na, nb1, nh, steps F, steps B, staged F, staged B, items
extern "C" int prb_geometry(int kind, uint32_t rows, uint32_t cr_max, uint32_t *out) {
    if ((kind != PRB_KIND_RS && kind != PRB_KIND_AG) || rows == 0 || rows % PRB_WORLD || cr_max < 1) return -1;
    Geo g;
    geo_make(&g, (uint32_t)kind, rows, cr_max);
    uint32_t v[11] = {g.q, g.cr, g.nc, g.na, g.nb1, g.nh, g.steps[0], g.steps[1], g.staged[0], g.staged[1], g.items};
    memcpy(out, v, sizeof(v));
    return 0;
}

// Item i of the merged work list -> (kind, j, step); send step s of direction d -> (chunk, src, blk_off, dep).
extern "C" int prb_item_at(int kind, uint32_t rows, uint32_t cr_max, uint32_t i, uint32_t *out) {
    Geo g;
    if (prb_geometry(kind, rows, cr_max, out)) return -1;
    geo_make(&g, (uint32_t)kind, rows, cr_max);
    if (i >= g.items) return -1;
    Streams S;
    make_streams(g, &S);
    item_at(S, i, &out[0], &out[1], &out[2]);
    return 0;
}
extern "C" int prb_send_step(int kind, uint32_t rows, uint32_t cr_max, int d, uint32_t s, int32_t *out) {
    Geo g;
    uint32_t tmp[11];
    if (prb_geometry(kind, rows, cr_max, tmp) || d < 0 || d > 1) return -1;
    geo_make(&g, (uint32_t)kind, rows, cr_max);
    if (s >= g.steps[d]) return -1;
    StepInfo st = send_step(g, d, s);
    out[0] = (int32_t)st.chunk; out[1] = st.src; out[2] = st.blk_off; out[3] = st.dep;
    return 0;
}

static struct ibv_qp *make_qp(prb_ctx *c, int h) {
    struct ibv_qp_init_attr at;
    memset(&at, 0, sizeof(at));
    at.send_cq = c->cq[h]; at.recv_cq = c->cq[h]; at.qp_type = IBV_QPT_RC;
    at.cap.max_send_wr = 2 * PRB_ITEM_RING + 16; at.cap.max_recv_wr = 1;
    at.cap.max_send_sge = 1; at.cap.max_recv_sge = 1; at.cap.max_inline_data = 16;
    struct ibv_qp *qp = ibv_create_qp(c->pd[h], &at);
    if (!qp) return NULL;
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT; a.pkey_index = 0; a.port_num = PRB_PORT;
    a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    if (ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        ibv_destroy_qp(qp);
        return NULL;
    }
    return qp;
}

extern "C" void prb_destroy(prb_ctx *c);

static int alloc_region(prb_ctx *c, char *err, uint64_t err_len) {
    const uint64_t total = c->L.total;
    if (c->mem_kind == PRB_MEM_PINNED) {
        void *p = NULL;
        cudaError_t ce = cudaHostAlloc(&p, total, cudaHostAllocDefault);
        if (ce != cudaSuccess) { snprintf(err, err_len, "cudaHostAlloc(%llu): %s", (unsigned long long)total, cudaGetErrorString(ce)); return -1; }
        void *dp = NULL;
        if (cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess || dp != p) {
            snprintf(err, err_len, "registered region needs identical host and device addresses");
            cudaFreeHost(p);
            return -1;
        }
        c->region = (uint8_t *)p;
    } else if (c->mem_kind == PRB_MEM_HOSTTHP) {
        int dev = 0, pma = 0, ats = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev) != cudaSuccess || !pma || !ats) {
            snprintf(err, err_len, "hostthp memory needs pageable memory access through the host page tables (ATS)");
            return -1;
        }
        const size_t align = 2u << 20;
        size_t len = total + align;
        void *m = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) { snprintf(err, err_len, "mmap(%zu): %s", len, strerror(errno)); return -1; }
        uint8_t *p = (uint8_t *)(((uintptr_t)m + align - 1) & ~(uintptr_t)(align - 1));
        if (madvise(p, total, MADV_HUGEPAGE)) {
            snprintf(err, err_len, "madvise(MADV_HUGEPAGE): %s", strerror(errno));
            munmap(m, len);
            return -1;
        }
        c->map_base = m; c->map_len = len; c->region = p;
    } else {
        snprintf(err, err_len, "unknown memory kind %d", c->mem_kind);
        return -1;
    }
    memset(c->region, 0, total);
    return 0;
}

extern "C" prb_ctx *prb_create(int rank, uint64_t max_rows, uint64_t cr_max, uint64_t nslot, uint32_t parts,
                               int gid_index, int grid, int cpu, uint64_t timeout_ns, int mem_kind, char *err,
                               uint64_t err_len) {
    prb_ctx *c = (prb_ctx *)calloc(1, sizeof(prb_ctx));
    if (!c) { snprintf(err, err_len, "out of memory"); return NULL; }
    if (rank < 0 || rank >= PRB_WORLD || make_layout(&c->L, max_rows, cr_max, nslot) || grid < 1 || grid > 256 ||
        parts < 1 || parts > 64) {
        snprintf(err, err_len, "invalid prefill ring geometry");
        free(c);
        return NULL;
    }
    c->rank = rank; c->next = (rank + 1) % PRB_WORLD; c->prev = (rank + 3) % PRB_WORLD;
    c->gid_index = gid_index; c->cpu = cpu; c->grid = grid; c->timeout_ns = timeout_ns; c->parts = parts;
    c->mem_kind = mem_kind;
    pthread_mutex_init(&c->mu, NULL);
    for (int d = 0; d < PRB_DIRS; d++) { c->h_send[d] = send_hca(rank, d); c->h_recv[d] = recv_hca(rank, d); }
    if (alloc_region(c, err, err_len)) { free(c); return NULL; }
    const size_t cnt_bytes = PRB_DIRS * nslot * sizeof(unsigned long long);
    if (cudaMalloc(&c->d_work, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMemset(c->d_work, 0, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&c->d_cnt, cnt_bytes) != cudaSuccess || cudaMemset(c->d_cnt, 0, cnt_bytes) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        snprintf(err, err_len, "device counter allocation failed");
        prb_destroy(c); return NULL;
    }
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (!list) { snprintf(err, err_len, "ibv_get_device_list: %s", strerror(errno)); prb_destroy(c); return NULL; }
    for (int h = 0; h < PRB_NHCA; h++) {
        struct ibv_device *dev = NULL;
        for (int i = 0; i < num; i++) if (!strcmp(ibv_get_device_name(list[i]), HCA_NAMES[h])) dev = list[i];
        if (!dev) { snprintf(err, err_len, "RDMA device %s not found", HCA_NAMES[h]); ibv_free_device_list(list); prb_destroy(c); return NULL; }
        c->ctx[h] = ibv_open_device(dev);
        struct ibv_port_attr pa;
        if (!c->ctx[h] || ibv_query_port(c->ctx[h], PRB_PORT, &pa) || pa.state != IBV_PORT_ACTIVE ||
            ibv_query_gid(c->ctx[h], PRB_PORT, gid_index, &c->gid[h])) {
            snprintf(err, err_len, "RDMA device %s unusable (open/port/gid)", HCA_NAMES[h]);
            ibv_free_device_list(list); prb_destroy(c); return NULL;
        }
        c->mtu[h] = pa.active_mtu;
        c->pd[h] = ibv_alloc_pd(c->ctx[h]);
        c->cq[h] = c->pd[h] ? ibv_create_cq(c->ctx[h], 2 * PRB_ITEM_RING + 64, NULL, NULL, 0) : NULL;
        c->mr[h] = c->cq[h] ? ibv_reg_mr(c->pd[h], c->region, c->L.total, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE) : NULL;
        if (!c->mr[h]) {
            snprintf(err, err_len, "pd/cq/ibv_reg_mr on %s: %s", HCA_NAMES[h], strerror(errno));
            ibv_free_device_list(list); prb_destroy(c); return NULL;
        }
    }
    ibv_free_device_list(list);
    for (int d = 0; d < PRB_DIRS; d++) {
        c->qp_send[d] = make_qp(c, c->h_send[d]);
        c->qp_recv[d] = make_qp(c, c->h_recv[d]);
        if (!c->qp_send[d] || !c->qp_recv[d]) { snprintf(err, err_len, "ibv_create_qp/INIT: %s", strerror(errno)); prb_destroy(c); return NULL; }
    }
    return c;
}

extern "C" int prb_local_blob(prb_ctx *c, void *out, uint64_t n) {
    if (n < sizeof(prb_blob)) return -1;
    prb_blob b;
    memset(&b, 0, sizeof(b));
    b.abi = PRB_ABI; b.rank = (uint32_t)c->rank; b.region = (uint64_t)(uintptr_t)c->region;
    for (int h = 0; h < PRB_NHCA; h++) {
        b.rkey[h] = c->mr[h]->rkey;
        memcpy(b.gid[h], c->gid[h].raw, 16);
        b.mtu[h] = (uint32_t)c->mtu[h];
    }
    for (int d = 0; d < PRB_DIRS; d++) { b.qpn_send[d] = c->qp_send[d]->qp_num; b.qpn_recv[d] = c->qp_recv[d]->qp_num; }
    b.layout_total = c->L.total; b.max_rows = c->L.max_rows; b.cr_max = c->L.cr_max; b.nslot = c->L.nslot;
    b.parts = c->parts;
    memcpy(out, &b, sizeof(b));
    return 0;
}

static int connect_one(prb_ctx *c, struct ibv_qp *qp, int h, const prb_blob *peer, uint32_t remote_qpn) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = peer->mtu[h] < (uint32_t)c->mtu[h] ? (enum ibv_mtu)peer->mtu[h] : c->mtu[h];
    a.dest_qp_num = remote_qpn; a.rq_psn = 0; a.max_dest_rd_atomic = 1; a.min_rnr_timer = 12;
    a.ah_attr.is_global = 1; a.ah_attr.port_num = PRB_PORT;
    memcpy(a.ah_attr.grh.dgid.raw, peer->gid[h], 16);
    a.ah_attr.grh.sgid_index = (uint8_t)c->gid_index; a.ah_attr.grh.hop_limit = 64;
    int rc = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                   IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc) { set_err(c, "%s: %s", "ibv_modify_qp(RTR)", rc); return -1; }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS; a.timeout = 14; a.retry_cnt = 7; a.rnr_retry = 7; a.sq_psn = 0; a.max_rd_atomic = 1;
    rc = ibv_modify_qp(qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                               IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc) { set_err(c, "%s: %s", "ibv_modify_qp(RTS)", rc); return -1; }
    return 0;
}

// blobs: next's then prev's
extern "C" int prb_connect(prb_ctx *c, const void *next_blob, const void *prev_blob) {
    const prb_blob *n = (const prb_blob *)next_blob, *p = (const prb_blob *)prev_blob;
    const prb_blob *peers[2] = {n, p};
    for (int i = 0; i < 2; i++) {
        const prb_blob *b = peers[i];
        if (b->abi != PRB_ABI || b->layout_total != c->L.total || b->max_rows != c->L.max_rows ||
            b->cr_max != c->L.cr_max || b->nslot != c->L.nslot || b->parts != c->parts) {
            snprintf(c->err, sizeof(c->err), "neighbour published an incompatible prefill ring record");
            return -1;
        }
    }
    if ((int)n->rank != c->next || (int)p->rank != c->prev) {
        snprintf(c->err, sizeof(c->err), "neighbour records out of ring order");
        return -1;
    }
    // forward: our send QP -> next's forward receive QP; backward: our send QP -> prev's backward receive QP. The same
    // function index at both ends of a cable, so the local HCA index also names the remote one.
    c->peer_region[PRB_FWD] = n->region; c->peer_rkey[PRB_FWD] = n->rkey[c->h_send[PRB_FWD]];
    c->peer_region[PRB_BWD] = p->region; c->peer_rkey[PRB_BWD] = p->rkey[c->h_send[PRB_BWD]];
    if (connect_one(c, c->qp_send[PRB_FWD], c->h_send[PRB_FWD], n, n->qpn_recv[PRB_FWD])) return -1;
    if (connect_one(c, c->qp_send[PRB_BWD], c->h_send[PRB_BWD], p, p->qpn_recv[PRB_BWD])) return -1;
    if (connect_one(c, c->qp_recv[PRB_FWD], c->h_recv[PRB_FWD], p, p->qpn_send[PRB_FWD])) return -1;
    if (connect_one(c, c->qp_recv[PRB_BWD], c->h_recv[PRB_BWD], n, n->qpn_send[PRB_BWD])) return -1;
    return 0;
}

static inline uint64_t load_acq_u64(const void *p) { return __atomic_load_n((const uint64_t *)p, __ATOMIC_ACQUIRE); }
static inline uint32_t load_acq_u32(const void *p) { return __atomic_load_n((const uint32_t *)p, __ATOMIC_ACQUIRE); }

static void fail(prb_ctx *c, const char *msg) {
    if (!c->err[0]) snprintf(c->err, sizeof(c->err), "%s", msg);
    __atomic_store_n(&c->failed, 1, __ATOMIC_RELEASE);
}

static int drain(prb_ctx *c) {
    struct ibv_wc wc[64];
    for (int d = 0; d < PRB_DIRS; d++) {
        int n = ibv_poll_cq(c->cq[c->h_send[d]], 64, wc);
        if (n < 0) { fail(c, "ibv_poll_cq failed"); return -1; }
        uint64_t send_done = 0;
        int any = 0;
        for (int k = 0; k < n; k++) {
            if (wc[k].status != IBV_WC_SUCCESS) {
                char msg[256];
                snprintf(msg, sizeof(msg), "RDMA write %s (%s) failed: %s (vendor 0x%x)", d == PRB_FWD ? "forward" : "backward",
                         HCA_NAMES[c->h_send[d]], ibv_wc_status_str(wc[k].status), wc[k].vendor_err);
                fail(c, msg);
                return -1;
            }
            int64_t s = c->ring[d][c->done[d] % PRB_ITEM_RING];
            c->done[d]++;
            if (s >= 0) { send_done = (uint64_t)s + 1; any = 1; }
        }
        if (any) __atomic_store_n((uint64_t *)(c->region + c->L.ctrl_off + d * PRB_FLAG_STRIDE), send_done, __ATOMIC_RELEASE);
    }
    return 0;
}

// Post one step of direction d: `nbytes` from local address `laddr` to the neighbour's receive slot, then its flag.
static int post_step(prb_ctx *c, int d, uint64_t laddr, uint32_t nbytes, uint64_t rdata, uint64_t rflag, uint32_t seq,
                     int64_t staged) {
    // a lost completion ends in an RC retry error within about a second, which drain() reports
    while (c->posted[d] - c->done[d] >= PRB_ITEM_RING - 1)
        if (drain(c)) return -1;
    uint32_t seq_copy = seq;
    struct ibv_sge fs = {(uint64_t)(uintptr_t)&seq_copy, 4, 0};
    struct ibv_send_wr fw, dw, *bad = NULL;
    memset(&fw, 0, sizeof(fw));
    fw.wr_id = (uint64_t)d; fw.sg_list = &fs; fw.num_sge = 1; fw.opcode = IBV_WR_RDMA_WRITE;
    fw.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    fw.wr.rdma.remote_addr = c->peer_region[d] + rflag; fw.wr.rdma.rkey = c->peer_rkey[d];
    struct ibv_send_wr *first = &fw;
    struct ibv_sge ds;
    if (nbytes) {
        ds.addr = laddr; ds.length = nbytes; ds.lkey = c->mr[c->h_send[d]]->lkey;
        memset(&dw, 0, sizeof(dw));
        dw.wr_id = (uint64_t)d; dw.sg_list = &ds; dw.num_sge = 1; dw.opcode = IBV_WR_RDMA_WRITE; dw.send_flags = 0;
        dw.wr.rdma.remote_addr = c->peer_region[d] + rdata; dw.wr.rdma.rkey = c->peer_rkey[d];
        dw.next = &fw;
        first = &dw;
        c->stat_bytes[d] += nbytes;
    }
    int rc = ibv_post_send(c->qp_send[d], first, &bad);
    if (rc) { set_err(c, "%s: %s", "ibv_post_send", rc); fail(c, c->err); return -1; }
    c->ring[d][c->posted[d] % PRB_ITEM_RING] = staged;
    c->posted[d]++;
    return 0;
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static void *proxy_main(void *arg) {
    prb_ctx *c = (prb_ctx *)arg;
    if (c->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(c->cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    const prb_layout *L = &c->L;
    prb_call cur[PRB_DIRS];
    int have[PRB_DIRS] = {0, 0};
    uint32_t step[PRB_DIRS] = {0, 0};
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    uint64_t stop_deadline = 0;
    for (;;) {
        if (__atomic_load_n(&c->stopping, __ATOMIC_ACQUIRE)) {
            // graceful stop: finish posting every queued call and wait for the completions (at most 2 s), so a rank
            // that closes right after its last kernel never strands data its neighbours still wait for
            const uint64_t t = now_ns();
            if (!stop_deadline) stop_deadline = t + 2000000000ull;
            const uint64_t tail = __atomic_load_n(&c->q_tail, __ATOMIC_ACQUIRE);
            const int quiet = !have[0] && !have[1] && c->q_head[0] == tail && c->q_head[1] == tail &&
                              c->posted[0] == c->done[0] && c->posted[1] == c->done[1];
            if (quiet || t > stop_deadline) break;
        }
        if (drain(c)) return NULL;
        int progressed = 0;
        for (int d = 0; d < PRB_DIRS; d++) {
            if (!have[d]) {
                uint64_t head = c->q_head[d], tail = __atomic_load_n(&c->q_tail, __ATOMIC_ACQUIRE);
                if (head == tail) continue;
                cur[d] = c->queue[head % PRB_QCAP];
                __atomic_store_n(&c->q_head[d], head + 1, __ATOMIC_RELEASE);
                have[d] = 1; step[d] = 0; progressed = 1;
                if (d == PRB_FWD) c->stat_calls++;
            }
            // post every step of the current call that is ready, in step order
            const prb_call &k = cur[d];
            while (have[d]) {
                const uint32_t s = step[d];
                if (s >= k.g.steps[d]) { have[d] = 0; break; }
                const StepInfo st = send_step(k.g, d, s);
                uint64_t laddr;
                int64_t staged = -1;
                if (s < k.g.staged[d]) {
                    const uint64_t gi = k.stage_base[d] + s;
                    const uint64_t slot = d * L->nslot + gi % L->nslot;
                    if (load_acq_u64(c->region + L->ready_off + slot * PRB_FLAG_STRIDE) != gi + 1) break;
                    laddr = (uint64_t)(uintptr_t)(c->region + L->stage_off + slot * L->chunk_bytes);
                    staged = (int64_t)gi;
                } else {
                    // relay: the incoming chunk of the same direction must have landed
                    if (load_acq_u32(c->region + rflag_at(L->rflag_off, L->steps_cap, k.parity, d, (uint32_t)st.dep)) != k.seq) break;
                    laddr = (uint64_t)(uintptr_t)(c->region + recv_off(L->dir_recv_bytes, k.parity, d) +
                                                  (uint64_t)st.dep * k.g.cr * PRB_ROW_BYTES);
                    c->stat_relays++;
                }
                uint32_t c0, c1;
                chunk_rows(k.g, st.chunk, &c0, &c1);
                const uint64_t rdata = recv_off(L->dir_recv_bytes, k.parity, d) + (uint64_t)s * k.g.cr * PRB_ROW_BYTES;
                const uint64_t rfl = rflag_at(L->rflag_off, L->steps_cap, k.parity, d, s);
                if (post_step(c, d, laddr, (c1 - c0) * PRB_ROW_BYTES, rdata, rfl, k.seq, staged)) return NULL;
                step[d]++;
                progressed = 1;
            }
        }
        if (progressed) {
            idle = 0;
        } else if (++idle > 2000000) {
            nanosleep(&nap, NULL);
            c->stat_idle_naps++;
        }
    }
    return NULL;
}

extern "C" int prb_start(prb_ctx *c) {
    if (c->started) return 0;
    __atomic_store_n(&c->running, 1, __ATOMIC_RELEASE);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc) { __atomic_store_n(&c->running, 0, __ATOMIC_RELEASE); set_err(c, "%s: %s", "pthread_create", rc); return -1; }
    c->started = 1;
    return 0;
}

// Enqueue one collective: the call descriptor for the proxy, then the kernel on `stream`.
extern "C" int prb_launch(prb_ctx *c, int kind, const void *in, void *out, uint64_t rows, void *stream) {
    if ((kind != PRB_KIND_RS && kind != PRB_KIND_AG) || rows == 0 || rows % PRB_WORLD || rows > c->L.max_rows ||
        ((uintptr_t)in % 16) || ((uintptr_t)out % 16)) {
        snprintf(c->err, sizeof(c->err), "unsupported prefill ring call (kind %d rows %llu)", kind, (unsigned long long)rows);
        return -1;
    }
    if (__atomic_load_n(&c->failed, __ATOMIC_ACQUIRE) || load_acq_u32(c->region + c->L.ctrl_off + 128)) return -2;
    pthread_mutex_lock(&c->mu);
    prb_call call;
    memset(&call, 0, sizeof(call));
    geo_make(&call.g, (uint32_t)kind, (uint32_t)rows, (uint32_t)c->L.cr_max);
    if (call.g.steps[0] > c->L.steps_cap || call.g.steps[1] > c->L.steps_cap || call.g.cr > c->L.cr_max) {
        pthread_mutex_unlock(&c->mu);
        snprintf(c->err, sizeof(c->err), "call geometry exceeds the layout");
        return -1;
    }
    call.seq = ++c->seq;
    call.parity = call.seq & 1u;
    call.stage_base[0] = c->stage_next[0];
    call.stage_base[1] = c->stage_next[1];
    uint64_t tail = c->q_tail;
    for (;;) {
        uint64_t h0 = __atomic_load_n(&c->q_head[0], __ATOMIC_ACQUIRE), h1 = __atomic_load_n(&c->q_head[1], __ATOMIC_ACQUIRE);
        if (tail - (h0 < h1 ? h0 : h1) < PRB_QCAP) break;
        if (__atomic_load_n(&c->failed, __ATOMIC_ACQUIRE)) { pthread_mutex_unlock(&c->mu); return -2; }
        sched_yield();
    }
    c->queue[tail % PRB_QCAP] = call;
    __atomic_store_n(&c->q_tail, tail + 1, __ATOMIC_RELEASE);
    KArgs a;
    memset(&a, 0, sizeof(a));
    a.in = (const uint4 *)in; a.out = (uint4 *)out; a.region = c->region; a.work = c->d_work; a.cnt = c->d_cnt;
    a.rflag_off = c->L.rflag_off; a.stage_off = c->L.stage_off; a.ready_off = c->L.ready_off; a.ctrl_off = c->L.ctrl_off;
    a.dir_recv_bytes = c->L.dir_recv_bytes; a.chunk_bytes = c->L.chunk_bytes; a.steps_cap = c->L.steps_cap;
    a.stage_base[0] = call.stage_base[0]; a.stage_base[1] = call.stage_base[1];
    a.work_base = c->work_next; a.timeout_ns = c->timeout_ns;
    a.nslot = (uint32_t)c->L.nslot; a.parts = c->parts; a.rank = (uint32_t)c->rank; a.seq = call.seq;
    a.parity = call.parity; a.g = call.g;
    prb_kernel<<<c->grid, PRB_THREADS, 0, (cudaStream_t)stream>>>(a);
    cudaError_t e = cudaGetLastError();
    c->stage_next[0] += call.g.staged[0];
    c->stage_next[1] += call.g.staged[1];
    c->work_next += (uint64_t)call.g.items * c->parts + (uint64_t)c->grid;
    pthread_mutex_unlock(&c->mu);
    if (e != cudaSuccess) {
        snprintf(c->err, sizeof(c->err), "kernel launch: %s", cudaGetErrorString(e));
        fail(c, c->err);
        return -1;
    }
    return 0;
}

// 0 healthy; 1 proxy failed; 2 poisoned (a GPU wait timed out). `msg` gets the detail.
extern "C" int prb_health(prb_ctx *c, char *msg, uint64_t n) {
    uint32_t poison = load_acq_u32(c->region + c->L.ctrl_off + 128);
    if (__atomic_load_n(&c->failed, __ATOMIC_ACQUIRE)) { snprintf(msg, n, "proxy failed: %s", c->err); return 1; }
    if (poison) {
        uint32_t where = load_acq_u32(c->region + c->L.ctrl_off + 132);
        snprintf(msg, n, "GPU wait timed out at call %u (where 0x%x); runtime poisoned", poison, where);
        return 2;
    }
    return 0;
}

extern "C" const char *prb_error(prb_ctx *c) { return c->err; }

// Base of the registered region (host address == device address). Used by the one-node smoke test, which plays the
// neighbours by writing incoming data and flags itself.
extern "C" void *prb_region(prb_ctx *c) { return c->region; }

extern "C" void prb_stats(prb_ctx *c, uint64_t *out) {
    out[0] = c->seq; out[1] = c->stat_calls; out[2] = c->posted[0]; out[3] = c->posted[1]; out[4] = c->done[0];
    out[5] = c->done[1]; out[6] = c->stat_bytes[0]; out[7] = c->stat_bytes[1];
    out[8] = load_acq_u64(c->region + c->L.ctrl_off); out[9] = load_acq_u64(c->region + c->L.ctrl_off + PRB_FLAG_STRIDE);
    out[10] = c->stage_next[0]; out[11] = c->stage_next[1]; out[12] = c->work_next; out[13] = c->stat_idle_naps;
    out[14] = c->stat_relays; out[15] = (uint64_t)c->mem_kind;
}

extern "C" void prb_stop(prb_ctx *c) {
    if (c->started) {
        __atomic_store_n(&c->stopping, 1, __ATOMIC_RELEASE);
        pthread_join(c->thread, NULL);
        __atomic_store_n(&c->running, 0, __ATOMIC_RELEASE);
        c->started = 0;
    }
}

extern "C" void prb_destroy(prb_ctx *c) {
    if (!c) return;
    prb_stop(c);
    for (int d = 0; d < PRB_DIRS; d++) {
        if (c->qp_send[d]) ibv_destroy_qp(c->qp_send[d]);
        if (c->qp_recv[d]) ibv_destroy_qp(c->qp_recv[d]);
    }
    for (int h = 0; h < PRB_NHCA; h++) {
        if (c->mr[h]) ibv_dereg_mr(c->mr[h]);
        if (c->cq[h]) ibv_destroy_cq(c->cq[h]);
        if (c->pd[h]) ibv_dealloc_pd(c->pd[h]);
        if (c->ctx[h]) ibv_close_device(c->ctx[h]);
    }
    if (c->d_work) cudaFree(c->d_work);
    if (c->d_cnt) cudaFree(c->d_cnt);
    if (c->region) {
        if (c->mem_kind == PRB_MEM_PINNED) cudaFreeHost(c->region);
        else if (c->map_base) munmap(c->map_base, c->map_len);
    }
    free(c);
}
