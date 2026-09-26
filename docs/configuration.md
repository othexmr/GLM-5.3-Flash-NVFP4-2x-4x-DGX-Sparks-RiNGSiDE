# Configuration reference

Every environment variable, docker option and server argument of the two profiles, exactly as their lab plans set
them (`launch/profiles/*/profile.json`). Values in `${...}` come from the site file (`site.env.example`).
"Read by" lists served files (install paths) or image files that contain the variable name; "no reader found"
means neither the served files nor the available copies of the image sources mention it, so it is kept only
because the measured plan set it.

## Fabric and collectives

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `LD_PRELOAD` | `/opt/switchless-nccl/libnccl.so.2` | - | Preloads the switchless NCCL build mounted at /opt/switchless-nccl. | the dynamic loader |
| `VLLM_NCCL_SO_PATH` | `/opt/switchless-nccl/libnccl.so.2` | - | Points vLLM at the same NCCL library. | image: `vllm/envs.py` |
| `TORCH_USE_RTLD_GLOBAL` | `1` | `1` | Loads torch libraries with RTLD_GLOBAL so the preloaded NCCL symbols are the ones used. | PyTorch |
| `NCCL_SWITCHLESS_RING_ONLY` | `1` | - | Switchless patch (patches/nccl/0001): skip Tree/PAT connection setup; the ring has no all-to-all links. | the NCCL library |
| `NCCL_ALGO` | `Ring` | - | Keep NCCL on the Ring algorithm; required with the switchless build. | the NCCL library |
| `NCCL_NET` | `IB` | `IB` | Use the IB verbs network plugin (RoCE v2 on the ConnectX-7 ports). | the NCCL library |
| `NCCL_IB_DISABLE` | `0` | `0` | Keep the IB transport enabled. | the NCCL library |
| `NCCL_IB_HCA` | `${SWITCHLESS_IB_HCA}` | `${SWITCHLESS_IB_HCA}` | RoCE functions NCCL may use (site value). TP4: all four PFs of both ports, exact match; TP2: both PFs of the one `f1` port (`fabric/README.md`). | the NCCL library |
| `NCCL_IB_ADDR_FAMILY` | `AF_INET` | `AF_INET` | Use IPv4 GIDs. | the NCCL library |
| `NCCL_IB_ADDR_RANGE` | `${SWITCHLESS_FABRIC_CIDR}` | `${SWITCHLESS_FABRIC_CIDR}` | Fabric address range (site value). The measured TP4 NCCL build only accepts the four cable subnets 10.100.224.0/24 to 10.100.227.0/24 (fabric/README.md). | the NCCL library |
| `NCCL_IB_ROCE_VERSION_NUM` | `2` | `2` | RoCE v2. | the NCCL library |
| `NCCL_IB_GID_INDEX` | - | `3` | GID table index of the RoCE v2 IPv4 address (TP2, stock NCCL). | the NCCL library |
| `NCCL_IB_SUBNET_AWARE_ROUTING` | `1` | - | NCCL's subnet-aware routing for direct connections on several subnets. | the NCCL library |
| `NCCL_IB_SUBNET_PREFIX_LEN` | `24` | - | Prefix length of each cable subnet. | the NCCL library |
| `NCCL_IB_MERGE_NICS` | `0` | `0` | Do not merge ports into one logical NIC. | the NCCL library |
| `NCCL_CROSS_NIC` | `1` | `0` | Allow (TP4) or forbid (TP2) rings that use different NICs on the two sides of a node. | the NCCL library |
| `NCCL_IB_EXTENDED_IPV4_GIDS` | `1` | - | Dual-PF patch (patches/nccl/0002): publish four IPv4 GIDs per listener. | the NCCL library |
| `NCCL_IB_PRESERVE_PCI_DOMAIN` | `1` | - | Dual-PF patch: prefer routes that keep the PCI root of the local PF. | the NCCL library |
| `NCCL_IB_ROUTE_DIAGNOSTICS` | `1` | - | Dual-PF patch: log the final QP routes. | the NCCL library |
| `NCCL_SOCKET_IFNAME` | `${SWITCHLESS_SOCKET_IFNAME}` | `${SWITCHLESS_SOCKET_IFNAME}` | Interface for NCCL bootstrap sockets (site value). | the NCCL library |
| `GLOO_SOCKET_IFNAME` | `${SWITCHLESS_SOCKET_IFNAME}` | `${SWITCHLESS_SOCKET_IFNAME}` | Interface for the Gloo CPU process group (site value). | PyTorch (Gloo) |
| `TP_SOCKET_IFNAME` | `${SWITCHLESS_SOCKET_IFNAME}` | `${SWITCHLESS_SOCKET_IFNAME}` | Interface for the tensor-parallel socket traffic (site value). | PyTorch (TensorPipe) |
| `MN_IF_NAME` | `${SWITCHLESS_SOCKET_IFNAME}` | - | Interface name for the multi-node launcher (site value). | no reader found |
| `VLLM_HOST_IP` | `${SWITCHLESS_RANK_HOST_IP}` | `${SWITCHLESS_RANK_HOST_IP}` | This rank's address on the bootstrap network (site value, one per rank). | image: `vllm/envs.py` |
| `NCCL_CUMEM_ENABLE` | `0` | `0` | NCCL cuMem allocations off. | the NCCL library |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | `1` | Ignore the process CPU affinity when NCCL picks cores. | the NCCL library |
| `NCCL_MAX_CTAS` | `4` | `4` | At most 4 CTAs per NCCL collective. | the NCCL library |
| `NCCL_NVLS_ENABLE` | `0` | `0` | NVLink SHARP off (not present). | the NCCL library |
| `NCCL_DEBUG` | `INFO` | `WARN` | NCCL log level. | the NCCL library |
| `NCCL_DEBUG_SUBSYS` | `INIT,NET` | - | NCCL log subsystems (initialisation and network). | the NCCL library |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `1` | `1` | Abort on asynchronous NCCL errors instead of hanging. | PyTorch |
| `VLLM_GLM53_TP4_LEAN_ALLREDUCE` | `lean` | - | TP4 small all-reduces through the lean neighbour all-reduce (src/tp4/glm53_lean_allreduce) instead of NCCL. | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `VLLM_GLM53_TP4_LEAN_MAX_BYTES` | `2097152` | - | Largest all-reduce (bytes) sent through the lean path. | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `GLM53_LEAN_CHUNKS` | `2` | - | The lean all-reduce pipelines large messages in this many chunks. | `glm53_lean_allreduce/lean_tp4.py` |
| `GLM53_LEAN_CHUNK_MIN_BYTES` | `262144` | - | Smallest message that is chunked. | `glm53_lean_allreduce/lean_tp4.py` |
| `VLLM_GLM53_TP2_ALLREDUCE` | `p2p` | `p2p` | TP2 small all-reduce path (p2p). | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `VLLM_GLM53_TP2_ALLREDUCE_MAX_BYTES` | `262144` | `262144` | Largest all-reduce (bytes) on the TP2 p2p path. | `vllm/distributed/device_communicators/cuda_communicator.py` |
| `GLM53_MHC_PREFILL_RDMA` | `1` | - | mHC prefill reduce-scatter and all-gather over the two-direction RDMA ring (src/tp4/glm53_prefill_rdma). | `glm53_prefill_rdma/ring.py`, `vllm/models/glm5next/nvidia/kda_fp8_handoff.py`, `vllm/models/glm5next/nvidia/mhc_prefill_sharding.py` |
| `GLM53_MHC_PREFILL_RDMA_MIN_ROWS` | `2304` | - | Smallest owned prefill forward (rows) that uses the RDMA ring. | `vllm/models/glm5next/nvidia/mhc_prefill_sharding.py` |

## Scheduler and speculation (glm53_speedup and the vLLM scheduler patches)

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `GLM53_SPEEDUP_MODE` | `saturation` | `saturation` | Verification-length policy; saturation = saturation-aware K. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_K` | `7` | `7` | Largest proposal length K of the DFlash2 drafter. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_OVERRIDE` | `/opt/glm53-speedup/runtime/override.json` | `/opt/glm53-speedup/runtime/override.json` | Runtime override file (launch/profiles/*/override.json, mounted read-only). | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_CAPTURE` | `all` | `all` | Capture CUDA graphs for the whole (requests, K) grid. | `glm53_speedup/geometry.py`, `glm53_speedup/graphs.py`, `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_CAPTURE_KS` | `1,2,3,4,5,6,7` | `1,2,3,4,5,6,7` | K values of the capture grid. | `glm53_speedup/geometry.py` |
| `GLM53_SPEEDUP_CAPTURE_REQUESTS` | `1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16` | `1,2,3,4,5,6` | Request counts of the capture grid (TP4 1-16, TP2 1-6). | `glm53_speedup/geometry.py` |
| `GLM53_SPEEDUP_GEOMETRY` | `1` | `1` | Derive the policy and graph geometry from the engine configuration. | `glm53_speedup/geometry.py`, `glm53_speedup/graphs.py`, `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_GRAPH_MEMORY` | `1` | `1` | Log per-rank graph memory. | `glm53_speedup/graphs.py` |
| `GLM53_SPEEDUP_TIMING` | `1` | `1` | Log CUDA-event step timings (drained without extra synchronisation). | `glm53_speedup/telemetry.py` |
| `GLM53_SPEEDUP_LOG_EVERY` | `1` | `1` | Scheduler record interval. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_COMM_TRACE` | `0` | `0` | Collective shape trace (off). | no reader found |
| `GLM53_SPEEDUP_MIXED_PREFILL` | `4608` | `4608` | Total prefill tokens granted in a step that also carries decodes (fair prefill cap). | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_MIXED_PREFILL_FAIR` | `1` | `1` | Share that cap fairly between waiting prefills, newcomers included. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_MIXED_PREFILL_RUNTIME` | `0` | `0` | Runtime changes of the cap (off). | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_PREFILL_ADAPTIVE` | `1` | `1` | Adaptive prefill admission. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_SPLIT_CADENCE` | `1` | `1` | Split cadence: a nonfinal prefill-only step is followed by a decode-only step. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_SPLIT_FAIR` | `1` | `1` | Plan the cadence before the fair split. | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_SPLIT_ADAPTIVE` | `0` | `0` | Adaptive split selector (off). | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_FAIR_COST_JOINT` | `0` | `0` | Joint cost model for the fair split (off). | `glm53_speedup/graphs.py` |
| `GLM53_SPEEDUP_PREFILL_TAILS` | `0` | `0` | Prefill-tail handling (off). | `glm53_speedup/graphs.py` |
| `GLM53_ADAPTIVE_FAIR_CHUNK` | `1` | `0` | Size the fair chunk from the number of running decoders (TP4 on, TP2 off). | `glm53_speedup/scheduler.py` |
| `GLM53_SPEEDUP_TP2_FAIR_CAP_GEOMETRY` | - | `1` | TP2: admit 9,216 as a second aligned prefill cap. | `glm53_speedup/scheduler.py` |
| `GLM53_PREFIX_CACHE_CONTRACT` | `1` | `1` | Check and log the prefix-cache retention contract at boot. | `vllm/v1/core/sched/scheduler.py` |
| `GLM53_KDA_CKPT` | `1` | - | KDA checkpoints saved inside a prefill chunk (scheduler, MambaManager, V2 runner and kda.py patches; src/tp4/glm53_kda_ckpt*.py). | `glm53_kda_ckpt_plan.py`, `vllm/models/glm5next/nvidia/kda.py`, `vllm/v1/core/sched/scheduler.py` ... |
| `GLM53_STARTUP_PREFILL_WARMUP` | `17` | `6` | Number of random-token warm-up prompts run before the HTTP server listens; the prefix cache is reset afterwards. | `vllm/entrypoints/launchers/api_server/entry.py` |
| `GLM53_STARTUP_PREFILL_WARMUP_TOKENS` | `28672,28672,1152,2304,2816,3328,3456,4608,5120,6656,8192,10240,13304,14456,14720,15104,15488` | `28672,28672,5120,5760,8192,10240` | Length of each warm-up prompt; a comma-separated list gives the lengths in order, cycled over the prompts (the profiles list the chunk row counts their benchmarks meet, so those shapes compile before readiness). | `vllm/entrypoints/launchers/api_server/entry.py` |
| `GLM53_SPEEDUP_FIRST_TOKEN_REPAY` | `1` | `1` | P1 first-token repay: a request that has just produced its first token is owed one decode-only turn before the next prefill-only offer (at most one converted turn per wave of new decoders; admission, KV and grants unchanged). | `glm53_speedup/cadence.py` |
| `GLM53_REPLAY_BOUNDARY` | `1` | `1` | Replay boundary: keep one complete hybrid state (MLA, the KDA state groups and the drafter blocks) at the latest position a DFlash2 replay or extension of the prompt can hit (glm53_replay_boundary.py and the scheduler and KV-manager patches). | `glm53_kda_ckpt_plan.py`, `glm53_replay_boundary.py`, `vllm/models/glm5next/nvidia/kda.py` ... |
| `GLM53_REPLAY_BOUNDARY_MAX` | `16` | `6` | Most replay-boundary KDA states retained at a time. | `glm53_replay_boundary.py` |
| `GLM53_GREEDY_ARGMAX_VERIFY` | `1` | - | TP4: verify a batch whose requests are all greedy (temperature 0, no logits processing) from per-rank (max, index) pairs instead of all-gathering the full logits (glm53_greedy_verify.py); any other batch takes the unchanged path. | `glm53_greedy_verify.py`, `vllm/v1/worker/gpu/model_runner.py` |
| `GLM53_GREEDY_ARGMAX_LOG_EVERY` | `500` | - | TP4: log interval (steps) of the greedy verification counters. | `glm53_greedy_verify.py` |
| `GLM53_SPEEDUP_MHC_SHARD` | `0` | `0` | Older mHC sharding switch (off; the token-sharded prefill uses GLM53_MHC_PREFILL_SHARD). | no reader found |
| `GLM53_SPEEDUP_NOPE_COPIES` | `0` | `0` | Carried over from earlier plans (0). | no reader found |
| `GLM53_SPEEDUP_KDA_STRIDES` | `0` | `0` | Carried over from earlier plans (0). | no reader found |
| `GLM53_SPEEDUP_ROCE` | `0` | `0` | Carried over from earlier plans (0). | no reader found |
| `VLLM_GLM53_DFLASH2_EDGE_SCALE` | `1.0` | `1.0` | Scale of the DFlash2 drafter's learned edge term. | image: `vllm/model_executor/models/qwen3_dflash2.py` |
| `VLLM_GLM53_DFLASH2_WALK` | `greedy` | `greedy` | DFlash2 drafter walk (greedy). | no reader found |

## Kernels: MoE (b12x)

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `B12X_GLM53_ATOMIC_PREFILL` | `1` | `1` | Atomic MoE output accumulation for eager pure prefill launches. | `glm53_sparse_mla/backend.py`, `sparse-MLA extension source (patches/sparse-mla/build)`, `vllm/model_executor/layers/fused_moe/b12x.py` |
| `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` | `1` | `1` | Deterministic MoE epilogue on the decode and graph paths. | `b12x/moe/fused_moe/_impl.py` |
| `B12X_GLM53_ATOMIC_DECODE_MIN_ROWS` | `10` | - | TP4: atomic accumulation for every MoE launch of at least this many rows. NON-DETERMINISTIC outputs. | `vllm/model_executor/layers/fused_moe/b12x.py` |
| `B12X_GLM53_DECODE_FASTPATH` | `1` | `1` | Small-launch decode fast path. | `b12x/moe/fused_moe/_impl.py` |
| `B12X_MICRO_DYNAMIC_CUTOVER_PAIRS` | `0` | `0` | Keep small launches on the dynamic kernel. | `b12x/moe/fused_moe/_impl.py` |
| `B12X_PRINT_COMPILE_PROGRESS` | `1` | `1` | Print b12x kernel compilation progress. | image: `b12x/_lib/compiler.py` |
| `B12X_TIMING` | `0` | `0` | b12x internal timing (off). | `b12x/moe/fused_moe/_impl.py` |
| `B12X_TP4_CLUSTER_MAP` | `{"15":32,"20":32,"25":32,"30":32}` | `0` | TP4: resident CTA grid per row count; 0 = off. | `b12x/moe/fused_moe/_impl.py` |
| `B12X_TP4_TILE_MAJOR_WEIGHTS` | `1` | `0` | TP4: tile-major expert-weight layout, repacked in place at load time. | `b12x/moe/fused_moe/_impl.py` |
| `B12X_TP2_TILE_MAJOR_WEIGHTS` | - | `1` | TP2: tile-major expert-weight layout. | `b12x/moe/fused_moe/_impl.py` |
| `B12X_TP2_CLUSTER_MAP` | - | `0` | TP2: cluster map (off). | `b12x/moe/fused_moe/_impl.py` |
| `B12X_GLM53_M128_MMA_WARPS` | `8` | - | Eight MMA warps for the M128 NVFP4 MoE tile. | `b12x/moe/_shared/kernels/dynamic.py`, `b12x/moe/fused_moe/_impl.py` |
| `B12X_GLM53_MIXED_SPLIT` | `1` | `1` | Split mixed prefill+decode MoE launches into a decode and a prefill launch. | `glm53_sparse_mla/backend.py`, `vllm/model_executor/layers/fused_moe/b12x.py` |
| `B12X_GLM53_MIXED_SPLIT_MIN_PREFILL` | `1536` | `1536` | Smallest prefill part (rows) that is split off. | `vllm/model_executor/layers/fused_moe/b12x.py` |
| `B12X_GLM53_PREFILL_PREFETCH` | `2a` | - | L2 prefetch of the FC2 reduction targets two tiles ahead (variant a). | `b12x/moe/_shared/kernels/dynamic.py` |

## Kernels: dense layers, attention, KDA and mHC

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `VLLM_GLM53_DUAL_FP8_DENSE` | `default,head,draft,shared_down,indexer` | `default,head,draft,shared_down,indexer` | Families of BF16 dense projections that get an online FP8 copy (glm53_dual_fp8_dense.py). | `vllm/model_executor/layers/linear.py`, `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `VLLM_GLM53_DUAL_FP8_DECODE` | `nvfp4` | `nvfp4` | Decode path of those families: NVFP4 (Marlin) copies. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `VLLM_GLM53_DUAL_FP8_DECODE_FP8_FAMILIES` | `shared_down,indexer,head` | `shared_down,indexer,head` | Families that decode from the FP8 copy instead. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `VLLM_GLM53_DUAL_FP8_DECODE_KERNEL` | `marlin` | `marlin` | Decode kernel of the dual path (Marlin). | no reader found |
| `VLLM_GLM53_DUAL_FP8_PREFILL` | `w8a8` | `w8a8` | Prefill path of the dual copies (W8A8). | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `VLLM_GLM53_DUAL_FP8_GEMV_CONFIG` | (empty) | (empty) | GEMV configuration override (empty = default). | no reader found |
| `GLM53_HUMMING_DENSE` | `shared_down` | - | TP4: Humming dense GEMM for these families. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `GLM53_HUMMING_NVFP4_M128` | `mla_qb,kda_in,attn_out,shared` | - | TP4: Humming NVFP4 GEMM for 64 < M <= 128 rows in these families. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `GLM53_DENSE_FP8_GEMM` | `kda_in=p64x128x128/2/1@129-1024,kda_in=p128x128x64/4/0@1025-3000,kda_in=vllm128/4/1@3001-6000,attn_out=p128x128x64/2/0@1025,mla_a=p64x128x128/2/1@129-1024,mla_a=vllm128/4/0@1025,mla_qb=p128x128x64/4/0@1025,indexer=p128x128x64/4/0@1025,shared=c256x128x64/1/0@6001,shared_down=p128x128x64/4/0@129` | - | TP4: CUTLASS SM120 FP8 GEMM configuration per family and row range (build/glm53_fp8_gemm). | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `GLM53_FP8_GEMM_SO` | `/opt/glm53_fp8_gemm/glm53_fp8_gemm.so` | - | TP4: path of the FP8 GEMM extension. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `GLM53_DENSE_CHUNK_ROWS` | `kda_in:3456,draft:2048:4096` | - | TP4: dense GEMM row chunks for kda_in and the drafter projection. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py` |
| `GLM53_DENSE_SHARED_QUANT` | `mla_qnorm>mla_qb,mla_qb>indexer,kda_onorm>attn_out` | - | TP4: share one activation quantization between the named families (`mla_qnorm>mla_qb`: q_b_proj takes the quantization that the q_a norm wrote, `GLM53_MLA_QNORM_FP8`). | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py`, `vllm/models/glm5next/nvidia/kda.py` |
| `GLM53_NVFP4_DENSE_CHUNK_ROWS` | `4608` | - | TP4: NVFP4 dense GEMM (layers 0-2 MLP) in chunks of this many rows... | `vllm/model_executor/kernels/linear/nvfp4/b12x.py` |
| `GLM53_NVFP4_DENSE_CHUNK_MODE` | `out` | - | ...each chunk written into one caller-owned output... | `vllm/model_executor/kernels/linear/nvfp4/b12x.py` |
| `GLM53_NVFP4_DENSE_CHUNK_MIN_ROWS` | `9216` | - | ...when the step has at least this many rows. | `vllm/model_executor/kernels/linear/nvfp4/b12x.py` |
| `GLM53_INDEXER_GATE_TC` | `1` | `1` | Indexer gate on tensor cores. | `vllm/models/glm5next/nvidia/attention.py` |
| `GLM53_INDEXER_FUSED_PROJ` | `1073741824` | - | TP4: one fused indexer projection GEMM for steps above 128 rows (value: row-chunk bound). | `vllm/models/glm5next/nvidia/attention.py` |
| `GLM53_INDEXER_POOL_LOGITS` | `1` | - | TP4: the sparse-attention indexer's decode logits are cdiv(max_model_len, 4) columns wide (one per kpool entry) instead of max_model_len. | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` |
| `GLM53_TOPK_ROWSPREAD` | `1` | - | TP4: the indexer's decode top-k through the rowspread extension (`nvfp4_topk_rowspread`, `build/topk_rowspread`): every row at or below the radix threshold is taken by one CTA of the whole grid. | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` |
| `GLM53_INDEXER_ROW_SPLIT` | `1` | - | TP4: the prefill indexer's query rows split across the ranks (`glm53_indexer_rowsplit.py`): a prefill chunk of at least `GLM53_INDEXER_ROW_SPLIT_MIN_ROWS` rows (not set: 1,024) over at least `GLM53_INDEXER_ROW_SPLIT_MIN_POOLS` compressed key positions (not set: 4,096) is cut into contiguous tiles of `GLM53_INDEXER_ROW_SPLIT_TILE` rows (not set: 64), one range per rank; each rank selects the top-k pools of its rows and the ranks all-gather the pool ids. `GLM53_INDEXER_ROW_SPLIT_CHECK=1` also runs the full-row selection and counts differing rows (a diagnostic). | `glm53_indexer_rowsplit.py`, `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` |
| `GLM53_EMBED_REPLICATE` | `1` | - | TP4: replicate the embedding table on every rank (no vocabulary all-reduce; about +0.9 GiB per rank). | `vllm/models/glm5next/nvidia/model.py` |
| `GLM53_TOPK_SMALL_ROWS` | `0` | - | TP4: take the top-k threshold path at every row count. | `deterministic_sparse_topk.py` |
| `VLLM_GLM53_CUDA_SPARSE_MLA` | `1` | `1` | Use the CUDA kernels of the sparse-MLA plugin. | `glm53_sparse_mla/backend.py`, `sparse-MLA extension source (patches/sparse-mla/build)` |
| `VLLM_GLM53_SPARSE_MLA_SUBGROUP_SOFTMAX` | `0` | `0` | Sparse-MLA subgroup softmax variant (off). | no reader found |
| `GLM53_SPARSE_MLA_SPLIT_KV` | `1` | `1` | Split-KV sparse MLA for short queries. | `sparse-MLA extension source (patches/sparse-mla/build)` |
| `GLM53_SPARSE_MLA_IDX_PIPE` | `1` | - | TP4: sparse-MLA index pipeline (A1c extension). | `sparse-MLA extension source (patches/sparse-mla/build)` |
| `GLM53_SPARSE_MLA_VALID_COLS` | `2051` | - | TP4: stop the tile loop after the valid index columns; exact because the later columns are always -1. | `sparse-MLA extension source (patches/sparse-mla/build)` |
| `GLM53_MLA_ABSORB_CHUNK` | `2304` | `2304` | MLA absorbed-weight layout, applied in chunks of this many rows. | `vllm/model_executor/layers/attention/mla_attention.py` |
| `GLM53_B12X_KDA_DIRECT_OUT` | `1` | - | TP4: b12x KDA prefill writes straight into the layer output. | `vllm/models/glm5next/nvidia/kda.py` |
| `GLM53_KDA_CONV_FUSED` | `1` | - | TP4: fold the KDA short convolution into b12x's prepare pass (src/tp4/glm53_kda_conv.py). | `vllm/models/glm5next/nvidia/kda.py` |
| `GLM53_KDA_CONV_FUSED_MIN_ROWS` | `512` | - | Smallest pure prefill (rows) for the fused convolution. | `vllm/models/glm5next/nvidia/kda.py` |
| `GLM53_KDA_CONTIGUOUS_MIXED` | `1` | - | TP4: contiguous KDA slices in mixed steps. | `vllm/models/glm5next/nvidia/kda.py` |
| `GLM53_KDA_STATE` | `fp16-slow32` | - | TP4: fp16-slow32 = the KDA recurrent-state pool in fp16 (round to nearest) with each head's 8 slowest key channels exact in fp32 side columns; storage only, every kernel computes in fp32 (kda_state_store.py, kda.py, recoverssm.py). | `vllm/models/glm5next/nvidia/kda.py`, `vllm/models/glm5next/nvidia/kda_state_store.py`, `vllm/models/glm5next/nvidia/ops/recoverssm.py` |
| `GLM53_KDA_FP8_HANDOFF` | `1` | - | TP4: quantize the owner-row mHC output once and all-gather FP8 bytes and scales for the KDA input projection. | `vllm/models/glm5next/nvidia/kda.py`, `vllm/models/glm5next/nvidia/kda_fp8_handoff.py`, `vllm/models/glm5next/nvidia/model.py` |
| `GLM53_KDA_FP8_HANDOFF_GATHER` | `two` | - | Gather mode (two all-gathers: bytes, then scales). | `vllm/models/glm5next/nvidia/kda_fp8_handoff.py` |
| `GLM53_KDA_FP8_HANDOFF_MIN_ROWS` | `2304` | - | Smallest owned forward (rows) for the FP8 handoff. | `vllm/models/glm5next/nvidia/kda_fp8_handoff.py` |
| `GLM53_KDA_FP8_HANDOFF_RING` | `1` | - | Run those gathers over the RDMA ring. | `vllm/models/glm5next/nvidia/kda_fp8_handoff.py` |
| `GLM53_KDA_FP8_HANDOFF_TUNED` | `1` | - | TP4: the handoff's input projection takes the dense FP8 module's tuned SM120 GEMM configuration (`GLM53_DENSE_FP8_GEMM`) per owner block, chunk or call instead of the generic scaled matrix multiply, from `GLM53_KDA_FP8_HANDOFF_TUNED_MIN_ROWS` rows per call (not set: 1,536). | `vllm/models/glm5next/nvidia/kda_fp8_handoff.py` |
| `GLM53_KDA_ONORM_FP8` | `2` | - | TP4: KDA o_norm with the o_proj FP8 quantization fused in. | `vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py`, `vllm/models/glm5next/nvidia/kda.py`, `vllm/models/glm5next/nvidia/ops/glm53_onorm_fp8.py` |
| `GLM53_MLA_QNORM_FP8` | `2` | - | TP4: the MLA q_a/kv_a RMSNorm writes q_b_proj's per-token FP8 quantization itself (`vllm/models/common/ops/glm53_qnorm_fp8.py`); mode 2 stores FP8 only (no BF16 q_c where every reader takes the FP8 copy), and a layer that misses the hand-over raises instead of recomputing. | `vllm/model_executor/layers/mla.py`, `vllm/models/common/ops/glm53_qnorm_fp8.py` |
| `GLM53_RECOVERSSM_FUSED_COMMIT` | `1` | - | TP4: `1` = the fused commit: the accepted KDA state is written by the next verify step, with the ordering fix of the RiNGSiDE release (rule 1 at commit time, its two launches skipped when the host proves that no row is within 8 tokens of a block end; rule 2 at step start) and the exact-landing column; `0` writes it after sampling (TP2, where the variable is unset and the default is `0`). | `vllm/models/glm5next/nvidia/ops/recoverssm.py` |
| `GLM53_MHC_PREFILL_SHARD` | `1` | - | TP4: token-sharded mHC prefill: each rank computes mHC for its share of the rows. | `vllm/models/glm5next/nvidia/mhc_prefill_sharding.py` |
| `GLM53_MHC_PREFILL_MIN_ROWS` | `2304` | - | Smallest eager forward (rows) that is sharded. | `vllm/models/glm5next/nvidia/kda_fp8_handoff.py`, `vllm/models/glm5next/nvidia/mhc_prefill_sharding.py` |
| `GLM53_MHC_PRE_BLOCKED` | `1` | - | TP4: blocked layer-0 mhc_pre. | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `GLM53_MHC_POST_PRENORM_FUSED` | `1` | - | TP4: fused mHC post and pre-norm. | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `GLM53_MHC_POST_PRENORM_MIN_ROWS` | `1152` | - | Smallest step (rows) for the fusion. | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `GLM53_MHC_POST_PRENORM_BLOCK_ROWS` | `432` | - | Row block of the fused kernel. | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `GLM53_MHC_POST_PRENORM_SPLIT` | `fill` | - | Block split policy (fill). | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `GLM53_TP4_MHC_GEOMETRY` | `1` | `0` | TP4 small-M mHC geometry table (TP4 on, TP2 off). | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `GLM53_TP2_MHC_GEOMETRY` | - | `0` | TP2 small-M mHC geometry table (off). | `vllm/model_executor/kernels/mhc/tilelang.py` |
| `VLLM_USE_B12X_MHC` | `0` | `0` | b12x mHC kernels (off; the TileLang path is used). | no reader found |

## Sampling

The V2 sampler files of the image that the TP4 profile patches (`topk_topp_triton.py`, `sample/sampler.py`,
`spec_decode/rejection_sampler.py`); with every variable unset they behave as the image's files except for the top-p
fix, which is on by default.

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `GLM53_TOPP_FIX` | `1` | `1` | TP4: exact top-p in the Triton top-k/top-p filter for every row whose estimated normaliser, search exit or reconstructed cutoff cannot guarantee the cut; other rows unchanged (the file's default is on too). | `vllm/v1/sample/ops/topk_topp_triton.py` |
| `GLM53_UTF8_GUARD` | `1` | `1` | TP4: UTF-8 continuation guard: in every sampled row (temperature > 0) whose sequence ends inside a character, each token that cannot continue it gets -inf; `1` masks only. | `glm53_utf8_guard.py`, `vllm/v1/worker/gpu/sample/sampler.py` |
| `GLM53_UTF8_GUARD_TOKENIZER` | `/models/target/tokenizer.json` | `/models/target/tokenizer.json` | TP4: the tokenizer the guard builds its continuation table from. | `glm53_utf8_guard.py` |
| `GLM53_UTF8_GUARD_FOLD` | - | - | Not set by either profile (the release plan asserts it unset): the guard's folding of mixed tokens stays off. | `glm53_utf8_guard.py` |
| `GLM53_UTF8_GUARD_GREEDY` | - | - | Not set by either profile (asserted unset): greedy rows (temperature 0) keep the argmax fast path and are not guarded. | `glm53_utf8_guard.py` |
| `GLM53_SAMPLER_AUDIT` | - | - | Not set by either profile (asserted unset): the sampler audit stays off; its module is not served. | `vllm/v1/sample/ops/topk_topp_triton.py`, `vllm/v1/worker/gpu/sample/sampler.py` |

## vLLM engine and model

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | `1` | vLLM V2 model runner. | `vllm/config/vllm.py` |
| `VLLM_PLUGINS` | `glm53_sparse_mla` | `glm53_sparse_mla` | Load the sparse-MLA plugin. | image: `vllm/envs.py` |
| `VLLM_GLM53_ROUTER_DEDUP` | `1` | `1` | Compute the MoE router once on the main stream. | `vllm/models/glm5next/nvidia/model.py` |
| `VLLM_GLM53_ROUTER_GATE_TRITON` | `0` | `0` | Carried over from earlier plans (0). | no reader found |
| `VLLM_GLM53_GROUPED_ROUTER_SKIP_PADDING` | `0` | `0` | Mask padding rows after grouped top-k (off). | no reader found |
| `VLLM_GLM53_V2_ASYNC_TP_EXPERIMENTAL` | `0` | `0` | Experimental asynchronous TP (off). | no reader found |
| `VLLM_GLM53_V2_SP_EXPERIMENTAL` | `0` | `0` | Experimental sequence parallelism (off). | no reader found |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` | `0` | `0` | Keep the shared experts on their own stream. | image: `vllm/envs.py` |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `0` | `0` | CUDA forward-compatibility libraries off. | image: `vllm/envs.py` |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` | `3600` | Engine start timeout; the first boot compiles kernels and runs the warm-up. | image: `vllm/envs.py` |
| `VLLM_KV_CACHE_LAYOUT_DUMP` | `1` | `1` | Log the KV cache layout at boot. | `vllm/v1/core/kv_cache_utils.py` (TP4: the served patch; TP2: the image file) |
| `VLLM_LOGGING_LEVEL` | `INFO` | `INFO` | vLLM log level. | `glm53_speedup/graphs.py`, `glm53_speedup/telemetry.py` |
| `VLLM_USAGE_SOURCE` | `production-docker-image` | `production-docker-image` | Usage-source label. | image: `vllm/envs.py` |
| `VLLM_NO_USAGE_STATS` | `1` | - | TP4: no usage statistics. | image: `vllm/envs.py` |
| `DO_NOT_TRACK` | `1` | - | TP4: no usage statistics. | image: `vllm/envs.py` |
| `INSTANTTENSOR_BACKEND` | `URING` | `URING` | InstantTensor weight loader (--load-format instanttensor) reads with io_uring. | the InstantTensor loader (image) |
| `INSTANTTENSOR_DEBUG` | `1` | `1` | InstantTensor loader log. | the InstantTensor loader (image) |
| `HF_HOME` | `/cache/huggingface` | `/cache/huggingface` | Hugging Face cache inside the /cache mount. | no reader found |
| `HF_HUB_OFFLINE` | `1` | `1` | No Hugging Face downloads at runtime. | image: `vllm/tokenizers/registry.py` |
| `TRANSFORMERS_OFFLINE` | `1` | `1` | No transformers downloads at runtime. | transformers (image) |
| `VLLM_CACHE_ROOT` | `/root/.cache/vllm` | `/root/.cache/vllm` | vLLM compile cache inside the per-rank /root/.cache mount. | image: `vllm/envs.py` |
| `XDG_CACHE_HOME` | `/root/.cache` | `/root/.cache` | Cache root inside the per-rank /root/.cache mount. | `glm53_lean_allreduce/_proxy.py` |
| `TILELANG_CACHE_DIR` | `/root/.cache/tilelang` | `/root/.cache/tilelang` | TileLang kernel cache inside /root/.cache. | TileLang (image) |
| `TILELANG_TMP_DIR` | `/root/.cache/tilelang/tmp` | `/root/.cache/tilelang/tmp` | TileLang temporary files inside /root/.cache. | TileLang (image) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `expandable_segments:True` | Expandable segments in the PyTorch allocator. | `vllm/config/vllm.py`, `vllm/v1/worker/gpu_worker.py` |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | `12.1a` | Build JIT kernels for SM 12.1a (GB10). | image: `vllm/collect_env.py` |
| `FLASHINFER_CUDA_ARCH_LIST` | `12.1a` | `12.1a` | FlashInfer JIT architecture. | image: `flashinfer/compilation_context.py` |
| `FLASHINFER_DISABLE_VERSION_CHECK` | `1` | `1` | Skip FlashInfer's version check. | image: `flashinfer/collect_env.py` |
| `VLLM_CANDIDATE_BOOT` | `tp4-ring-20260917` | `tp4-ring-20260917` | A label of the lab's launcher; nothing in the served code reads it. | no reader found |

## Image environment restated on the command line (same values as the image)

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `CUDA_VERSION` | `13.0.3` | `13.0.3` |  | image |
| `DEBIAN_FRONTEND` | `noninteractive` | `noninteractive` |  | image |
| `LD_LIBRARY_PATH` | `/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64` | `/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64` |  | image |
| `NVARCH` | `sbsa` | `sbsa` |  | image |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | `compute,utility` |  | image |
| `NVIDIA_REQUIRE_CUDA` | `cuda>=13.0 brand=... (image value)` | `cuda>=13.0 brand=... (image value)` |  | image |
| `NVIDIA_VISIBLE_DEVICES` | `all` | `all` |  | image |
| `NV_CUDA_CUDART_VERSION` | `13.0.96-1` | `13.0.96-1` |  | image |
| `PATH` | `/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` | `/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` |  | image |
| `UV_CACHE_DIR` | `/opt/uv/cache` | `/opt/uv/cache` |  | image |
| `UV_HTTP_TIMEOUT` | `500` | `500` |  | image |
| `UV_INDEX_STRATEGY` | `unsafe-best-match` | `unsafe-best-match` |  | image |
| `UV_LINK_MODE` | `copy` | `copy` |  | image |
| `UV_OVERRIDE` | `/etc/uv-overrides.txt` | `/etc/uv-overrides.txt` |  | image |
| `UV_PYTHON_INSTALL_DIR` | `/opt/uv/python` | `/opt/uv/python` |  | image |
| `VLLM_BUILD_COMMIT` | `98dff2a81d747d1dba01a47f939f48c3526d4206` | `98dff2a81d747d1dba01a47f939f48c3526d4206` |  | image |
| `VLLM_BUILD_PIPELINE` | `local-glm53-backport` | `local-glm53-backport` |  | image |
| `VLLM_BUILD_URL` | (empty) | (empty) |  | image |
| `VLLM_IMAGE_TAG` | `local/glm53-v029:ported-20260910` | `local/glm53-v029:ported-20260910` |  | image |

## Server arguments

| argument | TP4 | TP2 | meaning |
|---|---|---|---|
| `--served-model-name` | `nvidia/GLM-5.3-Flash-NVFP4` | `nvidia/GLM-5.3-Flash-NVFP4` | Model name in the OpenAI-compatible API. |
| `--trust-remote-code` | (flag) | (flag) | Allow the model repository's code. |
| `--tensor-parallel-size` | `4` | `2` | Ranks: one per Spark. |
| `--gpu-memory-utilization` | `0.87` | `0.87` | Share of unified memory vLLM may plan with. |
| `--max-model-len` | `262144` | `262144` | Context length per request. |
| `--max-num-seqs` | `16` | `6` | Concurrent requests admitted (TP4 16, TP2 6). |
| `--max-num-batched-tokens` | `14336` | `14336` | Token budget of one scheduler step. |
| `--block-size` | `2304` | `2304` | KV block size; the mamba (KDA) state block is the same size. |
| `--kda-prefill-backend` | `b12x` | `b12x` | b12x kernels for KDA prefill. |
| `--moe-backend` | `b12x` | `b12x` | b12x NVFP4 MoE kernels. |
| `--linear-backend` | `b12x` | `b12x` | b12x NVFP4 dense kernels. |
| `--attention-config` | `{"backend_per_kind":{"sliding_window":"FLASHINFER"},"use_trtllm_attention":true}` | `{"backend_per_kind":{"sliding_window":"FLASHINFER"},"use_trtllm_attention":true}` | FlashInfer for the sliding-window layers; TRT-LLM attention kernels. |
| `--speculative-config` | `{"method":"dflash","model":"/models/dflash2-repo/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe","num_speculative_tokens":7,"kv_cache_dtype":"auto"}` | `{"method":"dflash","model":"/models/dflash2-repo/snapshots/7d74cdd881ed7e32c31175984a67823127b66cfe","num_speculative_tokens":7,"kv_cache_dtype":"auto"}` | DFlash2 drafter from the drafter mount, 7 speculative tokens. |
| `--kv-cache-memory` | `34359738368` | `11274289152` | KV cache pool per rank in bytes (TP4 32 GiB, TP2 10.5 GiB). |
| `--kv-cache-dtype` | `fp8_e4m3` | `fp8_e4m3` | FP8 KV cache. |
| `--no-async-scheduling` | (flag) | (flag) | Synchronous scheduling; the scheduler patches require it. |
| `--prefix-cache-retention-interval` | `2304` | `4608` | Keep a prefix-cache state every this many tokens (TP4 2,304, TP2 4,608). |
| `--compilation-config` | `{"mode":3,"cudagraph_capture_sizes":[8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128],"compile_sizes":[8]}` | `{"mode":3,"cudagraph_capture_sizes":[8,16,24,32,40,48],"compile_sizes":[8]}` | torch.compile mode 3, CUDA graph capture sizes (TP4 8-128, TP2 8-48, step 8), compile size 8. |
| `--tool-call-parser` | `glm53` | `glm53` | GLM-5.3 tool-call parser. |
| `--enable-auto-tool-choice` | (flag) | (flag) | Automatic tool choice. |
| `--reasoning-parser` | `glm53` | `glm53` | GLM-5.3 reasoning parser. |
| `--default-chat-template-kwargs` | `{"clear_thinking": true}` | `{"clear_thinking": true}` | Clear earlier turns' thinking by default. |
| `--override-generation-config` | `{"top_p":0.95}` | `{"top_p":0.95}` | Generation defaults over the checkpoint's generation_config.json: top_p 0.95, NVIDIA's value, which RedHat's file lacks. |
| `--chat-template` | `/chat_template_mm.jinja` | `/chat_template_mm.jinja` | The mounted chat template: zai-org's GLM-5.3 template with Raymond Lucke's thinking gate, the file of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (`NOTICE`); `thinking` or `enable_thinking` in `chat_template_kwargs` switches the thinking block. |
| `--limit-mm-per-prompt` | `{"image":4,"video":1}` | `{"image":4,"video":1}` | At most 4 images and 1 video per prompt. |
| `--skip-mm-profiling` | (flag) | (flag) | Skip the multimodal memory profiling pass. |
| `--distributed-executor-backend` | `mp` | `mp` | Multiprocessing executor, one process group across the nodes. |
| `--nnodes` | `4` | `2` | Number of nodes. |
| `--node-rank` | `${RANK}` | `${RANK}` | This node's rank (rendered per rank). |
| `--master-addr` | `${SWITCHLESS_MASTER_ADDR}` | `${SWITCHLESS_MASTER_ADDR}` | Rank 0's address (site value). |
| `--master-port` | `${SWITCHLESS_MASTER_PORT}` | `${SWITCHLESS_MASTER_PORT}` | Rendezvous port (site value). |
| `--host` | `0.0.0.0` | `0.0.0.0` | Rank 0: API listen address. |
| `--port` | `${SWITCHLESS_API_PORT}` | `${SWITCHLESS_API_PORT}` | Rank 0: API port (site value). |
| `--headless` | ranks 1+ | ranks 1+ | Ranks above 0: no API server. |
| `--per-request-spec-decode-metrics` | `detailed` | `detailed` | Per-request speculative decoding metrics. |
| `--kernel-config` | `{"enable_flashinfer_autotune":false}` | `{"enable_flashinfer_autotune":false}` | FlashInfer autotuning off. |
| `--cudagraph-metrics` | (flag) | (flag) | CUDA graph metrics. |
| `--kv-cache-metrics` | (flag) | (flag) | KV cache metrics. |
| `--enable-prompt-tokens-details` | (flag) | (flag) | Cached-token details in usage. |
| `--enable-per-request-metrics` | (flag) | (flag) | Per-request metrics. |
| `--profiler-config` | `{"profiler": "torch", "torch_profiler_dir": "/profiles", "torch_profiler_with_stack": false, "torch_profiler_record_shapes": false, "torch_profiler_with_memory": false, "capture_torch_profiler": false, "detailed_trace_annotation": true, "ignore_frontend": true, "delay_iterations": 0, "max_iterations": 0}` | `{"profiler": "torch", "torch_profiler_dir": "/profiles", "torch_profiler_with_stack": false, "torch_profiler_record_shapes": false, "torch_profiler_with_memory": false, "capture_torch_profiler": false, "detailed_trace_annotation": true, "ignore_frontend": true, "delay_iterations": 0, "max_iterations": 0}` | Torch profiler into the /profiles mount; idle unless started through the API. |
| `--load-format` | `instanttensor` | `instanttensor` | InstantTensor loader. |
| `--use-replayssm` | (flag) | (flag) | RecoverSSM: speculative verify with accepted-state recovery for the KDA layers. |

## Optional: weightless GLP-44 steering (TP4, off by default)

Neither profile sets these. `SWITCHLESS_WEIGHTLESS_DIR` in the TP4 site file turns the option on
(`launch/options/weightless/prepare.py`, `docs/operations.md`); `launch/render.py` and `launch/compose.py` then mount
`DIR/model.py` over the served `model.py` and `DIR/control.gguf` at `/opt/weightless/control.gguf`, read-only, and set
the four variables just before the image. The TP2 profile refuses it.

| variable | TP4 | TP2 | meaning | read by |
|---|---|---|---|---|
| `SWITCHLESS_WEIGHTLESS_DIR` | - | - | Site file, unset by default: the directory `prepare.py` wrote (`model.py`, `control.gguf`), at the same path on every node. | `launch/render.py`, `launch/compose.py` |
| `WEIGHTLESS_STEER_PATH` | - | - | Weightless option: `/opt/weightless/control.gguf`, the GLP-44 vector. The steered file refuses to start without a `.gguf` path. | the steered `model.py` |
| `WEIGHTLESS_STEER_ALPHA` | - | - | Weightless option: `2.0`, the steering strength; the steered file refuses any other value. | the steered `model.py` |
| `WEIGHTLESS_STEER_HOOK` | - | - | Weightless option: `post_layer`, the post-layer stream of every layer; the steered file refuses any other hook. | the steered `model.py` |
| `WEIGHTLESS_VECTOR_SHA256` | - | - | Weightless option: `0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c`; the steered file checks the vector against it before loading. | the steered `model.py` |

## Docker options

| option | meaning |
|---|---|
| `-d` | Detached. |
| `--name` | Container name (site value). |
| `--network host` | Host networking. |
| `--ipc host` | Host IPC namespace. |
| `--gpus all` | The node's GPU. |
| `--cap-add IPC_LOCK` | Pin memory for RDMA. |
| `--security-opt label=disable` | No SELinux relabelling of mounts. |
| `--device /dev/infiniband` | RDMA devices. |
| `--ulimit memlock=-1:-1` | Unlimited locked memory for RDMA registration. |
| `--shm-size 32g` | Shared memory. |
| `--label nvfp4.candidate.boot=...` | A label of the lab's launcher; no runtime effect. |
| `--security-opt seccomp=unconfined` | io_uring for the weight loader. |

The mounts are listed in each profile (`mounts`) and in `sources/installed-files.json`.
