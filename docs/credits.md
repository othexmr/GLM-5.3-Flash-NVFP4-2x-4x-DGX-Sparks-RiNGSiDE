# Credits

This recipe stands on other people's work. Licence notices are in `NOTICE` and `licenses/`; this page credits the
projects by their owner and repository.

## Built on

| Project | What the recipe uses |
|---|---|
| [zai-org](https://huggingface.co/zai-org/GLM-5.3-Flash) GLM-5.3-Flash (Hugging Face) | the model; the base of the served chat template (MIT, `NOTICE`) |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | the served chat template file, its `files/chat_template.jinja` byte for byte: zai-org's template whose one change is the thinking gate written by Raymond Lucke ([rwl4](https://github.com/rwl4), commit `eaf90d0`); Mia's AI Lab restored his Reasoning Effort gate in `a5cc004` and Rusty's AI Lab ported zai-org's later corrections in `68ce597` (MIT, `NOTICE`); MiaAI-Lab's own adaptive verification length (`GLM53_ADAPTIVE_K`) that the recipe's verification-length policy follows; its mixed-prefill work and issue reports, which informed the fair-prefill experiments; a comparison recipe |
| [rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8](https://github.com/rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8) and [knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4) | the idea of the TP4 profile's prefill indexer row split (`GLM53_INDEXER_ROW_SPLIT`): it comes from rhys101's SG18 native prefill TP split, adapted to TP4 by knapcio; `glm53_indexer_rowsplit.py` is the recipe's own implementation, no code copied |
| [Red Hat AI](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) GLM-5.3-Flash-NVFP4 (Hugging Face) | the NVFP4 checkpoint the profiles serve, read with the `tokenizer.json` of [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |
| [NVIDIA/nccl](https://github.com/NVIDIA/nccl) | the collective library under both profiles; the TP4 build is patched |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | the GEMM templates of the FP8 GEMM extension, including two CUTLASS-derived epilogue headers vendored through vLLM |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | the serving engine (v0.29.0 in the base image); the token-sharded mHC idea first appeared in its DeepSeek-V4 sequence parallelism ([#46789](https://github.com/vllm-project/vllm/pull/46789) by WoosukKwon); RecoverSSM comes from its Kimi-K3 support ([#51855](https://github.com/vllm-project/vllm/pull/51855) by ZJY0516, with benchislett), which began as ReplaySSM support for Kimi-K3 and runs under the `--use-replayssm` option of ReplaySSM ([#48018](https://github.com/vllm-project/vllm/pull/48018) by Johnny-Liou, a Dao AI Lab and NVIDIA collaboration: keep a checkpoint and the recent inputs, rebuild the state from them); the DFlash2 auxiliary hidden-state capture of the TP4 profile and the base image is adapted from its `DeepseekV4Model.forward` (the capture of [#46995](https://github.com/vllm-project/vllm/pull/46995) by benchislett), and the drafter KV-cache group is built on its KV-cache group mechanisms (`NOTICE`); the TP4 profile's MLA q_a norm with fused FP8 quantization is adapted from its fused q/kv RMSNorm kernel (from its DeepSeek-V4 support, [#40860](https://github.com/vllm-project/vllm/pull/40860)), and its rowspread top-k extension is a change to its persistent top-k kernel ([#37421](https://github.com/vllm-project/vllm/pull/37421) by LopezCastroRoberto) with the hunks of [#55314](https://github.com/vllm-project/vllm/pull/55314) by Dovis01 (open), a fix whose concept comes from [sgl-project/sglang#37625](https://github.com/sgl-project/sglang/pull/37625) by ormandj and bold84, as its description says |
| [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention), Flash Linear Attention (Songlin Yang, Yu Zhang) | the gated RMS norm kernel whose body the TP4 profile's fused o_norm and FP8 quantization kernel copies, through vLLM's copy (MIT, `NOTICE`); FLA's gated, row-tiled form of Tri Dao's Triton layer norm, with vLLM's head-strided gate addressing ([#50089](https://github.com/vllm-project/vllm/pull/50089)) |
| [state-spaces/mamba](https://github.com/state-spaces/mamba) (Tri Dao) | the Triton layer norm (`mamba_ssm/ops/triton/layernorm.py`, Copyright (c) 2023 Tri Dao, Apache-2.0, after the Triton LayerNorm tutorial) that FLA's gated norm kernel derives from; its normalisation steps remain in the TP4 fused o_norm kernel (`NOTICE`) |
| [vllm-project/vllm#58454](https://github.com/vllm-project/vllm/pull/58454) by mmastrac, building on [#55219](https://github.com/vllm-project/vllm/pull/55219) by ivanium | the kpool tail ring sized for speculative decoding, which the TP4 profile serves in its `kpool_compress.py` and `attention.py` patches |
| [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx) | its port of vllm-project/vllm#58454 (commit `1d4b59e`), which the TP4 profile's kpool tail ring follows |
| [vllm-project/vllm#50843](https://github.com/vllm-project/vllm/pull/50843) by alexbi29 | the tile argmax of the sampling kernels clamped to the vocabulary, which the TP4 profile serves in its `gumbel.py` and `rejection_sampler_utils.py` patches |
| [ZJY0516/vllm](https://github.com/ZJY0516/vllm) | the GLM-5.3 model sources of its glm-release branch (commit `4500c80c`, merge base `ee17d0d8` with vllm-project/vllm main), written mostly by JaredforReal with ZJY0516, ivanium and Isotr0py and upstreamed as [vllm-project/vllm#53906](https://github.com/vllm-project/vllm/pull/53906), which the base image ports onto vLLM v0.29.0, and the vLLM source build of that commit that the image stages start from (`docs/build-provenance.md`) |
| [vllm-project/humming](https://github.com/vllm-project/humming) (Jinzhen Lin) | Humming NVFP4 and dense GEMMs in the TP4 dense dispatch |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | four FlashInfer 0.6.18 files (XQA sources, NVIDIA's) that the base image's drafter-batch fix changes (`docker/base/19-runtime-bugfixes/drafter-batch-fix`, `NOTICE`) |
| [scitix/InstantTensor](https://github.com/scitix/InstantTensor) | the checkpoint loader both profiles select (`--load-format instanttensor`), used unmodified from the base image |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) (Luke Alonso and contributors, among them Martin Vit) | KDA, MoE and dense kernels (its KDA prefill prepare and recurrence kernels, Luke Alonso's, are carried in `glm53_kda_conv.py` and `glm53_kda_ckpt.py`); RoCEnante, the RDMA transport by Jason Cook ([original-el8](https://github.com/original-el8), [#295](https://github.com/local-inference-lab/b12x/pull/295)) that the lean all-reduce is adapted from |
| [Libertai/vllm-sparse-mla-blackwell](https://github.com/Libertai/vllm-sparse-mla-blackwell) (Moshe Malawach, LibertAI Labs) | the sparse-MLA plugin |
| [incoai](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) GLM-5.3-Flash-DFlash2 (Hugging Face) | the speculative drafter: Inco AI's DFlash 2, which extends DFlash by Jian Chen, Yesheng Liang and Zhijian Liu ([z-lab/dflash](https://github.com/z-lab/dflash), ICML 2026), as its model card credits (CC BY-NC-ND 4.0 per its model card, for research and evaluation; downloaded by the operator, neither redistributed nor modified by the recipe) |
| [FujitsuPolycom/sparkring](https://github.com/FujitsuPolycom/sparkring) (SparkRing) | the switchless-cycle NCCL patches (their skip-Tree/skip-PAT approach was first published by Joseph Rose in [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless), which SparkRing and switchless-nccl credit; no code of his is included), the mHC prefill package, the KDA checkpoint export and the vendored RoCEnante changes |
| [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl) | the hardened switchless NCCL patch that the RiNGSiDE dual-PF patch extends |

## Benchmarks and optional components

| Project | Use |
|---|---|
| [alexellis/rigmark](https://github.com/alexellis/rigmark) | RigMark, the benchmark the results are scored with; the staggered-arrival and replay sections come from the fork [othexmr/rigmark](https://github.com/othexmr/rigmark) (`feat/staggered-arrival`, commit `40fabcaf`) |
| [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) | the decode benchmark protocol (DecodeBench) the sparkDash cells reproduce in Python |
| [eugr/llama-benchy](https://github.com/eugr/llama-benchy) | llama-benchy, run on the RiNGSiDE TP4 profiles in the same boots as their RigMark cells |
| [SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench) | tool-eval-bench, the tool-calling benchmark run on the final RiNGSiDE TP4 profile and on the MiaAI-Lab TP4 recipe; its scenario methodology is adapted from [stevibe/ToolCall-15](https://github.com/stevibe/ToolCall-15) (MIT), as its README credits |
| [msuiche/weightless](https://github.com/msuiche/weightless) | the optional GLP-44 projective activation steering of the TP4 profile (off by default): the method, the GLM-5.3 patcher (commit `15ed1373`) that `launch/options/weightless/prepare.py` fetches and applies on the operator's machine, and the vector [msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44](https://huggingface.co/msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44) (revision `ef85b016`, MIT, gated: each operator fetches it with their own token); used, not redistributed (no licence is stated for the repository); also measured as an additional arm on an earlier TP4 base |

## Compared with

The comparison recipes in `bench/results/README.md`: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks,
tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark and tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark. They are
compared as whole recipes, each with its own checkpoint, quantisation, drafter, image, runtime and admission limit
(`bench/results/evidence/comparison-arms.json`), on the same hardware and benchmarks.

Checkpoints the comparison recipes loaded (compared only; this recipe neither builds on nor redistributes them):

| Checkpoint (Hugging Face) | Revision | Loaded by |
|---|---|---|
| [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) | `5ab363a8dcf6405955fd5f99671e01a1c9fb124b` | the MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks arms (the default `MODEL` of their `.env.example` is a mirror with the same bytes) |
| [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4) | `18d55bfd` | the tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark arm (the lab's download is identical to that revision) |
| [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | `423acf37` | the tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark arm, converted locally with that recipe's own tools |
