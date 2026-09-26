# Operations

Two ways to launch a profile:
- **Self-built image with Docker Compose** (the Quick start of `README.md`): `docker/Dockerfile` builds one image per
  profile with the overlay baked in, `launch/compose.py` renders one Compose project per rank, and `launch/up.sh` starts
  the ranks and waits for the API. `launch/up.sh` and `launch/down.sh` print their commands and run them only with
  `--go`; the image is not the measured one until qualified (`docker/README.md`).
- **The measured form**: the base image with each profile's overlay mounted from the node, as `launch/render.py`
  renders it. Nothing in this repository runs a command on a node here: the operator builds the overlay, renders the
  commands, reviews them and runs them.

Both need the prerequisites below; the rows on the base image and the native artefacts apply to the measured form,
whose image and artefacts are not published.

## Prerequisites (every node)

| What | Measured setup | Check |
|---|---|---|
| Hardware | NVIDIA DGX Spark (GB10, 128 GB unified memory), one ConnectX-7 with two ports | `nvidia-smi` lists one GB10 |
| Driver and CUDA | a driver that runs CUDA 13.0 containers on SM 12.1; the image brings CUDA 13.0.3, Python 3.12, PyTorch 2.13.0+cu130 | `nvidia-smi` shows the driver and CUDA version |
| Docker with GPU access | Docker Engine with the NVIDIA Container Toolkit; the profiles run `docker run --gpus all --network host --ipc host` with `--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`, `--shm-size 32g` and `seccomp=unconfined` (`docs/configuration.md`) | `docker run --rm --gpus all sha256:017fd0ba... nvidia-smi` |
| RDMA access | `/dev/infiniband` and the RDMA devices under `/sys/class/infiniband` on the host (the container shares the host network); TP4 needs all four PCIe functions, TP2 the two of the `f1` port | `fabric/preflight_tp4.py` (TP4), `ibv_devices` |
| Fabric | cabled and addressed as `fabric/README.md` describes | `fabric/preflight_tp4.py --rank N` exits 0 on every TP4 node |
| Image | the self-built profile image (`docker/README.md`) with the same image ID on every node, or the measured base image `sha256:017fd0ba0a265dd81b78b352b14d33df03e9996f8f45870d4c3df84e08b1ebcb` (`docs/image.md`) | `docker image inspect --format '{{.Id}}' IMAGE` |
| Weights | the model and drafter at the pinned revisions (`release/assets.json`) | the model directory holds `model.safetensors.index.json` with sha256 `26765b26...` |

The TP4 NCCL build, the lean all-reduce and the prefill RDMA ring assume the TP4 cabling and addressing of
`fabric/README.md` exactly; the two TP4 transports also open the four functions by name and use GID index 3
(`fabric/README.md`, "Fixed device names and GID index").

## Preflight sequence

1. **Fabric.** On every TP4 node run `python3 fabric/preflight_tp4.py --rank N` (N is the node's rank in the ring).
   It reads sysfs only and checks the four device names, that GID index 3 is a RoCE v2 entry with an IPv4-mapped
   address, and that each address lies in the subnet of the function's cable. For TP2, check the two `f1` functions
   the same way by hand (`/sys/class/infiniband/<dev>/ports/1/gid_attrs/types/3`, `.../gids/3`).
2. **Image and GPU.** Load the base image and run the `nvidia-smi` check above in it.
3. **Weights.** Download `RedHatAI/GLM-5.3-Flash-NVFP4` at revision `18d55bfd5a2194887738da73753975c9d3842f46` into a
   directory (`SWITCHLESS_MODEL_DIR`), the `tokenizer.json` of `nvidia/GLM-5.3-Flash-NVFP4` at revision
   `423acf37583782c51c142d145aef733d72943d93` as a file (`SWITCHLESS_TOKENIZER_FILE`: the profiles mount it over the
   checkpoint's own) and `incoai/GLM-5.3-Flash-DFlash2` at revision `7d74cdd881ed7e32c31175984a67823127b66cfe` into a
   Hugging Face cache (`SWITCHLESS_DRAFTER_REPO_DIR` is the `models--incoai--GLM-5.3-Flash-DFlash2` repository
   directory; the server reads `snapshots/7d74cdd8.../` below it).
4. **Native artefacts and Humming** (`build/`): the NCCL library, the sparse-MLA and FP8 GEMM extensions and the
   prefill RDMA library in one artefacts directory (files named by their sha256 or their file name), and a
   humming-kernels 0.1.15 installation for TP4 (`build/humming/README.md`).
5. **Overlay.** Copy the image's Python packages out once (`docs/image.md`) and build the overlay on each node:
   `python3 -B sources/apply.py --profile tp4 --image-root IMAGE_ROOT --artefacts ARTEFACTS --third-party
   HUMMING_SITE_PACKAGES --rank N --out OVERLAY`. It must exit 0 (`OVERLAY/apply-receipt.json` lists every file as
   patched, copied or measured-artefact); `--rank` selects the prefill RDMA library of that rank.
6. **Directories.** Create, on every node, the bind-mount sources the site file names, owned by the user that runs
   Docker:
   - `SWITCHLESS_HF_CACHE_DIR` (mounted at `/cache`; `HF_HOME=/cache/huggingface`, the server runs offline);
   - `SWITCHLESS_JIT_CACHE_ROOT/rank<N>` (mounted at `/root/.cache`: kernel and compile caches, several GB after the
     first boot; one directory per rank and per profile, never shared);
   - `SWITCHLESS_PROFILES_ROOT/rank<N>` (mounted at `/profiles`: profiler and telemetry output).
7. **Commands.** Copy `site.env.example` to `site-tp4.env` (git ignores it), fill in the addresses, interfaces and
   paths, and render: `python3 -B launch/render.py --profile tp4 --site site-tp4.env`. Values may be quoted; a path
   used in a mount must not contain a comma; `SWITCHLESS_IMAGE`, if set, must be pinned by digest. Review the
   commands.

## Launching

- With Docker Compose: `launch/up.sh --profile tp4 --site site.env --go` copies each rank's project to
  `SWITCHLESS_REMOTE_REPO/deploy/<profile>/rank<N>` on its node and runs `docker compose up -d --no-build --pull never
  glm53` there, workers first; `launch/down.sh` runs `docker compose down`, rank 0 first. The compose service carries
  the same docker-run options as the rendered command, without the overlay mounts (`launch/compose.py`).
- Start ranks 1..N-1 (headless) first, then rank 0, which serves the OpenAI-compatible API on `SWITCHLESS_API_PORT`.
- The first boot fills the per-rank JIT caches; later boots reuse them. Expect the first boot to take much longer.
- Before the API listens, rank 0 runs the startup warm-up and resets the prefix cache: TP4 17 prompts, TP2 six (two
  of 28,672 tokens each, then one at each chunk row count the profile's benchmarks meet). The log shows
  `GLM53_STARTUP_WARMUP done`, and every worker logs `GLM53_STARTUP_WARMUP worker rank=<N>`.
- TP4 logs `GLM53_KDA_CKPT_ENGAGED`, `GLM53_KDA_STATE_ENGAGED mode=fp16-slow32`, `GLM53_GREEDY_ARGMAX_VERIFY engaged`,
  `GLM53_REPLAY_BOUNDARY on:` (rank 0) and `GLM53_MOE_ATOMIC_DECODE engaged` for the non-deterministic MoE path; TP2
  logs `GLM53_MOE_MIXED_SPLIT engaged` on every rank, and its prefix-cache contract and `GLM53_REPLAY_BOUNDARY on:` on
  rank 0.
- With `GLM53_INDEXER_ROW_SPLIT=1` (TP4) every rank logs `GLM53_INDEXER_ROW_SPLIT ready: rank` at start-up and
  `GLM53_INDEXER_ROW_SPLIT engaged: rank` on the first prefill chunk long enough to split; `GLM53_INDEXER_ROW_SPLIT
  settings differ across TP ranks` stops the start.
- Rank 0's engine configuration line names the checkpoint format: `quantization=compressed-tensors` with
  RedHatAI/GLM-5.3-Flash-NVFP4, `quantization=modelopt_fp4` (with the warning `Detected ModelOpt NVFP4 checkpoint
  (quant_algo=NVFP4)`) with nvidia/GLM-5.3-Flash-NVFP4.
- A listening port is not readiness: send a completion request and check that it returns a full answer, for example

```sh
curl -s http://HEAD_NODE:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nvidia/GLM-5.3-Flash-NVFP4",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 512,
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

  (`HEAD_NODE` is rank 0's address). `"thinking": false` in `chat_template_kwargs` switches the thinking block off
  (Raymond Lucke's thinking gate in the chat template of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, `NOTICE`).

## Optional: weightless GLP-44 steering (TP4)

Off by default; it changes the model's behaviour (msuiche/weightless GLP-44 steering, `README.md`), not its speed.
1. Request access to [msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44](https://huggingface.co/msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44)
   on Hugging Face with your account, and make your own token available on the machine that prepares the files:
   `HF_TOKEN` in the environment or `huggingface-cli login`. The script sends it only to huggingface.co and never
   prints or stores it.
2. `python3 -B launch/options/weightless/prepare.py --out DIR` (Python 3.11+, Git, network access to GitHub and
   Hugging Face). It rebuilds the TP4 `model.py` (`f826b431...`) from this repository, fetches the weightless patcher
   (msuiche/weightless `15ed1373`, `patches/hotfix-glm53-steering-projective.py`, sha256 `63520932...`; its strings
   are read, the script is not run), applies it with the recipe's guards, and writes `DIR/model.py` (sha256
   `8cbfea91c45de7f2f1f61ca4164f36647583bb812a3eecb1901accf432c27db7`: the file the lab's verification window served, `78693488...`, with the author note of one comment left out, token-equal with comments stripped), `DIR/control.gguf` (revision `ef85b016`,
   2,887,584 bytes, sha256 `0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c`) and
   `DIR/weightless-receipt.json`. Any other byte stops it before anything is written. `--cache DIR` uses local copies
   (named by file name or sha256), `--offline` never downloads, `--skip-vector` writes `model.py` only.
3. Copy `DIR` to the same path on every node and set `SWITCHLESS_WEIGHTLESS_DIR=DIR` in the TP4 site file. Render or
   compose as usual: every rank mounts `DIR/model.py` over the served `model.py` and `DIR/control.gguf` at
   `/opt/weightless/control.gguf`, read-only, with `WEIGHTLESS_STEER_PATH`, `WEIGHTLESS_STEER_ALPHA=2.0`,
   `WEIGHTLESS_STEER_HOOK=post_layer` and `WEIGHTLESS_VECTOR_SHA256` (`docs/configuration.md`). On the base image
   the mount replaces the overlay's `model.py`; on the self-built profile image it covers the baked one.
4. Check that every rank logs `weightless GLP steering active` at boot. The steered file fails closed: without the
   vector, with other bytes, another alpha, hook, layer set or topology it stops the rank
   (`Weightless overlay requires its pinned GGUF vector`, `Weightless vector identity mismatch`, ...).
5. To turn it off, remove the site line and render or compose again; the default profile comes back unchanged.

## API trust boundary

The server listens on `0.0.0.0:SWITCHLESS_API_PORT` of rank 0 with **no authentication and no TLS**, and the
containers share the host network. Anyone who can reach the port can use the model. Run it only on a
trusted network, restrict the port with a host firewall, or put an authenticating TLS reverse proxy in front of it
and let only the proxy reach the port. The bootstrap port (`SWITCHLESS_MASTER_PORT`) and the RDMA fabric carry
unauthenticated traffic between the ranks and belong on a private network.

## Troubleshooting

| Symptom | Where to look |
|---|---|
| `apply.py` reports "not the recorded preimage" | the image is not `sha256:017fd0ba...`, or `IMAGE_ROOT` was copied from another image (`docs/image.md`) |
| `apply.py` reports a missing or rebuilt artefact | the artefacts directory or the Humming installation (`build/`); a rebuilt artefact needs `--allow-rebuilt` and its own measurement |
| `render.py` refuses a value | quotes, a comma in a mounted path, or an image reference without a digest |
| a rank exits during startup with `missing RDMA device ...` | the TP4 transports did not find one of the four functions: run `fabric/preflight_tp4.py` |
| NCCL fails to connect, or the TP4 NCCL library rejects a listener | addresses outside the four cable subnets or a function without its IPv4 GID (`fabric/README.md`) |
| a TP4 rank exits with code 86 | the lean all-reduce or prefill RDMA transport lost a peer or timed out; check the fabric and restart all ranks |
| the first boot takes long | it compiles kernels into the per-rank cache directories; later boots with the same directories start faster |
| out of memory or host memory pressure | the KV pools are sized for the measured nodes (see Memory); nothing else should run on the nodes |

## Memory

The GB10 memory is unified. The KV pools are sized for the measured nodes (TP4 32 GiB per rank, TP2 10.5 GiB). When
changing them, move in small steps and watch host memory; a much larger pool starved the lab's nodes.
