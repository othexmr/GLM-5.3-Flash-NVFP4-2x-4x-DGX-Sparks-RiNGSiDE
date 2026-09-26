# Launch profiles

`profiles/tp4` and `profiles/tp2` each hold:

- `profile.json`: the rank-0 `docker run` argv of the lab plan as a template (`argv_template`), with `${...}`
  placeholders for site values, `${RANK}`, `${ROLE_ARGS}` (rank 0: `--host 0.0.0.0 --port ${SWITCHLESS_API_PORT}`;
  other ranks: `--headless`) and `${OVERLAY_ROOT}` for the files `sources/apply.py` builds; the mount table; the image;
  the determinism flag; the lab plan it was imported from.
- `override.json`: the runtime override the scheduler reads (`GLM53_SPEEDUP_OVERRIDE`).
- `chat_template_mm.jinja`: the chat template, MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks's thinking-gated GLM-5.3
  template (thinking gated by the request; `NOTICE`).

`render.py` prints the per-rank commands from a site file (`site.env.example`); it never runs them. Every variable
and argument is explained in `docs/configuration.md`. The import checked that this template reproduces every rank of
the lab plan when the lab's site values are filled in.

For the self-built profile image (`docker/README.md`):

- `compose.py` writes one Docker Compose project per rank, `deploy/<profile>/rank<N>/compose.yaml` and `.env`: the
  service `glm53` carries the rendered command's image, mounts, environment, network, IPC, GPUs, devices, capabilities,
  security options, ulimits, shared memory, labels, container name and server command, without the overlay mounts
  (the image bakes those files in at the same paths), and a `build:` section for `docker/Dockerfile`'s profile target;
  the build-only service `vllm-source` builds the vLLM source image. `tests/cpu/test_compose.py` checks every rank of
  both profiles against the canonical command, option by option. `deploy/` holds site data and is ignored by Git.
- `up.sh` starts a profile: optionally builds the image on rank 0's node and copies it to the others (`--build`),
  checks that every node has the same image ID, starts the ranks with `docker compose up`, workers before rank 0, and
  waits for `/health` and `/v1/models`. `down.sh` stops them, rank 0 first. Both print their commands and run them only
  with `--go`; they reach the nodes over ssh and change no host settings.

Optional, off by default: `options/weightless/prepare.py` prepares the weightless GLP-44 steering of the TP4 profile
(msuiche/weightless, fetched and applied on the operator's machine; `README.md`, `docs/operations.md`). With
`SWITCHLESS_WEIGHTLESS_DIR` in the TP4 site file, `render.py` and `compose.py` mount the steered `model.py` and the
vector and set the four `WEIGHTLESS_*` variables; without it their output is the profile's, byte for byte.
