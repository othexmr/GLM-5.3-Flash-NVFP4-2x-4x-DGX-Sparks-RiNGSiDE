# Stage 04: b12x KDA prefill

`b12x-kda-prefill-overlay/` holds the `b12x/policy` and `b12x/sequence` packages that this stage adds to b12x 1.3.0.
They are b12x's public KDA prefill work (local-inference-lab/b12x, Apache License 2.0), kept byte for byte as the lab
used them and pinned by `../SHA256SUMS` and `b12x-kda-prefill-overlay.SHA256SUMS`. This note is the recipe's, not a
stage input.

- 19 of the 22 files are byte-identical (Git blob) to local-inference-lab/b12x commit
  `4b389912412bed957cb5c1654b24c6c39d387316` (Luke Alonso, 2026-09-04, "fix(kda-prefill): order prepare before
  recurrence").
- The lab took the files from `c083e1d4`, its local rewrite of 2026-09-05 on b12x commit
  `287be2e40b43f5c18f914174b30ff75b3df7f25c`; `c083e1d4` itself is a local commit, not in the public repository (the
  lab's record).
- Three files differ from `4b389912`:
  - `b12x/sequence/kda_prefill/_cute_kernels.py` (sha256 `1580ca67...`) was changed in the lab of the GLM-5.3
    RiNGSiDE recipe (othexmr). It combines the two upstream fixes of the prefill pipeline: it is the file of
    `4b389912` with the prepare pipeline of `b794879d` (Martin Vit, 2026-09-03, "fix(kda-prefill): guarantee producer
    progress across windows"): a high-priority prepare stream, and the prepare of the next window enqueued after the
    recurrence of the current one, which waits for its own prepare.
  - `b12x/policy/_profiles/data/nvidia.gb10.48sm.json.gz` and `nvidia.rtx.pro.6000.blackwell.json.gz` (stored here as
    base64, `*.json.gz.b64`) are, decoded, byte-identical to the files of `287be2e4`, the base of the lab's rewrite
    (the same bytes are in `b794879d`). `4b389912` carries other versions of both.

Checked on 2026-09-25 against GitHub and against the lab's clone of local-inference-lab/b12x (fetched 2026-09-18);
the origin of the two profile data files in that clone only, where `287be2e4` is the tip of the branch
`integration/glm53-r20-release-20260903`. The vLLM integration patches, the static oracle and the GPU gate of this
stage are the lab's (`../README.md`, "Licences"; `NOTICE`, "Base image stages").
