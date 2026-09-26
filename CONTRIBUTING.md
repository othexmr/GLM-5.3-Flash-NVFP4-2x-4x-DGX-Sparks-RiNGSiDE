# Contributing

Contributions are welcome: bug reports, documentation fixes, new measurements and profile changes.

## Before you start
- For anything larger than a small fix, open an issue first, so the approach can be agreed before you put time in.
- Keep each pull request to one reviewable change, with commit messages that say what changed and why.

## How changes are made
- Profiles change only by importing a lab plan, measured or derived from a measured plan (`docs/updating.md`). The importer regenerates the profile, its
  patches, its `src/` files and the manifests, and checks every patch against its preimage and served hash.
- A change to an upstream-owned file is a patch against the base image's file. New files and the recipe's own modules go
  to `src/<profile>/`. Keep the upstream SPDX and copyright headers of every modified file.
- Site values (addresses, host names, paths, interfaces) belong in your own `site.env`, which git ignores. Never commit
  them.

## Checks
Run the CPU checks before opening a pull request:

```sh
python3 -B sources/verify.py
python3 -B tests/run_cpu.py
```

In the pull request, say which checks ran and which native, GPU or serving checks did not. A CPU check or a successful
build does not qualify a change for serving.

## Performance claims
- Back every performance claim with a measurement: the benchmark's own output (for RigMark, its JSON and card), the
  hardware, the profile or plan, and the date.
- Keep measured results apart from estimates.
- Don't put numbers from other checkpoints, drafters or runtimes next to these as if they were equal.
- Don't change a profile on the strength of a kernel-level speedup alone; the serving measurement decides.

## Licence
- By contributing, you agree that your contribution is licensed under the licence of the files it changes. Original
  recipe code and documentation use the repository licence, the Apache License 2.0 (`LICENSE`). Patches to and
  copies of third-party files keep their upstream licence (`NOTICE`, `licenses/README.md`).
- If you bring in third-party code, add its licence and notice.

## Security
Please don't report security problems in public issues. Use GitHub's private vulnerability reporting for this
repository instead.

## Upstream
Changes that belong in vLLM, b12x, the sparse-MLA plugin, NCCL, switchless-nccl or SparkRing follow the receiving
project's own contribution rules.
