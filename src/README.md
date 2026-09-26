# Files served in full

`src/tp4/` and `src/tp2/` hold the files a profile mounts that are not modifications of an upstream file in the base
image: the recipe's own modules (`glm53_speedup`, the KDA, top-k and dense modules, the lean all-reduce, the prefill
RDMA ring) and new files placed inside upstream package directories. Below each profile directory the path is the
install path under `/usr/local/lib/python3.12/dist-packages/`.

The two directories hold the served bytes of each profile; where both serve the same file the bytes are identical,
and Git stores them once. Every file's sha256, licence and attribution is in `sources/installed-files.json`
(summary in `licenses/README.md`). Some files carry comments that refer to the lab's research archive
("outputs/..."); that archive is not part of this repository. A `NOTICE` file beside
served files (for example in `tp4/glm53_lean_allreduce/`) documents provenance and is not part of the served bytes.
