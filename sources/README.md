# Sources and tools

| File | Purpose |
|---|---|
| `installed-files.json` | Every served path of both profiles: kind, sha256 (and bytes), where the bytes come from (patch, `src/`, profile file, artefact), licence, attribution, and for patches the preimage sha256 and its evidence. |
| `apply.py` | Builds a profile's overlay directory from the base image's files, the patches, `src/` and the native artefacts, checking every hash. |
| `verify.py` | CPU consistency checks of the lock, the patches, the manifests, the profiles (mount options and table, launch identity against the canonical launch), the licences and the artefacts across all manifests. `--release` always fails and lists what publication still needs. |
| `import_plan.py` | Imports a lab launch plan (measured, or derived from a measured plan) into a profile (`docs/updating.md`): builds every output in a staging directory, scans all of them for private values, checks every copy's sha256, and only then swaps them in. |
| `licence_table.py` | Regenerates the per-file table of `licenses/README.md` from the manifest. |

`../sources.lock.json` records the upstream bases (with Git trees), every patch with its sha256, the preimage and
postimage blob ids and sha256, the relation of each preimage to upstream, the NCCL series with the tree after each
patch, and the sparse-MLA build patches with their result trees.

## Import document (`switchless-import/1`)

`import_plan.py` reads one JSON document per profile. It is produced outside this repository from the lab plan,
because it holds private site values (they are used only to check the rendering and never written).

```text
schema        "switchless-import/1"
topology      "tp4" | "tp2"
plan          {sha256, label, deterministic, nondeterministic_note,
               derived_from (a plan derived from a measured plan: the base plan's sha256 and label, a note, the steps),
               qualification (that plan's own checks: plan_sha256, evidence, not_established)}
image         {id, tag}
argv          {rank: [docker, run, ...]}              the measured command of every rank
site          values        {SWITCHLESS_...: value}    shared site values of the plan
              per_rank      {rank: {SWITCHLESS_RANK_HOST_IP, SWITCHLESS_CONTAINER_NAME,
                                    SWITCHLESS_JIT_CACHE_DIR, SWITCHLESS_PROFILES_DIR}}
              mount_sources {container path: SWITCHLESS_...}   mounts supplied by the site
              env           {ENV_KEY: SWITCHLESS_...}           environment values supplied by the site
              args          {--flag: SWITCHLESS_...}            server arguments supplied by the site
files         [{target, sha256, bytes, local, kind: patch|src|template, component, licence, attribution,
                evidence, preimage: {sha256, local, evidence, origin, upstream_relation} (patch and template)}]
dirs          [{target, kind: src-dir|config-dir|binary-dir|third-party-dir, licence, attribution,
                files: {name: {sha256, local | binary+artefact(+per_rank_sha256)}}, symlinks,
                source_dir (third-party-dir: the directory of the installation that supplies the mount),
                distribution {name, version} (a dist-info source_dir)}]
binaries      [{target, sha256, bytes, artefact, local, evidence}]
```

Every mount of the plan must be either a site mount or one of the `files`, `dirs` or `binaries`; every rank's argv
must differ from rank 0's only in the declared site values, `--node-rank` and the role arguments.

Besides the profile's files, the importer writes `launch/profiles/<profile>/canonical-launch.json`: the commands
`launch/render.py` renders for a documentation site, with the profile's `launch_sha256` (a digest of the argv
template, role arguments, mount table, image and every served byte). The same digest and the lab plan's sha256 are
recorded in `profile.json` and `release/provenance.json`; `verify.py` recomputes the digest and re-renders the
fixture, so a hand edit of a flag, a mount or a served file fails until a new plan is imported.
