# Updating a profile

A profile changes by importing a lab launch plan and the files it serves; nothing is edited by hand. A plan is either
measured, or derived from a measured plan (the TP4 RiNGSiDE release plan: change notices, new code compared bit for
bit with the code it replaces, an environment change). The import document of a derived plan names the base plan and
each change (`derived_from`) and carries the checks the plan passed and what they did not establish
(`qualification`); the importer writes both into the profile (`source_plan`), and `sources/verify.py` requires the
qualification to be that plan's own.

1. Write an import document (schema `switchless-import/1`, described in `sources/README.md`) for the new plan: its
   per-rank argv, the site values it used, and for every served file its target, sha256 and a local copy, plus the
   image file (preimage) for files that replace upstream-owned image files. The lab produces this document from its
   plan and staging receipts; site values stay in the document and are never written to the repository.
2. Import it:

   ```sh
   python3 -B sources/import_plan.py import-tp4.json
   ```

   The importer re-renders every rank from the new template and compares it with the plan, regenerates
   `launch/profiles/<profile>/` (including `canonical-launch.json`), `patches/*/<profile>/`, `src/<profile>/`, the
   profile's entries in `sources/installed-files.json` and `sources.lock.json` and its lab plan and launch digest in
   `release/provenance.json`, and applies every patch to its preimage and requires the served sha256. Everything is
   first written to a staging directory; every staged file (patches, `src/`, the chat template, the profile and the
   manifests) is scanned for the document's site values (addresses, host paths, container names) and for private
   paths and addresses, and every copy is checked against its served sha256. Only then are the profile's directories
   and the shared files swapped in; if anything fails, the repository is left exactly as it was. The other profile is
   left as it is.
3. Update what the importer does not own: `docs/configuration.md` (the CPU tests fail if a variable of a profile is
   not documented there), `release/assets.json` for new or changed native artefacts, the descriptive fields of
   `release/provenance.json`, `NOTICE` and `licenses/components.json` for new third-party material, `CURRENT.md` and
   the results page.
4. Run `python3 -B sources/verify.py` and `python3 -B tests/run_cpu.py`.

The optional weightless steering (`launch/options/weightless/prepare.py`) pins the TP4 profile's served `model.py`
and the steered file built on it. An import that changes that `model.py` makes `tests/cpu/test_weightless.py` fail and
`prepare.py` refuse until the lab has built, and measured, a steered file for the new one and its pins and edits are
updated.
