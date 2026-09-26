# GPU and serving checks

No GPU or node test runs in CI. The CPU suite (`tests/run_cpu.py`) checks metadata, patches, rendering and the
overlay builder; it does not check numerics, collectives or serving. Before serving a rebuilt artefact or a changed
profile, measure it on the hardware against the recorded profile, as `bench/protocols/README.md` describes.
