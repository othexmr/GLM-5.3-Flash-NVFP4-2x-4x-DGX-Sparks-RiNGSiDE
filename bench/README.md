# Benchmarks

- `results/README.md`: the results page (TP4 and TP2 tables, method, caveats); `results/results.json`: its data;
  `results/evidence/`: the records behind it; `results/runs/`: the receipts of the final TP4 run (2 runs) and of the
  previous TP4 profile's 5-run window (RigMark JSON and cards, llama-benchy results).
- `protocols/README.md`: what each benchmark measures and how the runs were made.

The benchmark tools are separate projects: RigMark ([alexellis/rigmark](https://github.com/alexellis/rigmark)) with
the staggered-arrival and replay sections of the fork [othexmr/rigmark](https://github.com/othexmr/rigmark) (branch
`feat/staggered-arrival`, commit `40fabcaf`, published with the identical tree as the measured revision `6e0e24a6`;
`patches/rigmark/README.md`), [eugr/llama-benchy](https://github.com/eugr/llama-benchy),
[SeraphimSerapis/tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), and the lab's Python
reproduction of the decode-benchmark protocol of [MiaAI-Lab/sparkDash](https://github.com/MiaAI-Lab/sparkDash) (the
sparkDash cells). They are not needed to serve.
