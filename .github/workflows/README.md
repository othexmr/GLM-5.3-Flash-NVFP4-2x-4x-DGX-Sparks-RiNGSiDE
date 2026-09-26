# CI

`cpu-tests.yml` ("CPU checks") runs `sources/verify.py` and `tests/run_cpu.py` on hosted Ubuntu runners (Python 3.11
and 3.13) with read-only permissions and commit-pinned actions. It needs no secrets and runs no hardware job. The
workflow has run on GitHub and passed there on both Python versions. `tests/run_cpu.py` discovers every
`tests/cpu/test_*.py`, so the checks of `docker/`, `launch/compose.py`, `launch/up.sh` and `launch/down.sh` run in the
same workflow; none of them builds an image or reaches a node.
