# Humming 0.1.15 (TP4)

The TP4 profile mounts `humming-kernels` 0.1.15 over the image's 0.1.12: the `humming` package directory and its
`humming_kernels-0.1.15.dist-info` directory, which is mounted at the image's `humming_kernels-0.1.12.dist-info`
path. `sources/installed-files.json` names the source directory of each mount (`source_dir`) and the distribution
the dist-info must declare (`distribution`: humming-kernels 0.1.15); `release/assets.json` repeats the mapping.

Source: [vllm-project/humming](https://github.com/vllm-project/humming) at
`a74973b5079e42ef861720b62f847ce9d33447f5` (Apache-2.0). Runtime dependencies (torch, triton, numpy, safetensors,
jinja2, cuda-bindings) come from the base image; the package is installed without them.

## Build

Inside the base image (the lab built in a container of the image), in a virtual environment that sees the image's
packages, with the pinned build requirements of `constraints.txt` and without build isolation:

```sh
git clone https://github.com/vllm-project/humming.git humming-src && git -C humming-src checkout a74973b5079e42ef861720b62f847ce9d33447f5
python3 -m venv --system-site-packages venv
venv/bin/pip install -c build/humming/constraints.txt setuptools setuptools-scm wheel
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.15 venv/bin/pip install --no-deps --no-build-isolation ./humming-src
venv/bin/python -c 'import humming; assert humming.__version__ == "0.1.15"'
```

Pass the environment's `site-packages` directory (`venv/lib/python3.12/site-packages`, which holds `humming/` and
`humming_kernels-0.1.15.dist-info/`) to `sources/apply.py --third-party`. `apply.py` takes each mount from its
`source_dir`, checks that the dist-info's METADATA declares humming-kernels 0.1.15, and compares every file with the
measured bytes (`__pycache__` files excluded).

## What a rebuild reproduces

What the served files record about the lab's build:

- the build backend: the dist-info `WHEEL` file names setuptools 84.0.0, and `INSTALLER` names pip;
- the install source: `direct_url.json` names the directory pip installed from (`file:///scratch/humming-v3/source`
  in the lab's build container), and `RECORD` lists the hashes of the installed files, including `direct_url.json`;
- `humming/_version.py` is written by setuptools_scm, whose version the lab build did not record.

A rebuild from the same commit therefore reproduces the package sources, but `direct_url.json` and `RECORD` differ
unless the source directory has the same path, and `_version.py` may differ if setuptools_scm writes another
template. `apply.py` reports such files as rebuilt and accepts them only with `--allow-rebuilt`; a rebuilt Humming
tree is a new artefact and needs its own measurement.
