# Experimental TP8 native library

`docker/Dockerfile` target `nccl-tp8` builds the separately locked series in
`patches/nccl-tp8` with CUDA 13.0 and
`NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121'`, using `make -j4 src.build`.
`compile-receipt.json` records the first ARM64 build. Rebuilt bytes may differ;
the tester kit binds every test to the actual library hash. Compilation alone
does not establish collective correctness, TP8 serving or speed.
