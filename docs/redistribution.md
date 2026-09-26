# Redistribution

RiNGSiDE distributes source code, patches, recorded hashes and build scripts. It distributes no binaries: no container
images, no native artefacts and no model weights. Its own code and documentation are under the Apache License 2.0; the
patches and third-party files it carries keep their upstream licences (`NOTICE`, `licenses/README.md`).

Building the images with `docker/Dockerfile` downloads third-party software and combines it in the image you build.
You accept the terms of what you download, and anyone who passes a built image on redistributes all of it. Among the
components, each under its own terms:

- NVIDIA's CUDA container image (`nvidia/cuda`), the CUDA toolkit packages from NVIDIA's package repository, and the
  NVIDIA libraries that the Python packages bring (cuBLAS, cuDNN, cuSPARSELt, NCCL, the CUTLASS DSL and others),
  under NVIDIA's licences for them;
- PyTorch's builder image (used in build stages only) and its Python wheels; Ubuntu and deadsnakes packages;
- vLLM, FlashInfer, b12x, the sparse-MLA plugin (Libertai/vllm-sparse-mla-blackwell), Humming and NVIDIA NCCL, whose
  sources are Apache-2.0 or, for NCCL, its own licence (`licenses/nccl-LICENSE.txt`), and some two hundred other
  Python packages of the vLLM image;
- the model and drafter weights, which you download separately under their Hugging Face terms.

Check each licence before you share an image or run it for others; RiNGSiDE makes no statement about whether a
particular use or redistribution is allowed. A license notice list of the built image is not generated; the package
metadata inside the image (`pip show`, `/usr/share/doc`) is where its components state their licences.
