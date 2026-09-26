# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.3-Flash NVFP4 RiNGSiDE: one build definition for the base image, the native artefacts and one ready-to-run
# image per profile. docker/README.md has the commands; docs/build-provenance.md records every input.
#
#   base            the lab's base image (measured as sha256:017fd0ba...) rebuilt from pinned sources: the vLLM
#                   source image, the lab's image stages 01-19 (docker/base), the vLLM v0.29.0 port, stage 21,
#                   the thinking gate and the mixed-prefill repair, in the order the lab built them; stage 02 and
#                   the port carry the recipe's DFlash2 capture and KV-cache group (docker/base/README.md), so
#                   model.py and kv_cache_utils.py differ from the measured image
#   nccl, sparse-mla-tp4, sparse-mla-tp2, fp8-gemm, prefill-rdma, humming
#                   the native artefacts and the Humming package, built inside the base from their recipes in build/
#   profile-tp4     base + the TP4 overlay (sources/apply.py) baked in at the served paths
#   profile-tp2     base + the TP2 overlay baked in
#
# Build context: the repository root (docker/README.md). The first stage is the vLLM source image, built by
# ZJY0516/vllm's own docker/Dockerfile at 4500c80c and passed in as the named build context `vllm-source` (the compose
# file's `vllm-source` service, or `--build-context vllm-source=docker-image://...`).
#
# Every input is pinned by commit, Git tree, sha256 or image digest, except what docs/build-provenance.md lists as
# resolved at build time (Ubuntu, CUDA and Rocky Linux packages, the rustup installer, pip build tools of the build-only
# stages). A rebuild compiles its own native bytes: it is a new image and needs its own qualification (docker/README.md).
# Uses the Docker Engine's built-in Dockerfile frontend (BuildKit: RUN --mount, heredocs, ADD --checksum).

ARG MANYLINUX_BUILDER=pytorch/manylinuxaarch64-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359@sha256:f91599c49f526c77d01b68286f2bf943a5fd6a432d7e3f0afcc5784825908fe9

########################################################################################################################
# 00  vLLM source image, then its Python distributions pinned to the lab's build
########################################################################################################################
FROM vllm-source AS source-upstream

FROM source-upstream AS source
# exact: re-pin drifted releases to the lab's build of 2026-09-03 (default); check: fail on any difference;
# report: record differences and continue (a different image).
ARG SOURCE_PIN=exact
RUN --mount=type=bind,source=docker/base/00-source/distributions.txt,target=/ctx/distributions.txt \
    --mount=type=bind,source=docker/tools/pin_source.py,target=/ctx/pin_source.py \
    python3 -B /ctx/pin_source.py --recorded /ctx/distributions.txt --mode "${SOURCE_PIN}"

# The lab's stage inputs, checked against docker/base/SHA256SUMS. Two binary data files of stage 04 are stored as
# base64 text in Git and decoded here; stage 04 checks the decoded bytes against the lab's own sums.
FROM source AS inputs
COPY docker/base /inputs
RUN cd /inputs && sha256sum -c --quiet SHA256SUMS \
 && find . -name '*.gz.b64' -exec sh -c 'base64 -d "$1" > "${1%.b64}" && rm "$1"' _ {} \;

########################################################################################################################
# Source checkouts (build-only stages; git comes from the Ubuntu archive of the source image)
########################################################################################################################
FROM source AS fetch
RUN apt-get update -y && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

FROM fetch AS src-libertai
RUN git init -q /src/libertai && cd /src/libertai \
 && git fetch -q --depth 1 https://github.com/Libertai/vllm-sparse-mla-blackwell.git b1f20638d7f82f880e4bcb2136ee2f19c06f38a4 \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = b1f20638d7f82f880e4bcb2136ee2f19c06f38a4 \
 && test "$(git rev-parse 'HEAD^{tree}')" = f72c01f994fd1dbd4f6b52a5417eb9581d958b80

# The plugin files stage 02 builds, as the lab vendored them (the files its SOURCE.SHA256SUMS lists).
FROM src-libertai AS nope-plugin-src
RUN --mount=type=bind,from=inputs,source=/inputs/02-kernels/nope_plugin_b1f20638/SOURCE.SHA256SUMS,target=/sums \
    mkdir -p /opt/nope-plugin-src && cd /src/libertai \
 && for f in $(awk '{print $2}' /sums); do mkdir -p "/opt/nope-plugin-src/$(dirname "$f")" && cp -p "$f" "/opt/nope-plugin-src/$f"; done \
 && cp /sums /opt/nope-plugin-src/SOURCE.SHA256SUMS \
 && cd /opt/nope-plugin-src && sha256sum -c --quiet SOURCE.SHA256SUMS

FROM src-libertai AS src-sparse-mla-tp4
RUN --mount=type=bind,source=patches/sparse-mla/build,target=/patches \
    cd /src/libertai && git apply --index /patches/tp4-so-a1c.patch \
 && test "$(git write-tree)" = 442d83ded51b5a0226bd13fbb2e3711d45d64699

FROM src-libertai AS src-sparse-mla-tp2
RUN --mount=type=bind,source=patches/sparse-mla/build,target=/patches \
    cd /src/libertai && git apply --index /patches/tp2-so-m7e.patch \
 && test "$(git write-tree)" = ce9976030ba1b0d87566c78b8269ad73f4e25c01

FROM fetch AS src-nccl
RUN --mount=type=bind,source=patches/nccl,target=/patches \
    git init -q /src/nccl && cd /src/nccl \
 && git fetch -q --depth 1 https://github.com/NVIDIA/nccl.git 73cf112295c33aee2b895f329f592f2a9b4b0f97 \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse 'HEAD^{tree}')" = 3e7de6f92f0190d1afe9f05642e634cbf43ae4c9 \
 && git apply --index /patches/0001-switchless-ring-setup-and-listener-hardening.patch \
 && test "$(git write-tree)" = 560ba01b9becbc7d3daa1677f0216503fc3be631 \
 && git apply --index /patches/0002-dual-pf-four-gid-routing.patch \
 && test "$(git write-tree)" = 9edccf0677e35c9b451eb2f5e037dfb9e6677a93

# Cloned to the directory the lab's build installed from, so the dist-info's direct_url.json and RECORD agree.
FROM fetch AS src-humming
RUN mkdir -p /scratch/humming-v3 && git init -q /scratch/humming-v3/source && cd /scratch/humming-v3/source \
 && git fetch -q --depth 1 https://github.com/vllm-project/humming.git a74973b5079e42ef861720b62f847ce9d33447f5 \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = a74973b5079e42ef861720b62f847ce9d33447f5

# vLLM v0.29.0 with the GLM-5.3 port patch; the result must be the recorded ported tree file for file.
FROM fetch AS src-vllm-v029
RUN --mount=type=bind,from=inputs,source=/inputs/v029-port,target=/port \
    --mount=type=bind,source=docker/tools/verify_port.py,target=/tools/verify_port.py \
    git init -q /src/vllm && cd /src/vllm \
 && git fetch -q --depth 1 https://github.com/vllm-project/vllm.git 98dff2a81d747d1dba01a47f939f48c3526d4206 \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse 'HEAD^{tree}')" = d105bc357a256379b4132d47e4ed477d83c27977 \
 && git apply /port/v029-glm53-local.patch \
 && python3 -B /tools/verify_port.py --tree /src/vllm --manifest /port/source-port-manifest.json --files /port/vllm-tree.sha256

# The official vLLM v0.29.0 ARM64 CUDA 13.0 wheel: its native extensions are what the port serves.
FROM scratch AS v029-wheel
ADD --checksum=sha256:e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b0 \
    https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0-cp38-abi3-manylinux_2_28_aarch64.whl \
    /wheel/vllm-0.29.0-cp38-abi3-manylinux_2_28_aarch64.whl

# Rust artifacts of the ported source (vllm-rs, _rust_tool_parser), following the rust-build-cache stage of the ported
# source's docker/Dockerfile.csrc-fast without its unrelated PyTorch install. Toolchain 1.95.0 (what the lab's build
# resolved the source's channel "1.95" to), the Python build tools at the versions the lab's build installed.
FROM ${MANYLINUX_BUILDER} AS v029-rust
RUN dnf install -y --setopt=install_weak_deps=False git curl make perl-core && dnf clean all && rm -rf /var/cache/dnf
RUN /opt/python/cp312-cp312/bin/python3.12 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir 'setuptools>=77.0.3,<81.0.0' setuptools-rust==1.13.0 \
      setuptools-scm==10.2.3 vcs-versioning==2.3.4 semantic-version==2.10.0 wheel==0.48.0
ENV PATH=/opt/venv/bin:${PATH} CARGO_BUILD_JOBS=4
WORKDIR /workspace
COPY --from=src-vllm-v029 /src/vllm/rust rust
COPY --from=src-vllm-v029 /src/vllm/rust-toolchain.toml rust-toolchain.toml
COPY --from=src-vllm-v029 /src/vllm/tools/build_rust.py tools/build_rust.py
COPY --from=src-vllm-v029 /src/vllm/build_rust.sh build_rust.sh
RUN grep -qx 'channel = "1.95"' rust-toolchain.toml \
 && sed -i 's/^channel = "1.95"$/channel = "1.95.0"/' rust-toolchain.toml \
 && SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+docker.cache" bash build_rust.sh \
 && ls vllm/vllm-rs vllm/_rust_*.so

########################################################################################################################
# 01-19  the lab's image stages (docker/base/<stage>), in the order of its stage list; each keeps its hash gates
########################################################################################################################
FROM source AS s01-compat
RUN --mount=type=bind,from=inputs,source=/inputs/01-compat,target=/ctx \
    mkdir -p /opt/glm-release-compat \
 && cp /ctx/vllm_C_compat.py /opt/glm-release-compat/vllm_C_compat.py \
 && echo "e044fbab6a9bc472df50741f68f113d2c92007e1a9710b80c57d9400a732151d  /opt/glm-release-compat/vllm_C_compat.py" | sha256sum -c - \
 && compat_target="$(python3 -c 'import pathlib,vllm; print(pathlib.Path(vllm.__file__).parent / "_C.py")')" \
 && install -m 0644 /opt/glm-release-compat/vllm_C_compat.py "${compat_target}" \
 && python3 -m py_compile "${compat_target}"

FROM fetch AS wheels
RUN --mount=type=bind,from=inputs,source=/inputs/wheels.SHA256SUMS,target=/ctx/wheels.SHA256SUMS \
    --mount=type=bind,source=docker/tools/fetch_wheels.py,target=/ctx/fetch_wheels.py \
    python3 -B /ctx/fetch_wheels.py --sums /ctx/wheels.SHA256SUMS --out /wheels

FROM s01-compat AS s02-kernels
RUN --mount=type=bind,from=inputs,source=/inputs/02-kernels,target=/ctx \
    --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    --mount=type=bind,from=nope-plugin-src,source=/opt/nope-plugin-src,target=/nope \
    set -e \
 && mkdir -p /opt/b12x-wheels /opt/nope-plugin-src \
 && cp -a /wheels/. /opt/b12x-wheels/ \
 && cp /ctx/check_b12x.py /opt/b12x-wheels/check_b12x.py \
 && cp /ctx/test_glm5_dflash_group.py /opt/b12x-wheels/test_glm5_dflash_group.py \
 && cp /ctx/patches/glm_release_4500c80c_glm5next_dflash_aux.patch /opt/b12x-wheels/glm5next-dflash-aux.patch \
 && cp /ctx/patches/glm_release_4500c80c_glm5next_dflash_group.patch /opt/b12x-wheels/glm5next-dflash-group.patch \
 && cp /ctx/patches/glm_release_4500c80c_dflash_exact_fit_layout.patch /opt/b12x-wheels/dflash-exact-fit-layout.patch \
 && cp /ctx/patches/glm_release_4500c80c_kv_layout_dump.patch /opt/b12x-wheels/kv-layout-dump.patch \
 && cp /ctx/patches/nope_sparse_mla_metadata_buffer_growth.patch /opt/b12x-wheels/nope-sparse-mla-metadata-buffer-growth.patch \
 && cp -a /nope/. /opt/nope-plugin-src/ \
 && echo '4436911ea5b32e90dfef487509f5caabecce334bb87be3304372ca089047577d  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py' | sha256sum -c - \
 && echo '4d50afef11583100bb3b2e1e6d6e3a6a93cbeccc43c8105b6e6b62dbf646c44e  /opt/b12x-wheels/glm5next-dflash-aux.patch' | sha256sum -c - \
 && cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-wheels/glm5next-dflash-aux.patch \
 && echo '388d38b1fabacbe398973809f3b1f880f5cf6c6c2d947f7422ea3a9b416d1c8a  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py' | sha256sum -c - \
 && echo '7d299419aece21423962748d14101f60441a0f1c844c1bb37d53f6541d50da3a  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py' | sha256sum -c - \
 && echo '812960fa56801fe30936be49f40f6ff84cb6c118a563491edc282dbd492cb31c  /opt/b12x-wheels/glm5next-dflash-group.patch' | sha256sum -c - \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-wheels/glm5next-dflash-group.patch \
 && echo 'c4032115c8936f40cc5a59f6c73e9875ca990d1161192a131fd06ddd9c1c9c1d  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py' | sha256sum -c - \
 && echo '4803dcd6e0a594c1478ce335a436c4a33f8ccc874460776f2cc83aed24ca7c3c  /opt/b12x-wheels/dflash-exact-fit-layout.patch' | sha256sum -c - \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-wheels/dflash-exact-fit-layout.patch \
 && echo '1aac7e8f48e44d9b595e2a4beeff2befad32f0519a3407dc158758e8984f0eee  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py' | sha256sum -c - \
 && echo 'e099f6ec988f2d2db293fdbb9d9743e025ffa66f09dd82b03a8480b268bb871c  /opt/b12x-wheels/test_glm5_dflash_group.py' | sha256sum -c - \
 && python3 /opt/b12x-wheels/test_glm5_dflash_group.py \
 && echo '5309af1808357e0cf08629ecf93e252b0c7f32ffbef2fffb70eec3f2e78b263c  /opt/b12x-wheels/kv-layout-dump.patch' | sha256sum -c - \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-wheels/kv-layout-dump.patch \
 && echo '24b9771f321df234cf8673f1a11c60ea7f508e9a3e9cead703d40dd280a519e3  /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py' | sha256sum -c - \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
 && cd /opt/b12x-wheels \
 && sha256sum -c SHA256SUMS \
 && python3 -c 'import importlib.metadata as m; assert m.version("apache-tvm-ffi") == "0.1.11"; assert m.version("tilelang") == "0.1.12"' \
 && python3 -m pip install --no-index --no-deps --find-links /opt/b12x-wheels \
      cuda-pathfinder==1.8.1 cuda-bindings==13.3.1 cuda-core==1.0.1 cuda-python==13.3.1 \
      nvidia-cutlass-dsl-libs-base==4.6.2 nvidia-cutlass-dsl-libs-core==4.6.2 \
      nvidia-cutlass-dsl-libs-cu13==4.6.2 nvidia-cutlass-dsl==4.6.2 \
      b12x==1.3.0 \
 && python3 /opt/b12x-wheels/check_b12x.py \
 && python3 -c 'import pathlib,vllm.engine.arg_utils as a; assert "add_argument(\"--linear-backend\"" in pathlib.Path(a.__file__).read_text()' \
 && python3 -c 'import vllm; print(vllm.__version__)' > /opt/b12x-wheels/VLLM_VERSION \
 && cat /opt/b12x-wheels/VLLM_VERSION \
 && cd /opt/nope-plugin-src \
 && sha256sum -c SOURCE.SHA256SUMS \
 && cp -a /opt/nope-plugin-src /tmp/nope-plugin-build \
 && GLM53_ARCHS=121a MAX_JOBS=16 python3 -m pip install --no-index --no-deps --no-build-isolation /tmp/nope-plugin-build \
 && rm -rf /tmp/nope-plugin-build \
 && python3 -c 'import importlib.metadata as m; assert m.version("glm53_sparse_mla") == "0.1.0"; eps={e.name:e.value for e in m.entry_points(group="vllm.general_plugins")}; assert eps["glm53_sparse_mla"] == "glm53_sparse_mla.backend:register"' \
 && echo '3dcf476910517c85a87efb65bee94c99233a8438493928ab25bb0184563b8c2f  /usr/local/lib/python3.12/dist-packages/glm53_sparse_mla/backend.py' | sha256sum -c - \
 && echo 'b896a51838221133838859fbaa591c84f9eb23fcc8a088bedc50b3d3fc1cb2aa  /opt/b12x-wheels/nope-sparse-mla-metadata-buffer-growth.patch' | sha256sum -c - \
 && cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-wheels/nope-sparse-mla-metadata-buffer-growth.patch \
 && echo '7fe426aeadf323f040d336d68565119ba6748334f23411b951f13b22058536fb  /usr/local/lib/python3.12/dist-packages/glm53_sparse_mla/backend.py' | sha256sum -c - \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/glm53_sparse_mla/backend.py

FROM s02-kernels AS s03-flashkda
RUN --mount=type=bind,from=inputs,source=/inputs/03-flashkda,target=/ctx \
    mkdir -p /opt/glm5next-flashkda \
 && cp /ctx/glm_release_4500c80c_glm5next_flashkda_prefill.patch /opt/glm5next-flashkda/runtime.patch \
 && cp /ctx/test_glm5_flashkda_prefill.py /opt/glm5next-flashkda/test_glm5_flashkda_prefill.py \
 && echo '99cfc24678c0a99f2c813c20e3dcc30c8bfac4f55ec4299737689d5d51f472cd  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py' | sha256sum -c - \
 && echo 'bc1410febfabccb1986a0843931c8b0cdc2d8fe0cc7f7c230ef73c9a634886cb  /opt/glm5next-flashkda/runtime.patch' | sha256sum -c - \
 && echo '779a2cdf5adbed513e8c82248ffd53d4bd9d377a0398246d314950baa811f784  /opt/glm5next-flashkda/test_glm5_flashkda_prefill.py' | sha256sum -c - \
 && python3 -c 'import pathlib; assert list(pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm").glob("_flashkda_C*.so"))' \
 && python3 -c 'import pathlib; p=pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py"); assert "--kda-prefill-backend" in p.read_text()' \
 && cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --fuzz=0 -p1 < /opt/glm5next-flashkda/runtime.patch \
 && echo '8ea6607c1eafb515e6effc96e101d227f3655b74f191863f484adbc22883c8c2  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py' | sha256sum -c - \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py \
 && python3 /opt/glm5next-flashkda/test_glm5_flashkda_prefill.py

FROM s03-flashkda AS s04-b12x-kda
RUN --mount=type=bind,from=inputs,source=/inputs/04-b12x-kda,target=/ctx \
    mkdir -p /opt/b12x-kda \
 && cp -a /ctx/b12x-kda-prefill-overlay /opt/b12x-kda/overlay \
 && cp /ctx/b12x-kda-prefill-overlay.SHA256SUMS /opt/b12x-kda/overlay.SHA256SUMS \
 && cp /ctx/glm_release_4500c80c_b12x_kda_arg_utils.patch /opt/b12x-kda/arg_utils.patch \
 && cp /ctx/glm_release_4500c80c_b12x_kda_utils.patch /opt/b12x-kda/b12x_utils.patch \
 && cp /ctx/glm_release_4500c80c_glm5next_b12x_kda_prefill.patch /opt/b12x-kda/kda.patch \
 && cp /ctx/test_glm5_b12x_kda_prefill.py /opt/b12x-kda/static_oracle.py \
 && cp /ctx/gpu_gate_b12x_kda_prefill.py /opt/b12x-kda/gpu_gate.py \
 && echo 'c97d88635521a7fdd4f67717c1835a017d0dcea49d49cc7aa5b4299f290d19e3  /opt/b12x-wheels/b12x-1.3.0-py3-none-any.whl' | sha256sum -c - \
 && echo '7b6c613c3a839c0888cf5a8851d58618b69d7d33f2c764e831a3383f830a66f2  /opt/b12x-kda/overlay.SHA256SUMS' | sha256sum -c - \
 && echo '8380b57f9171dbef292997cc0ba9f40770d630a00cc75322de15bbab98ce68c2  /opt/b12x-kda/arg_utils.patch' | sha256sum -c - \
 && echo 'a509c909fdd20808e3b50286a61cba55bdd42837dd63396e4639d6af2ebca0d8  /opt/b12x-kda/b12x_utils.patch' | sha256sum -c - \
 && echo '41e603ac01355dbfad62429c74823639e5b7e1318059826698ac21defd5dd11f  /opt/b12x-kda/kda.patch' | sha256sum -c - \
 && echo '46fbf0e46c036efd4c80491dc549c44b06bd306e2a12b3dcc88fd8899c67a70d  /opt/b12x-kda/static_oracle.py' | sha256sum -c - \
 && echo '4828024b06b03f5b430b877b09506eeded4066671456db68f9f9b95252316823  /opt/b12x-kda/gpu_gate.py' | sha256sum -c - \
 && cd /opt/b12x-kda/overlay \
 && sha256sum -c /opt/b12x-kda/overlay.SHA256SUMS \
 && test ! -e /usr/local/lib/python3.12/dist-packages/b12x/policy \
 && test ! -e /usr/local/lib/python3.12/dist-packages/b12x/sequence \
 && echo '8ea6607c1eafb515e6effc96e101d227f3655b74f191863f484adbc22883c8c2  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py' | sha256sum -c - \
 && echo 'da89b2686e1fd25ae071cf3dba42b3bd32e810eb85c1899d8c29aa167704f2b4  /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py' | sha256sum -c - \
 && echo '99287526edeff891b8434024ad66a671b03b9eb1f89edf2dcd88797800806ec0  /usr/local/lib/python3.12/dist-packages/vllm/utils/b12x.py' | sha256sum -c - \
 && echo 'c5e54c5dcd3967ef389e9e7e34de610118097087d74682a37fabd9286f82884d  /usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/_impl.py' | sha256sum -c - \
 && echo '01a87c3e0e19ba3c7e4482a7dab318300493d3ee3dcd14bf87460e3ef149b46f  /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/dynamic.py' | sha256sum -c - \
 && echo 'abb74591a75307180761af3b71232412c40b8bf50d94a44f57b2f87e0e0ba2f5  /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/silu.py' | sha256sum -c - \
 && cp -a /opt/b12x-kda/overlay/b12x/policy /usr/local/lib/python3.12/dist-packages/b12x/ \
 && cp -a /opt/b12x-kda/overlay/b12x/sequence /usr/local/lib/python3.12/dist-packages/b12x/ \
 && cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-kda/arg_utils.patch \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-kda/b12x_utils.patch \
 && patch --batch --fuzz=0 -p1 < /opt/b12x-kda/kda.patch \
 && echo '6601dffb13ed76caebf4b4156cb1f0cfdaea1369e4924c6a4d433149b3e9d9d4  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py' | sha256sum -c - \
 && echo '01e07be29378ac38f0b6e8c52ee8dff061db6683ca91bbf525d90325ab12b56b  /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py' | sha256sum -c - \
 && echo 'b8a94398c7fba6aa498e56238b1e06632748029f9ce73eb71a2f3cd71d966565  /usr/local/lib/python3.12/dist-packages/vllm/utils/b12x.py' | sha256sum -c - \
 && echo 'c5e54c5dcd3967ef389e9e7e34de610118097087d74682a37fabd9286f82884d  /usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/_impl.py' | sha256sum -c - \
 && echo '01a87c3e0e19ba3c7e4482a7dab318300493d3ee3dcd14bf87460e3ef149b46f  /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/dynamic.py' | sha256sum -c - \
 && echo 'abb74591a75307180761af3b71232412c40b8bf50d94a44f57b2f87e0e0ba2f5  /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/silu.py' | sha256sum -c - \
 && python3 -m compileall -q \
      /usr/local/lib/python3.12/dist-packages/b12x/policy \
      /usr/local/lib/python3.12/dist-packages/b12x/sequence \
      /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/kda.py \
      /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py \
      /usr/local/lib/python3.12/dist-packages/vllm/utils/b12x.py \
      /opt/b12x-kda/static_oracle.py /opt/b12x-kda/gpu_gate.py \
 && python3 -c 'import b12x.policy; import b12x.sequence.kda_prefill as kda; assert all(hasattr(kda, name) for name in ("Caps", "plan", "bind", "run", "prewarm")); print("B12X_KDA_IMPORT_CLOSURE_OK")' \
 && python3 /opt/b12x-kda/static_oracle.py

FROM s04-b12x-kda AS s05-layout
RUN --mount=type=bind,from=inputs,source=/inputs/05-layout,target=/ctx \
    mkdir -p /opt/dflash-xqa-a75 \
 && cp /ctx/patches/glm_release_4500c80c_dflash_kv_layout_fanout.patch /opt/dflash-xqa-a75/candidate.patch \
 && cp /ctx/test_dflash_kv_layout_fanout.py /opt/dflash-xqa-a75/behavioral_oracle.py \
 && echo '2ac8ac727e53a2b75e6c3ffe1040f83d7d75f1f47bbe476f133bccb67f09ea08  /opt/dflash-xqa-a75/candidate.patch' | sha256sum -c - \
 && echo 'ee5b5b02d0c21a97c6a906f75b308f8d9b9e1eda8752e7a29fcf44618740797f  /opt/dflash-xqa-a75/behavioral_oracle.py' | sha256sum -c - \
 && echo '4a1a5266b912fe97991d8dc89d92b923990da9f6cb4b817d00d52e9c0d26dd7c  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py' | sha256sum -c - \
 && cd /usr/local/lib/python3.12/dist-packages \
 && patch --batch --fuzz=0 -p1 < /opt/dflash-xqa-a75/candidate.patch \
 && echo 'e9d5f2a5a6039143aa4a13aa01799f63034c7b50a4f35b0033a5bbd8e79cb596  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py' | sha256sum -c - \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py /opt/dflash-xqa-a75/behavioral_oracle.py \
 && python3 /opt/dflash-xqa-a75/behavioral_oracle.py

FROM s05-layout AS s06-dense
RUN --mount=type=bind,from=inputs,source=/inputs/06-dense,target=/ctx \
    set -e \
 && mkdir -p /tmp/dual_fp8 \
 && cp /ctx/SOURCE.SHA256SUMS /tmp/dual_fp8/SOURCE.SHA256SUMS \
 && cp /ctx/patch_vllm_linear_hook.py /tmp/dual_fp8/patch.py \
 && cp /ctx/patch_vllm_embedding_hook.py /tmp/dual_fp8/patch_embedding.py \
 && cp /ctx/glm53_dual_fp8_dense.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py \
 && sha256sum -c /tmp/dual_fp8/SOURCE.SHA256SUMS \
 && python3 /tmp/dual_fp8/patch.py \
 && python3 /tmp/dual_fp8/patch_embedding.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py \
 && sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/glm53_dual_fp8_dense.py \
      > /opt/dual_fp8_dense.PATCHED.SHA256SUMS \
 && rm -rf /tmp/dual_fp8

FROM s06-dense AS s07-mtp
RUN --mount=type=bind,from=inputs,source=/inputs/07-mtp,target=/ctx \
    set -e \
 && mkdir -p /tmp/mtp_fp8 \
 && cp /ctx/SOURCE.SHA256SUMS /tmp/mtp_fp8/SOURCE.SHA256SUMS \
 && cp /ctx/patch_vllm_fp8_oracle.py /tmp/mtp_fp8/patch.py \
 && sha256sum -c /tmp/mtp_fp8/SOURCE.SHA256SUMS \
 && python3 /tmp/mtp_fp8/patch.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/fp8.py \
 && sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/fp8.py > /opt/mtp_fp8_oracle.PATCHED.SHA256SUMS \
 && rm -rf /tmp/mtp_fp8

FROM s07-mtp AS s08-router
RUN --mount=type=bind,from=inputs,source=/inputs/08-router,target=/ctx \
    set -e \
 && mkdir -p /tmp/router_dedup \
 && cp /ctx/SOURCE.SHA256SUMS /tmp/router_dedup/SOURCE.SHA256SUMS \
 && cp /ctx/patch_vllm_glm5next_router_dedup.py /tmp/router_dedup/patch.py \
 && sha256sum -c /tmp/router_dedup/SOURCE.SHA256SUMS \
 && python3 /tmp/router_dedup/patch.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py \
 && sha256sum /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py > /opt/router_dedup.PATCHED.SHA256SUMS \
 && rm -rf /tmp/router_dedup

FROM s08-router AS s09-drafter
RUN --mount=type=bind,from=inputs,source=/inputs/09-drafter,target=/ctx \
    set -e \
 && mkdir -p /tmp/edge_scale \
 && cp /ctx/SOURCE.SHA256SUMS /tmp/edge_scale/SOURCE.SHA256SUMS \
 && cp /ctx/patch_vllm_dflash2_edge_scale.py /tmp/edge_scale/patch.py \
 && sha256sum -c /tmp/edge_scale/SOURCE.SHA256SUMS \
 && python3 /tmp/edge_scale/patch.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py \
 && sha256sum /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py > /opt/dflash2_edge_scale.PATCHED.SHA256SUMS \
 && rm -rf /tmp/edge_scale

FROM s09-drafter AS s10-collectives
RUN --mount=type=bind,from=inputs,source=/inputs/10-collectives,target=/ctx \
    set -e \
 && mkdir -p /tmp/tp2_allreduce \
 && cp /ctx/SOURCE.SHA256SUMS /tmp/tp2_allreduce/SOURCE.SHA256SUMS \
 && cp /ctx/patch_vllm_tp2_allreduce.py /tmp/tp2_allreduce/patch.py \
 && sha256sum -c /tmp/tp2_allreduce/SOURCE.SHA256SUMS \
 && python3 /tmp/tp2_allreduce/patch.py --check \
 && python3 /tmp/tp2_allreduce/patch.py \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py \
 && sha256sum /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py > /opt/tp2_allreduce.PATCHED.SHA256SUMS \
 && rm -rf /tmp/tp2_allreduce

FROM s10-collectives AS s11-moe
RUN --mount=type=bind,from=inputs,source=/inputs/11-moe,target=/ctx \
    set -e \
 && mkdir -p /tmp/b12x_moe_decode \
 && cp /ctx/SOURCE.SHA256SUMS /tmp/b12x_moe_decode/SOURCE.SHA256SUMS \
 && cp /ctx/patch_b12x_moe_decode.py /tmp/b12x_moe_decode/patch.py \
 && sha256sum -c /tmp/b12x_moe_decode/SOURCE.SHA256SUMS \
 && python3 /tmp/b12x_moe_decode/patch.py \
 && rm -rf /usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/__pycache__ \
           /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/__pycache__ \
 && python3 -m py_compile /usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/_impl.py \
                          /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/dynamic.py \
 && sha256sum /usr/local/lib/python3.12/dist-packages/b12x/moe/fused_moe/_impl.py \
              /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/dynamic.py \
      > /opt/b12x_moe_decode.PATCHED.SHA256SUMS \
 && rm -rf /tmp/b12x_moe_decode
RUN python3 - <<'EOF'
import os
os.environ.pop("B12X_GLM53_DECODE_FASTPATH", None)
import b12x.moe.fused_moe._impl as impl
assert impl._GLM53_DECODE_FASTPATH is False
assert impl._GLM53_DECODE_FASTPATH_MAX_ROWS == 64
assert callable(impl._glm53_decode_fastpath_selected)
from b12x.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
assert callable(getattr(MoEDynamicKernelBackend, "configure_decode_fastpath"))
print("b12x decode fast path hook present; default off")
EOF

FROM s11-moe AS s12-reasoning
RUN --mount=type=bind,from=inputs,source=/inputs/12-reasoning,target=/ctx \
    mkdir -p /opt/glm53-reasoning-fix && cp -a /ctx/. /opt/glm53-reasoning-fix/ \
 && python3 /opt/glm53-reasoning-fix/install.py \
 && python3 /opt/glm53-reasoning-fix/test_parser.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream_extended.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream_multi.py \
 && python3 /opt/glm53-reasoning-fix/test_required_ids.py \
 && python3 /opt/glm53-reasoning-fix/test_responses_boundary.py \
 && python3 /opt/glm53-reasoning-fix/test_responses_identity.py \
 && python3 /opt/glm53-reasoning-fix/test_strict_auto.py

FROM s12-reasoning AS s13-deterministic-fastpath
RUN --mount=type=bind,from=inputs,source=/inputs/13-deterministic-fastpath,target=/ctx \
    mkdir -p /opt/deterministic-fastpath && cp -a /ctx/. /opt/deterministic-fastpath/ \
 && python3 /opt/deterministic-fastpath/install.py

FROM s13-deterministic-fastpath AS s14-attention-tail
RUN --mount=type=bind,from=inputs,source=/inputs/14-attention-tail,target=/ctx \
    mkdir -p /opt/attention-tail && cp -a /ctx/. /opt/attention-tail/ \
 && python3 /opt/attention-tail/install.py

FROM s14-attention-tail AS s15-topk-repair
RUN --mount=type=bind,from=inputs,source=/inputs/15-topk-repair,target=/ctx \
    mkdir -p /opt/topk-repair && cp -a /ctx/. /opt/topk-repair/ \
 && MAX_JOBS=2 python3 /opt/topk-repair/build.py \
 && python3 /opt/topk-repair/install.py \
 && python3 /opt/topk-repair/test_install.py

FROM s15-topk-repair AS s16-deterministic-topk
RUN --mount=type=bind,from=inputs,source=/inputs/16-deterministic-topk,target=/ctx \
    mkdir -p /opt/deterministic-topk && cp -a /ctx/. /opt/deterministic-topk/ \
 && python3 /opt/deterministic-topk/test_install.py \
 && python3 /opt/deterministic-topk/install.py

FROM s16-deterministic-topk AS s17-tool-call-truncation
RUN --mount=type=bind,from=inputs,source=/inputs/17-tool-call-truncation,target=/ctx \
    mkdir -p /opt/tool-call-truncation && cp -a /ctx/. /opt/tool-call-truncation/ \
 && cd /opt/tool-call-truncation \
 && (python3 test_cpu.py --variant original > /tmp/tool-call-truncation-before.json 2>/dev/null || true) \
 && python3 install.py \
 && (python3 test_cpu.py --variant fixed > /tmp/tool-call-truncation-after.json 2>/dev/null || true) \
 && python3 gate.py /tmp/tool-call-truncation-before.json /tmp/tool-call-truncation-after.json \
 && python3 /opt/glm53-reasoning-fix/test_parser.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream_extended.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream_multi.py \
 && python3 /opt/glm53-reasoning-fix/test_required_ids.py \
 && python3 /opt/glm53-reasoning-fix/test_responses_boundary.py \
 && python3 /opt/glm53-reasoning-fix/test_responses_identity.py \
 && python3 /opt/glm53-reasoning-fix/test_strict_auto.py

# Stage 18 (a rejected top-k variant) is not part of the lineage: stage 19 follows stage 17 directly.
FROM s17-tool-call-truncation AS s19-runtime-bugfixes
RUN --mount=type=bind,from=inputs,source=/inputs/19-runtime-bugfixes,target=/ctx \
    mkdir -p /opt/runtime-bugfixes && cp -a /ctx/. /opt/runtime-bugfixes/ \
 && python3 /opt/runtime-bugfixes/install.py \
 && python3 /opt/runtime-bugfixes/test_runtime_repairs.py \
 && python3 /opt/glm53-reasoning-fix/test_parser.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream_extended.py \
 && python3 /opt/glm53-reasoning-fix/test_required_stream_multi.py \
 && python3 /opt/glm53-reasoning-fix/test_required_ids.py \
 && python3 /opt/glm53-reasoning-fix/test_responses_boundary.py \
 && python3 /opt/runtime-bugfixes/test_responses_identity.py \
 && python3 /opt/glm53-reasoning-fix/test_strict_auto.py \
 && python3 /opt/runtime-bugfixes/drafter-batch-fix/install.py \
 && python3 /opt/runtime-bugfixes/drafter-batch-fix/test_cpu.py

########################################################################################################################
# v029  vLLM v0.29.0 with the GLM-5.3 port: official wheel natives + ported Python tree + rebuilt Rust artifacts,
#       installed over stage 19 (the lab's Dockerfile.runtime-official)
########################################################################################################################
FROM s19-runtime-bugfixes AS v029-assemble
COPY --from=v029-wheel /wheel/vllm-0.29.0-cp38-abi3-manylinux_2_28_aarch64.whl /native-wheels/
COPY --from=v029-rust /workspace/vllm /rust-artifacts
COPY --from=src-vllm-v029 /src/vllm/vllm /ported-source/vllm
COPY --from=inputs /inputs/v029-port/assemble-wheel.py /assemble-wheel.py
RUN python3 /assemble-wheel.py

FROM s19-runtime-bugfixes AS v029-port
COPY --from=v029-assemble /assembled-wheels /opt/vllm-v029-build
RUN uv pip install --system --no-deps --reinstall /opt/vllm-v029-build/*.whl
ENV VLLM_BUILD_COMMIT=98dff2a81d747d1dba01a47f939f48c3526d4206 \
    VLLM_BUILD_PIPELINE=local-glm53-backport \
    VLLM_IMAGE_TAG=local/glm53-v029:ported-20260910
LABEL org.opencontainers.image.version="0.29.0+glm53.local"

########################################################################################################################
# 21, thinking gate, mixed-prefill repair: the lab's layers over the port, then the base
########################################################################################################################
FROM v029-port AS s21-atomic-prefill
RUN --mount=type=bind,from=inputs,source=/inputs/21-atomic-prefill,target=/ctx \
    mkdir -p /opt/atomic-prefill-candidate && cp -a /ctx/. /opt/atomic-prefill-candidate/ \
 && python3 /opt/atomic-prefill-candidate/test_phase_policy.py \
 && python3 /opt/atomic-prefill-candidate/install.py

FROM s21-atomic-prefill AS thinking-gate
RUN --mount=type=bind,from=inputs,source=/inputs/thinking-gate,target=/ctx \
    mkdir -p /opt/glm53-thinking-gate && cp -a /ctx/. /opt/glm53-thinking-gate/ \
 && python3 -B /opt/glm53-thinking-gate/check_preimage.py \
 && cp /opt/glm53-thinking-gate/glm53_moe.py /usr/local/lib/python3.12/dist-packages/vllm/parser/glm53_moe.py \
 && cp /opt/glm53-thinking-gate/test_*.py /opt/glm53-reasoning-fix/ \
 && python3 -B /opt/glm53-thinking-gate/thinking_contract.py --template /opt/glm53-thinking-gate/chat_template_mm.jinja \
 && python3 -B /opt/glm53-reasoning-fix/test_parser.py \
 && python3 -B /opt/glm53-reasoning-fix/test_required_ids.py \
 && python3 -B /opt/glm53-reasoning-fix/test_required_stream.py \
 && python3 -B /opt/glm53-reasoning-fix/test_required_stream_extended.py \
 && python3 -B /opt/glm53-reasoning-fix/test_required_stream_multi.py \
 && python3 -B /opt/glm53-reasoning-fix/test_responses_boundary.py \
 && python3 -B /opt/glm53-reasoning-fix/test_responses_identity.py \
 && python3 -B /opt/glm53-reasoning-fix/test_strict_auto.py

FROM thinking-gate AS mixed-prefill-repair
RUN --mount=type=bind,from=inputs,source=/inputs/mixed-prefill-repair,target=/tmp/glm53-build \
    python3 -B /tmp/glm53-build/verify_install.py --stage source --receipt /tmp/glm53-build/overlay-receipt.json \
 && mkdir -p /opt/glm53-speedup \
 && cp -a /tmp/glm53-build/. /opt/glm53-speedup/ \
 && cp -a /opt/glm53-speedup/overlay/. /usr/local/lib/python3.12/dist-packages/ \
 && python3 -B /opt/glm53-speedup/verify_install.py --stage overlay
ENV GLM53_SPEEDUP_MODE=off GLM53_SPEEDUP_CAPTURE=off \
    GLM53_SPEEDUP_KDA_STRIDES=0 GLM53_SPEEDUP_NOPE_COPIES=0 \
    GLM53_SPEEDUP_NOPE_LAYOUT=0 GLM53_SPEEDUP_NOPE_CONCAT=0 \
    GLM53_SPEEDUP_MIXED_PREFILL=0 GLM53_SPEEDUP_PREFILL_TAILS=0
ENV GLM53_SPEEDUP_MIXED_PREFILL_FAIR=0 GLM53_SPEEDUP_MIXED_PREFILL_RUNTIME=0

# The rebuilt base: every file a profile patch replaces must have its recorded preimage (docker/tools/check_base.py).
FROM mixed-prefill-repair AS base
RUN --mount=type=bind,source=sources/installed-files.json,target=/recipe/installed-files.json \
    --mount=type=bind,source=docker/tools/check_base.py,target=/recipe/check_base.py \
    python3 -B /recipe/check_base.py --manifest /recipe/installed-files.json

########################################################################################################################
# Native artefacts and Humming, built inside the base (build/ has the recipes)
########################################################################################################################
FROM base AS nccl
ARG NCCL_JOBS=4
COPY --from=src-nccl /src/nccl /build/nccl
RUN cd /build/nccl \
 && make -j"${NCCL_JOBS}" src.build CUDA_HOME=/usr/local/cuda-13.0 NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121' \
 && mkdir -p /artefacts && cp build/lib/libnccl.so.2.30.7 /artefacts/libnccl.so.2.30.7

FROM base AS sparse-mla-tp4
COPY --from=src-sparse-mla-tp4 /src/libertai /build/sparse-mla
RUN cd /build/sparse-mla \
 && GLM53_ARCHS=120a,121a python3 setup.py build_ext --inplace \
 && mkdir -p /artefacts && cp glm53_sparse_mla/_C.cpython-312-aarch64-linux-gnu.so /artefacts/

FROM base AS sparse-mla-tp2
COPY --from=src-sparse-mla-tp2 /src/libertai /build/sparse-mla
RUN cd /build/sparse-mla \
 && GLM53_ARCHS=120a,121a python3 setup.py build_ext --inplace \
 && mkdir -p /artefacts && cp glm53_sparse_mla/_C.cpython-312-aarch64-linux-gnu.so /artefacts/

# PyTorch's extension loader imports the module after linking it; the CUDA driver stub stands in for libcuda.so.1
# during this GPU-less build only (the image does not keep it).
FROM base AS fp8-gemm
COPY build/glm53_fp8_gemm /src/glm53_fp8_gemm
COPY docker/tools/build_fp8_gemm.py /tools/build_fp8_gemm.py
RUN if [ ! -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        apt-get update -y && apt-get install -y --no-install-recommends cuda-driver-dev-13-0 && rm -rf /var/lib/apt/lists/*; \
    fi \
 && mkdir -p /tmp/cuda-stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
 && LIBRARY_PATH=/usr/local/cuda/lib64/stubs LD_LIBRARY_PATH="/tmp/cuda-stub:${LD_LIBRARY_PATH}" \
    python3 -B /tools/build_fp8_gemm.py --src /src/glm53_fp8_gemm --work /build/fp8-gemm --out /artefacts

FROM base AS prefill-rdma
COPY src/tp4/glm53_prefill_rdma/_prb.cu /build/prb/_prb.cu
RUN cd /build/prb \
 && test "$(sha256sum _prb.cu | cut -c1-16)" = 7ffd99abacd3e746 \
 && /usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode arch=compute_121,code=sm_121 -Xcompiler -fPIC -shared \
      -o libprb-7ffd99abacd3e746.so _prb.cu -libverbs -lpthread \
 && mkdir -p /artefacts && cp libprb-7ffd99abacd3e746.so /artefacts/

# The rowspread top-k extension (TP4, build/topk_rowspread): stage 15's persistent top-k sources with the rowspread
# persistent_topk.cuh and binding. The loader refuses a library that its build receipt does not name, so both come
# from this build. The extension loader loads the library after linking it; as in the fp8-gemm stage, the CUDA driver
# stub stands in for libcuda.so.1 during this GPU-less build only.
FROM base AS topk-rowspread
COPY build/topk_rowspread /src/topk_rowspread
RUN if [ ! -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        apt-get update -y && apt-get install -y --no-install-recommends cuda-driver-dev-13-0 && rm -rf /var/lib/apt/lists/*; \
    fi \
 && mkdir -p /tmp/cuda-stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
 && LIBRARY_PATH=/usr/local/cuda/lib64/stubs LD_LIBRARY_PATH="/tmp/cuda-stub:${LD_LIBRARY_PATH}" \
    python3 -B /src/topk_rowspread/build.py --src /opt/topk-repair/src --work /build/topk-rowspread --out /artefacts

FROM base AS humming
COPY --from=src-humming /scratch/humming-v3/source /scratch/humming-v3/source
COPY build/humming/constraints.txt /build/humming/constraints.txt
RUN python3 -m venv --system-site-packages /opt/humming-build \
 && /opt/humming-build/bin/pip install --no-cache-dir -c /build/humming/constraints.txt setuptools setuptools-scm wheel \
 && SETUPTOOLS_SCM_PRETEND_VERSION=0.1.15 /opt/humming-build/bin/pip install --no-cache-dir --no-deps --no-build-isolation \
      /scratch/humming-v3/source \
 && /opt/humming-build/bin/python -c 'import importlib.metadata as m; assert m.version("humming-kernels") == "0.1.15"' \
 && site="$(/opt/humming-build/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
 && mkdir -p /third-party \
 && cp -a "${site}/humming" "${site}/humming_kernels-0.1.15.dist-info" /third-party/ \
 && find /third-party -name __pycache__ -prune -exec rm -rf {} +

FROM scratch AS artefacts-tp4
COPY --from=nccl /artefacts/ /artefacts/
COPY --from=sparse-mla-tp4 /artefacts/ /artefacts/
COPY --from=fp8-gemm /artefacts/glm53_fp8_gemm.so /artefacts/
COPY --from=prefill-rdma /artefacts/ /artefacts/
COPY --from=topk-rowspread /artefacts/ /artefacts/

FROM scratch AS artefacts-tp2
COPY --from=sparse-mla-tp2 /artefacts/ /artefacts/

########################################################################################################################
# Profiles: sources/apply.py builds the overlay against the rebuilt base, the overlay is baked in at the served paths
########################################################################################################################
FROM base AS overlay-tools
RUN apt-get update -y && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

FROM overlay-tools AS overlay-tp4
RUN --mount=type=bind,source=.,target=/recipe \
    --mount=type=bind,from=artefacts-tp4,source=/artefacts,target=/artefacts \
    --mount=type=bind,from=humming,source=/third-party,target=/third-party \
    python3 -B /recipe/sources/apply.py --profile tp4 --image-root / --artefacts /artefacts \
      --third-party /third-party --rank 0 --allow-rebuilt --out /overlay

FROM overlay-tools AS overlay-tp2
RUN --mount=type=bind,source=.,target=/recipe \
    --mount=type=bind,from=artefacts-tp2,source=/artefacts,target=/artefacts \
    python3 -B /recipe/sources/apply.py --profile tp2 --image-root / --artefacts /artefacts --allow-rebuilt --out /overlay

FROM base AS profile-tp4
RUN --mount=type=bind,from=overlay-tp4,source=/overlay,target=/overlay \
    --mount=type=bind,source=sources,target=/recipe/sources \
    --mount=type=bind,source=docker/tools,target=/recipe/docker/tools \
    python3 -B /recipe/docker/tools/bake_overlay.py --profile tp4 --overlay /overlay \
      --manifest /recipe/sources/installed-files.json \
 && mkdir -p /opt/glm53-switchless/recipe/docker \
 && cp -a /recipe/sources /opt/glm53-switchless/recipe/ \
 && cp -a /recipe/docker/tools /opt/glm53-switchless/recipe/docker/ \
 && python3 -B /opt/glm53-switchless/recipe/docker/tools/check_served.py --profile tp4 --rank 0 \
      --manifest /opt/glm53-switchless/recipe/sources/installed-files.json
LABEL org.opencontainers.image.title="GLM-5.3-Flash NVFP4 RiNGSiDE TP4" \
      org.opencontainers.image.source="https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE" \
      org.opencontainers.image.description="Self-built image of the TP4 profile: rebuilt base, rebuilt native artefacts, overlay baked in; qualify before serving"

FROM base AS profile-tp2
RUN --mount=type=bind,from=overlay-tp2,source=/overlay,target=/overlay \
    --mount=type=bind,source=sources,target=/recipe/sources \
    --mount=type=bind,source=docker/tools,target=/recipe/docker/tools \
    python3 -B /recipe/docker/tools/bake_overlay.py --profile tp2 --overlay /overlay \
      --manifest /recipe/sources/installed-files.json \
 && mkdir -p /opt/glm53-switchless/recipe/docker \
 && cp -a /recipe/sources /opt/glm53-switchless/recipe/ \
 && cp -a /recipe/docker/tools /opt/glm53-switchless/recipe/docker/ \
 && python3 -B /opt/glm53-switchless/recipe/docker/tools/check_served.py --profile tp2 \
      --manifest /opt/glm53-switchless/recipe/sources/installed-files.json
LABEL org.opencontainers.image.title="GLM-5.3-Flash NVFP4 RiNGSiDE TP2" \
      org.opencontainers.image.source="https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE" \
      org.opencontainers.image.description="Self-built image of the TP2 profile: rebuilt base, rebuilt native artefact, overlay baked in; qualify before serving"
