# syntax=docker/dockerfile:1
#
# Oracle 2 (strengthened-test labels) in one image: the SWE-ABS harness (Yu et al., ICML 2026) plus Keel.
#
# The harness runs SWE-bench's per-instance images as SIBLING containers through the host's Docker
# daemon (the socket is mounted at run time), so this image only holds Python code, not the
# test environments. Build from the keel repository root:
#
#   keel eval abs-docker-build                      # or:
#   docker build -f docker/oracle2/Dockerfile -t keel-oracle2 .
#
# and run through `keel eval abs-run --docker ...`, which adds the mounts (see keel/eval/abs_docker.py).
# Needs an x86-64 Docker engine with Linux containers (Linux, or Docker Desktop with the WSL 2 backend).

FROM python:3.11-slim

# Pinned for reproducibility (top of SWE-ABS main on 2026-10-26). Override with --build-arg.
ARG SWEABS_REPO=https://github.com/OpenAgentEval/SWE-ABS
ARG SWEABS_REF=26b4c736ca3d566d8395d7999304d0264abe7992
# SWE-ABS keeps its *.json files in Git LFS. The harness reads the strengthened tests from Hugging Face,
# not from those files, so LFS objects are skipped by default; set SWEABS_LFS=1 to fetch them.
ARG SWEABS_LFS=0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/hf

RUN apt-get update \
 && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN if [ "$SWEABS_LFS" = "1" ]; then git lfs install; else export GIT_LFS_SKIP_SMUDGE=1; fi \
 && git clone "$SWEABS_REPO" /opt/SWE-ABS \
 && cd /opt/SWE-ABS \
 && git checkout "$SWEABS_REF" \
 && if [ "$SWEABS_LFS" = "1" ]; then git lfs pull; fi \
 && bash setup.sh \
 && python -c "import swebench.runtest.run_evaluation_test, sweabs_utils, docker"

# Keel itself (for `keel eval abs-labels` inside the container and the version stamp).
COPY pyproject.toml README.md /opt/keel/
COPY keel /opt/keel/keel
RUN pip install /opt/keel && keel eval --help > /dev/null

RUN mkdir -p /data/submissions /cache/hf /opt/SWE-ABS/swe-bench/swe_abs_res /opt/SWE-ABS/swe-bench/logs
WORKDIR /opt/SWE-ABS/swe-bench
LABEL org.opencontainers.image.title="keel-oracle2" \
      org.opencontainers.image.description="SWE-ABS strengthened-test harness + Keel, for thesis Oracle 2"

CMD ["python", "-m", "swebench.runtest.run_evaluation_test", "--help"]
