# Python Unsloth side. GPU passthrough at run time. Venv is inside the image.
FROM docker.io/nvidia/cuda:12.8.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && pip install --upgrade pip

# Torch first (cu128). Do not bake unsloth here — the combined layer OOMs
# /var/tmp on commit. run.sh pip-installs into the image venv at runtime
# (plus cu128 torchvision) and records a failed import as FAIL_ENV.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch \
    && pip install numpy

# Triton JIT needs a host C compiler; python3-dev does not pull gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY compare/py /opt/compare/py
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["/opt/compare/py/generate_and_unsloth.py"]
