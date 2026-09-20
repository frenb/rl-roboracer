# fly.ai connectome service (see docs/flybrain-driver-plan.md, step 1).
#
# Build context: this directory.
#   docker compose build fly-brain
#
# WHY A SEPARATE SERVICE: flybrain needs Python 3.10+, and sim-controller is
# Python 3.8 (nvidia/cuda:11.0.3-runtime-ubuntu20.04 + TF 2.7). The two cannot
# share an interpreter, so the brain runs here and the trainer talks to it.
#
# CUDA 12 is safe: the host driver is 566.36, which reports CUDA 12.7. Ubuntu
# 22.04 ships Python 3.10.12, which already satisfies flybrain's floor, so
# there is no deadsnakes PPA to maintain.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
        ca-certificates \
        vim \
        iputils-ping \
        net-tools \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN python -m pip install --no-cache-dir --upgrade pip

# cupy-cuda12x is pinned explicitly rather than left to the [gpu] extra, which
# may resolve to a CUDA 11 build.
RUN pip install --no-cache-dir \
        "flybrain[gpu]" \
        cupy-cuda12x \
        numpy

# gRPC pinned to the trainer's runtime: sim-controller reports grpcio 1.51.1 /
# protobuf 3.20.1 on Python 3.8, and stubs must load there.
#
# grpcio-tools is 1.48.2, NOT 1.51.1: the 1.51 line requires protobuf>=4.21,
# which cannot coexist with the trainer's 3.20.1, and protobuf 4.x codegen is
# unloadable on a 3.x runtime. 1.48.2 is the last release whose protobuf floor
# is <4.0, and its protoc emits builder-style code that 3.20.1 reads fine.
# All three have cp310 wheels. If you bump any of them, regenerate the stubs
# and bump sim-controller in the same change.
RUN pip install --no-cache-dir \
        grpcio==1.51.1 \
        grpcio-tools==1.48.2 \
        protobuf==3.20.1

# The ~260 MB prebuilt connectome lands here. Mounted as a named volume in
# compose so it survives image rebuilds.
ENV FLY_DATA=/fly_data
ENV FLY_DEVICE=cuda
RUN mkdir -p /fly_data

WORKDIR /fly_brain
COPY . /fly_brain

# Serve the connectome over gRPC. Loading it takes about a second, so the
# server is up almost immediately; clients should still expect the first Info
# call to fail if they race the container.
#
# To poke at the brain by hand instead, override:
#   docker compose run --rm fly-brain sleep infinity
CMD ["python", "-u", "/fly_brain/fly_brain_server.py"]
