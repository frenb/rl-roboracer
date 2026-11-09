# Microduck trainer container.
#
# Runs the pollen-robotics microduck_rl training stack (mjlab on MuJoCo Warp,
# rsl_rl PPO, PyTorch) from the third_party/microduck_rl submodule. Kept
# separate from sim_controller: that image is TensorFlow on CUDA 11, while
# warp-lang 1.12 and torch 2.9 need a CUDA 12 driver stack and Python 3.12.
#
# Build context is the submodule, so the image only carries its ~60 MB:
#
#   docker build -f docker/duck_trainer/Dockerfile third_party/microduck_rl -t duck_trainer
#
# CUDA itself comes from the torch and warp wheels; the base image only needs
# the driver libraries the NVIDIA runtime injects, plus EGL for headless
# MuJoCo rendering (rollout videos).

FROM nvidia/cuda:12.9.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libegl1 libgl1 libglvnd0 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04's own 3.12 satisfies microduck_rl's ">=3.12,<3.13" pin, so uv
# never has to download a standalone interpreter from GitHub at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

WORKDIR /microduck_rl

# Dependencies first so source edits don't bust the multi-GB torch/warp layer.
# The uv cache lives in a BuildKit cache mount, not the image: kept in a layer
# it doubles the image (~8 GB of wheels next to the ~8 GB venv), and as a
# mount it survives uv.lock bumps so only changed wheels are downloaded.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY . .
# pymongo is for the rl-roboracer job worker (rl_agent/duck), not microduck_rl.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
    && uv pip install --python /opt/venv/bin/python "pymongo~=4.7"

ENV PATH=/opt/venv/bin:$PATH

CMD ["sleep", "infinity"]
