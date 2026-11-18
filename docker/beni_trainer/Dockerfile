# OpenBeni trainer container.
#
# Runs Mondo-Robotics/open-beni (mjlab >=1.6 on MuJoCo Warp, rsl_rl PPO with
# RNNModel GRU-512, PyTorch 2.7 cu128) from the third_party/open-beni submodule.
# Kept separate from duck_trainer: microduck pins mjlab==1.3.0 + torch 2.9,
# while open-beni needs mjlab>=1.6.0 + torch>=2.7,<2.8 + optional MNN export.
#
#   docker build -f docker/beni_trainer/Dockerfile third_party/open-beni -t beni_trainer
#
# CUDA comes from the torch/warp wheels; the base image only needs the NVIDIA
# driver libraries the runtime injects, plus EGL for headless MuJoCo rendering.

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
    WANDB_MODE=disabled \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libegl1 libgl1 libglvnd0 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /usr/local/bin/

WORKDIR /openbeni_rl

# Dependencies first so source edits don't bust the multi-GB torch/warp layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group export --group dev

COPY . .
# pymongo + pytest for the rl-roboracer job worker (rl_agent/openbeni);
# onnx for validate_onnx in mjlab_core.worker (dev group already has onnxruntime).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group export --group dev \
    && uv pip install --python /opt/venv/bin/python \
         "pymongo~=4.7" pytest "onnx>=1.16" "imageio-ffmpeg>=0.5"

ENV PATH=/opt/venv/bin:$PATH

CMD ["sleep", "infinity"]
