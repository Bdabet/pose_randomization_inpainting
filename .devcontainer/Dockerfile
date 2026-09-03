# Base toolchain for the Phantom/Masquerade pipeline (MuJoCo/robosuite,
# Detectron2, mmcv-full, E2FGVI, HaMeR, SAM2). Project-specific Python deps
# are installed by .devcontainer/post_create.sh via the repo's own
# install.sh, not baked in here, so this stays in sync with install.sh
# automatically.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Headless GPU rendering for MuJoCo (robosuite's recommended backend) --
# nvidia-container-toolkit injects the matching EGL/GL libraries at
# `docker run` time when these capabilities are requested.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip ca-certificates \
        build-essential cmake ninja-build \
        ffmpeg \
        libgl1 libglu1-mesa libglew-dev libosmesa6-dev libglfw3 libgles2-mesa-dev libegl1 \
        libspatialindex-dev \
        libsm6 libxext6 libxrender-dev \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the Dev Containers convention, so the bind-mounted
# workspace keeps host-matching file ownership.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Miniconda -- install.sh (run in postCreateCommand) manages its own
# "phantom" conda env and even installs its own CUDA toolkit via conda,
# so we only need conda itself here, not a system CUDA/toolkit setup.
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && chown -R $USERNAME:$USERNAME $CONDA_DIR
ENV PATH=$CONDA_DIR/bin:$PATH

USER $USERNAME
RUN conda init bash
