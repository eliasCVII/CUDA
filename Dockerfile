FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools, clangd for LSP, and bear for compile_commands.json
RUN apt-get update && apt-get install -y \
    build-essential \
    clang-tools \
    clangd \
    bear \
    cmake \
    git \
    mpich \
    libmpich-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
