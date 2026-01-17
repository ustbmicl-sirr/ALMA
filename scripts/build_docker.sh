#!/bin/bash
# Build Docker image for ALMA
# Usage: ./build_docker.sh [--use-wandb]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR/docker"

USE_UPDATED=true
IMAGE_NAME="alma-task-allocation"

echo "=== Building ALMA Docker Image ==="

# Check if WANDB_API_KEY is set
if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY detected, will configure W&B in container"
    WANDB_ARG="--build-arg WANDB_API_KEY=${WANDB_API_KEY}"
else
    echo "No WANDB_API_KEY set, skipping W&B configuration"
    WANDB_ARG=""
fi

# Build with updated Dockerfile
if [ "$USE_UPDATED" = true ] && [ -f "Dockerfile.updated" ]; then
    echo "Using updated Dockerfile (CUDA 11.8 + PyTorch 1.13)"
    docker build $WANDB_ARG -f Dockerfile.updated -t $IMAGE_NAME .
else
    echo "Using original Dockerfile (CUDA 8.0 + PyTorch 1.1)"
    docker build $WANDB_ARG -t $IMAGE_NAME .
fi

echo ""
echo "=== Build Complete ==="
echo "Image: $IMAGE_NAME"
echo ""
echo "To run training, use: ./scripts/train.sh"
