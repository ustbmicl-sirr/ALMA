#!/bin/bash
# ALMA Evaluation Script
# Usage: ./evaluate.sh <GPU_ID> <ENV> <ALG> <CHECKPOINT_NAME> [additional args]
#
# Examples:
#   ./evaluate.sh 0 ff qmix_atten my_run_name
#   ./evaluate.sh 0 sc2multiarmy refil my_sc2_run --scenario=6-8sz_maxsize4_maxarmies3_symmetric

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Arguments
GPU=${1:-0}
ENV=${2:-ff}
ALG=${3:-qmix_atten}
CHECKPOINT=${4:-""}
shift 4 2>/dev/null || true

if [ -z "$CHECKPOINT" ]; then
    echo "Error: Please provide a checkpoint run name"
    echo "Usage: ./evaluate.sh <GPU_ID> <ENV> <ALG> <CHECKPOINT_NAME> [args]"
    exit 1
fi

IMAGE_NAME="alma-task-allocation"
CONTAINER_NAME="${USER}_alma_eval_GPU_${GPU}_$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)"

# Check SC2PATH
if [ -z "$SC2PATH" ]; then
    SC2PATH="$PROJECT_DIR/3rdparty/StarCraftII"
fi

echo "=== ALMA Evaluation ==="
echo "GPU: $GPU"
echo "Environment: $ENV"
echo "Algorithm: $ALG"
echo "Checkpoint: $CHECKPOINT"
echo "Additional args: $@"
echo ""

# Run evaluation in Docker
docker run \
    --gpus "device=$GPU" \
    --rm \
    --name "$CONTAINER_NAME" \
    --user $(id -u):$(id -g) \
    -v "$PROJECT_DIR":/task-allocation \
    -v "$SC2PATH":/task-allocation/3rdparty/StarCraftII \
    -t "$IMAGE_NAME" \
    python3 src/main.py \
    --config="$ALG" \
    --env-config="$ENV" \
    --evaluate=True \
    --checkpoint_run_name="$CHECKPOINT" \
    "$@"
