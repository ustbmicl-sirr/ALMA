#!/bin/bash
# ALMA Training Script
# Usage: ./train.sh <GPU_ID> <ENV> <ALG> [additional args]
#
# Examples:
#   ./train.sh 0 ff qmix_atten                           # SaveTheCity with QMIX
#   ./train.sh 0 sc2multiarmy refil --scenario=6-8sz_maxsize4_maxarmies3_symmetric  # StarCraft
#
# ALMA method:
#   ./train.sh 0 ff qmix_atten --agent.subtask_cond=mask --hier_agent.task_allocation=aql
#
# With EA optimization:
#   ./train.sh 0 ff qmix_atten --ea.enabled=True --agent.subtask_cond=mask --hier_agent.task_allocation=aql

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default values
GPU=${1:-0}
ENV=${2:-ff}
ALG=${3:-qmix_atten}
shift 3 2>/dev/null || true

IMAGE_NAME="alma-task-allocation"
CONTAINER_NAME="${USER}_alma_GPU_${GPU}_$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)"

# Check SC2PATH
if [ -z "$SC2PATH" ]; then
    SC2PATH="$PROJECT_DIR/3rdparty/StarCraftII"
fi

echo "=== ALMA Training ==="
echo "GPU: $GPU"
echo "Environment: $ENV"
echo "Algorithm: $ALG"
echo "SC2PATH: $SC2PATH"
echo "Additional args: $@"
echo ""

# Set environment-specific defaults
EXTRA_ARGS=""
if [ "$ENV" = "ff" ]; then
    EXTRA_ARGS="--epsilon_anneal_time=2000000 --hier_agent.action_length=5"
elif [ "$ENV" = "sc2multiarmy" ]; then
    EXTRA_ARGS="--hier_agent.action_length=3"
fi

# Run in Docker
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
    $EXTRA_ARGS \
    "$@"
