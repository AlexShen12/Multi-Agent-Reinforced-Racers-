#!/bin/bash
# Script to evaluate trained MAPPO agents

# Default parameters
CHECKPOINT_PATH=""
NUM_EPISODES=10
SEED=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if checkpoint path is provided
if [ -z "$CHECKPOINT_PATH" ]; then
  echo "Error: Checkpoint path is required. Use --checkpoint PATH"
  exit 1
fi

# Build the command
CMD="python evaluate_mappo_racing.py --checkpoint $CHECKPOINT_PATH --num-episodes $NUM_EPISODES --seed $SEED"

# Print the command
echo "Running: $CMD"

# Execute the command
$CMD
