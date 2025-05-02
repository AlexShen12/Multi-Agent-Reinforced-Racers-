#!/bin/bash
# Script to train MAPPO agents with common parameters

# Default parameters
NUM_GPUS=0
NUM_WORKERS=4
ITERATIONS=1000
CHECKPOINT_FREQ=10
RESUME_PATH=""
EVAL=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --checkpoint-freq)
      CHECKPOINT_FREQ="$2"
      shift 2
      ;;
    --resume)
      RESUME_PATH="$2"
      shift 2
      ;;
    --eval)
      EVAL=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python train_mappo_racing.py --num-gpus $NUM_GPUS --num-workers $NUM_WORKERS --iterations $ITERATIONS --checkpoint-freq $CHECKPOINT_FREQ"

# Add resume path if provided
if [ -n "$RESUME_PATH" ]; then
  CMD="$CMD --resume $RESUME_PATH"
fi

# Add eval flag if set
if [ "$EVAL" = true ]; then
  CMD="$CMD --eval"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
$CMD
