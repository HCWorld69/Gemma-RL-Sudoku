#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Export PYTHONPATH so Python knows where to find the local modules (env, agent, rl)
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the main script
python main.py "$@"
