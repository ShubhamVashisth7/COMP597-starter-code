#!/bin/bash

# ==============================
# Run file for GPT2 example - how to use the launch script to run GPT2 with simple trainer
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

# Use only locally cached Hugging Face files; do not access the network.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

### run GPT2 with CodeCarbon tracking
${SCRIPTS_DIR}/launch.sh \
    --logging.level INFO \
    --model gpt2 \
    --trainer simple \
    --epochs 2 \
    --batch_size 20 \
    --learning_rate 1e-6 \
    --data_configs.dataset.split "train[:10000]" \
    --data_configs.dataset.name "json" \
    --data_configs.dataset.train_files "${REPO_DIR}/data/c4-en.tfrecord-00000-of-01024.json.gz" \
    --trainer_stats codecarbon \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir "${REPO_DIR}/codecarbonlogs"
