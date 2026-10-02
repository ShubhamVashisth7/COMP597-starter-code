#!/bin/bash
#SBATCH --job-name=gpt2
#SBATCH --account=def-kemme
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:20:00
#SBATCH --output=compute_canada/gpt2-%j.out
#SBATCH --error=compute_canada/gpt2-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

# Compute Canada modules. Arrow must be loaded before the virtual environment.
module load StdEnv/2023
module load python/3.11
module load arrow/22.0.0

source "${REPO_DIR}/energy-env/bin/activate"

# Keep package/model caches and temporary build files off the repository.
export TMPDIR=/home/vashisth/scratch/tmp
export PIP_CACHE_DIR=/home/vashisth/scratch/pip-cache
export HF_HOME=/home/vashisth/scratch/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}"

echo "Running on host: $(hostname)"
echo "Python: $(which python)"
python --version
python -c 'import torch, pyarrow; print("Torch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("PyArrow:", pyarrow.__version__)'
nvidia-smi

python "${REPO_DIR}/launch.py" \
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
