#!/bin/bash
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --job-name=esmfold2_server_test
#SBATCH --output=esmfold2_server_test.out
#SBATCH --error=esmfold2_server_test.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --export=NONE

export WORKDIR=/groups/umcg-gcc/tmp04/users/umcg-tniemeijer
export HF_HOME="$WORKDIR/.cache/huggingface" # Just to make sure /home is not flooded
export HF_HUB_DISABLE_XET=True

## Environment
# Load latest CUDA environment module
ml CUDA
# Load right Python == 3.12.3
ml Python/3.12.3-GCCcore-13.3.0 

source esmfold2_env/bin/activate

cd esmfold2-server

uvicorn esmfold-server:app --host 0.0.0.0 --port 8000 --reload --log-level debug
