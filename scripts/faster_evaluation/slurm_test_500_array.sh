#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration
#SBATCH --job-name=vq2d_test
#SBATCH --output=./logs/slurm_eval/eval_vq2d-%j.out
#SBATCH --error=./logs/slurm_eval/eval_vq2d-%j.err

#SBATCH --partition=learnfair
#SBATCH --array=0-499%32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB

#SBATCH --time=10:00:00
#SBATCH --comment="vq2d baseline evaluation"

# echo `localhost`
### Section 2: Setting environment variables for the job
module purge

# Load what we need
ROOT_DIR=$PWD

module load anaconda3/2020.11
module load cuda/10.2
module load cudnn/v7.6.5.32-cuda.10.2
module load gcc/7.3.0
module load cmake/3.15.3/gcc.7.3.0

### Section 3:

source activate ego4d_vq2d

TRAIN_ROOT=<PATH to training log dir>
EVAL_ROOT=<PATH to evaluation experiment dir>
CLIPS_ROOT=<PATH to extracted clips dir>
VQ2D_ROOT=$PWD
VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data
PYTRACKING_ROOT="$VQ2D_ROOT/dependencies/pytracking"
# CKPT_FLAG="v8"
# ITER='final'

cd $VQ2D_ROOT

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"

which python

sleep $((RANDOM%30+1))

python evaluate_vq2d.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="test" \
  +data.part=$SLURM_ARRAY_TASK_ID \
  +data.n_part=500 \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=2 \
  data.rcnn_batch_size=4 \
  +signals.height=0.6 \
  model.config_path="$TRAIN_ROOT/output/config.yaml" \
  model.checkpoint_path="$TRAIN_ROOT/output/model_final.pth" \
  logging.save_dir="$EVAL_ROOT" \
  logging.stats_save_path="$EVAL_ROOT/vq_stats_test_$SLURM_ARRAY_TASK_ID.json.gz" \
