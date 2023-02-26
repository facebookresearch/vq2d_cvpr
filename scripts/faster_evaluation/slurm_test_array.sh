#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration
#SBATCH --job-name=vq2d_test
#SBATCH --output=./logs/slurm_eval/eval_vq2d-%j.out
#SBATCH --error=./logs/slurm_eval/eval_vq2d-%j.err

##SBATCH --partition=learnfair
#SBATCH --array=0-499
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128GB

#SBATCH --time=3:00:00
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
# conda deactivate
# source activate ego4d_vq2d_vit
# conda deactivate
# conda deactivate
conda activate vq2d

VQ2D_ROOT=$PWD
# /private/home/frostxu/VQ2D_CVPR22/checkpoint/train_log/slurm_8gpus_4nodes_v11_set_bijit_t128_rmgt0.5_frame_llr
TRAIN_ROOT=$VQ2D_ROOT/checkpoint/eccv_model
EVAL_ROOT=$VQ2D_ROOT/result/eccv_model
CLIPS_ROOT=$VQ2D_ROOT/data/clips

VQ2D_SPLITS_ROOT=$VQ2D_ROOT/data/vq_splits
PYTRACKING_ROOT=$VQ2D_ROOT/dependencies/pytracking

N_PART=100.0
ITER='0064999'

export PYTHONPATH="$PYTHONPATH:$VQ2D_ROOT"
export PYTHONPATH="$PYTHONPATH:$PYTRACKING_ROOT"

cd $VQ2D_ROOT
which python

sleep $((RANDOM%30+1))
# SLURM_ARRAY_TASK_ID is from 1 to 100
# SLURM_ARRAY_TASK_ID=1.0
python get_test_challenge_predictions.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="test" \
  +data.part=$SLURM_ARRAY_TASK_ID \
  +data.n_part=$N_PART \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=2 \
  data.rcnn_batch_size=4 \
  +signals.height=0.6 \
  model.config_path="$TRAIN_ROOT/output/config.yaml" \
  model.checkpoint_path="$TRAIN_ROOT/output/model_${ITER}.pth" \
  logging.save_dir="$EVAL_ROOT/${ITER}/" \
  logging.stats_save_path="$EVAL_ROOT/${ITER}/vq_stats_test_$SLURM_ARRAY_TASK_ID.json.gz" \
