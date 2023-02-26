#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration
#SBATCH --job-name=vq2d_val
#SBATCH --output=./logs/slurm_eval/eval_vq2d-%j.out
#SBATCH --error=./logs/slurm_eval/eval_vq2d-%j.err


##SBATCH --partition=learnfair
#SBATCH --array=1-100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
##SBATCH -C volta
#SBATCH --mem=64GB

#SBATCH --time=6:00:00
#SBATCH --comment="vq2d baseline evaluation"

# echo `localhost`

### Section 2: Setting environment variables for the job
module purge
# Load what we need
# module load anaconda3/2020.11
module load cuda/11.4.4
module load gcc
# module load cudnn/v7.6.5.32-cuda.10.2
# module load gcc/7.3.0
# module load cmake/3.15.3/gcc.7.3.0


### Section 3:
# conda deactivate
# source activate ego4d_vq2d
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
python evaluate_vq2d.py \
  data.data_root="$CLIPS_ROOT" \
  data.split="val" \
  +data.part=$SLURM_ARRAY_TASK_ID \
  +data.n_part=$N_PART \
  data.annot_root="$VQ2D_SPLITS_ROOT" \
  data.num_processes=2 \
  data.rcnn_batch_size=4 \
  +signals.height=0.6 \
  model.config_path="$TRAIN_ROOT/output/config.yaml" \
  model.checkpoint_path="$TRAIN_ROOT/output/model_${ITER}.pth" \
  logging.save_dir="$EVAL_ROOT/model_${ITER}_kys/" \
  logging.stats_save_path="$EVAL_ROOT/model_${ITER}_kys/vq_stats_val_$SLURM_ARRAY_TASK_ID.json.gz" tracker.type='kys'

echo $EVAL_ROOT/model_${ITER}
  # signals.distance=5 signals.width=3 
