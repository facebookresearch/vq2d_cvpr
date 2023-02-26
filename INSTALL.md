# Installation

## Requirements

- Linux or macOS with Python ≥ 3.7
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional but needed by demo and visualization


## Basic installation
This session is to support train and validation of the visual query detector
1. Clone our repository from [here](https://github.com/facebookresearch/vq2d_cvpr.git).
    ```
    git clone https://github.com/facebookresearch/vq2d_cvpr.git
    cd vq2d_cvpr
    export VQ2D_ROOT=$PWD
    
    ```
2. Create conda environment.
    ```
    conda create -n vq2d python=3.8
    ```
3. Install [pytorch](https://pytorch.org/) using conda. We rely on cuda-10.2 and cudnn-7.6.5.32 for our experiments.
    ```
    conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
    ```

4. Install additional requirements using `pip`.
    ```
    pip install -r requirements.txt
    ```

5. Install [detectron2](https://github.com/facebookresearch/detectron2).
    ```
    python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
    ```

6. Install [submitit](https://github.com/facebookincubator/submitit/blob/main/README.md) for muliple node training. 
    ```
    pip install submitit
    ```

## Tracking installation
You will need to following steps to apply VQ2D evaluation. These are not required for the visual query detection task.

7.  Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQ2D_ROOT/pretrained_models/kys.pth`.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:visionml/pytracking.git
    git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
    ```

8. For installing the [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) dependency for pytracking, follow these steps if the pip install fails.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:ClementPinard/Pytorch-Correlation-extension.git
    cd Pytorch-Correlation-extension
    python setup.py install
    ```

## Preparing dataset 



1. Download the annotations and videos as instructed [here](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) to `$VQ2D_ROOT/data`.
    ```
    ego4d --output_directory="$VQ2D_ROOT/data" --datasets full_scale annotations
    # Define ego4d videos directory
    export EGO4D_VIDEOS_DIR=$VQ2D_ROOT/data/v1/full_scale
    # Move out vq annotations to $VQ2D_ROOT/data
    mv $VQ2D_ROOT/data/v1/annotations/vq_*.json $VQ2D_ROOT/data
    ```
2. Use the updated v1.0.5 data:
    ```
    # Download the data using the Ego4D CLI. 
    ego4d --output_directory="$VQ2D_ROOT/data" --datasets annotations -y --version v1_0_5

    # Move out vq annotations to $VQ2D_ROOT/data
    mv $VQ2D_ROOT/data/v1_0_5/annotations/vq_*.json $VQ2D_ROOT/data
    ```

3. Process the VQ dataset.
    ```
    python process_vq_dataset.py --annot-root data --save-root data
    ```

4. Extract clips for validation and test data from videos.
    ```
    python convert_videos_to_clips.py \
        --annot-paths data/vq_val.json data/vq_test_unannotated.json \
        --save-root data/clips \
        --ego4d-videos-root $EGO4D_VIDEOS_DIR \
        --num-workers 10 # Increase this for speed
    ```

5. Extract images for train and validation data from videos. Please be noted that the frame extraction will take longer time because we also sample some negative frames in the video.
    ```
    python convert_videos_to_images.py \
        --annot-paths data/vq_train.json data/vq_val.json \
        --save-root data/images \
        --ego4d-videos-root $EGO4D_VIDEOS_DIR \
        --num-workers 10 # Increase this for speed
    ```


## Running experiments


1. Training a model in multiple nodes by following script. It loads images from `INPUT.VQ_IMAGES_ROOT`, and the training log and checkpoints are save in `--job-dir`. You could also use the original command for single node training. 
    ```
   python slurm_8node_launch.py \
        --num-gpus 8 --use-volta32 \
        --config-file configs/siam_rcnn_8_gpus_e2e_125k.yaml \
        --resume --num-machines 4 --name ivq2d \
        --job-dir <PATH to training log dir> \
        INPUT.VQ_IMAGES_ROOT <PATH to extracted frames dir> \
        INPUT.VQ_DATA_SPLITS_ROOT data 
    ```

2. Evaluating the our model for visual queries 2D localization on val set. 

    1. Download the model ckpt and configuration from [google drive](https://drive.google.com/drive/folders/1Q8lAZocw3k7niWX-gQtThevdgAlCplzA?usp=share_link).

    2. Query all the validation videos in parallel. To do so, pleas edit `slurm_eval_500_array.sh` to specify the paths, then submit the job array to slurm. NB 1, `TRAIN_ROOT` is the folder for the downloaded checkpoint and configuration file, and `EVAL_ROOT` saves the evaluation from each run. NB 2, `N_PART` is the number of the splits. The script will produce `N_PART` results.
        ```
        sbatch scripts/faster_evaluation/slurm_eval_array.sh
        ```

    3. Merge all the predictions and evaluate the result. Note that <the evaluation experiment dir> is the output of the evalution script (i.e. `EVAL_ROOT`). It is different from training log dir.
        ```
        PYTHONPATH=. python scripts/faster_evaluation/merge_results.py \
            --stats-dir <PATH to evaluation experiment dir>
        ```

3. Making predictions for Ego4D challenge. This is similar to step 2, but we will use different script.
    1. Ensure that `vq_test_unannotated.json` is copied to `$VQ2D_ROOT`.
    2. Query all the test videos in parallel. To do so, pleas edit `slurm_test_500_array.sh` to specify the paths, then submit the job array to slurm.
        ```
        sbatch scripts/faster_evaluation/slurm_test_array.sh
        ```
    3. Merge all the predictions.
        ```
        PYTHONPATH=. python scripts/faster_evaluation/merge_results_test.py \
            --stats-dir <the evaluation experiment dir>
        ```
    4. The file `$EXPT_ROOT/visual_queries_log/test_challenge_predictions.json` should be submitted on the EvalAI server.
    5. Before submission, you can validate the format of the predictions using the following:
        ```
        cd $VQ2D_ROOT
        python validate_challenge_predictions.py \
            --test-unannotated-path <PATH TO vq_test_unannotated.json> \
        ```
        