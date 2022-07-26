# Visual Queries 2D localization

## Installation instructions

1. Clone the Ego4d episodic memory repository from [here](https://github.com/EGO4D/episodic-memory).
    ```
    git clone git@github.com:EGO4D/episodic-memory.git
    cd episodic-memory/VQ2D
    export VQ2D_ROOT=$PWD
    
    ```
2. Create conda environment.
    ```
    conda create -n ego4d_vq2d python=3.8
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

6.  Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQ2D_ROOT/pretrained_models/kys.pth`.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:visionml/pytracking.git
    git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
    ```

    Note: For installing the [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) dependency for pytracking, follow these steps if the pip install fails.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:ClementPinard/Pytorch-Correlation-extension.git
    cd Pytorch-Correlation-extension
    python setup.py install
    ```

## Running experiments

1. Download the annotations and videos as instructed [here](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) to `$VQ2D_ROOT/data`.
    ```
    python -m ego4d.cli.cli --output_directory="$VQ2D_ROOT/data" --datasets full_scale annotations
    # Define ego4d videos directory
    export EGO4D_VIDEOS_DIR=$VQ2D_ROOT/data/v1/full_scale
    # Move out vq annotations to $VQ2D_ROOT/data
    mv $VQ2D_ROOT/data/v1/annotations/vq_*.json $VQ2D_ROOT/data
    ```

2. Process the VQ dataset.
    ```
    python process_vq_dataset.py --annot-root data --save-root data
    ```

3. Extract clips for validation and test data from videos.
    ```
    python convert_videos_to_clips.py \
        --annot-paths data/vq_val.json data/vq_test_unannotated.json \
        --save-root data/clips \
        --ego4d-videos-root $EGO4D_VIDEOS_DIR \
        --num-workers 10 # Increase this for speed
    ```

4. Extract images for train and validation data from videos.
    ```
    python convert_videos_to_images.py \
        --annot-paths data/vq_train.json data/vq_val.json \
        --save-root data/images \
        --ego4d-videos-root $EGO4D_VIDEOS_DIR \
        --num-workers 10 # Increase this for speed
    ```

5. Training a model. Copy `scripts/train_2_gpus.sh` or `scripts/train_8_gpus.sh` to the required experiment directory and execute it.
    ```
    EXPT_ROOT=<experiment path>
    cp $VQ2D_ROOT/scripts/train_2_gpu.sh $EXPT_ROOT
    cd $EXPT_ROOT
    chmod +x train_2_gpu.sh && ./train_2_gpu.sh
    ```

6. Evaluating the baseline for visual queries 2D localization. Copy `scripts/evaluate_vq.sh` to the exxperiment directory, update the paths and checkpoint id, and execute it. Note: To evaluate with the particle filter tracker, add the commandline argument `tracker.type="pfilter"`.
    ```
    EXPT_ROOT=<experiment path>
    cp $VQ2D_ROOT/scripts/evaluate_vq.sh $EXPT_ROOT
    cd $EXPT_ROOT
    <UPDATE PATHS in evaluate_vq.sh>
    chmod +x evaluate_vq.sh && ./evaluate_vq.sh
    ```
    We provide pre-trained models for reproducibility. They can be downloaded using the ego4d CLI as follows:
    ```
    python -m ego4d.cli.cli -y --output_directory /path/to/output/ --datasets vq2d_models
    ```

## [NEW] Making predictions for Ego4D challenge
1. Ensure that `vq_test_unannotated.json` is copied to `$VQ2D_ROOT`.
2. Copy `scripts/get_challenge_predictions.sh` to the experiment directory, update the paths and checkpoint id, and execute it. The arguments are similar to baseline evaluation in the previous section, but the script has been modified to output predictions consistent with the challenge format.
    ```
    EXPT_ROOT=<experiment path>
    cp $VQ2D_ROOT/scripts/get_challenge_predictions.sh $EXPT_ROOT
    cd $EXPT_ROOT
    <UPDATE PATHS in get_challenge_predictions.sh>
    chmod +x get_challenge_predictions.sh && ./get_challenge_predictions.sh
    ```
3. Note: For faster evaluation, increase `data.num_processes`.
4. The file `$EXPT_ROOT/visual_queries_log/test_challenge_predictions.json` should be submitted on the EvalAI server.
5. Before submission you can validate the format of the predictions using the following:
    ```
    cd $VQ2D_ROOT
    python validate_challenge_predictions.py --test-unannotated-path <PATH TO vq_test_unannotated.json> --test-predictions-path <PATH to test_challenge_predictions.json>
    ```

## Acknowledgements
This codebase relies on [detectron2](https://github.com/facebookresearch/detectron2), [Ego4d](https://github.com/EGO4D)) repositories.
