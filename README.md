# Improved Baseline for Visual Queries 2D Localization
This repo holds the solution of our submission to the VQ2D task in Ego4D Challenge 2022. 

## Introduction
The recently released Ego4D dataset and benchmark significantly scales and diversifies the first-person visual perception data. 
Episodic memory is an interesting conception in Ego4D that aims to understand the past visual experience from the recording in the first-person view. It is distinguished from semantic memory in that episodic memory gives responses based on specific first-person experiences.

Our focus is the Ego4D Visual Queries 2D Localization problem in episodic memory tasks. This task requires a system to spatially and temporally localize the most recent appearance of a given object in an egocentric view. 
The query is registered by a single tight visual crop of the object in a different scene. 
Our study is based on the three-stage baseline introduced in the Ego4D benchmark suite. The baseline solves the problem by detection+tracking: detect the similar objects in all the frames, then run a tracker from the most confident detection result. 
In the VQ2D challenge, we identify two limitations of the current baseline: (1) the training configuration has redundant computation which leads to a low convergence rate; (2) the false positive rate is high on background frames. 
To this end, we developed a more efficient and effective solution. Concretely, we bring the training loop from ~15 days to less than 24h, and we achieve $0.17\%$ spatial-temporal AP, which is $31\%$ higher than the baseline. Our solution got the first ranking on the public leaderboard. 

## Installation instructions

1. Please follow the full Installation Instuction in [the VQ2D baseline](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D). 

2. Install [submitit](https://github.com/facebookincubator/submitit/blob/main/README.md) for muliple node training.
    ```
    pip install submitit
    ```

## Running experiments

1. Please follow the step 1-4 in [the VQ2D baseline](https://github.com/EGO4D/episodic-memory/tree/main/VQ2D) to pre-process the data. 
Please be noted that the frame extraction will take longer time because we also sample some negative frames in the video.

2. Training a model in multiple nodes by following script. You could also use the original command for single node training. 
    ```
   python slurm_8node_launch.py \
        --num-gpus 8 --use-volta32 \
        --config-file configs/siam_rcnn_8_gpus_e2e_125k.yaml \
        --resume --num-machines 4 --name ivq2d \
        --job-dir <PATH to log dir> \
        INPUT.VQ_IMAGES_ROOT <PATH to extracted frames dir> \
        INPUT.VQ_DATA_SPLITS_ROOT data 
    ```

3. Evaluating the baseline for visual queries 2D localization on val set. 

    1. Query all the validation videos in parallel. To do so, pleas edit `slurm_eval_500_array.sh` to specify the paths, then submit the job array to slurm.
        ```
        sbatch scripts/faster_evaluation/slurm_eval_500_array.sh
        ```

    2. Merge all the predictions and evaluate the result.
        ```
        PYTHONPATH=. python scripts/faster_evaluation/merge_results.py --stats-dir <PATH to log dir>
        ```

4. Making predictions for Ego4D challenge. This is similar to step 3, but we will use different script.
    1. Ensure that `vq_test_unannotated.json` is copied to `$VQ2D_ROOT`.
    2. Query all the test videos in parallel. To do so, pleas edit `slurm_test_500_array.sh` to specify the paths, then submit the job array to slurm.
        ```
        sbatch scripts/faster_evaluation/slurm_test_500_array.sh
        ```
    3. Merge all the predictions.
        ```
        PYTHONPATH=. python scripts/faster_evaluation/merge_results_test.py --stats-dir <PATH to log dir>
        ```
    4. The file `$EXPT_ROOT/visual_queries_log/test_challenge_predictions.json` should be submitted on the EvalAI server.
    5. Before submission you can validate the format of the predictions using the following:
        ```
        cd $VQ2D_ROOT
        python validate_challenge_predictions.py --test-unannotated-path <PATH TO vq_test_unannotated.json> --test-predictions-path <PATH to test_challenge_predictions.json>
        ```

## Acknowledgements
This codebase relies on the Visual Queries 2D baseline in [ego4d](https://github.com/EGO4D/episodic-memory) and [detectron2](https://github.com/facebookresearch/detectron2) repositories.
