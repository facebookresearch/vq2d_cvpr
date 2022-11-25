import gzip
import json
import multiprocessing as mp
import os
import os.path as osp
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pims
import skimage.io
import torch
import tqdm
from detectron2.utils.logger import setup_logger
from detectron2_extensions.config import get_cfg as get_detectron_cfg
from scipy.signal import find_peaks, medfilt
from vq2d.baselines import (
    create_similarity_network,
    convert_annot_to_bbox,
    get_clip_name_from_clip_uid,
    perform_retrieval,
    SiamPredictor,
)
from vq2d.metrics import compute_visual_query_metrics
from vq2d.structures import ResponseTrack

setup_logger()

import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.insert(0,'/home/xum/vq2d_cvpr/dependencies/pytracking')
from vq2d.tracking.utils import draw_bbox
from vq2d.tracking import Tracker
import cv2

SKIP_UIDS = [None]


def get_images_at_peak(all_bboxes, all_scores, all_imgs, peak_idx, topk=5):
    bboxes = all_bboxes[peak_idx]
    scores = all_scores[peak_idx]
    image = all_imgs[peak_idx]
    # Visualize the top K retrievals from peak image
    bbox_images = []
    for bbox in bboxes[:topk]:
        bbox_images.append(image[bbox.y1 : bbox.y2 + 1, bbox.x1 : bbox.x2 + 1])
    return bbox_images


def evaluate_one_vq(annotation, cfg, device_id=0, use_tqdm=False):

    data_cfg = cfg.data
    sig_cfg = cfg.signals

    visual_crop_boxes = []
    gt_response_track = []
    pred_response_track = []
    n_accessed_frames_per_sample = []
    n_total_frames_per_sample = []
    dataset_uids = []

    device = torch.device(f"cuda:{device_id}")

    # Create detector
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.set_new_allowed(True)
    detectron_cfg.merge_from_file(cfg.model.config_path)
    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.model.score_thresh
    detectron_cfg.MODEL.WEIGHTS = cfg.model.checkpoint_path
    detectron_cfg.MODEL.DEVICE = f"cuda:{device_id}"
    detectron_cfg.INPUT.FORMAT = "RGB"
    detectron_cfg.INPUT.CLS_EMB='/ibex/ai/home/xum/vq2d_cvpr/data/class_clip_embedding.pth'
    predictor = SiamPredictor(detectron_cfg)

    # Create tracker
    similarity_net = create_similarity_network()
    similarity_net.eval()
    similarity_net.to(device)
    tracker = Tracker(cfg)

    # Visualization
    os.makedirs(cfg.logging.save_dir, exist_ok=True)
    if cfg.logging.visualize:
        OmegaConf.save(cfg, os.path.join(cfg.logging.save_dir, "config.yaml"))

    # run inference
    start_time = time.time()
    clip_uid = annotation["clip_uid"]
    # Load clip from file
    clip_path = os.path.join(
        data_cfg.data_root, get_clip_name_from_clip_uid(clip_uid)
    )
    video_reader = pims.Video(clip_path)
    query_frame = annotation["query_frame"]
    visual_crop = annotation["visual_crop"]
    object_title = annotation["object_title"]
    vcfno = annotation["visual_crop"]["frame_number"]
    clip_frames = video_reader[0 : max(query_frame, vcfno) + 1]
    clip_read_time = time.time() - start_time
    start_time = time.time()
    
    # Retrieve nearest matches and their scores per image
    ret_bboxes, ret_scores, ret_imgs, visual_crop_im = perform_retrieval(
        clip_frames,
        visual_crop,
        query_frame,
        net=predictor,
        object_title = object_title,
        batch_size=data_cfg.rcnn_batch_size,
        recency_factor=cfg.model.recency_factor,
        subsampling_factor=cfg.model.subsampling_factor,
    )
    detection_time_taken = time.time() - start_time
    start_time = time.time()
    # Generate a time signal of scores
    score_signal = []
    for scores in ret_scores:
        if len(scores) == 0:
            score_signal.append(0.0)
        else:
            score_signal.append(np.max(scores).item())
    # Smooth the signal using median filtering
    kernel_size = sig_cfg.smoothing_sigma
    if kernel_size % 2 == 0:
        kernel_size += 1
    score_signal_sm = medfilt(score_signal, kernel_size=kernel_size)
    # normalize curve
    score_signal_sm = (score_signal_sm-min(score_signal_sm))/(max(score_signal_sm)-min(score_signal_sm))
    # Identify the latest peak in the signal
    peaks, _ = find_peaks(
        score_signal_sm,
        height=0.5,
        distance=sig_cfg.distance,
        width=sig_cfg.width,
        prominence=sig_cfg.prominence,
    )
    peak_signal_time_taken = time.time() - start_time
    peaks = np.where(score_signal_sm>0.6)[0]
    start_time = time.time()
    # Perform tracking to predict response track
    search_frames = clip_frames[: query_frame - 1]
    pred_response_track = []
    pred_rts = []
    if len(peaks) > 0:
        init_state = ret_bboxes[peaks[-1]][0]
        init_frame = clip_frames[init_state.fno]
        pred_rt, pred_rt_vis = tracker(
            init_state, init_frame, search_frames, similarity_net, device
        )
        pred_rts = [ResponseTrack(pred_rt, score=1.0)]
        pred_response_track.append(pred_rts)
    else:
        pred_rt = [ret_bboxes[-1][0]]
        pred_rt_vis = []
        pred_rts = [ResponseTrack(pred_rt, score=1.0)]
        pred_response_track.append(pred_rts)
    # Get GT response window
    gt_response_track.append(
        ResponseTrack(
            [convert_annot_to_bbox(rf) for rf in annotation["response_track"]]
        )
    )
    visual_crop_boxes.append(convert_annot_to_bbox(visual_crop))
    # Timeliness metrics
    accessed_frames = set()
    for bboxes in ret_bboxes:
        accessed_frames.add(bboxes[0].fno)
    for rt in pred_rts:
        for bbox in rt.bboxes:
            accessed_frames.add(bbox.fno)
    n_accessed_frames = len(accessed_frames)
    n_total_frames = query_frame
    n_accessed_frames_per_sample.append(n_accessed_frames)
    n_total_frames_per_sample.append(n_total_frames)
    dataset_uids.append(annotation["dataset_uid"])

    tracking_time_taken = time.time() - start_time
    print(
        "====> Data uid: {} | search window :{:>8d} frames | "
        "clip read time: {:>6.2f} mins | "
        "detection time: {:>6.2f} mins | "
        "peak signal time: {:>6.2f} mins | "
        "tracking time: {:>6.2f} mins".format(
            annotation["dataset_uid"],
            annotation["query_frame"],
            clip_read_time / 60.0,
            detection_time_taken / 60.0,
            peak_signal_time_taken / 60.0,
            tracking_time_taken / 60.0,
        )
    )

    # Note: This visualization does not work for subsampled evaluation.
    if cfg.logging.visualize:
        ####################### Visualize the peaks ########################
        plt.figure(figsize=(6, 3))
        # Plot raw signals
        plt.plot(score_signal, color="gray", label="Original signal")
        plt.plot(score_signal_sm, color="blue", label="Similarity scores")
        # Plot highest-scoring pred response track
        pred_rt_start, pred_rt_end = pred_response_track[-1][0].temporal_extent
        rt_signal = np.zeros((query_frame,))
        rt_signal[pred_rt_start : pred_rt_end + 1] = 1
        # plt.plot(rt_signal, color="red", label="Pred response track")
        # Plot peak in signal
        plt.plot(peaks, score_signal_sm[peaks], "rx", label="prediction")
        # Plot gt response track
        gt_rt_start, gt_rt_end = gt_response_track[-1].temporal_extent
        rt_signal = np.zeros((query_frame,))
        rt_signal[gt_rt_start : gt_rt_end + 1] = 1
        plt.plot(rt_signal, color="green", label="GT Response track")
        plt.legend()
        save_path = os.path.join(
            cfg.logging.save_dir, f"example_{clip_uid}_{object_title}_graph.png"
        )
        plt.savefig(save_path, dpi=500)
        plt.close()
        print('graph is saved to', save_path)
        ###################### Visualize retrievals ########################
        # Visualize crop
        save_path = os.path.join(
            cfg.logging.save_dir, f"example_{clip_uid}_{object_title}_visual_crop.png"
        )
        skimage.io.imsave(save_path, visual_crop_im)
        # Visualize retrievals at the peaks
        for peak_idx in peaks[-3:]:
            peak_images = get_images_at_peak(
                ret_bboxes, ret_scores, ret_imgs, peak_idx, topk=2
            )
            for image_idx, image in enumerate(peak_images):
                save_path = os.path.join(
                    cfg.logging.save_dir,
                    f"example_{clip_uid}_{object_title}_peak_{peak_idx:05d}_rank_{image_idx:03d}.png",
                )
                skimage.io.imsave(save_path, image)
        ################## Visualize response track ########################
        save_path = os.path.join(cfg.logging.save_dir, f"example_{clip_uid}_{object_title}_rt.mp4")
        writer = imageio.get_writer(save_path)
        for rtf in pred_rt_vis:
            writer.append_data(rtf)
        writer.close()
        ################## Visualize search window #########################
        save_path = os.path.join(cfg.logging.save_dir, f"example_{clip_uid}_{object_title}_sw.mp4")
        writer = imageio.get_writer(save_path)
        for sf in search_frames:
            writer.append_data(sf)
        writer.close()
        ################## Visualize detection ######################
        save_path = os.path.join(cfg.logging.save_dir, f"example_{clip_uid}_{object_title}_vqd.mp4")
        writer = imageio.get_writer(save_path)
        for frame, bbox, score in zip(clip_frames[:-1],ret_bboxes,ret_scores):
            frame_vis = np.copy(frame) 
            draw_bbox(frame_vis,bbox[0],text=str(score[0]))
            frame_vis = cv2.resize(frame_vis, None, fx=0.5, fy=0.5)
            writer.append_data(frame_vis)
        writer.close()

    return (
        pred_response_track,
        gt_response_track,
        visual_crop_boxes,
        dataset_uids,
        n_accessed_frames_per_sample,
        n_total_frames_per_sample,
    )



@hydra.main(config_path="vq2d", config_name="config")
def main(cfg: DictConfig) -> None:
    # Load annotations
    annot_path = osp.join(cfg.data.annot_root, f"{cfg.data.split}_annot.json.gz")
    with gzip.open(annot_path, "rt") as fp:
        annotations = json.load(fp)

    # evaluation for a part of video
    
    query_idx = 1243 # 127 # mug 77 # toothpaste # 106 # metal tool box # cfg.data.query_idx
    print('Recieved',len(annotations), 'queries, use {}-th'.format(query_idx) )
    annotation = annotations[query_idx]

    predictions = evaluate_one_vq(annotation, cfg)



if __name__ == "__main__":
    main()
