import copy
import gzip
import json
import os
import os.path as osp
import random
from typing import Any, Optional, Sequence, List, Dict

import imagesize
import numpy as np
import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils

import pickle as pk
import multiprocessing as mp

import torch

from matplotlib import pyplot as plt

from ..constants import DATASET_FILE_TEMPLATE
from .utils import (
    get_image_name_from_clip_uid,
    get_bbox_from_data,
    get_image_id_from_data,
    get_clip_name_from_clip_uid,
    _extract_image_from_clip,
    # extract_object_from_image_file,
    extract_square_with_context,
    img2occlusion
)


def _register_visual_query_dataset(
    data_id: str,
    data_path: str,
    images_root: str,
    **kwargs: Any,
) -> None:
    """Helper function to register visual query datasets."""

    def visual_query_dataset_function():
        return visual_query_dataset(data_path, images_root, **kwargs)

    try:
        DatasetCatalog.register(data_id, visual_query_dataset_function)
    except AssertionError:
        # Skip this step if it is already registered
        pass

    MetadataCatalog.get(data_id).thing_classes = ["visual_crop"]
    MetadataCatalog.get(data_id).thing_dataset_id_to_contiguous_id = {0: 0}


def register_visual_query_datasets(
    data_root: str,
    images_root: str,
    data_key: str,
    bbox_aspect_scale: Optional[float] = None,
    bbox_area_scale: Optional[float] = None,
    **kwargs: Any,
) -> None:
    """
    Given dataset paths and other configuration arguments, it registers a
    visual query dataset with the specified arguments.
    """
    for split in ["train", "val", "test","minitrain","debug"]:
        data_path = osp.join(data_root, DATASET_FILE_TEMPLATE.format(split))
        bbox_aspect_ratio_range = None
        bbox_area_ratio_range = None
        data_id = f"{data_key}_{split}"
        if bbox_aspect_scale is not None:
            assert 0.0 < bbox_aspect_scale < 1.0
            bbox_aspect_ratio_range = (bbox_aspect_scale, 1.0 / bbox_aspect_scale)
        if bbox_area_scale is not None:
            assert 0.0 < bbox_area_scale < 1.0
            bbox_area_ratio_range = (bbox_area_scale, 1.0 / bbox_area_scale)
        _register_visual_query_dataset(
            data_id,
            data_path,
            images_root,
            bbox_aspect_ratio_range=bbox_aspect_ratio_range,
            bbox_area_ratio_range=bbox_area_ratio_range,
            **kwargs,
        )


def visual_query_dataset(
    data_path: str,
    images_root: str,
    bbox_aspect_ratio_range: Optional[Sequence[float]] = None,
    bbox_area_ratio_range: Optional[Sequence[float]] = None,
    perform_response_augmentation: bool = False,
    augmentation_limit: bool = 10,
    include_empty_image: bool = False,
    pose_jitter: str = "",
    jitter_score: bool = False,
    debug = False,
    split_dataset = None,
) -> List[Dict[str, Any]]:
    # decode pose jitter
    if type(pose_jitter) is list or type(pose_jitter) is tuple:
        pose_jitter, jitter_aug_ratio = pose_jitter
    else:
        jitter_aug_ratio = 1.0 # default

    with gzip.open(data_path, "rt") as fp:
        annotations = json.load(fp)
    data_samples = []
    cache_path = './.cache/data_samples/{}_{}{}{}{}{}{}.pk'.format(
        data_path.replace('/','_'), 
        perform_response_augmentation, augmentation_limit, include_empty_image, pose_jitter, jitter_aug_ratio, debug
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path) and not debug and split_dataset is None: # force to recompute in debug mode
        print("=======> Using cached dataset from {}".format(cache_path))
        print("=======> Warning! remove cache folder if you need a new dataset!")
        with open(cache_path, 'rb') as f:
            data_samples = pk.load(f)
        print('Size of data: ', len(data_samples), flush=True)
        return data_samples
    else:
        print("=======> No found cached dataset at {}".format(cache_path))
        print("=======> Computing dataset")
    # annotations = annotations[:int(len(annotations)/2)]
    if debug:
        print("DEBUG => Computing dataset")
        annotations = annotations[:10]
    if split_dataset:
        part, total_part = split_dataset
        total_dataset = len(annotations)
        start_idx = int(total_dataset*part/total_part)
        end_idx = int(total_dataset*(part+1)/total_part)
        annotations = annotations[start_idx:end_idx]
    for annot_ix, annot in tqdm.tqdm(enumerate(annotations), total=len(annotations)):
        clip_uid = annot["clip_uid"]
        vc_bbox = get_bbox_from_data(annot["visual_crop"])
        vc_fno = annot["visual_crop"]["frame_number"]
        vc_path = get_image_name_from_clip_uid(clip_uid, vc_fno)
        vc_path = osp.join(images_root, vc_path)
        q_fno = annot["query_frame"] # query frame
        if not os.path.isfile(vc_path):
            continue
        # Is aspect ratio correction needed?
        actual_width, actual_height = imagesize.get(vc_path)
        vc_width = annot["visual_crop"]["original_width"]
        vc_height = annot["visual_crop"]["original_height"]
        vc_arc = False
        if (vc_width, vc_height) != (actual_width, actual_height):
            vc_arc = True
            print("=======> VC needs aspect ratio correction")
        # Sort response track by frame number
        annot["response_track"] = sorted(
            annot["response_track"], key=lambda x: x["frame_number"]
        )
 
        # Duration of a response track
        rt_dur = annot["response_track"][-1]["frame_number"] - annot["response_track"][0]["frame_number"] + 1

        # Get aspect ratio for the largest response track BBoxes
        bbox_areas = []
        aspect_ratios = []
        for rf_idx, rf_data in enumerate(annot["response_track"]):
            bba = rf_data["width"] * rf_data["height"]
            ar = float(rf_data["width"]) / float(rf_data["height"] + 1e-10)
            bbox_areas.append(bba)
            aspect_ratios.append(ar)
        bbox_areas = np.array(bbox_areas)
        aspect_ratios = np.array(aspect_ratios)
        bbox_idxs = np.argsort(-bbox_areas)[:5]
        std_bbox_area = np.median(bbox_areas[bbox_idxs]).item()
        std_aspect_ratio = np.median(aspect_ratios[bbox_idxs]).item()
        # Create one sample for every (visual query, response frame) pairs
        curr_data_samples = []
        for rf_idx, rf_data in enumerate(annot["response_track"]):
            rf_bbox = get_bbox_from_data(rf_data)
            rf_fno = rf_data["frame_number"]
            rf_path = get_image_name_from_clip_uid(clip_uid, rf_fno)
            rf_path = osp.join(images_root, rf_path)
            if not os.path.isfile(rf_path):
                continue
            # NOTE: By default, the category_id will be 0 always. This is because there is only
            # one class, corresponding to the right match. Within detectron2, the unmatched
            # bbox proposals will automatically be set to 1, the background class.
            category_id = 0
            # Is aspect ratio correction needed?
            actual_width, actual_height = imagesize.get(rf_path)
            rf_width = rf_data["original_width"]
            rf_height = rf_data["original_height"]
            response_arc = False
            if (rf_width, rf_height) != (actual_width, actual_height):
                response_arc = True
                print("=======> RF needs aspect ratio correction")
            # Clean dataset
            bbox_area = rf_data["width"] * rf_data["height"]
            aspect_ratio = float(rf_data["width"]) / float(rf_data["height"] + 1e-10)
            clean = True
            if clean and (bbox_aspect_ratio_range is not None):
                ratio = aspect_ratio / (std_aspect_ratio + 1e-10)
                clean = (
                    bbox_aspect_ratio_range[0] <= ratio <= bbox_aspect_ratio_range[1]
                )
            if clean and (bbox_area_ratio_range is not None):
                ratio = bbox_area / (std_bbox_area + 1e-10)
                clean = bbox_area_ratio_range[0] <= ratio <= bbox_area_ratio_range[1]
            if not clean:
                continue
            if include_empty_image != 'only':
                curr_data_samples.append({
                    "image_id": get_image_id_from_data(annot, annot_ix, rf_idx),
                    "file_name": rf_path,
                    "info": {
                        "aspect_ratio": aspect_ratio,
                        "bbox_area": bbox_area,
                        "std_aspect_ratio": std_aspect_ratio,
                        "std_bbox_area": std_bbox_area,
                    },
                    "width": rf_width,
                    "height": rf_height,
                    "incorrect_aspect_ratio": response_arc,
                    "annotations": [
                        {
                            "bbox": rf_bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": category_id,
                        }
                    ],
                    "reference": {
                        "file_name": vc_path,
                        "width": vc_width,
                        "height": vc_height,
                        "incorrect_aspect_ratio": vc_arc,
                        "annotations": [
                            {
                                "bbox": vc_bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "textual_name": annot["object_title"],
                            }
                        ],
                    },
                })

            if include_empty_image:
                image_id = get_image_id_from_data(annot, annot_ix, rf_idx) + '_neg'
                offset = rt_dur # int((q_fno - rf_fno)//2)
                if rf_fno+offset<q_fno: # and rf_fno%2==0: # dont make sure there is a big gap between last rt and query
                    rfn_path = get_image_name_from_clip_uid(clip_uid, rf_fno+offset)
                    rfn_path = osp.join(images_root, rfn_path)
                    # print('adding', image_id)
                    
                    if not os.path.isfile(rfn_path):
                        continue
                    empty_image = {
                        "image_id": image_id,
                        "file_name": rfn_path,
                        "info": {
                            "aspect_ratio": aspect_ratio,
                            "bbox_area": bbox_area,
                            "std_aspect_ratio": std_aspect_ratio,
                            "std_bbox_area": std_bbox_area,
                        },
                        "width": rf_width,
                        "height": rf_height,
                        "incorrect_aspect_ratio": response_arc,
                        "annotations": [
                            # No annotation
                            # {
                            #     "bbox": rf_bbox,
                            #     "bbox_mode": BoxMode.XYXY_ABS,
                            #     "category_id": category_id,
                            # }
                        ],
                        "reference": {
                            "file_name": vc_path,
                            "width": vc_width,
                            "height": vc_height,
                            "incorrect_aspect_ratio": vc_arc,
                            "annotations": [
                                {
                                    "bbox": vc_bbox,
                                    "bbox_mode": BoxMode.XYXY_ABS,
                                    "textual_name": annot["object_title"],
                                }
                            ],
                        },
                    }
                    curr_data_samples.append(empty_image)

        data_samples += curr_data_samples

        # Optionally, augment dataset by creating (response frame, response frame) pairs
        if not perform_response_augmentation:
            continue
        ## Get a list of good response frames to serve as dummy visual queries.
        ## A good response frame is one that is clean according to bbox ratio criteria.
        clean_response_frames = []
        for rf_idx, rf_data in enumerate(annot["response_track"]):
            rf_bbox = get_bbox_from_data(rf_data)
            rf_fno = rf_data["frame_number"]
            rf_path = get_image_name_from_clip_uid(clip_uid, rf_fno)
            rf_path = osp.join(images_root, rf_path)
            if not os.path.isfile(rf_path):
                continue
            # Is aspect ratio correction needed?
            actual_width, actual_height = imagesize.get(rf_path)
            rf_width = rf_data["original_width"]
            rf_height = rf_data["original_height"]
            response_arc = False
            if (rf_width, rf_height) != (actual_width, actual_height):
                response_arc = True
                print("=======> RF needs aspect ratio correction")
            bbox_area = rf_data["width"] * rf_data["height"]
            aspect_ratio = float(rf_data["width"]) / float(rf_data["height"] + 1e-10)
            clean = True
            if clean:
                ratio = aspect_ratio / (std_aspect_ratio + 1e-10)
                clean = 0.85 <= ratio <= (1.0 / 0.85)
            if clean:
                ratio = bbox_area / (std_bbox_area + 1e-10)
                clean = 0.50 <= ratio <= (1.0 / 0.5)
            if not clean:
                continue

            clean_response_frames.append(
                {
                    "file_name": rf_path,
                    "width": rf_width,
                    "height": rf_height,
                    "incorrect_aspect_ratio": response_arc,
                    "annotations": [
                        {
                            "bbox": rf_bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "textual_name": annot["object_title"],
                        }
                    ],
                }
            )
        random.shuffle(clean_response_frames)
        clean_response_frames = clean_response_frames[:augmentation_limit]

        ## Add new data with augmented samples
        for ds in curr_data_samples:
            for crf in clean_response_frames:
                ds = copy.deepcopy(ds)
                ds["reference"] = crf
                data_samples.append(ds)

    # pose jitter
    # to be tested
    # The basic idea of pose jitter is to add more positive examples
    if pose_jitter:
        jitter_path = data_path[:-8]+'_{}.json.gz'.format(pose_jitter)
        print('load jittering result from', jitter_path)
        print('number of pairs so far:', len(data_samples))
        with gzip.open(jitter_path, "rt") as fp:
            jitter_annotations = json.load(fp)
        total = len(jitter_annotations)
        if debug:
            print("DEBUG => Computing dataset")
            total = int(0.01*total)
        if split_dataset:
            raise ValueError('split_dataset is unsafe')
            part, total_part = split_dataset
            total_dataset = len(jitter_annotations)
            start_idx = int(total_dataset*part/total_part)
            end_idx = int(total_dataset*(part+1)/total_part)
            jitter_annotations = jitter_annotations[start_idx:end_idx]
            total = len(jitter_annotations)
        else:
            start_idx,end_idx = 0, len(jitter_annotations)

        if jitter_score:
            jitter_score_cache_path = data_path[:-8]+'_{}_score.pk'.format(
                pose_jitter)
            assert os.path.exists(jitter_score_cache_path), \
                "{} not exist for jitter score!".format(jitter_score_cache_path)
            with open(jitter_score_cache_path,'rb') as f:
                jitter_score_cache = pk.load(f)
                
        for annot_ix, annot in tqdm.tqdm(enumerate(jitter_annotations[:total]), total=total):
            clip_uid = annot['clip_uid']
            clean_response_frames = []
            curr_data_samples = []
            
            for rf_idx, rf_data in enumerate(annot["bboxes"]):
                rf_bbox = rf_data['x1'],rf_data['y1'],rf_data['x2'],rf_data['y2'] # x1,y1,x2,y2
                rf_fno = rf_data["fno"]
                rf_path = get_image_name_from_clip_uid(clip_uid, rf_fno)
                rf_path = osp.join(images_root, 'jitter', rf_path)
                if not os.path.isfile(rf_path) or os.path.getsize(rf_path)==0:
                    # continue
                    print('Warning! File {} not found, skip')
                    continue
               
                # Is aspect ratio correction needed? - NO!
                # rf_width, rf_height = imagesize.get(rf_path)
                rf_width = 1440
                rf_height = 1080
                if rf_width<0:
                    print('os.remove for invalid image file ',rf_path)
                    os.remove(rf_path)
                    continue

                width = rf_bbox[2]-rf_bbox[0]
                height = rf_bbox[3]-rf_bbox[1]
                bbox_area = width * height
                aspect_ratio = float(width) / float(height + 1e-10)
                
                # add to samples
                curr_data_samples.append(
                    {
                        "image_id": f"clip-uid_{clip_uid}_idx_{annot_ix}_pose_jitter_{rf_idx}_response-fno_{rf_fno}", # rf_path, # get_image_id_from_data(annot, annot_ix, rf_idx),
                        "file_name": rf_path,
                        "info": {
                            "aspect_ratio": aspect_ratio,
                            "bbox_area": bbox_area,
                            "std_aspect_ratio": std_aspect_ratio,
                            "std_bbox_area": std_bbox_area,
                        },
                        "width": rf_width,
                        "height": rf_height,
                        "incorrect_aspect_ratio": False,
                        "annotations": [
                            {
                                "bbox": rf_bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": category_id,
                            }
                        ],
                        "reference": None # will be filled later
                    }
                )

                ratio = aspect_ratio / (std_aspect_ratio + 1e-10)
                clean = 0.85 <= ratio <= (1.0 / 0.85)
                ratio = bbox_area / (std_bbox_area + 1e-10)
                clean = 0.50 <= ratio <= (1.0 / 0.5)
                if not clean:
                    continue # skip unclear visual query

                clean_response_frames.append(
                    {
                        "file_name": rf_path,
                        "width": rf_width,
                        "height": rf_height,
                        "incorrect_aspect_ratio": response_arc,
                        "annotations": [
                            {
                                "bbox": rf_bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "textual_name": 'object',
                            }
                        ],
                    }
                )

            max_jit = int(augmentation_limit*jitter_aug_ratio)           
            if jitter_score: # pick jittering image which has highest jittering score
                pass
            else: # randomly pick jittering image
                random.shuffle(clean_response_frames)
                clean_response_frames = clean_response_frames[:max_jit]
            

            ## Add new data with augmented samples
            new_data_samples = []
            for ds in curr_data_samples:
                for crf in clean_response_frames:
                    ds = copy.deepcopy(ds)
                    ds["reference"] = crf
                    
                    if jitter_score:
                        jitter_uid = "{}_{}_{}".format(
                            ds["file_name"].split('/')[-2],
                            ds["file_name"].split('/')[-1],
                            ds["reference"]["file_name"].split('/')[-1],
                        )    
                        ds["jit_score"] = jitter_score_cache.get(jitter_uid,0)   

                    new_data_samples.append(ds)

                        # ONLY RUN ONCE       
                        # img1 = ds["file_name"]
                        # img2 = ds["reference"]["file_name"]
                        # x1, y1, x2, y2 = ds["annotations"][0]["bbox"]
                        # rx1, ry1, rx2, ry2 = ds["reference"]["annotations"][0]["bbox"]
                        # w,h = ds['width'],ds['height']
                        # # remove corner object
                        # safe_dist = 128 # pixels
                        # if min(x1,y1,w-x2,h-y2)<safe_dist:
                        #     score = 0
                        # elif min(rx1,ry1,w-rx2,h-ry2)<safe_dist:
                        #     score = 0
                        # else:
                        #     # compute occlusion
                        #     try: 
                        #         # print(jitter_uid)
                        #         occlusion, im1, im2 = img2occlusion(img1,img2)
                        #     except Exception as e:
                        #         print(e)
                        #         print(img1)
                        #         print(img2)
                        #         jitter_score_cache[jitter_uid] = (0,None)
                        #         continue

                        #     # crop the object region of occlusion
                        #     nh,nw=occlusion.shape
                        #     x1,y1,x2,y2 = x1*nw/w,y1*nh/h,x2*nw/w,y2*nh/h
                        #     obj = extract_square_with_context(
                        #         image = occlusion.unsqueeze(0).unsqueeze(0),
                        #         bbox = (x1, y1, x2, y2),
                        #         p=16, size=64, pad_value=1
                        #     )
                        #     # jitter score is the occlusion over crop
                        #     score = 1 - obj.sum()/(64*64)

              
                        #     jitter_score_cache[jitter_uid] = (
                        #         score, 
                        #         obj[0][0].cpu().numpy()
                        #     )
                        #     # if score>0.2 and score<0.8:
                        #     #     folder = int(score*10)/10
                        #     #     jitter_uid = "{}_{}_{}".format(
                        #     #         ds["file_name"].split('/')[-2],
                        #     #         ds["file_name"].split('/')[-1],
                        #     #         ds["reference"]["file_name"].split('/')[-1],
                        #     #     )
                        #     #     fig = plt.figure()
                        #     #     plt.imshow(obj[0][0].cpu().numpy())
                        #     #     plt.savefig('jit_score_vis/{}/{}_{}.png'.format(folder,score,jitter_uid),dpi=fig.dpi)
                        #     #     print('The jitter score is:', score)
                        #     #     print(ds["file_name"])
                        #     #     print(ds["reference"]["file_name"])
                        #     #     print(ds['width'],ds['height'])
                        #     #     print(ds["annotations"][0]["bbox"],ds["reference"]["annotations"][0]["bbox"])

            if jitter_score: # pick jittering image which has highest jittering score
                max_pairs = len(curr_data_samples)*max_jit
                new_data_samples = sorted(new_data_samples,key=lambda x: x['jit_score'],reverse=True)
                new_data_samples = new_data_samples[:max_pairs] # TODO: sparse sample
            else: # already randomly picked jittering image
                pass
                
            data_samples += new_data_samples
        # if jitter_score:
        #     with open(jitter_score_cache_path, 'wb') as f:
        #         pk.dump(jitter_score_cache, f)
        #     print("=======> Saving cached jitter score to {}".format(jitter_score_cache_path))
        with open(cache_path, 'wb') as f:
            pk.dump(data_samples, f)
        print("=======> Saving cached dataset to {}".format(cache_path))

        print('number of pairs so far:', len(data_samples))

    with open(cache_path, 'wb') as f:
        pk.dump(data_samples, f)
        print("=======> Saving cached dataset to {}".format(cache_path))
    return data_samples


