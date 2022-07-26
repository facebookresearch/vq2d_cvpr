import argparse
import glob
import gzip
import json
from collections import defaultdict
from unittest import result

import tqdm
from vq2d.metrics import compute_visual_query_metrics
from vq2d.structures import (
    BBox,
    ResponseTrack,
)


NUM_PARTS = 500
invalid_anno = [
    'a48c3e00-e5e4-4f76-93c3-036d4fc103dc',
]

def merge_results(args):
    stats_paths = sorted(glob.glob(f"{args.stats_dir}/vq_stats_test_*.json.gz"))
    if len(stats_paths) != NUM_PARTS:
        print('Expect {} parts, got {}'.format(NUM_PARTS, len(stats_paths)))

    results = dict()

    for path in tqdm.tqdm(stats_paths):
        with open(path, "r") as fp:
            data = json.load(fp)
        for video in data['results']['videos']:
            for clip in video["clips"]:
                for pred in clip["predictions"]:
                    if not pred['is_empty']:
                        auid = pred['annotation_uid']
                        if auid in results:
                            results[auid]['query_sets'].update(pred['query_sets'])
                        else:
                            results[auid] = pred

                        # remove list
                        for k, v in results[auid]['query_sets'].items():
                            if isinstance(v, list):
                                results[auid]['query_sets'][k] = v[0]
        

    # Format results
    predictions = {
        "version": "1.0",
        "challenge": "ego4d_vq2d_challenge",
        "results": {
            "videos": []
        }
    }
    annot_path = "data/vq_test_unannotated.json"
    with open(annot_path) as fp:
        annotations = json.load(fp)

    for v in annotations["videos"]:
        video_predictions = {"video_uid": v["video_uid"], "clips": []}
        for c in v["clips"]:
            clip_predictions = {"clip_uid": c["clip_uid"], "predictions": []}
            for a in c["annotations"]:
                auid = a["annotation_uid"]
                if auid not in results:
                    apred = {
                        "query_sets": {
                            qid: {"bboxes": [], "score": 0.0}
                            for qid in a["query_sets"].keys()
                        }, 
                        'annotation_uid': auid,
                        'is_empty': True
                    }
                    print('generate empty prediction for', auid)
                else:
                    apred = results[auid] # this should be an error when not finished
                clip_predictions["predictions"].append(apred)
            video_predictions["clips"].append(clip_predictions)
        predictions["results"]["videos"].append(video_predictions)
    ################################################################################################
    # Save results
    with open(f"{args.stats_dir}/merged_output_test.json", "w") as fp:
        json.dump(predictions, fp)
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=str, required=True)

    args = parser.parse_args()

    merge_results(args)


