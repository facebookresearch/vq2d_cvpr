from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tmodels
from einops import rearrange

import os
import imageio,pims
import multiprocessing as mp

Numeric = Union[int, float]

from ..constants import (
    CLIP_NAME_PATTERN,
    IMAGE_NAME_PATTERN,
)
from ..structures import BBox

from torchvision.transforms.functional import InterpolationMode, resize

def convert_image_np2torch(image: np.ndarray) -> torch.Tensor:
    """
    image - (H, W, 3) numpy array
    """
    mean = torch.Tensor([[[0.485, 0.456, 0.406]]])
    std = torch.Tensor([[[0.229, 0.224, 0.225]]])
    image = torch.from_numpy(image).float() / 255.0
    image = (image - mean) / std
    image = rearrange(image, "h w c -> () c h w")
    return image


def convert_annot_to_bbox(annot: Dict[str, Any]) -> BBox:
    return BBox(
        annot["frame_number"],
        annot["x"],
        annot["y"],
        annot["x"] + annot["width"],
        annot["y"] + annot["height"],
    )


def get_clip_name_from_clip_uid(clip_uid: str) -> str:
    return CLIP_NAME_PATTERN.format(clip_uid)


def get_image_name_from_clip_uid(clip_uid: str, fno: int) -> str:
    return IMAGE_NAME_PATTERN.format(clip_uid, fno + 1)


def create_similarity_network(pretrained: bool = True) -> nn.Sequential:
    resnet50 = tmodels.resnet50(pretrained=pretrained)
    net = nn.Sequential(
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3,
        resnet50.layer4,
        resnet50.avgpool,
        nn.Flatten(),
    )

    return net


def extract_window_with_context(
    image: torch.Tensor,
    bbox: Sequence[Union[int, float]],
    p: int = 16,
    size: int = 256,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Extracts region from a bounding box in the image with some context padding.

    Arguments:
        image - (1, c, h, w) Tensor
        bbox - bounding box specifying (x1, y1, x2, y2)
        p - number of pixels of context to include around window
        size - final size of the window
        pad_value - value of pixels padded
    """
    H, W = image.shape[2:]
    bbox = tuple([int(x) for x in bbox])
    x1, y1, x2, y2 = bbox
    x1 = max(x1 - p, 0)
    y1 = max(y1 - p, 0)
    x2 = min(x2 + p, W)
    y2 = min(y2 + p, H)
    window = image[:, :, y1:y2, x1:x2]
    H, W = window.shape[2:]
    # Zero pad and resize
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if H > W:
        left_pad = (H - W) // 2
        right_pad = (H - W) - left_pad
    elif H < W:
        top_pad = (W - H) // 2
        bot_pad = (W - H) - top_pad
    if H != W:
        window = nn.functional.pad(
            window, (left_pad, right_pad, top_pad, bot_pad), value=pad_value
        )
    window = nn.functional.interpolate(
        window, size=size, mode="bilinear", align_corners=False
    )

    return window

def extract_square_with_context(
    image: torch.Tensor,
    bbox: Sequence[Union[int, float]],
    p: int = 16,
    size: int = 256,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Extracts region from a bounding box in the image with some context padding.

    Arguments:
        image - (1, c, h, w) Tensor
        bbox - bounding box specifying (x1, y1, x2, y2)
        p - number of pixels of context to include around window
        size - final size of the window
        pad_value - value of pixels padded
    """
    H, W = image.shape[2:]
    bbox = tuple([int(x) for x in bbox])
    x1, y1, x2, y2 = bbox
    a2 = max(x2-x1, y2-y1)/2
    x_mid,y_mid = (x1+x2)/2, (y1+y2)/2
    x1 = int(max(x_mid - a2 - p, 0))
    y1 = int(max(y_mid - a2 - p, 0))
    x2 = int(min(x_mid + a2 + p, W))
    y2 = int(min(y_mid + a2 + p, H))
    window = image[:, :, y1:y2, x1:x2]
    H, W = window.shape[2:]
    # pad and resize
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if H > W:
        left_pad = (H - W) // 2
        right_pad = (H - W) - left_pad
    elif H < W:
        top_pad = (W - H) // 2
        bot_pad = (W - H) - top_pad
    if H != W:
        window = nn.functional.pad(
            window, (left_pad, right_pad, top_pad, bot_pad), value=pad_value
        )
    window = nn.functional.interpolate(
        window, size=size, mode="bilinear", align_corners=False
    )

    return window

def get_bbox_from_data(data: Dict[str, Any]) -> List[Numeric]:
    return [data["x"], data["y"], data["x"] + data["width"], data["y"] + data["height"]]


def get_image_id_from_data(data: Dict[str, Any], data_ix: int, rno: int) -> str:
    """
    Defines a unique image id for a given VQ data point.
    """
    clip_uid = data["clip_uid"]
    qset = data["query_set"]
    return f"clip-uid_{clip_uid}_idx_{data_ix}_query-set_{qset}_response-idx_{rno}"


def _extract_image_from_clip(input):
    clip_save_path, rf_fno, rf_path,rf_data = input
    # try:
    reader = pims.Video(clip_save_path)
    dirname = os.path.dirname(rf_path)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except:
            print('cannot create folder for', dirname)
    try:
        if not os.path.exists(rf_path) or os.path.getsize(rf_path)==0:
            f = reader[rf_fno]
            imageio.imwrite(rf_path, f)
    except Exception as e:
        print(e)
    # except:
        print('Clip {} has only {} frames, require {}'.format(clip_save_path, len(reader), rf_fno))
        print('rf_data:', rf_data)


def random_masking(image, grid_size, mask_ratio, mean=[103.53,116.28,123.675]):
    '''
    image - (1, c, h, w) Tensor
    '''
    B,C,H,W = image.shape
    assert C == 3, "RandomMasking only works on RGB images"
    # print(image.shape) # H,W,3
    L = (H//grid_size)*(W//grid_size)
    len_keep = int(L * (1 - mask_ratio))
    # create the rnd mask
    noise = torch.randperm(L, device=image.device)
    mask = noise < len_keep # True (1) for keep
    mask = mask.view(1, H//grid_size, W//grid_size).float() # dim 0 is for batch
    # scale it up
    # print(mask.shape)
    mask = resize(img=mask, size=[H,W], interpolation=InterpolationMode.NEAREST)
    mask = mask.unsqueeze(0) # 1,1,W,H
    # print(mask.shape)
    mean_image = torch.tensor(mean, device=image.device).view(1,3,1,1)
    o_image = image*mask
    o_mean = mean_image*(1-mask)
    return o_image + o_mean

#### jittering score

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

from torchvision.io import read_image
# from torchvision.models.optical_flow import Raft_Large_Weights
# from torchvision.models.optical_flow import raft_large

plt.rcParams["savefig.bbox"] = "tight"



def scatter2d(input,flow):
    '''
    input: H,W
    flow: H,W,2
    '''
    H,W = input.shape
    input1d = input.view(H*W)
    xx = torch.arange(0, W).view(1 ,-1).repeat(H ,1)
    yy = torch.arange(0, H).view(-1 ,1).repeat(1 ,W)
    grid = torch.stack((xx ,yy) ,2).to(flow.device)
    idx = grid + flow
    idx = idx.round().long() #.clip()
    idx_w = idx[...,0].clip(0,W-1)
    idx_h = idx[...,1].clip(0,H-1)
    idx1d = (idx_w+idx_h*W).long().view(-1)
    output = torch.zeros_like(input1d)
    output.scatter_(dim=0, index=idx1d, src=input1d)
    output = output.view(H,W)
    return output
    
def flow2occlusion(flow_f, flow_b):
    """
    flow should be with size [2, H, W]
    """
    _,H,W = flow_f.shape
    initial_mask = torch.ones_like(flow_f[0,...])
    flow = flow_f.permute(1,2,0)
    forward_mask = scatter2d(initial_mask,flow)
    flow = flow_b.permute(1,2,0)
    return_mask = scatter2d(forward_mask,flow)

    return return_mask

def img2occlusion(img1_path, img2_path, scale=0.5):
    """
    img path is a string
    """
    try: 
        im1 = read_image(img1_path) # C,H,W
    except:
        print(img1_path)
    try:
        im2 = read_image(img2_path)
    except:
        print(img2_path)

    img1_batch = torch.stack([im1, im2])
    img2_batch = torch.stack([im2, im1])

    weights = Raft_Large_Weights.DEFAULT

    # pre-process
    B,C,H,W = img1_batch.shape    
    new_size = int(H*scale//8)*8, int(W*scale//8)*8
    transforms = weights.transforms()
    img1_batch,img2_batch = transforms(
        F.resize(img1_batch, size=new_size),
        F.resize(img2_batch, size=new_size)
    )

    # flow model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = raft_large(weights=weights, progress=False).to(device)
    model = model.eval()

    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    predicted_flows = list_of_flows[-1]
    occlusion = flow2occlusion(predicted_flows[0,...], predicted_flows[1,...])

    return occlusion, img1_batch[0,...],img1_batch[1,...]
