from detectron2.config import CfgNode
from detectron2.config import get_cfg as get_default_cfg


# Extend detectron2 defaults
_C = get_default_cfg()


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
# Image augmentations for SiamRCNN
_C.INPUT.ENABLE_RANDOM_BRIGHTNESS = False
_C.INPUT.RANDOM_BRIGHTNESS_VALS = [0.75, 1.33333]
_C.INPUT.ROTATION_ANGLES = [-60, 60]
_C.INPUT.ROTATION_EXPAND = True
_C.INPUT.ENABLE_RANDOM_ROTATION = False
_C.INPUT.TRANSFORM_VISUAL_CROP = False
_C.INPUT.ENABLE_RANDOM_ROTATION_VISUAL_CROP = False
# Visual crop generation
_C.INPUT.REFERENCE_CONTEXT_PAD = 16  # Pixel padding around visual crop
_C.INPUT.REFERENCE_SIZE = 256  # Visual crop size after padding
_C.INPUT.VISUAL_CROP_MASK_RATIO = 0.  # randomly masking out visual crop
_C.INPUT.GT_RM_RATIO = 0.5  # randomly remove positive
# Dataset paths for SiamRCNN training
_C.INPUT.VQ_IMAGES_ROOT = "./data/images"
_C.INPUT.VQ_DATA_SPLITS_ROOT = "./data/vq_splits"
_C.INPUT.CLS_EMB = "/private/home/frostxu/em_public/VQ2D/data/class_clip_embedding.pth"
_C.INPUT.POS_JITTER = "jitter_8h8s"
_C.INPUT.JITTER_AUG_RATIO = 0.5
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
# Config for pre-trained weights for SiamRCNN
_C.MODEL.SIAMESE_PRETRAINED_CONFIG = (
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
_C.MODEL.WEIGHTS_FEATURE = None
_C.MODEL.VITDET_PRETRAINED_WEIGHT = None

# ---------------------------------------------------------------------------- #
# VIT and SFP
# ---------------------------------------------------------------------------- #
_C.MODEL.VIT = CfgNode()
_C.MODEL.SFP = CfgNode()
_C.MODEL.SFP.IN_FEATURE="last_feat"
_C.MODEL.SFP.OUT_CHANNELS=256
_C.MODEL.SFP.SCALE_FACTORS=(4.0, 2.0, 1.0, 0.5)

_C.MODEL.TEACHER = None

# ---------------------------------------------------------------------------- #
# Siamese Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_SIAMESE_HEAD = CfgNode()
_C.MODEL.ROI_SIAMESE_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.ROI_SIAMESE_HEAD.QUERY_FEATURE = "p3"
_C.MODEL.ROI_SIAMESE_HEAD.TXT_FEATURE = "clip_txt"
_C.MODEL.ROI_SIAMESE_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_SIAMESE_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_SIAMESE_HEAD.POOLER_TYPE = "ROIAlignV2"
# Hidden size for siamese similarity head
_C.MODEL.ROI_SIAMESE_HEAD.HIDDEN_SIZE = 1024
# txt size from clip encoder
_C.MODEL.ROI_SIAMESE_HEAD.TXT_SIZE = 512
# Projection layer hyperparameters
_C.MODEL.ROI_SIAMESE_HEAD.PROJECTOR_TYPE = "basic"
_C.MODEL.ROI_SIAMESE_HEAD.N_RESIDUAL_LAYERS = 1
# Hard negative mining for loss computation
_C.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING = CfgNode()
_C.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING.ENABLE = False
# Number of hard negatives to mine from the set of all negatives
_C.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING.NUM_NEGATIVES = 16
# Loss type to use [ bce | kl_div | metric ] --- metric applies only for "dot"
_C.MODEL.ROI_SIAMESE_HEAD.LOSS_TYPE = "bce"
# Share the projection layers for siamese head?
_C.MODEL.ROI_SIAMESE_HEAD.SHARE_PROJECTION = False
# Compare layer type [ bilinear | dot ]
_C.MODEL.ROI_SIAMESE_HEAD.COMPARE_TYPE = "bilinear"
# Set transformer type [ mab | mab_global for frame level loss ]
_C.MODEL.ROI_SIAMESE_HEAD.TRANS_TYPE = "mab"
# reduce number of tokens to reduce the memory
_C.MODEL.ROI_SIAMESE_HEAD.TOKEN_NUMBER_PER_IMAGE = 0
# dropout in transformer to avoid overfitting
_C.MODEL.ROI_SIAMESE_HEAD.TRANS_DROPOUT = 0.0
# Margin value for triplet-margin loss
_C.MODEL.ROI_SIAMESE_HEAD.TRIPLET_MARGIN = 0.25
# Enable cross batch negatives
_C.MODEL.ROI_SIAMESE_HEAD.USE_CROSS_BATCH_NEGATIVES = False
# frame loss rate
_C.MODEL.ROI_SIAMESE_HEAD.FRAME_LOSS_RATE = 0.1

_C.SOLVER.BACKBONE_LR = 0.02

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from .defaults import _C

    return _C.clone()
