import inspect
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from detectron2.config import CfgNode, configurable
from detectron2.layers import cat, ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import ImageList, Instances
from detectron2_extensions.layers import (
    binary_cross_entropy_with_logits,
    kl_div,
    triplet_margin,
)


class DotProduct(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x - (N, F)
            y - (N, F)
        Output:
            output - (N, 1) dot-product output
        """
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert x.shape == y.shape
        x = nn.functional.normalize(x, dim=1)
        y = nn.functional.normalize(y, dim=1)
        output = torch.matmul(x.unsqueeze(1), y.unsqueeze(2)).squeeze(2)
        return output


# 3x3 convolution
def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SiameseHead(nn.Module):
    """
    Compares two image features to see if they correspond to the same object
    instance.
    """

    valid_loss_types: List[str] = ["kl_div", "bce", "metric"]
    valid_compare_types: List[str] = ["bilinear", "dot"]

    def __init__(
        self,
        in_features_1: int,
        in_features_2: int,
        hidden_size: int,
        test_score_thresh: float,
        test_nms_thresh: float,
        test_topk_per_image: int,
        use_hard_negative_mining: bool = False,
        num_hard_negatives: int = 16,
        loss_type: str = "bce",
        use_shared_projection: bool = False,
        compare_type: str = "bilinear",
        margin_value: float = 0.25,
        projector_type: str = "basic",
        n_residual_layers: int = 1,
    ) -> None:
        super().__init__()
        self.in_features_1 = in_features_1
        self.in_features_2 = in_features_2
        self.hidden_size = hidden_size
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.use_hard_negative_mining = use_hard_negative_mining
        self.num_hard_negatives = num_hard_negatives
        self.margin_value = margin_value
        self.projector_type = projector_type
        self.n_residual_layers = n_residual_layers
        assert loss_type in self.valid_loss_types
        if loss_type == "metric":
            assert compare_type == "dot"
        self.loss_type = loss_type
        self.use_shared_projection = use_shared_projection
        if self.use_shared_projection:
            self.projector = self._create_projection_layers(in_features_1, hidden_size)
        else:
            self.projector_1 = self._create_projection_layers(
                in_features_1, hidden_size
            )
            self.projector_2 = self._create_projection_layers(
                in_features_2, hidden_size
            )
        assert compare_type in self.valid_compare_types
        self.compare_type = compare_type
        if self.compare_type == "bilinear":
            self.compare = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        else:
            self.compare = DotProduct()

    def _create_projection_layers(
        self,
        in_features: int,
        hidden_size: int,
    ) -> nn.Sequential:
        if self.projector_type == "basic":
            projector = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(in_features, in_features, 3, 1, 1),
                nn.ReLU(True),
                nn.Conv2d(in_features, in_features, 3, 1, 1),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
            )
        elif self.projector_type == "residual":
            projector = nn.Sequential(
                *[
                    ResidualBlock(in_features, in_features)
                    for _ in range(self.n_residual_layers)
                ],
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_features, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
            )
        else:
            raise ValueError(f"Projector type {self.projector_type} is invalid!")

        return projector

    def forward(
        self, x: List[torch.Tensor], y: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Each element of x and y are (N, F, H, W) and (1, F, H, W) tensors.
        The required output is a (N, 1) matrix of similarities.

            sim(x_i, y_j) = sigmoid( P(x_i)^T W P(y_j) + b )

        Outputs:
            List(Nx1 similarity tensors)

        Note:
            N is the number of proposal bboxes.
            M is the number of visual crop features.
            Each visual crop is compared with every proposal bbox.
        """

        out = []
        for x_, y_ in zip(x, y):
            if self.use_shared_projection:
                xp = self.projector(x_)  # NxH
                yps = self.projector(y_)  # 1xH
            else:
                xp = self.projector_1(x_)  # NxH
                yps = self.projector_2(y_)  # 1xH
            sims = []
            for yp in yps:
                yp = yp.unsqueeze(0).expand(xp.shape[0], -1)  # NxH
                sim = self.compare(xp, yp)  # Nx1
                sims.append(sim)
            if len(sims) == 1:
                sim = sims[0]  # Nx1
            else:
                sims = torch.stack(sims, dim=1)  # Nx1x1
                assert sims.shape[1] == 1
                sim = sims.squeeze(1)  # Nx1
            out.append(sim)
        return out

    def losses(
        self,
        predictions: List[torch.Tensor],
        gt_classes: List[Instances],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: return values of :meth:`forward()`
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        if self.use_hard_negative_mining:
            return self._losses_hnm(predictions, gt_classes)
        else:
            return self._losses(predictions, gt_classes)

    def _losses(
        self,
        predictions: List[torch.Tensor],
        gt_classes: List[Instances],
    ) -> Dict[str, torch.Tensor]:
        """
        Standard binary cross entropy loss between predictions and ground-truth boxes.
        Args: same as :meth: `losses()`
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        if self.loss_type == "bce":
            scores = cat(predictions).squeeze(1)
            # parse classification outputs
            gt_classes = cat(gt_classes, dim=0) if len(gt_classes) else torch.empty(0)
            # _log_classification_stats(scores, gt_classes)
            loss_cls = binary_cross_entropy_with_logits(
                scores, gt_classes, reduction="mean", flip_class=True
            )
        elif self.loss_type == "kl_div":
            loss_cls = 0.0
            for pred, gt in zip(predictions, gt_classes):
                pred = pred.squeeze(1).unsqueeze(0)
                gt = gt.unsqueeze(0)
                loss_cls = loss_cls + kl_div(
                    pred, gt, reduction="mean", flip_class=True
                )
        else:
            loss_cls = 0.0
            for pred, gt in zip(predictions, gt_classes):
                loss_cls = loss_cls + triplet_margin(
                    pred, gt, reduction="mean", flip_cls=True, margin=self.margin_value
                )
        losses = {"loss_cls": loss_cls}

        return losses

    def _losses_hnm(
        self,
        predictions: List[torch.Tensor],
        gt_classes: List[Instances],
    ) -> Dict[str, torch.Tensor]:
        """
        Performs hard-negative mining before applying the standard BCE loss.
        Args: same as :meth: `losses()`
        Returns:
            Dict[str, Tensor]: dict of losses
        """
        # Perform hard negative mining
        mined_predictions = []
        mined_gt_classes = []
        for predictions_i, gt_classes_i in zip(predictions, gt_classes):
            neg_mask = gt_classes_i == 1  # (N, )
            neg_preds = predictions_i[neg_mask]  # (N, 1)
            if len(neg_preds) <= self.num_hard_negatives:
                mined_predictions.append(predictions_i)
                mined_gt_classes.append(gt_classes_i)
            else:
                # Get indices of topk negative activations
                hard_neg_idxs = torch.topk(
                    neg_preds[:, 0], self.num_hard_negatives
                ).indices
                # Select all positives + topk negatives
                data_mask = gt_classes_i == 0  # (P, )
                data_mask[hard_neg_idxs] = 1
                mined_predictions.append(predictions_i[data_mask])
                mined_gt_classes.append(gt_classes_i[data_mask])

        return self._losses(mined_predictions, mined_gt_classes)

    def inference(self, predictions: List[torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.
        Returns:
            outputs are processed versions of inputs
        """
        boxes = [x.proposal_boxes.tensor for x in proposals]
        # Convert predictions to the appropriate format
        if self.loss_type == "bce":
            predictions = [torch.sigmoid(pred) for pred in predictions]
        elif self.loss_type == "kl_div":
            predictions = [torch.softmax(pred, dim=0) for pred in predictions]
        # The scores must include K + 1 classes, where the last class is a
        # background class.# Here, the first class is the actual visual crop.
        # The second class is 1 - that value.
        scores = [torch.cat([pred, 1 - pred], dim=1) for pred in predictions]
        image_shapes = [x.image_size for x in proposals]

        outputs = fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

        return outputs


@ROI_HEADS_REGISTRY.register()
class SiameseROIHeads(ROIHeads):
    """
    A Standard ROIHeads which contains an addition of SiamesePredictor head.
    """

    @configurable
    def __init__(
        self,
        *args: Any,
        siam_in_features: List[str],
        siam_q_feature: str,
        siam_pooler: ROIPooler,
        siam_head: nn.Module,
        ref_pooler: nn.Module,
        use_cross_batch_negatives: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        NOTE: this interface is experimental.
        Args:
            siam_in_features (list[str]): list of feature names to use for the siam head.
            siam_pooler (ROIPooler): pooler to extra region features for siam head
            siam_head (nn.Module): transform features to make siam predictions
            siam_predictor (nn.Module): make siam predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            use_cross_batch_negatives
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.siam_in_features = siam_in_features
        self.q_feature = self.siam_q_feature = siam_q_feature
        self.siam_pooler = siam_pooler
        self.siam_head = siam_head
        self.ref_pooler = ref_pooler
        self.use_cross_batch_negatives = use_cross_batch_negatives

    @classmethod
    def from_config(cls, cfg: CfgNode, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg)
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_siamese_head):
            ret.update(cls._init_siamese_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_siamese_head(
        cls,
        cfg: CfgNode,
        input_shape: Dict[str, ShapeSpec],
    ) -> Dict[str, Any]:
        # fmt: off
        in_features          = cfg.MODEL.ROI_SIAMESE_HEAD.IN_FEATURES
        q_feature            = cfg.MODEL.ROI_SIAMESE_HEAD.QUERY_FEATURE
        pooler_resolution    = cfg.MODEL.ROI_SIAMESE_HEAD.POOLER_RESOLUTION
        pooler_scales        = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio       = cfg.MODEL.ROI_SIAMESE_HEAD.POOLER_SAMPLING_RATIO
        pooler_type          = cfg.MODEL.ROI_SIAMESE_HEAD.POOLER_TYPE
        hidden_size          = cfg.MODEL.ROI_SIAMESE_HEAD.HIDDEN_SIZE
        projector_type       = cfg.MODEL.ROI_SIAMESE_HEAD.PROJECTOR_TYPE
        n_residual_layers    = cfg.MODEL.ROI_SIAMESE_HEAD.N_RESIDUAL_LAYERS
        test_score_thresh    = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        test_nms_thresh      = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        test_topk_per_image  = cfg.TEST.DETECTIONS_PER_IMAGE
        hnm_enable           = cfg.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING.ENABLE
        hnm_num_negatives    = cfg.MODEL.ROI_SIAMESE_HEAD.HARD_NEGATIVE_MINING.NUM_NEGATIVES
        loss_type            = cfg.MODEL.ROI_SIAMESE_HEAD.LOSS_TYPE
        share_projection     = cfg.MODEL.ROI_SIAMESE_HEAD.SHARE_PROJECTION
        compare_type         = cfg.MODEL.ROI_SIAMESE_HEAD.COMPARE_TYPE
        margin_value         = cfg.MODEL.ROI_SIAMESE_HEAD.TRIPLET_MARGIN
        use_cross_batch_negs = cfg.MODEL.ROI_SIAMESE_HEAD.USE_CROSS_BATCH_NEGATIVES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        q_channels = input_shape[q_feature].channels

        siam_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ref_pooler = nn.AdaptiveAvgPool2d(pooler_resolution)
        siam_head = SiameseHead(
            in_channels,
            q_channels,
            hidden_size,
            test_score_thresh,
            test_nms_thresh,
            test_topk_per_image,
            use_hard_negative_mining=hnm_enable,
            num_hard_negatives=hnm_num_negatives,
            loss_type=loss_type,
            use_shared_projection=share_projection,
            compare_type=compare_type,
            margin_value=margin_value,
            projector_type=projector_type,
            n_residual_layers=n_residual_layers,
        )
        return {
            "siam_in_features": in_features,
            "siam_q_feature": q_feature,
            "siam_pooler": siam_pooler,
            "siam_head": siam_head,
            "ref_pooler": ref_pooler,
            "use_cross_batch_negatives": use_cross_batch_negs,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        ref_features: Dict[str, torch.Tensor],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_siam(features, ref_features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            return proposals, losses
        else:
            pred_instances = self._forward_siam(features, ref_features, proposals)
            return pred_instances, {}

    def _forward_siam(
        self,
        features: Dict[str, torch.Tensor],
        ref_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
    ):
        """
        Forward logic of the siam prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            ref_features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of NxM similarity scores to each proposal.
        """
        features = [features[f] for f in self.siam_in_features]
        ref_features = ref_features[self.siam_q_feature]  # (N, R, *)
        siam_features = self.siam_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        # siam_features contains (M, C, H, W) tensor combining all images and proposals.
        # these need to be split into batches for siam_head.
        siam_features_joint = siam_features
        siam_features_split = []
        itr = 0
        for proposals_per_image in proposals:
            nprops = proposals_per_image.proposal_boxes.tensor.size(0)
            siam_features_split.append(siam_features[itr : itr + nprops])
            itr += nprops
        siam_features = siam_features_split
        # Pool ref_features
        N, R = ref_features.shape[:2]
        ref_features = ref_features.view(-1, *ref_features.shape[2:])
        ref_features = self.ref_pooler(ref_features)
        ref_features = ref_features.view(N, R, *ref_features.shape[1:])
        # ref_features contains (N, R, C, H, W) tensor with one reference per batch
        # multiple rotations of the same reference may be provided (R)
        # these need to be split into list of batch elements for siam_head
        ref_features = [rf for rf in ref_features]
        if self.training:
            gt_classes = [proposals_i.gt_classes for proposals_i in proposals]
            gt_classes_joint = cat(gt_classes, dim=0)
            if self.use_cross_batch_negatives:
                siam_features_cb = []
                gt_classes_cb = []
                batch_idxs = torch.Tensor(
                    [
                        i
                        for i, gt_classes_i in enumerate(gt_classes)
                        for _ in range(len(gt_classes_i))
                    ]
                )
                for i in range(len(gt_classes)):
                    pos_mask = batch_idxs == i
                    neg_mask = ~pos_mask
                    gt = torch.zeros_like(gt_classes_joint)
                    gt[pos_mask] = gt_classes_joint[pos_mask]
                    gt[neg_mask] = 1
                    siam_features_cb.append(siam_features_joint)
                    gt_classes_cb.append(gt)
                siam_features = siam_features_cb
                gt_classes = gt_classes_cb

        predictions = self.siam_head(siam_features, ref_features)

        del siam_features

        if self.training:
            losses = self.siam_head.losses(predictions, gt_classes)
            return losses
        else:
            pred_instances, _ = self.siam_head.inference(predictions, proposals)
            return pred_instances
