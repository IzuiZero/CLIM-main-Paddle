import copy
from typing import Dict, List, Optional, Sequence, Tuple

import paddle
import paddle.nn as nn
from mmcv.cnn import Scale
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from mmdet.models.dense_heads import CenterNetUpdateHead
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2distance
from mmdet.utils import ConfigType, InstanceList, OptConfigType, reduce_mean
from .iou_loss import IOULoss

INF = 1000000000
RangeType = Sequence[Tuple[int, int]]


@MODELS.register_module()
class CenterNetRPNHead(CenterNetUpdateHead):
    """CenterNetUpdateHead is an improved version of CenterNet in CenterNet2.

    Paper link `<https://arxiv.org/abs/2103.07461>`_.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channel in the input feature map.
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        hm_min_radius (int): Heatmap target minimum radius of cls branch.
            Defaults to 4.
        hm_min_overlap (float): Heatmap target minimum overlap of cls branch.
            Defaults to 0.8.
        more_pos_thresh (float): The filtering threshold when the cls branch
            adds more positive samples. Defaults to 0.2.
        more_pos_topk (int): The maximum number of additional positive samples
            added to each gt. Defaults to 9.
        soft_weight_on_reg (bool): Whether to use the soft target of the
            cls branch as the soft weight of the bbox branch.
            Defaults to False.
        loss_cls (:obj:`ConfigDict` or dict): Config of cls loss. Defaults to
            dict(type='GaussianFocalLoss', loss_weight=1.0)
        loss_bbox (:obj:`ConfigDict` or dict): Config of bbox loss. Defaults to
             dict(type='GIoULoss', loss_weight=2.0).
        norm_cfg (:obj:`ConfigDict` or dict, optional): dictionary to construct
            and config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config.
            Unused in CenterNet. Reserved for compatibility with
            SingleStageDetector.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config
            of CenterNet.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((0, 80), (64, 160), (128, 320),
                                              (256, 640), (512, INF)),
                 hm_min_radius: int = 4,
                 hm_min_overlap: float = 0.8,
                 more_pos: bool = False,
                 more_pos_thresh: float = 0.2,
                 more_pos_topk: int = 9,
                 soft_weight_on_reg: bool = False,
                 not_clamp_box: bool = False,
                 loss_cls: ConfigType = dict(
                     type='HeatmapFocalLoss',
                     alpha=0.25,
                     beta=4.0,
                     gamma=2.0,
                     pos_weight=1.0,
                     neg_weight=1.0,
                     sigmoid_clamp=1e-4,
                     ignore_high_fp=-1.0,
                     loss_weight=1.0,
                 ),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss', loss_weight=2.0),
                 norm_cfg: OptConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            norm_cfg=norm_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.soft_weight_on_reg = soft_weight_on_reg
        self.hm_min_radius = hm_min_radius
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.more_pos = more_pos
        self.not_clamp_box = not_clamp_box
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        self.loss_bbox = IOULoss('giou')

        self.use_sigmoid_cls = True
        self.cls_out_channels = num_classes

        self.regress_ranges = regress_ranges
        self.scales = nn.LayerList([Scale(1.0) for _ in self.strides])

    def _init_layers(self) -> None:
        self._init_reg_convs()
        self._init_predictor()

    def forward_single(self, x: paddle.Tensor, scale: Scale,
                       stride: int) -> Tuple[paddle.Tensor, paddle.Tensor]:
        for m in self.reg_convs:
            x = m(x)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_pred = scale(bbox_pred).astype('float32')
        bbox_pred = paddle.clip(bbox_pred, min=0)
        return cls_score, bbox_pred

    def loss_by_feat(
        self,
        cls_scores: List[paddle.Tensor],
        bbox_preds: List[paddle.Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, paddle.Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is 4.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = cls_scores[0].shape[0]
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [cls_score.shape[-2:] for cls_score in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=str(bbox_preds[0].device))

        flatten_cls_scores = [
            cls_score.transpose((0, 2, 3, 1)).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.transpose((0, 2, 3, 1)).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = paddle.concat(flatten_cls_scores)
        flatten_bbox_preds = paddle.concat(flatten_bbox_preds)

        flatten_points = paddle.concat(
            [points.tile((num_imgs, 1)) for points in all_level_points])

        assert paddle.isfinite(flatten_bbox_preds).all().item()

        cls_targets, bbox_targets = self.get_targets(all_level_points,
                                                     batch_gt_instances)

        featmap_sizes = flatten_points.new_tensor(featmap_sizes)

        if self.more_pos:
            pos_inds, cls_labels = self.add_cls_pos_inds(
                flatten_points, flatten_bbox_preds, featmap_sizes,
                batch_gt_instances)
        else:
            pos_inds = self._get_label_inds(batch_gt_instances,
                                            batch_img_metas, featmap_sizes)

        if pos_inds is None:
            num_pos_cls = bbox_preds[0].new_tensor(0, dtype='float32')
        else:
            num_pos_cls = bbox_preds[0].new_tensor(
                len(pos_inds), dtype='float32')
        num_pos_cls = max(reduce_mean(num_pos_cls), 1.0)

        cat_agn_cls_targets = cls_targets.max(axis=1)[0]

        cls_pos_loss, cls_neg_loss = self.loss_cls(
            flatten_cls_scores.squeeze(1), cat_agn_cls_targets, pos_inds,
            num_pos_cls)

        pos_bbox_inds = paddle.nonzero(
            bbox_targets.max(axis=1)[0] >= 0).squeeze(1)
        pos_bbox_preds = flatten_bbox_preds[pos_bbox_inds]
        pos_bbox_targets = bbox_targets[pos_bbox_inds]

        bbox_weight_map = cls_targets.max(axis=1)[0]
        bbox_weight_map = bbox_weight_map[pos_bbox_inds]
        bbox_weight_map = bbox_weight_map if self.soft_weight_on_reg \
            else paddle.ones_like(bbox_weight_map)

        num_pos_bbox = max(reduce_mean(bbox_weight_map.sum()), 1.0)

        if len(pos_bbox_inds) > 0:
            bbox_loss = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                bbox_weight_map,
                reduction='sum') / num_pos_bbox
        else:
            bbox_loss = flatten_bbox_preds.sum() * 0

        return dict(
            loss_bbox=bbox_loss,
            loss_cls_pos=cls_pos_loss,
            loss_cls_neg=cls_neg_loss)

    def loss_and_predict(
        self,
        x: Tuple[paddle.Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions

    def _predict_by_feat_single(self,
                                cls_score_list: List[paddle.Tensor],
                                bbox_pred_list: List[paddle.Tensor],
                                score_factor_list: List[paddle.Tensor],
                                mlvl_priors: List[paddle.Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]

            bbox_pred = bbox_pred * self.strides[level_idx]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.transpose((1, 2, 0)).reshape(-1, dim)
            cls_score = cls_score.transpose((1, 2,
                                             0)).reshape(-1, self.cls_out_channels)
            heatmap = cls_score.sigmoid()
            score_thr = cfg.get('score_thr', 0)

            candidate_inds = heatmap > score_thr
            pre_nms_top_n = candidate_inds.sum()
            pre_nms_top_n = min(pre_nms_top_n.item(), nms_pre) if nms_pre > 0 else pre_nms_top_n

            heatmap = heatmap[candidate_inds]
            bbox_pred = bbox_pred[candidate_inds]
            priors = priors[candidate_inds]

            if pre_nms_top_n > 0 and pre_nms_top_n < len(heatmap):
                _, topk_inds = heatmap.topk(pre_nms_top_n)
                heatmap = heatmap[topk_inds]
                bbox_pred = bbox_pred[topk_inds]
                priors = priors[topk_inds]

            det_bboxes, det_labels = self.bbox_head.refine_bboxes(
                bbox_pred, priors, img_meta, cfg)
            mlvl_bbox_preds.append(det_bboxes)
            mlvl_scores.append(heatmap)
            mlvl_labels.append(det_labels)

        if len(mlvl_bbox_preds) == 0:
            mlvl_bbox_preds = [paddle.empty((0, 4), dtype='float32')]
            mlvl_scores = [paddle.empty((0, self.cls_out_channels),
                                        dtype='float32')]
            mlvl_labels = [paddle.empty((0,), dtype='int32')]

        mlvl_bbox_preds = paddle.concat(mlvl_bbox_preds, axis=0)
        mlvl_scores = paddle.concat(mlvl_scores, axis=0)
        mlvl_labels = paddle.concat(mlvl_labels, axis=0)

        if with_nms:
            mlvl_bboxes, mlvl_scores, mlvl_labels = self.bbox_head.nms(mlvl_bbox_preds, mlvl_scores, mlvl_labels,
                    score_thr, cfg.max_per_img)
            mlvl_bboxes, mlvl_scores, mlvl_labels = mlvl_bboxes[
                :nms_post], mlvl_scores[:nms_post], mlvl_labels[:nms_post]

        return InstanceData(
            bbox=mlvl_bboxes,
            scores=mlvl_scores,
            labels=mlvl_labels,
            img_shape=img_meta['pad_shape'],
            ori_shape=img_meta['ori_shape'],
            scale_factor=img_meta.get('scale_factor', 1.0),
            pad_shape=img_meta['pad_shape'])

    def predict_by_feat(
        self,
        cls_scores: List[paddle.Tensor],
        bbox_preds: List[paddle.Tensor],
        score_factors: List[paddle.Tensor],
        batch_img_metas: List[dict],
        cfg: ConfigDict,
        rescale: bool = False
    ) -> InstanceList:
        mlvl_priors = self.prior_generator.grid_priors(
            [cls_score.shape[-2:] for cls_score in cls_scores],
            dtype=cls_scores[0].dtype,
            device=str(cls_scores[0].device))
        return [
            self._predict_by_feat_single(cls_score, bbox_pred, score_factor,
                                         mlvl_priors, img_meta, cfg, rescale)
            for cls_score, bbox_pred, score_factor, img_meta in zip(
                cls_scores, bbox_preds, score_factors, batch_img_metas)
        ]

    def get_targets(
        self,
        points: paddle.Tensor,
        batch_gt_instances: InstanceList,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        gt_bboxes, gt_labels = unpack_gt_instances(batch_gt_instances)
        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]

        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]

        num_points = points.shape[0]

        bbox_targets = paddle.full((num_points, 4), -1, dtype='float32')
        bbox_weights = paddle.full((num_points, 4), 0, dtype='float32')
        agn_cls_targets = paddle.zeros(num_points, dtype='int32')

        num_gt = gt_bboxes.shape[0]

        if num_gt == 0:
            return agn_cls_targets, bbox_targets

        distances = bbox2distance(points, gt_bboxes)
        min_distances, min_indices = distances.min(axis=1)

        for i in range(num_points):
            if min_distances[i] < self.hm_min_radius:
                bbox_targets[i] = gt_bboxes[min_indices[i]]
                bbox_weights[i] = paddle.ones(4)
                agn_cls_targets[i] = gt_labels[min_indices[i]]

        return agn_cls_targets, bbox_targets

    def _get_label_inds(
        self,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        featmap_sizes: List[List[int]]
    ) -> Optional[paddle.Tensor]:
        gt_bboxes, gt_labels = unpack_gt_instances(batch_gt_instances)
        return self.prior_generator.match_points(gt_bboxes, gt_labels,
                                                 featmap_sizes)

    def add_cls_pos_inds(
        self,
        points: paddle.Tensor,
        bbox_preds: paddle.Tensor,
        featmap_sizes: paddle.Tensor,
        batch_gt_instances: InstanceList
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        gt_bboxes, gt_labels = unpack_gt_instances(batch_gt_instances)

        gt_bboxes = gt_bboxes[0]
        gt_labels = gt_labels[0]

        num_points = points.shape[0]

        bbox_targets = paddle.full((num_points, 4), -1, dtype='float32')
        bbox_weights = paddle.full((num_points, 4), 0, dtype='float32')
        agn_cls_targets = paddle.zeros(num_points, dtype='int32')

        num_gt = gt_bboxes.shape[0]

        if num_gt == 0:
            return None, agn_cls_targets

        distances = bbox2distance(points, gt_bboxes)
        min_distances, min_indices = distances.min(axis=1)

        ind = (min_distances < self.hm_min_radius)
        min_indices = min_indices[ind]

        bbox_preds = bbox_preds[ind]

        min_bboxes = gt_bboxes[min_indices]

        dets = bbox2distance(points, bbox_preds, min_bboxes)

        dets_max, _ = dets.max(axis=1)

        dets_max_ind = (dets_max > self.more_pos_thresh)
        dets_max, _ = dets_max[dets_max_ind].topk(min(dets_max_ind.sum().item(),
                                                      self.more_pos_topk))

        bbox_targets[dets_max_ind] = min_bboxes[dets_max[:, None].tile(
            (1, 4)), dets[dets_max_ind].argmax(axis=1)]

        bbox_weights[dets_max_ind] = paddle.ones(4)
        agn_cls_targets[dets_max_ind] = gt_labels[min_indices[
            dets[dets_max_ind].argmax(axis=1)]]

        return dets_max, agn_cls_targets
