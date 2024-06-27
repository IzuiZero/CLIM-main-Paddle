import json
from typing import List, Optional
import paddle
from paddle import Tensor
import paddle.nn.functional as F
from mmcv.cnn import Caffe2Xavier
from mmcv.runner import force_fp32

from mmdet.models.utils import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.utils import empty_instances
from mmdet.core import InstanceData
from mmdet.core.bbox.geometry import get_box_tensor, scale_boxes
from mmcv import ConfigDict
from mmdet.utils import ConfigType, InstanceList


def load_class_freq(path='datasets/metadata/lvis_v1_train_cat_info.json',
                    freq_weight=0.5):
    with open(path, 'r') as f:
        cat_info = json.load(f)
    cat_info = paddle.to_tensor(
        [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])])
    freq_weight = cat_info.cast('float32')**freq_weight
    return freq_weight


def get_fed_loss_inds(labels, num_sample_cats, C, weight=None):
    appeared = paddle.unique(labels)  # C'
    prob = paddle.ones([C + 1], dtype='float32')
    prob[-1] = 0
    if len(appeared) < num_sample_cats:
        if weight is not None:
            prob[:C] = weight.cast('float32').clone()
        prob[appeared] = 0
        more_appeared = paddle.multinomial(
            prob, num_sample_cats - len(appeared), replacement=False)
        appeared = paddle.concat([appeared, more_appeared])
    return appeared


@MODELS.register_module()
class DeticBBoxHead(Shared2FCBBoxHead):
    def __init__(self,
                 use_fed_loss: bool = False,
                 cat_freq_path: str = '',
                 fed_loss_freq_weight: float = 0.5,
                 fed_loss_num_cat: int = 50,
                 cls_predictor_cfg: ConfigType = dict(type='ZeroShotClassifier'),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # reconstruct fc_cls and fc_reg since input channels are changed
        assert self.with_cls

        self.cls_predictor_cfg = cls_predictor_cfg
        cls_channels = self.num_classes
        self.cls_predictor_cfg.update(
            in_features=self.cls_last_dim, out_features=cls_channels)
        self.fc_cls = MODELS.build(self.cls_predictor_cfg)

        self.init_cfg += [
            dict(type='Caffe2Xavier', override=dict(name='reg_fcs'))
        ]

        self.use_fed_loss = use_fed_loss
        self.cat_freq_path = cat_freq_path
        self.fed_loss_freq_weight = fed_loss_freq_weight
        self.fed_loss_num_cat = fed_loss_num_cat

        if self.use_fed_loss:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.freq_weight = self.create_parameter(freq_weight.shape, default_initializer=lambda _: paddle.to_tensor(freq_weight))
        else:
            self.freq_weight = None

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Tensor,
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            roi (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_meta (dict): image information.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None

        Returns:
            :obj:`InstanceData`: Detection results of each image\
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        results = InstanceData()
        if roi.shape[0] == 0:
            return empty_instances([img_meta],
                                   roi.device,
                                   task_type='bbox',
                                   instance_results=[results],
                                   box_type=self.predict_box_type,
                                   use_box_type=False,
                                   num_classes=self.num_classes,
                                   score_per_cls=rcnn_test_cfg is None)[0]
        scores = cls_score
        img_shape = img_meta['img_shape']
        num_rois = roi.shape[0]

        num_classes = 1 if self.reg_class_agnostic else self.num_classes
        roi = paddle.tile(roi, [1, num_classes])
        bbox_pred = bbox_pred.reshape([-1, self.bbox_coder.encode_size])
        bboxes = self.bbox_coder.decode(
            roi[..., 1:], bbox_pred, max_shape=img_shape)

        if rescale and bboxes.shape[0] > 0:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            bboxes = scale_boxes(bboxes, scale_factor)

        bboxes = get_box_tensor(bboxes)
        box_dim = bboxes.shape[-1]
        bboxes = bboxes.reshape([num_rois, -1])

        if rcnn_test_cfg is None:
            results.bboxes = bboxes
            results.scores = scores
        else:
            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                rcnn_test_cfg.score_thr,
                rcnn_test_cfg.nms,
                rcnn_test_cfg.max_per_img,
                box_dim=box_dim)
            results.bboxes = det_bboxes[:, :-1]
            results.scores = det_bboxes[:, -1]
            results.labels = det_labels
        return results

    def loss(self,
             cls_score: Tensor,
             bbox_pred: Tensor,
             rois: Tensor,
             labels: Tensor,
             label_weights: Tensor,
             bbox_targets: Tensor,
             bbox_weights: Tensor,
             reduction_override: Optional[str] = None) -> dict:
        """Calculate the loss based on the network predictions and targets.

        Args:
            cls_score (Tensor): Classification prediction
                results of all class, has shape
                (batch_size * num_proposals_single_image, num_classes)
            bbox_pred (Tensor): Regression prediction results,
                has shape
                (batch_size * num_proposals_single_image, 4), the last
                dimension 4 represents [tl_x, tl_y, br_x, br_y].
            rois (Tensor): RoIs with the shape
                (batch_size * num_proposals_single_image, 5) where the first
                column indicates batch id of each RoI.
            labels (Tensor): Gt_labels for all proposals in a batch, has
                shape (batch_size * num_proposals_single_image, ).
            label_weights (Tensor): Labels_weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, ).
            bbox_targets (Tensor): Regression target for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4),
                the last dimension 4 represents [tl_x, tl_y, br_x, br_y].
            bbox_weights (Tensor): Regression weights for all proposals in a
                batch, has shape (batch_size * num_proposals_single_image, 4).
            reduction_override (str, optional): The reduction
                method used to override the original reduction
                method of the loss. Options are "none",
                "mean" and "sum". Defaults to None,

        Returns:
            dict: A dictionary of loss.
        """

        losses = dict()

        if cls_score is not None:

            if cls_score.numel() > 0:
                loss_cls_ = self.sigmoid_cross_entropy_loss(cls_score, labels)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                    bbox_pred = get_box_tensor(bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.shape[0], -1)[pos_inds.astype(paddle.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.shape[0], self.num_classes, -1)[
                        pos_inds.astype(paddle.bool), labels[pos_inds.astype(paddle.bool)]]

                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.astype(paddle.bool)],
                    bbox_weights[pos_inds.astype(paddle.bool)],
                    avg_factor=bbox_targets.shape[0],
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

    def sigmoid_cross_entropy_loss(self, cls_score, labels):
        if cls_score.numel() == 0:
            return cls_score.new_zeros(
                [1])[0]
        B = cls_score.shape[0]
        C = cls_score.shape[1] - 1

        target = paddle.zeros([B, C + 1], dtype=cls_score.dtype)
        target[paddle.arange(len(labels)), labels] = 1
        target = target[:, :C]

        weight = paddle.ones([1], dtype='float32')
        if self.use_fed_loss and (self.freq_weight is not None):
            appeared = get_fed_loss_inds(
                labels,
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = paddle.zeros([C + 1], dtype=labels.dtype)
            appeared_mask[appeared] = 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.unsqueeze(0).expand([B, C])
            weight = weight * fed_w.cast('float32')
        
        cls_loss = F.binary_cross_entropy_with_logits(
            cls_score[:, :-1], target, reduction='none')
        loss = paddle.sum(cls_loss * weight) / B
        return loss

    @force_fp32(apply_to=('bbox_pred',))
    def regress(self, priors: Tensor, bbox_pred: Tensor,
                img_meta: dict) -> Tensor:
        reg_dim = self.bbox_coder.encode_size
        assert bbox_pred.shape[1] == reg_dim

        max_shape = img_meta['img_shape']
        regressed_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=max_shape)
        return regressed_bboxes

    def run_ovd(self, x) -> tuple:
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.ndim > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.ndim > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        region_embeddings = self.fc_cls.linear(x_cls)
        bbox_pred = self.fc_reg(x_reg)
        return region_embeddings, bbox_pred

    def refine_bboxes_ovd(self, bbox_results: dict,
                          batch_img_metas: List[dict]) -> InstanceList:
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']
        img_ids = rois[:, 0].unique()
        assert img_ids.shape[0] <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = paddle.nonzero(
                rois[:, 0] == i, as_tuple=False).squeeze()

            bboxes_ = rois[inds, 1:]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]

            bboxes = self.regress(bboxes_, bbox_pred_, img_meta_)

            results = InstanceData(bboxes=bboxes)
            results_list.append(results)

        return results_list
