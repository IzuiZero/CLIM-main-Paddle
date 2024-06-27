from typing import Optional, Tuple
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from paddle import Tensor
from mmdet.models import ConvFCBBoxHead
from mmdet.models.layers import multiclass_nms
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.bbox import get_box_tensor, scale_boxes


@MODELS.register_module()
class FVLMConvFCBBoxHead(ConvFCBBoxHead):
    def __init__(self,
                 alpha=0.35,
                 beta=0.65,
                 clip_temp=50.0,
                 cls_temp=50.0,
                 learn_cls_temp=True,
                 cls_embeddings_path=None, bg_embedding='learn',
                 invalid_classes=None,
                 *args, **kwargs):
        super(FVLMConvFCBBoxHead, self).__init__(*args, **kwargs)
        if learn_cls_temp:
            self.cls_temp = self.create_parameter(shape=[1],
                                                 default_initializer=paddle.nn.initializer.Assign(cls_temp))
        else:
            self.cls_temp = cls_temp
        self.clip_temp = clip_temp
        self.alpha = alpha
        self.beta = beta
        assert self.with_cls
        assert self.reg_class_agnostic
        assert not self.custom_cls_channels

        if invalid_classes is not None:
            self.register_buffer('invalid_classes', paddle.to_tensor(invalid_classes))
        else:
            self.invalid_classes = None

        cls_embeddings = paddle.to_tensor(np.load(cls_embeddings_path), dtype='float32')
        self.learn_bg = False
        if bg_embedding == 'zero':
            assert self.num_classes == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings)
            self.register_buffer('bg_embedding', paddle.zeros_like(cls_embeddings[:1]))
        elif bg_embedding == 'learn':
            assert self.num_classes == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings)
            self.bg_embedding = nn.Linear(1, cls_embeddings.shape[1])
            self.bg_embedding.weight.uniform_(-1.0 / np.sqrt(cls_embeddings.shape[1]),
                                              1.0 / np.sqrt(cls_embeddings.shape[1]))
            self.bg_embedding.bias.zero_()
            self.learn_bg = True
        elif bg_embedding == 'clip':
            assert (self.num_classes + 1) == cls_embeddings.shape[0]
            self.register_buffer('cls_embeddings', cls_embeddings[:-1])
            self.register_buffer('bg_embedding', cls_embeddings[-1:])
        else:
            raise ValueError(f"{bg_embedding} not supported.")

        self.fc_cls = nn.Linear(self.cls_last_dim, cls_embeddings.shape[1])
        self.class_weight = paddle.to_tensor(self.loss_cls.class_weight[:-1] + [1.0])

    def pred_cls_logits(self, region_embeddings, clip_embeddings=None):
        region_embeddings = F.normalize(region_embeddings, p=2, axis=-1)
        if self.learn_bg:
            input_ones = paddle.ones((1, 1), dtype=region_embeddings.dtype)
            bg_embedding = self.bg_embedding(input_ones)
            bg_embedding = F.normalize(bg_embedding, p=2, axis=-1)
        else:
            bg_embedding = self.bg_embedding
        cls_embeddings = paddle.concat([self.cls_embeddings, bg_embedding], axis=0)
        cls_logits = self.cls_temp * paddle.matmul(region_embeddings, cls_embeddings.t())
        assert cls_logits.shape[1] == self.num_classes + 1

        if not self.training:
            if self.invalid_classes is not None:
                cls_logits[:, self.invalid_classes > 0] = float('-inf')
            cls_scores = F.softmax(cls_logits, axis=-1)
            assert clip_embeddings is not None
            clip_embeddings = F.normalize(clip_embeddings, p=2, axis=-1)
            clip_logits = self.clip_temp * paddle.matmul(clip_embeddings, cls_embeddings.t())
            if self.invalid_classes is not None:
                clip_logits[:, self.invalid_classes > 0] = float('-inf')
            clip_scores = F.softmax(clip_logits, axis=-1)

            base_idx = self.class_weight > 0.0
            novel_idx = paddle.logical_not(base_idx)

            cls_scores[:, base_idx] = (cls_scores[:, base_idx] ** (1 - self.alpha)
                                       * clip_scores[:, base_idx] ** self.alpha)
            cls_scores[:, novel_idx] = (cls_scores[:, novel_idx] ** (1 - self.beta)
                                        * clip_scores[:, novel_idx] ** self.beta)

            return cls_scores

        return cls_logits

    def forward(self, x: Tuple[Tensor], clip_embeddings=None):
        region_embeddings, bbox_pred = super().forward(x)
        cls_score = self.pred_cls_logits(region_embeddings,
                                         clip_embeddings=clip_embeddings)
        return cls_score, bbox_pred

    def _predict_by_feat_single(
            self,
            roi: Tensor,
            cls_score: Tensor,
            bbox_pred: Optional[Tensor],
            img_meta: dict,
            rescale: bool = False,
            rcnn_test_cfg: Optional[ConfigDict] = None) -> InstanceData:
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
        if bbox_pred is not None:
            num_classes = 1 if self.reg_class_agnostic else self.num_classes
            roi = paddle.tile(roi, repeat_times=[num_classes, 1])
            bbox_pred = bbox_pred.reshape([-1, self.bbox_coder.encode_size])
            bboxes = self.bbox_coder.decode(
                roi[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = roi[:, 1:].clone()
            if img_shape is not None and bboxes.shape[-1] == 4:
                bboxes[:, [0, 2]].clip_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clip_(min=0, max=img_shape[0])

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
