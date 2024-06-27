from typing import List, Sequence, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from mmcv.cnn import normal_init

from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule
from mmcv.utils import build_from_cfg

from mmdet.models.roi_heads import CascadeRoIHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.test_time_augs import merge_aug_masks
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
from mmdet.utils import ConfigType, InstanceList, MultiConfig

from ovdet.methods.builder import OVD


@MODELS.register_module()
class DeticRoIHead(CascadeRoIHead):
    def __init__(self, clip_cfg=None, ovd_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.clip = None
        if clip_cfg is not None:
            self.clip = build_from_cfg(clip_cfg, MODELS)

        if ovd_cfg is not None:
            for k, v in ovd_cfg.items():
                setattr(self, k, OVD.build(v))

    def run_ovd(self, x, batch_data_samples, rpn_results_list, ovd_name, batch_inputs, *args, **kwargs) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        ovd_method = getattr(self, ovd_name)

        losses = dict()
        results_list = rpn_results_list

        for stage in range(self.num_stages):
            self.current_stage = stage
            stage_loss_weight = self.stage_loss_weights[stage]

            sampling_results = [
                ovd_method.sample(results_list[i], batch_data_samples[i])
                for i in range(len(results_list))
            ]
            for sampling_result in sampling_results:
                sampling_result['bboxes'] = paddle.tensor.detach(sampling_result['bboxes'])

            bbox_results = self._bbox_run_ovd(stage, x, sampling_results)
            stage_losses = ovd_method.get_losses(
                bbox_results['region_embeddings'],
                sampling_results=sampling_results,
                clip_model=self.clip,
                images=batch_inputs,
                update_queue=stage == (self.num_stages - 1)   # update queue only in the last stage
            )

            for k, v in stage_losses.items():
                losses[f'{stage}.{k}'] = v * stage_loss_weight

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with paddle.no_grad():
                    results_list = [
                        bbox_head.refine_bboxes_ovd(
                            bbox_results, [data_sample['metainfo'] for data_sample in batch_data_samples])
                        for data_sample in batch_data_samples
                    ]
                    # Empty proposal
                    assert results_list is not None

        return losses

    def _bbox_run_ovd(self, stage: int, x: Tuple[paddle.Tensor], sampling_results):
        rois = bbox2roi([res['bboxes'] for res in sampling_results])
        ovd_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = ovd_roi_extractor(x[:ovd_roi_extractor.num_inputs],
                                       rois)
        region_embeddings, bbox_pred = bbox_head.run_ovd(bbox_feats)

        bbox_results = dict(rois=rois, region_embeddings=region_embeddings,
                            bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        return bbox_results

    def init_mask_head(self, mask_roi_extractor: MultiConfig,
                       mask_head: MultiConfig) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_head (dict): Config of mask in mask head.
            mask_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of mask roi extractor.
        """
        self.mask_head = build_from_cfg(mask_head, MODELS)

        if mask_roi_extractor is not None:
            self.share_roi_extractor = False
            self.mask_roi_extractor = build_from_cfg(mask_roi_extractor, MODELS)
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def _refine_roi(self, x: Tuple[paddle.Tensor], rois: paddle.Tensor,
                    batch_img_metas: List[dict],
                    num_proposals_per_img: Sequence[int], **kwargs) -> tuple:
        """Multi-stage refinement of RoI.

        Args:
            x (tuple[paddle.Tensor]): List of multi-level img features.
            rois (paddle.Tensor): shape (n, 5), [batch_ind, x1, y1, x2, y2]
            batch_img_metas (list[dict]): List of image information.
            num_proposals_per_img (sequence[int]): number of proposals
                in each image.

        Returns:
            tuple:

               - rois (paddle.Tensor): Refined RoI.
               - cls_scores (list[paddle.Tensor]): Average predicted
                   cls score per image.
               - bbox_preds (list[paddle.Tensor]): Bbox branch predictions
                   for the last stage of per image.
        """
        # "ms" in variable names means multi-stage
        ms_scores = []
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage=stage, x=x, rois=rois, **kwargs)

            # split batch bbox prediction back to each image
            cls_scores = bbox_results['cls_score'].sigmoid()
            bbox_preds = bbox_results['bbox_pred']

            rois = paddle.split(rois, num_proposals_per_img, axis=0)
            cls_scores = paddle.split(cls_scores, num_proposals_per_img, axis=0)
            ms_scores.append(cls_scores)
            bbox_preds = paddle.split(bbox_preds, num_proposals_per_img, axis=0)

            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                refine_rois_list = []
                for i in range(len(batch_img_metas)):
                    if rois[i].shape[0] > 0:
                        bbox_label = cls_scores[i][:, :-1].argmax(axis=1)
                        # Refactor `bbox_head.regress_by_class` to only accept
                        # box tensor without img_idx concatenated.
                        refined_bboxes = bbox_head.regress_by_class(
                            rois[i][:, 1:], bbox_label, bbox_preds[i],
                            batch_img_metas[i])
                        refined_bboxes = get_box_tensor(refined_bboxes)
                        refined_rois = paddle.concat(
                            [rois[i][:, [0]], refined_bboxes], axis=1)
                        refine_rois_list.append(refined_rois)
                rois = paddle.concat(refine_rois_list)
        # ms_scores aligned
        # average scores of each image by stages
        cls_scores = [
            paddle.sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(len(batch_img_metas))
        ]  # aligned
        return rois, cls_scores, bbox_preds

    def predict_bbox(self,
                     x: Tuple[paddle.Tensor],
                     batch_img_metas: List[dict],
                     rpn_results_list: InstanceList,
                     rcnn_test_cfg: ConfigType,
                     rescale: bool = False,
                     **kwargs) -> InstanceList:
        """Perform forward propagation of the bbox head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[paddle.Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (paddle.Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (paddle.Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (paddle.Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        proposals = [res.bboxes for res in rpn_results_list]
        proposal_scores = [res.scores for res in rpn_results_list]
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            return empty_instances(
                batch_img_metas,
                rois.device,
                task_type='bbox',
                box_type=self.bbox_head[-1].predict_box_type,
                num_classes=self.bbox_head[-1].num_classes,
                score_per_cls=rcnn_test_cfg is None)
        # rois aligned
        rois, cls_scores, bbox_preds = self._refine_roi(
            x=x,
            rois=rois,
            batch_img_metas=batch_img_metas,
            num_proposals_per_img=num_proposals_per_img,
            **kwargs)

        # score reweighting in centernet2
        cls_scores = [(s * ps[:, None])**0.5
                      for s, ps in zip(cls_scores, proposal_scores)]
        # # for demo
        # cls_scores = [
        #     s * (s == s[:, :-1].max(axis=1)[0][:, None]).astype('float32')
        #     for s in cls_scores
        # ]

        # fast_rcnn_inference
        results_list = self.bbox_head[-1].predict_by_feat(
            rois=rois,
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            rcnn_test_cfg=rcnn_test_cfg)
        return results_list

    def _mask_forward(self, x: Tuple[paddle.Tensor], rois: paddle.Tensor) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[paddle.Tensor]): Tuple of multi-level img features.
            rois (paddle.Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (paddle.Tensor): Mask prediction.
        """
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], rois)
        # do not support caffe_c4 model anymore
        mask_preds = self.mask_head(mask_feats)

        mask_results = dict(mask_preds=mask_preds)
        return mask_results

    def mask_loss(self, x, sampling_results: List[SamplingResult],
                  batch_gt_instances: InstanceList) -> dict:
        """Run forward function and calculate loss for mask head in training.

        Args:
            x (tuple[paddle.Tensor]): Tuple of multi-level img features.
            sampling_results (list["SamplingResult"]): Sampling results.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict: Usually returns a dictionary with keys:

                - `mask_preds` (paddle.Tensor): Mask prediction.
                - `loss_mask` (dict): A dictionary of mask loss components.
        """
        pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        mask_results = self._mask_forward(x, pos_rois)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg[-1])
        mask_results.update(mask_loss_and_target)

        return mask_results

    def loss(self, x: Tuple[paddle.Tensor], rpn_results_list: InstanceList,
             batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[paddle.Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, paddle.Tensor]: A dictionary of loss components
        """
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs

        num_imgs = len(batch_data_samples)
        losses = dict()
        results_list = rpn_results_list

        for stage in range(self.num_stages):
            self.current_stage = stage
            stage_loss_weight = self.stage_loss_weights[stage]
            if hasattr(batch_gt_instances[0], 'bboxes'):
                # assign gts and sample proposals
                sampling_results = []
                if self.with_bbox or self.with_mask:
                    bbox_assigner = self.bbox_assigner[stage]
                    bbox_sampler = self.bbox_sampler[stage]

                    for i in range(num_imgs):
                        results = results_list[i]
                        # rename rpn_results.bboxes to rpn_results.priors
                        results.priors = results.pop('bboxes')

                        assign_result = bbox_assigner.assign(
                            results, batch_gt_instances[i],
                            batch_gt_instances_ignore[i])

                        sampling_result = bbox_sampler.sample(
                            assign_result,
                            results,
                            batch_gt_instances[i],
                            feats=[lvl_feat[i][None] for lvl_feat in x])

                        sampling_results.append(sampling_result)

                # bbox head forward and loss
                bbox_results = self.bbox_loss(stage, x, sampling_results)

                for name, value in bbox_results['loss_bbox'].items():
                    losses[f's{stage}.{name}'] = (
                        value * stage_loss_weight if 'loss' in name else value)

                # mask head forward and loss
                # D2 only forward stage.0
                if self.with_mask and stage == 0:
                    mask_results = self.mask_loss(x, sampling_results,
                                                  batch_gt_instances)
                    for name, value in mask_results['loss_mask'].items():
                        losses[name] = (
                            value *
                            stage_loss_weight if 'loss' in name else value)

            else:
                for name in ['loss_cls', 'loss_bbox']:
                    losses[f's{stage}.{name}'] = paddle.zeros([1])[0]
                if stage == 0:
                    losses['loss_mask'] = paddle.zeros([1])[0]

            # refine bboxes
            if stage < self.num_stages - 1:
                bbox_head = self.bbox_head[stage]
                with paddle.no_grad():
                    results_list = [
                        bbox_head.refine_bboxes(
                            bbox_results, batch_img_metas)
                        for bbox_results in results_list
                    ]
                    # Empty proposal
                    if results_list is None:
                        break

        return losses

    def predict_mask(self,
                     x: Tuple[paddle.Tensor],
                     batch_img_metas: List[dict],
                     results_list: List[InstanceData],
                     rescale: bool = False) -> List[InstanceData]:
        """Perform forward propagation of the mask head and predict detection
        results on the features of the upstream network.

        Args:
            x (tuple[paddle.Tensor]): Feature maps of all scale level.
            batch_img_metas (list[dict]): List of image information.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (paddle.Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (paddle.Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (paddle.Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (paddle.Tensor): Has a shape (num_instances, H, W).
        """
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        num_mask_rois_per_img = [len(res) for res in results_list]
        aug_masks = []
        mask_results = self._mask_forward(x, mask_rois)
        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        mask_preds = paddle.split(mask_preds, num_mask_rois_per_img, axis=0)
        aug_masks.append([m.sigmoid().detach() for m in mask_preds])

        merged_masks = []
        for i in range(len(batch_img_metas)):
            aug_mask = [mask[i] for mask in aug_masks]
            merged_mask = merge_aug_masks(aug_mask, batch_img_metas[i])
            merged_masks.append(merged_mask)
        results_list = self.mask_head.predict_by_feat(
            mask_preds=merged_masks,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
            activate_map=True)
        return results_list
