import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from paddle.static import InputSpec
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmcv.runner import ConfigDict
from mmcv.utils import deprecated_api_warning
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.models import RPNHead
from mmdet.structures.bbox import empty_box_as, get_box_tensor, get_box_wh, scale_boxes
@MODELS.register_module()
class CustomRPNHead(RPNHead):
    def __init__(self, norm_cfg=None, *args, **kwargs):
        self.norm_cfg = norm_cfg
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        inplace=False))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2D(
                self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2D(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2D(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: paddle.Tensor = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation.

        Args:
            results (InstanceData): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (paddle.Tensor, optional): Image meta info. Defaults to None.

        Returns:
            InstanceData: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        deprecated_api_warning(self._bbox_post_process, 'bbox_post_process', 'mmdet', 0.17)
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)

            det_bboxes, keep_idxs = batched_nms(bboxes, results.scores.astype('float32'),
                                                results.level_ids, cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = paddle.zeros((len(results),), dtype='int64')
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = empty_box_as(results.bboxes)
            results_.scores = paddle.zeros((0,), dtype='float32')
            results_.labels = paddle.zeros((0,), dtype='int64')
            results = results_
        return results
