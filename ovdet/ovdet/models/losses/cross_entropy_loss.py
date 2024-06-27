import paddle
import paddle.nn.functional as F
from paddle import Tensor
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models import CrossEntropyLoss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
from ovdet.utils import load_class_freq

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False, **kwargs):
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.shape[-1], ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).astype(paddle.float32)
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.astype(paddle.float32)
    loss = F.binary_cross_entropy_with_logits(pred, label.astype(paddle.float32), reduction='none')
    if class_weight is not None:
        loss = loss * paddle.to_tensor(class_weight, dtype=paddle.float32)[None]
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    if class_weight is not None:
        mask_out = paddle.to_tensor(class_weight, dtype=paddle.float32) < 0.00001
        pred[:, mask_out] = -float('inf')
    loss = F.cross_entropy(
        pred,
        label,
        weight=paddle.to_tensor(class_weight, dtype=paddle.float32),
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = paddle.numel(label) - paddle.sum((label == ignore_index).astype(paddle.int32)).item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.astype(paddle.float32)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss
@MODELS.register_module()
class CustomCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, bg_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sigmoid:
            del self.cls_criterion
            self.cls_criterion = binary_cross_entropy
        elif not self.use_mask:
            del self.cls_criterion
            self.cls_criterion = cross_entropy

        if isinstance(self.class_weight, str):
            cat_freq = load_class_freq(self.class_weight, min_count=0)
            self.class_weight = (cat_freq > 0.0).astype(paddle.float32).tolist() + [bg_weight]
