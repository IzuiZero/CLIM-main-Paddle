import paddle
import paddle.nn as nn

class IOULoss(nn.Layer):
    def __init__(self, loc_loss_type='iou'):
        super(IOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None, reduction='sum'):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = paddle.minimum(pred_left, target_left) + paddle.minimum(pred_right, target_right)
        h_intersect = paddle.minimum(pred_bottom, target_bottom) + paddle.minimum(pred_top, target_top)

        g_w_intersect = paddle.maximum(pred_left, target_left) + paddle.maximum(pred_right, target_right)
        g_h_intersect = paddle.maximum(pred_bottom, target_bottom) + paddle.maximum(pred_top, target_top)

        ac_union = g_w_intersect * g_h_intersect

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_union - area_union) / ac_union

        if self.loc_loss_type == 'iou':
            losses = -paddle.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None:
            losses = losses * weight
        else:
            losses = losses

        if reduction == 'sum':
            return losses.sum()
        elif reduction == 'batch':
            return losses.sum(axis=[1])
        elif reduction == 'none':
            return losses
        else:
            raise NotImplementedError

def giou_loss(
    boxes1: paddle.Tensor,
    boxes2: paddle.Tensor,
    reduction: str = 'none',
    eps: float = 1e-7,
) -> paddle.Tensor:
    x1, y1, x2, y2 = paddle.unbind(boxes1, axis=-1)
    x1g, y1g, x2g, y2g = paddle.unbind(boxes2, axis=-1)

    assert paddle.all(x2 >= x1), 'bad box: x1 larger than x2'
    assert paddle.all(y2 >= y1), 'bad box: y1 larger than y2'

    # Intersection keypoints
    xkis1 = paddle.maximum(x1, x1g)
    ykis1 = paddle.maximum(y1, y1g)
    xkis2 = paddle.minimum(x2, x2g)
    ykis2 = paddle.minimum(y2, y2g)

    intsctk = paddle.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsctk = paddle.where(mask, (xkis2 - xkis1) * (ykis2 - ykis1), intsctk)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk
    iouk = intsctk / (unionk + eps)

    # smallest enclosing box
    xc1 = paddle.minimum(x1, x1g)
    yc1 = paddle.minimum(y1, y1g)
    xc2 = paddle.maximum(x2, x2g)
    yc2 = paddle.maximum(y2, y2g)

    area_c = (xc2 - xc1) * (yc2 - yc1)
    miouk = iouk - ((area_c - unionk) / (area_c + eps))

    loss = 1 - miouk

    if reduction == 'mean':
        loss = paddle.mean(loss)
    elif reduction == 'sum':
        loss = paddle.sum(loss)

    return loss
