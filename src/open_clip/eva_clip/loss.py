import math
import paddle
import paddle.nn as nn
from paddle.nn import functional as F

from timm.loss import LabelSmoothingCrossEntropy


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    if paddle.distributed.ParallelEnv().nranks > 1:
        if gather_with_grad:
            all_image_features = paddle.distributed.all_gather(image_features)
            all_text_features = paddle.distributed.all_gather(text_features)
        else:
            with paddle.no_grad():
                all_image_features = paddle.distributed.all_gather(image_features)
                all_text_features = paddle.distributed.all_gather(text_features)
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, axis=0))
                gathered_text_features = list(all_text_features.chunk(world_size, axis=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = paddle.concat(gathered_image_features, axis=0)
                all_text_features = paddle.concat(gathered_text_features, axis=0)
    else:
        all_image_features = image_features
        all_text_features = text_features

    return all_image_features, all_text_features


class ClipLoss(nn.Layer):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            smoothing=0.,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.label_smoothing_cross_entropy = LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale=1.):
        device = image_features.place
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * paddle.matmul(image_features, all_text_features, transpose_y=True)
                logits_per_text = logit_scale * paddle.matmul(text_features, all_image_features, transpose_y=True)
            else:
                logits_per_image = logit_scale * paddle.matmul(all_image_features, all_text_features, transpose_y=True)
                logits_per_text = logits_per_image.t()
        else:
            logits_per_image = logit_scale * paddle.matmul(image_features, text_features, transpose_y=True)
            logits_per_text = logit_scale * paddle.matmul(text_features, image_features, transpose_y=True)

        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = paddle.arange(num_logits, dtype='int64')
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if self.label_smoothing_cross_entropy:
            total_loss = (
                self.label_smoothing_cross_entropy(logits_per_image, labels) +
                self.label_smoothing_cross_entropy(logits_per_text, labels)
                ) / 2
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
                ) / 2

        acc = None
        i2t_acc = (logits_per_image.argmax(-1) == labels).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == labels).sum() / len(logits_per_text)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc}
        return total_loss, acc
