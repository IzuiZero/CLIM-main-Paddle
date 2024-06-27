import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn

def get_fed_loss_inds(gt_classes, num_sample_cats, C):
    appeared = paddle.unique(gt_classes)  # C'
    prob = paddle.ones([C], dtype='float32')
    if appeared.shape[0] < num_sample_cats:
        prob[appeared] = 0
        more_appeared = paddle.multinomial(
            prob, num_sample_cats - appeared.shape[0],
            replacement=False)
        appeared = paddle.concat([appeared, more_appeared])
    return appeared

class RegionCLIP(nn.Layer):
    def __init__(self, args):
        super().__init__()
        embed_path = args.train_embed_path
        noun_embeddings = np.load(embed_path)
        noun_embeddings = paddle.to_tensor(noun_embeddings, dtype='float32')
        noun_embeddings = F.normalize(noun_embeddings, axis=-1)
        self.noun_embeddings = self.create_parameter(shape=noun_embeddings.shape, dtype='float32', default_initializer=paddle.nn.initializer.Assign(noun_embeddings))
        self.place_holder = self.create_parameter(shape=[1], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))

    def forward(self, batch, model, dist_model, loss, device, cast_dtype,
                distributed, args):
        if distributed:
            model = model._layers['model']
        images, boxes = batch
        images = paddle.to_tensor(images, dtype=cast_dtype)
        boxes = paddle.to_tensor(boxes)

        boxes_list = []
        boxes_label_list = []

        for boxes_per_image in boxes:
            boxes_per_image = boxes_per_image[boxes_per_image[:, -1] > 0.5]
            boxes_label_list.append(boxes_per_image[:, 4].astype('int64'))
            boxes_list.append(boxes_per_image[:, :4])
        boxes_labels = paddle.concat(boxes_label_list)
        box_features = model.encode_pseudo_boxes(images, boxes_list, normalize=True,
                                                 extract_type=args.extract_type)
        temp = model.logit_scale.exp().detach()
        boxes2nouns = paddle.matmul(box_features, self.noun_embeddings.T) * temp
        target = paddle.zeros_like(boxes2nouns)
        target[paddle.arange(boxes_labels.shape[0]), boxes_labels] = 1.0

        appeared = get_fed_loss_inds(boxes_labels, 100, self.noun_embeddings.shape[0])
        target = target[:, appeared]
        boxes2nouns = boxes2nouns[:, appeared]

        loss_cls = F.binary_cross_entropy_with_logits(boxes2nouns, target, reduction='none')  # B x C
        loss_cls = paddle.sum(loss_cls, axis=-1).mean()

        image_size = model.visual.image_size
        if isinstance(image_size, int):
            tar_h = tar_w = image_size
        else:
            tar_h, tar_w = image_size
        images = F.interpolate(images, size=(tar_h, tar_w), mode='bilinear')

        losses = dict(loss_contrast=loss_cls * args.contrast_weight)

        return losses, len(images), temp
