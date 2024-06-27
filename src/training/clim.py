import paddle
import paddle.nn.functional as F

class CLIM:
    mosaic_choices = [2, 3, 4]

    def __init__(self):
        super().__init__()

    def __call__(self, batch, model, dist_model, loss, device, cast_dtype,
                 distributed, args):
        if distributed:
            model = model._layers  # Assuming model is an instance of nn.Layer in PaddlePaddle
        images, texts = batch
        images = paddle.to_tensor(images, dtype=cast_dtype)
        texts = paddle.to_tensor(texts)

        mosaicked_images, pseudo_boxes_list, single_images \
            = self.split_a_batch(images, args.train_image_size)
        single_image_features = model.encode_image(single_images, normalize=True)
        with paddle.no_grad():
            text_features = model.encode_text(texts, normalize=True)
        logit_scale = model.logit_scale.exp()

        pseudo_region_features = model.encode_pseudo_boxes(
            mosaicked_images, pseudo_boxes_list, normalize=True, extract_type=args.extract_type)
        image_features = paddle.concat([pseudo_region_features, single_image_features], axis=0)

        contrast_loss = loss(image_features,
                             text_features,
                             logit_scale,
                             output_dict=False)

        losses = dict(loss_contrast=contrast_loss * args.contrast_weight)

        return losses, len(images), logit_scale

    @staticmethod
    def _generate_normed_boxes(M, N):
        grid_x, grid_y = paddle.meshgrid(paddle.linspace(0, 1, N + 1), paddle.linspace(0, 1, M + 1))
        x0y0s = paddle.stack([grid_x[:M, :N], grid_y[:M, :N]], axis=-1)
        x1y1s = paddle.stack([grid_x[1:, 1:], grid_y[1:, 1:]], axis=-1)
        pseudo_boxes = paddle.concat([x0y0s, x1y1s], axis=-1).reshape([-1, 4])
        return pseudo_boxes

    def split_a_batch(self, images, train_image_size):
        batch_size = images.shape[0]
        choices = self.mosaic_choices
        min_images = sum([c**2 for c in choices])

        assert batch_size >= min_images
        num_single = batch_size % min_images
        num_groups = batch_size // min_images

        split = [c for c in choices for _ in range(num_groups)]
        pseudo_boxes_list = [self._generate_normed_boxes(s, s).astype(images.dtype) for s in split]

        images_list = paddle.split(images, [s**2 for s in split] + [num_single], axis=0)

        mosaicked_images_list = [
            F.interpolate(self._mosaic_a_minibatch(imgs, s, s), size=train_image_size, mode='bicubic')
            for imgs, s in zip(images_list[:-1], split)]

        mosaicked_images = paddle.concat(mosaicked_images_list, axis=0)

        return mosaicked_images, pseudo_boxes_list, images_list[-1]

    @staticmethod
    def _mosaic_a_minibatch(images, M, N):
        bs, _, h, w = images.shape
        assert bs % (M * N) == 0
        num_mosaic = bs // (M * N)
        images_for_mosaic = images.transpose([0, 2, 3, 1])
        images_for_mosaic = images_for_mosaic.reshape([num_mosaic, M, N, h, w, 3])
        images_for_mosaic = images_for_mosaic.transpose([0, 1, 3, 2, 4, 5]).reshape([num_mosaic, M * h, N * w, 3])
        mosaicked_images = images_for_mosaic.transpose([0, 3, 1, 2])

        return mosaicked_images
