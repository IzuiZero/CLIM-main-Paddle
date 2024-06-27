import logging
import paddle
import paddle.nn.functional as F
from tqdm import tqdm
from .distributed import all_gather
from .distributed import is_master
from open_clip import get_cast_dtype
from .precision import get_autocast

def run_panoptic(model, dataloader, args):
    cls_embeddings = dataloader.dataset.embeddings
    cls_embeddings = paddle.to_tensor(cls_embeddings).astype('float32')
    cls_embeddings = F.normalize(cls_embeddings, axis=-1)
    cls_embeddings = cls_embeddings.to(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if cast_dtype is not None:
        cls_embeddings = cls_embeddings.astype(cast_dtype)
    with paddle.no_grad():
        correct_rois = []
        correct_maskpool = []
        correct_crops = []
        similarity_crops = []
        similarity_rois = []
        similarity_maskpool = []
        all_box_sizes = []
        all_is_thing = []
        all_cls_labels = []
        for images, bboxes, image_crops, gt_masks, masked_image_crops in tqdm(dataloader, disable=not is_master(args)):
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            image_crops = image_crops.to(args.device)
            masked_image_crops = masked_image_crops.to(args.device)
            gt_masks = gt_masks.to(args.device)
            if cast_dtype is not None:
                images = images.astype(cast_dtype)
                bboxes = bboxes.astype(cast_dtype)
                image_crops = image_crops.astype(cast_dtype)
                masked_image_crops = masked_image_crops.astype(cast_dtype)
                gt_masks = gt_masks.astype(cast_dtype)
            image_crops_list = []
            gt_masks_list = []
            cls_labels = []
            rois = []
            box_sizes = []
            is_thing = []
            for bboxes_per_image, crops_per_image, gt_mask, masked_crops_per_image in zip(bboxes, image_crops, gt_masks, masked_image_crops):
                valid = bboxes_per_image[:, 5] > 0.5
                rois.append(bboxes_per_image[valid, :4])
                cls_labels.append(bboxes_per_image[valid, 4])
                image_crops_list.append(crops_per_image[valid])
                gt_masks_list.append(gt_mask[valid])
                box_sizes.append(bboxes_per_image[valid, 6])
                is_thing.append(bboxes_per_image[valid, 7])
            cls_labels = paddle.concat(cls_labels, axis=0).astype('int64')
            if cls_labels.shape[0] == 0:
                continue
            image_crops = paddle.concat(image_crops_list)
            box_sizes = paddle.concat(box_sizes, axis=0).astype('float32')
            is_thing = paddle.concat(is_thing, axis=0)
            all_box_sizes.append(box_sizes)
            all_is_thing.append(is_thing)
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    module = model.module
                else:
                    module = model
                roi_extractor = module.encode_pseudo_boxes
                roi_features = roi_extractor(images, rois, normalize=True, extract_type=args.extract_type)
                mask_pooler = module.encode_masks
                maskpool_features = mask_pooler(images, gt_masks_list, normalize=True, mask_attn=args.extract_type == "v1")
                # New way to obtain crop features
                if args.image_ave_pool:
                    feature_map = module.visual.encode_dense(image_crops, keep_shape=True)
                    crop_features = paddle.mean(feature_map, axis=(-2, -1))
                    crop_features = F.normalize(crop_features, axis=-1)
                else:
                    crop_features = module.encode_image(image_crops, normalize=True)

                if cast_dtype is not None:
                    roi_features = roi_features.astype(cast_dtype)
                    crop_features = crop_features.astype(cast_dtype)
                    maskpool_features = maskpool_features.astype(cast_dtype)

                roi_logits = paddle.matmul(roi_features, cls_embeddings, transpose_y=True)
                crop_logits = paddle.matmul(crop_features, cls_embeddings, transpose_y=True)
                maskpool_logits = paddle.matmul(maskpool_features, cls_embeddings, transpose_y=True)

            _, roi_top5_inds = paddle.topk(roi_logits, k=5)
            _, crop_top5_inds = paddle.topk(crop_logits, k=5)
            _, maskpool_top5_inds = paddle.topk(maskpool_logits, k=5)
            correct_rois.append(roi_top5_inds == cls_labels.unsqueeze(1))
            correct_crops.append(crop_top5_inds == cls_labels.unsqueeze(1))
            correct_maskpool.append(maskpool_top5_inds == cls_labels.unsqueeze(1))

            similarity_rois.append(paddle.gather(roi_logits, index=cls_labels.unsqueeze(1), axis=1)[:, 0])
            similarity_crops.append(paddle.gather(crop_logits, index=cls_labels.unsqueeze(1), axis=1)[:, 0])
            similarity_maskpool.append(paddle.gather(maskpool_logits, index=cls_labels.unsqueeze(1), axis=1)[:, 0])

            all_cls_labels.append(cls_labels)

        # TODO: gather correct matrix across gpus
        correct_rois = paddle.concat(correct_rois, axis=0).astype('float32')
        correct_crops = paddle.concat(correct_crops, axis=0).astype('float32')
        correct_maskpool = paddle.concat(correct_maskpool, axis=0).astype('float32')
        similarity_rois = paddle.concat(similarity_rois, axis=0).astype('float32')
        similarity_crops = paddle.concat(similarity_crops, axis=0).astype('float32')
        similarity_maskpool = paddle.concat(similarity_maskpool, axis=0).astype('float32')
        all_box_sizes = paddle.concat(all_box_sizes, axis=0)
        all_is_thing = paddle.concat(all_is_thing, axis=0)
        all_cls_labels = paddle.concat(all_cls_labels, axis=0)
        if args.distributed and not args.horovod:
            correct_rois = multi_gpu_sync(correct_rois)
            correct_crops = multi_gpu_sync(correct_crops)
            correct_maskpool = multi_gpu_sync(correct_maskpool)
            all_box_sizes = multi_gpu_sync(all_box_sizes)
            all_is_thing = multi_gpu_sync(all_is_thing)
            similarity_rois = multi_gpu_sync(similarity_rois)
            similarity_crops = multi_gpu_sync(similarity_crops)
            similarity_maskpool = multi_gpu_sync(similarity_maskpool)
            all_cls_labels = multi_gpu_sync(all_cls_labels)

    return correct_rois, correct_crops, correct_maskpool, \
        similarity_rois, similarity_crops, similarity_maskpool, \
        all_box_sizes, all_is_thing, all_cls_labels


def run_det(model, dataloader, args):
    cls_embeddings = dataloader.dataset.embeddings
    cls_embeddings = paddle.to_tensor(cls_embeddings).astype('float32')
    cls_embeddings = F.normalize(cls_embeddings, axis=-1)
    cls_embeddings = cls_embeddings.to(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    if cast_dtype is not None:
        cls_embeddings = cls_embeddings.astype(cast_dtype)
    with paddle.no_grad():
        correct_rois = []
        correct_crops = []
        all_box_sizes = []
        all_cls_labels = []
        for images, bboxes, image_crops in tqdm(dataloader, disable=not is_master(args)):
            images = images.to(args.device)
            bboxes = bboxes.to(args.device)
            image_crops = image_crops.to(args.device)
            if cast_dtype is not None:
                images = images.astype(cast_dtype)
                bboxes = bboxes.astype(cast_dtype)
                image_crops = image_crops.astype(cast_dtype)
            image_crops_list = []
            cls_labels = []
            rois = []
            box_sizes = []
            for bboxes_per_image, crops_per_image in zip(bboxes, image_crops):
                valid = bboxes_per_image[:, 5] > 0.5
                rois.append(bboxes_per_image[valid, :4])
                cls_labels.append(bboxes_per_image[valid, 4])
                image_crops_list.append(crops_per_image[valid])
                box_sizes.append(bboxes_per_image[valid, 6])
            cls_labels = paddle.concat(cls_labels, axis=0).astype('int64')
            if cls_labels.shape[0] == 0:
                continue
            image_crops = paddle.concat(image_crops_list)
            box_sizes = paddle.concat(box_sizes, axis=0).astype('float32')
            all_box_sizes.append(box_sizes)
            with autocast():
                # predict
                if args.distributed and not args.horovod:
                    module = model.module
                else:
                    module = model
                roi_extractor = module.encode_pseudo_boxes
                roi_features = roi_extractor(images, rois, normalize=True, extract_type=args.extract_type)
                crop_features = module.encode_image(image_crops, normalize=True)

                if cast_dtype is not None:
                    roi_features = roi_features.astype(cast_dtype)
                    crop_features = crop_features.astype(cast_dtype)

                roi_logits = paddle.matmul(roi_features, cls_embeddings, transpose_y=True)
                crop_logits = paddle.matmul(crop_features, cls_embeddings, transpose_y=True)

            _, roi_top5_inds = paddle.topk(roi_logits, k=5)
            _, crop_top5_inds = paddle.topk(crop_logits, k=5)
            correct_rois.append(roi_top5_inds == cls_labels.unsqueeze(1))
            correct_crops.append(crop_top5_inds == cls_labels.unsqueeze(1))

            all_cls_labels.append(cls_labels)

        # TODO: gather correct matrix across gpus
        correct_rois = paddle.concat(correct_rois, axis=0).astype('float32')
        correct_crops = paddle.concat(correct_crops, axis=0).astype('float32')
        all_box_sizes = paddle.concat(all_box_sizes, axis=0)
        all_cls_labels = paddle.concat(all_cls_labels, axis=0)
        if args.distributed and not args.horovod:
            correct_rois = multi_gpu_sync(correct_rois)
            correct_crops = multi_gpu_sync(correct_crops)
            all_box_sizes = multi_gpu_sync(all_box_sizes)
            all_cls_labels = multi_gpu_sync(all_cls_labels)

    return correct_rois, correct_crops, all_box_sizes, all_cls_labels


def multi_gpu_sync(x):
    device = x.place
    x_list = all_gather(x.numpy())
    x = paddle.concat([paddle.to_tensor(res) for res in x_list], axis=0)
    return x


def macc_with_is_thing(correct_matrix, is_thing, all_cls_labels, prefix):
    def _macc(corrects, cls_labels):
        min_id = paddle.min(cls_labels).item()
        max_id = paddle.max(cls_labels).item()
        cand_labels = list(range(min_id, max_id+1))

        acc_per_cls = []

        for lb in cand_labels:
            corrects_per_cls = corrects[cls_labels == lb]
            if corrects_per_cls.shape[0] == 0:
                continue
            acc_per_cls.append(paddle.mean(corrects_per_cls.astype('float16')).item())

        return sum(acc_per_cls) / len(acc_per_cls)

    results = {}
    thing_correct_matrix = correct_matrix[is_thing > 0]
    stuff_correct_matrix = correct_matrix[is_thing < 1]

    thing_cls_labels = all_cls_labels[is_thing > 0].astype('int64')
    stuff_cls_labels = all_cls_labels[is_thing < 1].astype('int64')

    thing_top1_acc = _macc(thing_correct_matrix[:, 0], thing_cls_labels)
    thing_top5_acc = _macc(paddle.sum(thing_correct_matrix, axis=-1), thing_cls_labels)

    stuff_top1_acc = _macc(stuff_correct_matrix[:, 0], stuff_cls_labels)
    stuff_top5_acc = _macc(paddle.sum(stuff_correct_matrix, axis=-1), stuff_cls_labels)

    results[f'{prefix}.thing.macc1'] = thing_top1_acc
    results[f'{prefix}.thing.macc5'] = thing_top5_acc
    results[f'{prefix}.stuff.macc1'] = stuff_top1_acc
    results[f'{prefix}.stuff.macc5'] = stuff_top5_acc

    return results


def macc_with_det(correct_matrix, all_cls_labels, prefix):
    def _macc(corrects, cls_labels):
        min_id = paddle.min(cls_labels).item()
        max_id = paddle.max(cls_labels).item()
        cand_labels = list(range(min_id, max_id+1))

        acc_per_cls = []

        for lb in cand_labels:
            corrects_per_cls = corrects[cls_labels == lb]
            if corrects_per_cls.shape[0] == 0:
                continue
            acc_per_cls.append(paddle.mean(corrects_per_cls.astype('float16')).item())

        return sum(acc_per_cls) / len(acc_per_cls)

    results = {}
    top1_acc = _macc(correct_matrix[:, 0], all_cls_labels)
    top5_acc = _macc(paddle.sum(correct_matrix, axis=-1), all_cls_labels)

    results[f'{prefix}.macc1'] = top1_acc
    results[f'{prefix}.macc5'] = top5_acc

    return results


def zero_shot_eval(model, data, epoch, args):
    if 'val' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    logging.info('Region classifier')
    results = {}
    if args.test_type == "coco_panoptic":
        correct_rois, correct_crops, correct_maskpool, \
            similarity_rois, similarity_crops, similarity_maskpool, \
            all_box_sizes, all_is_thing, all_cls_labels = run_panoptic(model, data['val'].dataloader, args)
        results.update(macc_with_is_thing(correct_rois, all_is_thing, all_cls_labels, 'rois'))
        results.update(macc_with_is_thing(correct_crops, all_is_thing, all_cls_labels, 'crops'))
        results.update(macc_with_is_thing(correct_maskpool, all_is_thing, all_cls_labels, 'maskpool'))
    else:
        assert args.test_type == "coco_detection"
        correct_rois, correct_crops, all_box_sizes, all_cls_labels = run_det(model, data['val'].dataloader, args)
        results.update(macc_with_det(correct_rois, all_cls_labels, 'rois'))
        results.update(macc_with_det(correct_crops, all_cls_labels, 'crops'))

    return results
