import collections.abc
import logging
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingNormal

# open CLIP
def resize_clip_pos_embed(state_dict, model, interpolation='bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info(f'Resizing position embedding grid-size from {old_grid_size} to {grid_size}')
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).transpose((0, 3, 1, 2))
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.transpose((0, 2, 3, 1)).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = paddle.concat([pos_emb_tok, pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_visual_pos_embed(state_dict, model, interpolation='bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info(f'Resizing position embedding grid-size from {old_grid_size} to {grid_size}')
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).transpose((0, 3, 1, 2))
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.transpose((0, 2, 3, 1)).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = paddle.concat([pos_emb_tok, pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['positional_embedding'] = new_pos_embed


def resize_evaclip_pos_embed(state_dict, model, interpolation='bicubic', seq_dim=1):
    all_keys = list(state_dict.keys())
    # interpolate position embedding
    if 'visual.pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['visual.pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = F.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.transpose((0, 2, 3, 1)).reshape((-1, embedding_size))
            new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
            state_dict['visual.pos_embed'] = new_pos_embed

            patch_embed_proj = state_dict['visual.patch_embed.proj.weight']
            patch_size = model.visual.patch_embed.patch_size
            state_dict['visual.patch_embed.proj.weight'] = F.interpolate(
                patch_embed_proj.astype('float32'), size=patch_size, mode='bicubic', align_corners=False)


def resize_eva_pos_embed(state_dict, model, interpolation='bicubic', seq_dim=1):
    all_keys = list(state_dict.keys())
    # interpolate position embedding
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = F.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.transpose((0, 2, 3, 1)).reshape((-1, embedding_size))
            new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
            state_dict['pos_embed'] = new_pos_embed

            patch_embed_proj = state_dict['patch_embed.proj.weight']
            patch_size = model.visual.patch_embed.patch_size
            state_dict['patch_embed.proj.weight'] = F.interpolate(
                patch_embed_proj.astype('float32'), size=patch_size, mode='bicubic', align_corners=False)


def resize_rel_pos_embed(state_dict, model, interpolation='bicubic', seq_dim=1):
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.shape
            dst_num_pos, _ = model.visual.state_dict()[key].shape
            dst_patch_shape = model.visual.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                print(f"Position interpolate for {key} from {src_size}x{src_size} to {dst_size}x{dst_size}")
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                print("Original positions =", x)
                print("Target positions =", dx)

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].reshape((src_size, src_size)).astype('float32')
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        paddle.to_tensor(f(dx, dy)).reshape((-1, 1)).to(rel_pos_bias.place))

                rel_pos_bias = paddle.concat(all_rel_pos_bias, axis=-1)

                new_rel_pos_bias = paddle.concat((rel_pos_bias, extra_tokens), axis=0)
                state_dict[key] = new_rel_pos_bias

    # interpolate position embedding
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = F.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.transpose((0, 2, 3, 1)).reshape((-1, embedding_size))
            new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
            state_dict['pos_embed'] = new_pos_embed

            patch_embed_proj = state_dict['patch_embed.proj.weight']
            patch_size = model.visual.patch_embed.patch_size
            state_dict['patch_embed.proj.weight'] = F.interpolate(
                patch_embed_proj.astype('float32'), size=patch_size, mode='bicubic', align_corners=False)


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (paddle.nn.Layer): Any PaddlePaddle module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        paddle.nn.Layer: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.BatchNorm2D, nn.SyncBatchNorm)):
        res = FrozenBatchNorm2D(module.num_features)
        res.num_features = module.num_features
        res.weight.set_value(module.weight.numpy().copy())
        res.bias.set_value(module.bias.numpy().copy())
        res.running_mean.set_value(module._mean.numpy().copy())
        res.running_var.set_value(module._variance.numpy().copy())
        res.eps = module.eps
    else:
        for child_name, child in module.named_sublayers():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                setattr(res, child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


def is_logging(args):
    def is_global_master(args):
        return args.rank == 0

    def is_local_master(args):
        return args.local_rank == 0

    def is_master(args, local=False):
        return is_local_master(args) if local else is_global_master(args)
    return is_master


class AllGather(paddle.autograd.Function):
    """An autograd function that performs allgather on a tensor.
    Performs all_gather operation on the provided tensors.
    *** Warning ***: paddle.distributed.all_gather has no gradient.
    """

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        tensors_gather = [paddle.empty_like(tensor) for _ in range(world_size)]
        paddle.distributed.all_gather(tensors_gather, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return paddle.concat(tensors_gather, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply
