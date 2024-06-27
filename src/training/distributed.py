import os
import paddle
import paddle.distributed as dist

try:
    import horovod.paddle as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    args.distributed = False
    args.world_size = 1
    args.rank = 0
    args.local_rank = 0
    if args.horovod:
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        args.local_rank = int(hvd.local_rank())
        args.rank = hvd.rank()
        args.world_size = hvd.size()
        args.distributed = True
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif is_using_distributed():
        if 'SLURM_PROCID' in os.environ:
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            dist.init_parallel_env()
        else:
            args.local_rank, _, _ = world_info_from_env()
            dist.init_parallel_env()
            args.world_size = dist.get_world_size()
            args.rank = dist.get_rank()
        args.distributed = True

    if paddle.device.is_compiled_with_cuda() and paddle.device.get_device_count() > 0:
        if args.distributed and not args.no_set_device_rank:
            device = paddle.set_device(f'gpu:{args.local_rank}')
        else:
            device = paddle.set_device('gpu:0')
    else:
        device = paddle.set_device('cpu')
    args.device = device
    return device


def broadcast_object(args, obj, src=0):
    if args.horovod:
        return hvd.broadcast_object(obj, root_rank=src)
    else:
        if args.rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, root=src)
        return objects[0]


def all_gather_object(args, obj, dst=0):
    if args.horovod:
        return hvd.allgather_object(obj)
    else:
        objects = [None] * args.world_size
        dist.all_gather_object_list(objects, obj)
        return objects
