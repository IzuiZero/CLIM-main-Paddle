import logging
import os
import time
import multiprocessing
import subprocess
from tqdm import tqdm
import paddle
import paddle.distributed.fleet as fleet
import paddle.distributed.fleet.base as base
import paddle.distributed.fleet.meta_optimizers as meta_optimizers

def remote_sync_s3(local_dir, remote_dir):
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, '--exclude', '*epoch_latest.pt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False
        
    logging.info(f"Successfully synced with S3 bucket")
    return True

def remote_sync_fsspec(local_dir, remote_dir):
    logging.warning("PaddlePaddle does not directly support fsspec. You may need to use another method for file synchronization.")
    return False

def remote_sync(local_dir, remote_dir, protocol):
    logging.info('Starting remote sync.')
    if protocol == 's3':
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == 'fsspec':
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error('Remote protocol not known')
        return False

def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)

def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(target=keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol))
    return p

def pt_save(pt_obj, file_path):
    logging.warning("PaddlePaddle does not directly support torch.save. You may need to use another method for saving checkpoints.")
    return False

def pt_load(file_path, map_location=None):
    logging.warning("PaddlePaddle does not directly support torch.load. You may need to use another method for loading checkpoints.")
    return None

def check_exists(file_path):
    logging.warning("PaddlePaddle does not directly support fsspec. You may need to use another method for file existence check.")
    return False
