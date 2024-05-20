import time
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data

from opensora.dataset.bucket import Bucket
from opensora.dataset.bucket_configs import bucket_webvid_latent_v257x288x512


def check_bucket():
    bucket_config = bucket_webvid_latent_v257x288x512
    bucket = Bucket(bucket_config)

    num_frames = [1, 2, 4, 8, 16, 32, 33, 64]
    heights = [36] * len(num_frames)
    widths = [64] * len(num_frames)

    for i in range(len(num_frames)):
        t, h, w = num_frames[i], heights[i], widths[i]
        bucket_id = bucket.get_bucket_id(t, h, w, seed=int(time.time()))
        thw = bucket.get_thw(bucket_id)
        bs = bucket.get_batch_size(bucket_id)
        print(f"({t}, {h}, {w}): {bucket_id}, thw={thw}, bs={bs}")


def init_logger():
    import logging
    import datetime
    from pathlib import Path
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logger = get_logger(__name__)
    logging_dir = Path("out_webvidlatent_129x288x512", "logs")
    accelerator_project_config = ProjectConfiguration(project_dir="out_webvidlatent_129x288x512", logging_dir=str(logging_dir))
    accelerator = Accelerator(
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    project_name = "tmp_check"
    accelerator.init_trackers(
        project_name,
        init_kwargs={
            "wandb":
                {
                    "entity": None,
                    "name": f"({now})check_bucket",
                }
        },
    )
    return accelerator, logger


def check_sampler():
    from transformers import AutoTokenizer
    from opensora.dataset.webvid_datasets import WebVidHFWebDataset, WebVidLatentDataset
    from opensora.dataset.sampler import VariableVideoBatchSampler
    from check_env.check_datasets import print_batch

    accelerator, logger = init_logger()
    tokenizer = AutoTokenizer.from_pretrained(
        "DeepFloyd/t5-v1_1-xxl", cache_dir="/public/home/201810101923/models/opensora/v1.0.0")
    WEBVID_LATENT_META = "/public/home/201810101923/datasets/webvid/latents_v257x288x512/latents_meta_all.csv"
    WEBVID_LATENT_DIR = "/public/home/201810101923/datasets/webvid/latents_v257x288x512/latents"
    train_dataset = WebVidLatentDataset(
        WEBVID_LATENT_META, WEBVID_LATENT_DIR, logger=logger,
        tokenizer=tokenizer,
    )
    logger.info("[DEBUG] dataset got")

    rank = 0
    world_size = 32
    sampler = VariableVideoBatchSampler(
        dataset=train_dataset,
        bucket_config=bucket_webvid_latent_v257x288x512,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        drop_last=True,
        verbose=True,
        num_bucket_build_workers=6,
        logger=logger,
    )
    num_steps_per_epoch = sampler.get_num_batch() // world_size
    print("[DEBUG] num_steps_per_epoch:", num_steps_per_epoch)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=4,
        batch_sampler=sampler,  # NOT sampler=xxx,
        pin_memory=True,
    )

    for idx, batch in enumerate(tqdm(train_dataloader)):
        print_batch(batch)


if __name__ == "__main__":
    # check_bucket()

    check_sampler()
