import os
import logging
from pathlib import Path
from tqdm import tqdm
import argparse

import numpy as np
import torch
from torch.utils import data
from datasets import load_dataset, load_from_disk, Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import AutoTokenizer

from opensora.dataset.t2v_datasets import T2V_dataset
from opensora.dataset import getdataset, ae_denorm, ae_norm
from opensora.train.train_t2v import parser_args


def download_dataset():
    target = "opensora"  # in ["opensora", "imdb"]
    download = True  # in [True, False]

    if target == "opensora":
        hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
        cache_dir = "/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest"
        save_dir = cache_dir + "_sorted"
    elif target == "imdb":
        hf_repo_name = "stanfordnlp/imdb"
        cache_dir = "/public/home/201810101923/datasets/opensora/imdb"
        save_dir = cache_dir + "_sorted"
    else:
        raise NotImplementedError(f"Target ({target}) not supported!")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if download:

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo_name,
            repo_type="dataset",
            # filename="mixkit.tar.gz",
            resume_download=True,
            max_workers=20,
            etag_timeout=500,
            cache_dir=cache_dir
        )
        target_dataset = load_dataset(
            path=hf_repo_name,
            cache_dir=cache_dir,
        )

        # 1. Download, error: 8,9,10,12,13,15,18
        # data_files = ["pixabay.tar.gz.%02d" % x for x in [18]]
        # data_files = ["pexels.tar.gz.%02d" % x for x in range(6)]
        # target_dataset = load_dataset(
        #     path=hf_repo_name,
        #     cache_dir=cache_dir,
        #     # num_proc=8,
        #     # token="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",
        #     # data_files=data_files,
        #     # download_mode="force_redownload",
        #     # verification_mode="all_checks"
        # )
        # print(f"[Check Datasets] Dataset loaded to cache={cache_dir}.")
        # exit()

        # 2. Reorder
        target_dataset.save_to_disk(
            dataset_dict_path=save_dir,
            max_shard_size="40GB"
        )
        print(f"[Check Datasets] Dataset saved to {save_dir}.")
    else:
        # 3. Check loading offline
        target_dataset = load_from_disk(
            dataset_path=save_dir
        )
        print(f"[Check Datasets] Dataset loaded from {save_dir}.")

    suffix = "online" if download else "offline"
    print(f"Dataset ({target}) loaded! mode={suffix}")


def print_batch(batch):
    def print_k_v(k, v):
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(f"({k}):{type(v)}, {v.shape}")
        elif isinstance(v, list):
            print(f"({k}):{type(v)}, len={len(v)}, [0]type:{type(v[0])}")
        elif isinstance(v, str) or isinstance(v, np.int64):
            print(f"({k}):{type(v)}, {v}")
        elif isinstance(v, bytes):
            print(f"({k}):{type(v)}, len={len(v)}")
        else:
            print(f"({k}):{type(v)}")

    if isinstance(batch, dict):
        for k, v in batch.items():
            print_k_v(k, v)
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for i in range(len(batch)):
            print_k_v(i, batch[i])
    elif isinstance(batch, torch.Tensor):
        print("all", batch.shape)
    else:
        raise TypeError(f"Batch type not supported: {type(batch)}")


def check_batch():
    args = parser_args()
    logger = get_logger(__name__)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # 1. Opensora Dataset
    # train_dataset = getdataset(args)

    # 2. InternVid Dataset
    # from opensora.dataset.internvid_datasets import InternVidDataset
    # from opensora.dataset.split_json import split_jsonl
    # INTERNVID_DIR = "/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-FLT-clip"
    # INTERNVID_META = "/exthome/future-technology-college-data/Internvid_dataset/InternVid-10M-flt-clips1.jsonl"
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.text_encoder_name, cache_dir=args.cache_dir)
    # train_dataset = InternVidDataset(
    #     INTERNVID_META, INTERNVID_DIR, logger=logger,
    #     tokenizer=tokenizer,
    #     norm_fun=ae_norm[args.ae],
    #     num_frames=32,
    #     max_frame_stride=args.sample_rate,
    # )

    # # 3. Panda70M Dataset
    # from opensora.dataset.panda70m_datasets import Panda70MPytorchDataset
    # PANDA70M_DIR = "/public/home/201810101923/datasets/panda70m/clips_0"
    # PANDA70M_META = "/public/home/201810101923/datasets/panda70m/panda70m_training_clips_0.csv"
    # tokenizer = AutoTokenizer.from_pretrained(
    #         args.text_encoder_name, cache_dir=args.cache_dir)
    # train_dataset = Panda70MPytorchDataset(
    #     PANDA70M_META, PANDA70M_DIR, logger=logger,
    #     tokenizer=tokenizer,
    #     norm_fun=ae_norm[args.ae],
    #     num_frames=32,
    #     max_frame_stride=args.sample_rate,
    # )
    # total_len = len(train_dataset)
    # print("len=", total_len)
    # print("[0]=")
    # print_batch(train_dataset[0])
    # print("[-1]=")
    # print_batch(train_dataset[total_len - 1])

    # 4. WebVid Dataset
    from opensora.dataset.webvid_datasets import WebVidHFWebDataset, WebVidLatentDataset
    WEBVID_DIR = "/exthome/future-technology-college-data/202321063560/webvid_data/webvid_train_data"
    # WEBVID_DIR = "/public/home/201810101923/datasets/webvid/data_demo"
    tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir)
    # train_dataset = WebVidHFWebDataset(
    #     WEBVID_DIR, logger=logger,
    #     tokenizer=tokenizer,
    #     norm_fun=ae_norm[args.ae],
    #     num_frames=129,
    #     target_size=(512, 288),
    #     max_frame_stride=args.sample_rate,
    # )
    WEBVID_META = "/public/home/201810101923/datasets/webvid/total.csv"
    WEBVID_LATENT_DIR = "/public/home/201810101923/datasets/webvid/latents"
    webvid_files = os.listdir(WEBVID_DIR)
    webvid_files = [os.path.join(WEBVID_DIR, f) for f in webvid_files if 'tar' in f]
    webvid_dataset = load_dataset(
        'webdataset', data_files=webvid_files, split='train', streaming=True)
    webvid_dataset = webvid_dataset.filter(lambda x: x["mp4"] is not None)
    webvid_dataset = webvid_dataset.map(lambda x: x, remove_columns=["mp4"])
    iterator = iter(webvid_dataset)
    for item in tqdm(iterator):
        pass
    exit()
    # train_dataset = WebVidLatentDataset(
    #     WEBVID_META, WEBVID_DIR, logger=logger,
    #     tokenizer=tokenizer,
    #     # norm_fun=ae_norm[args.ae],
    #     # num_frames=129,
    #     # target_size=(512, 288),
    #     # max_frame_stride=args.sample_rate,
    # )

    logger.info("[DEBUG] dataset got")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=4,
        drop_last=True,
    )
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)

    for idx, batch in enumerate(tqdm(train_dataloader)):
        print_batch(batch)


if __name__ == "__main__":
    """
    bash check_env/sh_check_all.sh
    """
    check_batch()
