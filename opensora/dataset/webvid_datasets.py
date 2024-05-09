import json
import os, io, csv, math, random
from typing import List, Dict, Iterator
import cv2
import numpy as np
import pandas as pd
import imageio as iio

import torchvision
from einops import rearrange
import decord
from decord import VideoReader
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from datasets import IterableDataset
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from litdata import StreamingDataset
from accelerate.logging import MultiProcessAdapter

from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo
from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


class WebVidHFWebDataset(torch.utils.data.IterableDataset):
    """ WebDataset format: https://huggingface.co/docs/hub/en/datasets-webdataset """
    def __init__(self,
                 dataset_dir: str,
                 tokenizer=None,
                 dataset_meta: str = None,
                 transform=None,
                 norm_fun=None,
                 logger: MultiProcessAdapter = None,
                 max_image_size: int = 512,
                 target_size: tuple = (512, 512),
                 num_frames: int = 16,
                 max_frame_stride: int = 8,
                 llm_max_length: int = 300,
                 proportion_empty_prompts: float = 0.,
                 rank: int = None,
                 world_size: int = None,
                 ):
        # super(WebVidHFWebDataset, self).__init__()
        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir
        self.logger = logger

        ''' load parquet files '''
        if self.logger is not None:
            self.logger.info(f"[WebVidHFWebDataset] loading webdataset from: {dataset_dir}")
        webvid_files = os.listdir(self.dataset_dir)
        webvid_files = [os.path.join(self.dataset_dir, f) for f in webvid_files if 'tar' in f]
        webvid_dataset: IterableDataset = load_dataset(
            'webdataset', data_files=webvid_files, split='train', streaming=True)
        if world_size is not None and rank is not None:
            webvid_dataset = split_dataset_by_node(
                webvid_dataset, rank=rank, world_size=world_size)
            if self.logger is not None:
                self.logger.info(f"[WebVidHFWebDataset] distributed split by rank: {rank}/{world_size}")

        def webvid_map(example):
            if example['txt'][-1] in ['\n', '.', ',', ' ']:
                example['txt'] = example['txt'][:-1]
            example["caption"] = example['txt'] + ", watermark"
            example["dataset"] = "webvid"
            return example  # "mp4", "caption", "dataset", "txt"

        webvid_columns = ["__key__", "__url__", "json", "txt"]
        webvid_dataset = webvid_dataset.map(
            webvid_map, remove_columns=webvid_columns[1:])  # "__key__", "mp4", "caption", "dataset"
        webvid_dataset = webvid_dataset.filter(lambda x: x["mp4"] != None)
        # webvid_dataset = webvid_dataset.map(
        #     self.iterate_map, remove_columns=["mp4", "video", "input_ids", "cond_mask", "caption", "dataset"]
        # )
        if self.logger is not None:
            logger.info(f"[WebVidHFWebDataset] webvid_dataset n_shards : {webvid_dataset.n_shards}")

        self.webvid_dataset = webvid_dataset

        ''' others '''
        self.tokenizer = tokenizer
        self.max_frame_stride = max_frame_stride
        self.target_size = target_size
        self.num_frames = num_frames
        self.max_frame_stride = max_frame_stride
        self.llm_max_length = llm_max_length
        self.proportion_empty_prompts = proportion_empty_prompts

        if transform is None:
            transform = transforms.Compose([
                ToTensorVideo(),
                CenterCropResizeVideo(max_image_size),
                RandomHorizontalFlipVideo(p=0.5),
                norm_fun
            ])
        self.transform = transform

        self.fail_cnt = 0
        self.success_cnt = 0

    # def __iter__(self):
    #     return iter(self.webvid_dataset)

    def __iter__(self):
        return self._sample_generator()

    def _sample_generator(self):
        webvid_iterator = iter(self.webvid_dataset)
        for idx, sample in enumerate(webvid_iterator):
            if idx == 0:
                print(f"[DEBUG] iterating {idx}: {sample['caption'][:20]}")
            # if idx >= 15:
            #     break
            try:
                mp4_data = sample["mp4"]
                caption = sample["caption"]
                video_id = int(sample["__key__"])

                video = self.decord_read(mp4_data)  # (T,C,H,W)
                video = self.transform(video)  # T C H W -> T C H W
                video = video.transpose(0, 1)  # T C H W -> C T H W
                assert (video.shape[1] == self.num_frames), f'{len(video.shape[1])} != video_length:{self.num_frames}'

                # Text
                if self.tokenizer is not None:
                    text = caption
                    text = text_preprocessing(text)
                    text_tokens_and_mask = self.tokenizer(
                        text,
                        max_length=self.llm_max_length,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        add_special_tokens=True,
                        return_tensors='pt'
                    )
                    input_ids = text_tokens_and_mask['input_ids'].squeeze(0)
                    cond_mask = text_tokens_and_mask['attention_mask'].squeeze(0)
                    self.success_cnt += 1
                    yield video_id, video, input_ids, cond_mask
                else:
                    self.success_cnt += 1
                    yield video_id, video
            except Exception as e:
                self.process_error(idx, sample, e)
                continue

    def process_error(self, index, sample, error=None):
        self.fail_cnt += 1
        self.logger.warning(f'Catch {error}, {index}: {sample["__key__"]} - {sample["caption"]}, get next item instead, '
                            f'fail={self.fail_cnt}, success={self.success_cnt}')

    def iterate_map(self, sample):
        mp4_data = sample["mp4"]
        caption = sample["caption"]

        video = self.decord_read(mp4_data)  # (T,C,H,W)
        video = self.transform(video)  # T C H W -> T C H W
        video = video.transpose(0, 1)  # T C H W -> C T H W
        assert (video.shape[1] == self.num_frames), f'{len(video.shape[1])} != video_length:{self.num_frames}'

        # Text
        text = caption
        text = text_preprocessing(text)
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.llm_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids'].squeeze(0)
        cond_mask = text_tokens_and_mask['attention_mask'].squeeze(0)

        self.success_cnt += 1
        sample["video"] = video
        sample["input_ids"] = input_ids
        sample["cond_mask"] = cond_mask
        sample["caption"] = caption
        sample["caption_len"] = torch.from_numpy(np.array([len(caption)]))
        return sample

    def decord_read(self, byte_data: bytes):
        decord_vr = VideoReader(io.BytesIO(byte_data), ctx=decord.cpu(0))
        total_frames = len(decord_vr)

        # If too short
        if total_frames < self.num_frames:
            raise Warning("[WebVidHFWebDataset][Warning] frames not enough!")

        # Sample frames by stride
        frame_stride = random.randint(1, self.max_frame_stride)
        all_frames = list(range(0, total_frames, frame_stride))
        if len(all_frames) < self.num_frames:  # if too sparse, reduce stride
            frame_stride = total_frames // self.num_frames
            assert (frame_stride > 0)
            all_frames = list(range(0, total_frames, frame_stride))

        # Select a random clip
        rand_idx = random.randint(0, len(all_frames) - self.num_frames)
        frame_indice = all_frames[rand_idx:rand_idx + self.num_frames]
        video_data = decord_vr.get_batch(frame_indice).asnumpy()  # (T,H,W,C)

        # If too small resolution
        if video_data.shape[1] < 288 and video_data.shape[2] < 512:
            raise Warning("[WebVidHFWebDataset][Warning] resolution too small!")

        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        return video_data


class WebVidLatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_meta: str,
                 dataset_dir: str,
                 tokenizer,
                 logger: MultiProcessAdapter = None,

                 ):
        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir
        self.logger = logger

        ''' load meta '''
        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] loading csv: {dataset_meta}")
        all_columns = ["videoid", "contentUrl", "duration", "page_dir", "name"]
        data = pd.read_csv(dataset_meta, encoding='utf-8')
        data = data[['videoid', 'name']]
        video_ids = data['videoid'].values.tolist()
        captions = data['name'].values.tolist()
        latent_fns = [f"{x}.npy" for x in video_ids]
        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] csv loaded data_len={len(video_ids)}")
        samples = []
        good, bad = 0, 0
        for i in tqdm(range(len(latent_fns)), desc="Check latent files"):
            fn = os.path.join(dataset_dir, latent_fns[i])
            if os.path.exists(fn):
                good += 1
            else:
                bad += 1
        self.samples: List[Dict] = samples
        print(f"good={good}, bad={bad}")

        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] loaded cnt={len(self.samples)}")

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.samples)
