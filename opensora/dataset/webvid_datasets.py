import json
import os, io, csv, math, random
from typing import List, Dict, Iterator
import multiprocessing
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
                 dataset_meta: str = None,  # not used
                 transform=None,
                 norm_fun=None,
                 logger: MultiProcessAdapter = None,
                 max_image_size: int = 512,
                 target_size: tuple = (512, 512),
                 num_frames: int = 16,
                 use_smaller_frames: bool = False,
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
        self.use_smaller_frames = use_smaller_frames
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
            if self.success_cnt == 0 and self.fail_cnt == 0 and os.environ.get("RANK") is not None:
                global_gpus = int(os.environ["WORLD_SIZE"])
                global_rank = int(os.environ["RANK"])
                print(f"[DEBUG] rank({global_rank}/{global_gpus}) "
                      f"iterating {idx}: {sample['caption'][:40]}")
            try:
                mp4_data = sample["mp4"]
                caption = sample["caption"]
                video_id = int(sample["__key__"])

                video, video_n_frames = self.decord_read(mp4_data)  # (T,C,H,W)
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
                    yield {
                        "video_id": video_id,
                        "video": video,
                        "video_n_frames": video_n_frames,
                        "text_ids": input_ids,
                        "cond_mask": cond_mask,
                    }
                else:
                    self.success_cnt += 1
                    yield {
                        "video_id": video_id,
                        "video": video,
                        "video_n_frames": video_n_frames,
                    }
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

        if not self.use_smaller_frames:  # default
            target_n_frames = self.num_frames
        else:
            target_n_frames = total_frames

        # If too short
        if total_frames < target_n_frames:
            raise Warning("[WebVidHFWebDataset][Warning] frames not enough!")

        # Sample frames by stride
        frame_stride = random.randint(1, self.max_frame_stride)
        all_frames = list(range(0, total_frames, frame_stride))
        if len(all_frames) < target_n_frames:  # if too sparse, reduce stride
            frame_stride = total_frames // target_n_frames
            assert (frame_stride > 0)
            all_frames = list(range(0, total_frames, frame_stride))

        # Select a random clip
        rand_idx = random.randint(0, len(all_frames) - target_n_frames)
        frame_indices = all_frames[rand_idx:rand_idx + target_n_frames]
        if total_frames < self.num_frames:  # If existing frames < needed max frames:
            frame_indices += [all_frames[-1]] * (self.num_frames - target_n_frames)  # repeat last frames as padding
        assert len(frame_indices) == self.num_frames, "[WebVidHFWebDataset] num_frames no consistent!"
        video_data = decord_vr.get_batch(frame_indices).asnumpy()  # (T,H,W,C)

        # If too small resolution
        if video_data.shape[1] < 288 and video_data.shape[2] < 512:
            raise Warning("[WebVidHFWebDataset][Warning] resolution too small!")

        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        video_n_frames = torch.LongTensor([target_n_frames])
        return video_data, video_n_frames


class WebVidLatentDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_meta: str,
                 dataset_dir: str,
                 tokenizer,
                 logger: MultiProcessAdapter = None,
                 llm_max_length: int = 300,
                 rank: int = None,
                 world_size: int = None,
                 ):
        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.rank = rank
        self.world_size = world_size

        ''' load meta '''
        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] loading csv: {dataset_meta}")
        all_columns = ["video_id", "caption"]
        data = pd.read_csv(dataset_meta, encoding='utf-8')
        data = data[['video_id', 'caption']]
        video_ids = data['video_id'].values.tolist()
        captions = data['caption'].values.tolist()
        video_id_to_caption = {}
        for i in range(len(video_ids)):
            video_id_to_caption[int(video_ids[i])] = captions[i]
        latent_fns = os.listdir(dataset_dir)
        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] csv loaded data_len={len(video_ids)}")
        samples = []
        good, bad = 0, 0
        video_ids_sorted = set([int(x) for x in video_ids])
        for i in tqdm(range(len(latent_fns)), desc="Check latent files"):
            fn = latent_fns[i]
            latent_fn_video_id = int(os.path.splitext(fn)[0])
            if latent_fn_video_id in video_ids_sorted:
                good += 1
                samples.append(
                    {
                        "video_id": int(latent_fn_video_id),
                        "caption": str(video_id_to_caption[int(latent_fn_video_id)]),
                        "latent_fn": fn,
                    }
                )
            else:
                bad += 1
        self.samples: List[Dict] = self._split_samples_by_rank(samples)

        self.tokenizer = tokenizer
        self.llm_max_length = llm_max_length

        self.mem_bad_indices = []
        self.fail_cnt = 0
        self.success_cnt = 0

        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] loaded cnt={len(self.samples)}, "
                             f"meta missed cnt={bad}, meta exist cnt={good}.")

    def _split_samples_by_rank(self, samples):
        if self.rank is None or self.world_size is None:
            return samples
        rank, world_size = self.rank, self.world_size
        all_indices = np.arange(len(samples))
        rank_indices = np.array_split(all_indices, world_size)[rank]
        if self.logger is not None:
            self.logger.info(f"[WebVidLatentDataset] split by rank=({self.rank}/{self.world_size}), "
                             f"range:{rank_indices[0]}-{rank_indices[-1]}", main_process_only=False)
        return [samples[i] for i in rank_indices]

    def __getitem__(self, index):
        if index in self.mem_bad_indices:
            return self.process_error(index, f"Skip bad index={index}")
        try:
            example = self.samples[index]
            latent_fn = example["latent_fn"]
            caption = example['caption']
            latent_path = os.path.join(self.dataset_dir, latent_fn)

            # Read latent from path
            latent = np.load(latent_path)  # (B,C,T,H,W)

            # Text
            text = f"{caption}, random flipped watermark: SHUTTERSTOCK"
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
            return latent, input_ids, cond_mask
        except Exception as e:
            return self.process_error(index, e)

    def process_error(self, index, error=None):
        self.fail_cnt += 1
        self.logger.warning(f'Catch {error}, {self.samples[index]}, get random item once again, '
                            f'fail={self.fail_cnt}, success={self.success_cnt}')
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.samples)


def merge_csv(file_list: list, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = None
    data_frames = []
    for idx, csv_file in enumerate(tqdm(file_list, desc="Checking headers")):
        data = pd.read_csv(csv_file, encoding='utf-8')
        if idx == 0:
            header = data.columns.tolist()
        else:
            assert header == data.columns.tolist(), f"{header} != {data.columns.tolist()}"
        data_frames.append(data)
    print(f"[merge_csv] read csv from: {file_list}, done.")
    df_merged = pd.concat(data_frames, ignore_index=True)
    df_merged.to_csv(out_path, index=False)
    print(f"[merge_csv] merged csv saved to: {out_path}.")

    data = pd.read_csv(out_path, encoding='utf-8')
    data = data[['video_id', 'caption']]
    video_ids = data['video_id'].values.tolist()
    captions = data['caption'].values.tolist()
    assert len(video_ids) == len(captions)
    print(f"[merge_csv] saved file checked: ok, len(w/o header)={len(video_ids)}")


def worker_extract_meta(rank, world_size):
    WEBVID_DIR = "/exthome/future-technology-college-data/202321063560/webvid_data/webvid_train_data"
    WEBVID_LATENT_META_FN = "/public/home/201810101923/datasets/webvid/latents_meta"
    webvid_files = os.listdir(WEBVID_DIR)
    webvid_files = [os.path.join(WEBVID_DIR, f) for f in webvid_files if 'tar' in f]
    webvid_dataset = load_dataset(
        'webdataset', data_files=webvid_files, split='train', streaming=True)
    webvid_dataset = split_dataset_by_node(webvid_dataset, rank, world_size)
    print(f"[worker_extract_meta:{rank}] split {rank}/{world_size}")
    webvid_dataset = webvid_dataset.filter(lambda x: x["mp4"] is not None)
    # webvid_dataset = webvid_dataset.map(lambda x: x, remove_columns=["mp4", "__url__", "json"])
    iterator = iter(webvid_dataset)
    meta = {"video_id": [], "caption": []}
    for idx, sample in tqdm(enumerate(iterator), disable=(rank != 0)):
        video_id = sample["__key__"]
        caption = sample["txt"]
        meta["video_id"].append(video_id)
        meta["caption"].append(caption)
    df = pd.DataFrame(meta)
    save_fn = f"{WEBVID_LATENT_META_FN}_{rank:04d}.csv"
    df.to_csv(save_fn, index=False)
    print(f"[worker_extract_meta:{rank}] finished. meta saved to: {save_fn}")


def multi_worker_start(worker_func, worker_cnt: int = 8):
    processes = []
    iterable_args = [(i, worker_cnt) for i in range(worker_cnt)]
    for i in range(worker_cnt):
        process = multiprocessing.Process(
            target=worker_func,
            args=iterable_args[i],
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    print("All processes are done.")
