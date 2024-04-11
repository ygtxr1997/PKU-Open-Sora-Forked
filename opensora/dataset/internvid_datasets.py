import json
import os, io, csv, math, random
from typing import List, Dict
import numpy as np

import torchvision
from einops import rearrange
from decord import VideoReader
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
from accelerate.logging import MultiProcessAdapter

from .split_json import split_jsonl
from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo
from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


class InternVidDataset(Dataset):
    def __init__(self, dataset_meta: str,
                 dataset_dir: str,
                 tokenizer,
                 transform=None,
                 norm_fun=None,
                 split_json_save_dir: str = "train_datasets/internvid",
                 logger: MultiProcessAdapter = None,
                 force_split: bool = False,
                 max_image_size: int = 512,
                 target_size: tuple = (512, 512),
                 num_frames: int = 16,
                 max_frame_stride: int = 8,
                 llm_max_length: int = 300,
                 proportion_empty_prompts: float = 0.,
                 ):
        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir
        self.split_json_save_dir = split_json_save_dir
        self.logger = logger
        self.force_split = force_split

        # Need to split?
        # if not os.path.exists(split_json_save_dir) or force_split:
        #     split_jsonl(self.dataset_meta, split_json_save_dir)
        useless_columns = [
            "YoutubeID", "Start_timestamp", "End_timestamp", "Caption",
            "UMT_Score", "Aesthetic_Score"]  # "video_path" is kept
        # with open(dataset_meta, 'r') as f:
        #     samples = json.load(f)
        samples = []
        with open(dataset_meta, 'r') as file:
            for line in file:
                samples.append(json.loads(line.strip()))
        for i in range(len(samples)):
            samples[i]["caption"] = samples[i]["Caption"]
            samples[i]["video_path"] = os.path.join(self.dataset_dir, samples[i]['video_path'])
            for k in useless_columns:
                samples[i].pop(k)
        self.samples: List[Dict] = samples

        # metas = [os.path.join(self.split_json_save_dir, f)
        #          for f in os.listdir(split_json_save_dir) if f.endswith('.jsonl')]
        # dataset = load_dataset(
        #     'json', data_files=metas, split='train', streaming=True)
        # dataset = dataset.map(self.dataset_map, remove_columns=columns)

        # dataset = dataset.filter(lambda x: x["video_data"] != None)
        # self.len = len(dataset)
        # self.dataset = dataset.with_format("torch")
        # assert isinstance(self.dataset, torch.utils.data.IterableDataset)

        self.tokenizer = tokenizer
        self.max_frame_stride = max_frame_stride
        self.target_size = target_size
        self.num_frames = num_frames
        self.max_frame_stride = max_frame_stride
        self.llm_max_length = llm_max_length
        self.proportion_empty_prompts = proportion_empty_prompts

        self.v_decoder = DecordInit()
        if transform is None:
            transform = transforms.Compose([
                ToTensorVideo(),
                CenterCropResizeVideo(max_image_size),
                RandomHorizontalFlipVideo(p=0.5),
                norm_fun
            ])
        self.transform = transform

        self.mem_bad_indices = []
        self.fail_cnt = 0
        self.success_cnt = 0

        if self.logger is not None:
            self.logger.info(f"[InterVidDataset] loaded cnt={len(self.samples)}")

    def dataset_map(self, example):
        example["caption"] = example["Caption"]
        example["video_data"] = os.path.join(self.dataset_dir, example['video_path'])
        example["video_data"] = example["video_data"] if os.path.exists(example["video_data"]) else None
        example["dataset"] = "internvid"
        return example

    def __getitem__(self, index):
        if index in self.mem_bad_indices:
            return self.process_error(index, f"Skip bad index={index}")
        try:
            example = self.samples[index]
            video_path = example["video_path"]
            caption = example['caption']

            # Read video from path
            video = self.decord_read(video_path)  # (T,C,H,W)
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
            return video, input_ids, cond_mask

            fps_ori = video_reader.get_avg_fps()
            fps_clip = fps_ori // frame_stride

            if fps_max is not None and fps_clip > fps_max:
                fps_clip = fps_max
            if fps_min is not None and fps_clip < fps_min:
                fps_clip = fps_min

            fps = torch.from_numpy(np.array([float(fps_clip)]))

            data = {
                'video_data': frames,
                'repeat_times': torch.from_numpy(np.array([1])),
                'motion_free_mask': motion_free_mask,
                'fps': fps,
                **additional_info
            }

            return data
        except Exception as e:
            return self.process_error(index, e)

    def process_error(self, index, error=None):
        self.fail_cnt += 1
        self.logger.warning(f'Catch {error}, {self.samples[index]}, get random item once again, '
                            f'fail_rate={(self.fail_cnt / (self.success_cnt + self.fail_cnt) * 100):.2f}%')
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def decord_read(self, path):
        if not os.path.exists(path):
            raise Warning("[InterVidDataset][Warning] file not existing!")

        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)

        # If too short
        if total_frames < self.num_frames:
            raise Warning("[InterVidDataset][Warning] frames not enough!")

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
        if video_data.shape[1] < 720 and video_data.shape[2] < 1280:  # TODO: is it ok?
            raise Warning("[InterVidDataset][Warning] resolution too small!")

        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        return video_data

    def __len__(self):
        return len(self.samples)
