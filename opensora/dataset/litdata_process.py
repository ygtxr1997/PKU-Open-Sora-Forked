import json
import os, io, csv, math, random
from typing import List, Dict
import cv2
import numpy as np
import argparse

import torchvision
from einops import rearrange
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from litdata import StreamingDataset

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


class LitDataDataset(StreamingDataset):
    def __init__(self,
                 dataset_meta: str,
                 dataset_dir: str,
                 tokenizer,
                 drop_last: bool = True,
                 max_cache_size: str = "10GB",
                 seed: int = 42,
                 llm_max_length: int = 300,
                 proportion_empty_prompts: float = 0.,
                 resume: int = None,
                 ):
        super().__init__(input_dir=f"local:{dataset_dir}",
                         drop_last=drop_last, seed=seed,
                         max_cache_size=max_cache_size)

        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir
        self.resume = resume

        self.meta_id_to_captions = {}
        if dataset_meta is not None:
            self.meta_id_to_captions = self.load_meta(dataset_meta)

        self.tokenizer = tokenizer
        self.llm_max_length = llm_max_length
        self.proportion_empty_prompts = proportion_empty_prompts

        self.mem_bad_indices = []
        self.fail_cnt = 0
        self.success_cnt = 0

        print(f"[LitDataDataset] loaded cnt={len(self)}")

    def load_meta(self, meta_path: str):
        print("[LitDataDataset] Reading meta file...")
        with open(meta_path, "r") as csv_f:
            csv.field_size_limit(9000000)
            reader = csv.reader(csv_f, delimiter=',')
            data = [[x[0], x[3]] for x in reader]  # 'videoID', 'url', 'timestamp', 'caption', 'matching_score'
        id_to_captions = {}
        cnt_id = 0
        cnt_captions = 0
        data = data[1:]  # skip 1st row
        for row in data:
            video_id = row[0]
            video_id = video_id.replace("-", "_")  # replacing all "-" with "_"
            caption_list_str = row[1]
            caption_list = caption_list_str.replace("\"", "\'")[2:-2].split("', '")
            id_to_captions[video_id] = caption_list
            cnt_id += 1
            cnt_captions += len(caption_list)
        print(f"[LitDataDataset] Meta file loaded, id_cnt={cnt_id}, caption_cnt={cnt_captions}")
        return id_to_captions

    def __getitem__(self, index):
        if self.resume is not None:
            if index.index < self.resume:
                return -1, -1, -1, -1
        all_data = super().__getitem__(index)  # "video_id", "clip_id", "video", "caption"
        return all_data["video_id"], all_data["clip_id"], all_data["video"], all_data["caption"]

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
                            f'fail={self.fail_cnt}, success={self.success_cnt}')
        return self.__getitem__(random.randint(0, self.__len__() - 1))


def print_batch(batch):
    def print_k_v(k, v):
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(f"({k}):{type(v)}, {v.shape}")
        elif isinstance(v, list) or isinstance(v, tuple):
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
    elif isinstance(batch, list):
        for i in range(len(batch)):
            print_k_v(i, batch[i])
    elif isinstance(batch, torch.Tensor):
        print("all", batch.shape)
    else:
        raise TypeError(f"Batch type not supported: {type(batch)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_folder", type=str, required=True)
    parser.add_argument("--input_meta_path", type=str, default=None)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--save_ext", type=str, default="mp4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=-1, help="-1 means all;")
    parser.add_argument("--resume", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    lit_dataset = LitDataDataset(
        args.input_meta_path, args.input_data_folder,
        tokenizer=None,
        drop_last=False,
        resume=None if args.resume is None else (args.resume - 1) * args.batch_size,
    )
    train_dataloader = torch.utils.data.DataLoader(
        lit_dataset,
        batch_size=args.batch_size,
        num_workers=12,
    )

    os.makedirs(args.save_folder, exist_ok=True)
    if args.max_len == -1:
        args.max_len = len(train_dataloader)
    for idx, batch in enumerate(tqdm(train_dataloader)):
        if args.resume is not None and idx <= args.resume:
            continue
        video_id, clip_id, video, caption = batch
        bs = len(video_id)
        for k in range(bs):
            if args.input_meta_path is not None:
                one_video_id = video_id[k].replace("-", "_")
                one_clip_id = clip_id[k]
                one_caption = lit_dataset.meta_id_to_captions[one_video_id][one_clip_id]
                print(one_video_id, one_clip_id, one_caption)
            save_fn = f"{video_id[k]}_{clip_id[k]:03d}.{args.save_ext}"
            with open(os.path.join(args.save_folder, save_fn), "wb") as f:
                f.write(video[k])
        if idx >= args.max_len:
            exit()


if __name__ == "__main__":
    """
    python opensora/dataset/litdata_process.py  \
      --input_data_folder /mnt/dongxu-fs1/data-ssd/mingyang/datasets/Panda-70M/litdata_0/  \
      --save_folder /home/geyuan/datasets/Panda-70M/clips_0/  \
      --max_len 100
    """
    opts = parse_args()
    main(opts)
