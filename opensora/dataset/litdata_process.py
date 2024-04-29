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
                 ):
        super().__init__(input_dir=f"local:{dataset_dir}",
                         drop_last=drop_last, seed=seed,
                         max_cache_size=max_cache_size)

        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir

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


def task_unzip_litdata(args):
    lit_dataset = LitDataDataset(
        args.input_meta_path, args.input_data_folder,
        tokenizer=None,
        drop_last=False,
    )

    train_dataloader = torch.utils.data.DataLoader(
        lit_dataset,
        batch_size=args.batch_size,
        num_workers=12,
    )

    os.makedirs(args.save_folder, exist_ok=True)
    if args.max_len == -1:
        args.max_len = len(train_dataloader)
    if args.resume is not None:
        for idx in tqdm(range(args.resume), desc="skipping..."):
            next(iter(train_dataloader))
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


def task_gen_txt(args):
    root = args.input_data_folder
    save_fn = f"{os.path.basename(os.path.abspath(root))}_list.txt"
    print(f"[TaskGenTXT] Save filename: {save_fn}")

    print(f"[TaskGenTXT] Reading directory: {root}")
    file_names = os.listdir(root)
    file_names.sort()
    if args.max_len == -1:
        args.max_len = len(file_names)
    file_names = file_names[:args.max_len]

    print(f"[TaskGenTXT] List will be saved to: {args.save_folder}, len={len(file_names)}")
    os.makedirs(args.save_folder, exist_ok=True)
    with open(os.path.join(args.save_folder, save_fn), "w") as f:
        file_lines = [f"{fn}\n" for fn in file_names]
        f.writelines(file_lines)

    print("[TaskGenTXT] Prepare to filter csv.")
    video_ids = []
    for fn in file_names:
        name, ext = os.path.splitext(fn)
        if len(name) == 15:
            video_id = str(name[:11])
            clip_id = int(name[12:15])
        else:
            print(f"[Warning] {fn}'s name={name} longer than 15!")
            assert len(name) > 15
            i = len(name) - 1
            while '0' <= name[i] <= '9':
                i -= 1
            assert name[i] == "_"
            video_id = str(name[:i])
            clip_id = int(name[i + 1: len(name)])
        video_ids.append(video_id)
    task_filter_csv(
        args.input_meta_path,
        save_folder=args.save_folder,
        save_fn=f"panda70m_training_{os.path.basename(os.path.abspath(root))}.csv",
        good_keys=video_ids,
    )

    print("[TaskGenTXT] Finished!")


def task_filter_csv(csv_path: str,
                    save_folder: str,
                    save_fn: str,
                    max_len: int = -1,
                    good_keys: list = None
                    ):
    csv.field_size_limit(9000000)
    print(f"[TaskFilterCSV] Reading csv file: {csv_path}")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = [x for x in reader]

    if max_len == -1:
        max_len = len(data)
    good_data = [data[0]]  # save 1st row, ['videoID', 'url', 'timestamp', 'caption', 'matching_score']
    print("[TaskFilterCSV] Total count (w/o 1st row):", len(data) - 1)
    print("[TaskFilterCSV] First row:", data[0])
    print("[TaskFilterCSV] Second row:", data[1])

    def filter_csv_row(csv_row: list, index: int = 0, filter_keys: set = None) -> bool:
        val = csv_row[index]
        val = val
        if val in filter_keys:
            filter_keys.remove(val)
            return True
        return False

    good_keys = set(good_keys)
    print(f"[TaskFilterCSV] Prepare to filter good data: len={len(good_keys)}")
    for i in tqdm(range(1, len(data)), desc="TaskFilterCSV Walk CSV File"):  # skip 1st row
        if len(good_data) >= max_len:
            break
        row = data[i]
        video_id = row[0]
        if filter_csv_row(row, filter_keys=good_keys):
            good_data.append(row)
        else:
            pass
    print(f"[TaskFilterCSV] Filter good data count: {len(good_data)}")

    def save_csv_and_check(save_path: str):
        with open(save_path, "w") as fid:
            writer = csv.writer(fid)
            writer.writerows(good_data)
        print(f"[TaskFilterCSV] Data saved to csv: {save_path}")
        with open(save_path, "r") as fid:
            tmp_reader = csv.reader(fid, delimiter=',')
            tmp_data = [x for x in tmp_reader]
            print(f"[TaskFilterCSV] [Check saved csv]: {save_path}")
            print("row[0]:", tmp_data[0])
            print("row[1]:", tmp_data[1])
            print("row[-1]:", tmp_data[-1])
            print(f"len={len(tmp_data)}")

    save_csv_and_check(os.path.join(save_folder, save_fn))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="unzip_lit", choices=["unzip_lit", "gen_txt"])
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
    if args.mode == "unzip_lit":
        task_unzip_litdata(args)
    elif args.mode == "gen_txt":
        task_gen_txt(args)
    else:
        raise NotImplementedError(f"Mode {args.mode} not supported!")


if __name__ == "__main__":
    """
    [Task 1 - Unzip LitData]
    python opensora/dataset/litdata_process.py  \
      -m unzip_lit  \
      --input_data_folder /mnt/dongxu-fs1/data-ssd/mingyang/datasets/Panda-70M/litdata_0/  \
      --save_folder /home/geyuan/datasets/Panda-70M/clips_0/  \
      --max_len 100
    [Task 2 - Generate txt list]
    python opensora/dataset/litdata_process.py  \
      -m gen_txt  \
      --input_data_folder /public/home/201810101923/datasets/panda70m/clips_0/  \
      --input_meta_path /public/home/201810101923/datasets/panda70m/panda70m_training_full.csv  \
      --save_folder /public/home/201810101923/datasets/panda70m/  \
      --max_len 100
    """
    opts = parse_args()
    main(opts)
