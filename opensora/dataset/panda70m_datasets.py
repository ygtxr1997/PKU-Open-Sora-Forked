import json
import os, io, csv, math, random
from typing import List, Dict
import cv2
import numpy as np
import imageio as iio

import torchvision
from einops import rearrange
import decord
from decord import VideoReader
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from datasets import load_dataset
from litdata import StreamingDataset
from accelerate.logging import MultiProcessAdapter

from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo
from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing


class Panda70MPytorchDataset(Dataset):
    def __init__(self, dataset_meta: str,
                 dataset_dir: str,
                 tokenizer,
                 transform=None,
                 norm_fun=None,
                 logger: MultiProcessAdapter = None,
                 max_image_size: int = 512,
                 target_size: tuple = (512, 512),
                 num_frames: int = 16,
                 max_frame_stride: int = 8,
                 llm_max_length: int = 300,
                 proportion_empty_prompts: float = 0.,
                 ):
        self.dataset_meta = dataset_meta
        self.dataset_dir = dataset_dir
        self.logger = logger

        ''' load meta '''
        if self.logger is not None:
            self.logger.info(f"[Panda70MPytorchDataset] loading csv: {dataset_meta}")
        useless_columns = [
            "url", "timestamp",
        ]  # all: ['videoID', 'url', 'timestamp', 'caption', 'matching_score']
        with open(dataset_meta, 'r') as fid:
            reader = csv.reader(fid, delimiter=',')
            data = [x for x in reader]
        samples = []
        for i in range(1, len(data)):  # skip 1st row
            row = data[i]
            video_id: str = row[0]
            captions_str = row[3]
            captions: list = captions_str.replace("\"", "\'")[2:-2].split("', '")
            for clip_id, caption in enumerate(captions):
                samples.append(
                    {
                        "video_id": str(video_id),
                        "caption": str(caption),
                        "clip_id": int(clip_id),
                        "video_fn": f"{video_id}_{clip_id:03d}.mp4",
                    }
                )
        self.samples: List[Dict] = samples

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
            self.logger.info(f"[Panda70MPytorchDataset] loaded cnt={len(self.samples)}")

    def __getitem__(self, index):
        if self.success_cnt == 0 and self.fail_cnt == 0:
            global_gpus = int(os.environ["WORLD_SIZE"])
            global_rank = int(os.environ["RANK"])
            print(f"[DEBUG] rank({global_rank}/{global_gpus}) iterating {index}: {self.samples[index]['caption'][:40]}")
        if index in self.mem_bad_indices:
            return self.process_error(index, f"Skip bad index={index}")
        try:
            example = self.samples[index]
            video_fn = example["video_fn"]
            caption = example['caption']
            video_path = os.path.join(self.dataset_dir, video_fn)

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
            return os.path.splitext(video_fn)[0], video, input_ids, cond_mask

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

    def decord_read(self, path):
        if not os.path.exists(path):
            raise Warning("[Panda70MPytorchDataset][Warning] file not existing!")

        decord_vr = self.v_decoder(path)
        total_frames = len(decord_vr)

        # If too short
        if total_frames < self.num_frames:
            raise Warning("[Panda70MPytorchDataset][Warning] frames not enough!")

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
        if video_data.shape[1] < 720 and video_data.shape[2] < 1280:
            raise Warning("[Panda70MPytorchDataset][Warning] resolution too small!")

        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
        return video_data

    def __len__(self):
        return len(self.samples)


# class Panda70MStreamingDataset(StreamingDataset):
#     def __init__(self,
#                  dataset_meta: str,
#                  dataset_dir: str,
#                  tokenizer,
#                  drop_last: bool = True,
#                  max_cache_size: str = "10GB",
#                  seed: int = 42,
#                  transform=None,
#                  norm_fun=None,
#                  logger: MultiProcessAdapter = None,
#                  max_image_size: int = 512,
#                  target_size: tuple = (512, 512),
#                  num_frames: int = 16,
#                  max_frame_stride: int = 8,
#                  llm_max_length: int = 300,
#                  proportion_empty_prompts: float = 0.,
#                  ):
#         super().__init__(input_dir=f"local:{dataset_dir}",
#                          drop_last=drop_last, seed=seed,
#                          max_cache_size=max_cache_size)
# 
#         self.dataset_meta = dataset_meta
#         self.dataset_dir = dataset_dir
#         self.logger = logger
# 
#         # useless_columns = [
#         #     "YoutubeID", "Start_timestamp", "End_timestamp", "Caption",
#         #     "UMT_Score", "Aesthetic_Score"]  # "video_path" is kept
#         # # with open(dataset_meta, 'r') as f:
#         # #     samples = json.load(f)
#         # samples = []
#         # with open(dataset_meta, 'r') as file:
#         #     for line in file:
#         #         samples.append(json.loads(line.strip()))
#         # for i in range(len(samples)):
#         #     samples[i]["caption"] = samples[i]["Caption"]
#         #     samples[i]["video_path"] = os.path.join(self.dataset_dir, samples[i]['video_path'])
#         #     for k in useless_columns:
#         #         samples[i].pop(k)
#         # self.samples: List[Dict] = samples
# 
#         # self.meta_id_to_captions: Dict[str, List[str]] = self.load_meta(self.dataset_meta)
# 
#         self.tokenizer = tokenizer
#         self.max_frame_stride = max_frame_stride
#         self.target_size = target_size
#         self.num_frames = num_frames
#         self.max_frame_stride = max_frame_stride
#         self.llm_max_length = llm_max_length
#         self.proportion_empty_prompts = proportion_empty_prompts
# 
#         if transform is None:
#             transform = transforms.Compose([
#                 ToTensorVideo(),
#                 CenterCropResizeVideo(max_image_size),
#                 RandomHorizontalFlipVideo(p=0.5),
#                 norm_fun
#             ])
#         self.transform = transform
# 
#         self.mem_bad_indices = []
#         self.fail_cnt = 0
#         self.success_cnt = 0
# 
#         if self.logger is not None:
#             self.logger.info(f"[Panda70MPytorchDataset] loaded cnt={len(self.samples)}")
# 
#     def dataset_map(self, example):
#         example["caption"] = example["Caption"]
#         example["video_data"] = os.path.join(self.dataset_dir, example['video_path'])
#         example["video_data"] = example["video_data"] if os.path.exists(example["video_data"]) else None
#         example["dataset"] = "internvid"
#         return example
# 
#     def load_meta(self, meta_path: str):
#         print("[Panda70M] Reading meta file...")
#         with open(meta_path, "r") as csv_f:
#             csv.field_size_limit(9000000)
#             reader = csv.reader(csv_f, delimiter=',')
#             data = [[x[0], x[3]] for x in reader]  # 'videoID', 'url', 'timestamp', 'caption', 'matching_score'
#         id_to_captions = {}
#         cnt_id = 0
#         cnt_captions = 0
#         data = data[1:]  # skip 1st row
#         for row in data:
#             video_id = row[0]
#             video_id = video_id.replace("-", "_")  # replacing all "-" with "_"
#             caption_list_str = row[1]
#             caption_list = caption_list_str.replace("\"", "\'")[2:-2].split("', '")
#             id_to_captions[video_id] = caption_list
#             cnt_id += 1
#             cnt_captions += len(caption_list)
#         print(f"[Panda70M] Meta file loaded, id_cnt={cnt_id}, caption_cnt={cnt_captions}")
#         return id_to_captions
# 
#     def __getitem__(self, index):
#         all_data = super().__getitem__(index)  # "video_id", "clip_id", "video", "caption"
#         video_data = iio.v2.imread(all_data["video"])
#         video_id = all_data["video_id"].replace("-", "_")
#         clip_id = all_data["clip_id"]
#         # caption = self.meta_id_to_captions[video_id][clip_id]
# 
#         # all_data["video"] = video_data
#         # all_data["caption"] = caption
#         # print(video_data.shape)
#         return all_data
# 
#         if index in self.mem_bad_indices:
#             return self.process_error(index, f"Skip bad index={index}")
#         try:
#             example = self.samples[index]
#             video_path = example["video_path"]
#             caption = example['caption']
# 
#             # Read video from path
#             video = self.decord_read(video_path)  # (T,C,H,W)
#             video = self.transform(video)  # T C H W -> T C H W
#             video = video.transpose(0, 1)  # T C H W -> C T H W
#             assert (video.shape[1] == self.num_frames), f'{len(video.shape[1])} != video_length:{self.num_frames}'
# 
#             # Text
#             text = caption
#             text = text_preprocessing(text)
#             text_tokens_and_mask = self.tokenizer(
#                 text,
#                 max_length=self.llm_max_length,
#                 padding='max_length',
#                 truncation=True,
#                 return_attention_mask=True,
#                 add_special_tokens=True,
#                 return_tensors='pt'
#             )
#             input_ids = text_tokens_and_mask['input_ids'].squeeze(0)
#             cond_mask = text_tokens_and_mask['attention_mask'].squeeze(0)
# 
#             self.success_cnt += 1
#             return video, input_ids, cond_mask
# 
#             fps_ori = video_reader.get_avg_fps()
#             fps_clip = fps_ori // frame_stride
# 
#             if fps_max is not None and fps_clip > fps_max:
#                 fps_clip = fps_max
#             if fps_min is not None and fps_clip < fps_min:
#                 fps_clip = fps_min
# 
#             fps = torch.from_numpy(np.array([float(fps_clip)]))
# 
#             data = {
#                 'video_data': frames,
#                 'repeat_times': torch.from_numpy(np.array([1])),
#                 'motion_free_mask': motion_free_mask,
#                 'fps': fps,
#                 **additional_info
#             }
# 
#             return data
#         except Exception as e:
#             return self.process_error(index, e)
# 
#     def process_error(self, index, error=None):
#         self.fail_cnt += 1
#         self.logger.warning(f'Catch {error}, {self.samples[index]}, get random item once again, '
#                             f'fail={self.fail_cnt}, success={self.success_cnt}')
#         return self.__getitem__(random.randint(0, self.__len__() - 1))
# 
#     def decord_read(self, path):
#         if not os.path.exists(path):
#             raise Warning("[Panda70MPytorchDataset][Warning] file not existing!")
# 
#         decord_vr = self.v_decoder(path)
#         total_frames = len(decord_vr)
# 
#         # If too short
#         if total_frames < self.num_frames:
#             raise Warning("[Panda70MPytorchDataset][Warning] frames not enough!")
# 
#         # Sample frames by stride
#         frame_stride = random.randint(1, self.max_frame_stride)
#         all_frames = list(range(0, total_frames, frame_stride))
#         if len(all_frames) < self.num_frames:  # if too sparse, reduce stride
#             frame_stride = total_frames // self.num_frames
#             assert (frame_stride > 0)
#             all_frames = list(range(0, total_frames, frame_stride))
# 
#         # Select a random clip
#         rand_idx = random.randint(0, len(all_frames) - self.num_frames)
#         frame_indice = all_frames[rand_idx:rand_idx + self.num_frames]
#         video_data = decord_vr.get_batch(frame_indice).asnumpy()  # (T,H,W,C)
# 
#         # If too small resolution
#         if video_data.shape[1] < 720 and video_data.shape[2] < 1280:
#             raise Warning("[Panda70MPytorchDataset][Warning] resolution too small!")
# 
#         video_data = torch.from_numpy(video_data)
#         video_data = video_data.permute(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
#         return video_data
