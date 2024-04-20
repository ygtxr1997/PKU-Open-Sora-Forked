import json
import os
import io
import csv
import math
import random
import argparse

import numpy as np
import torchvision
from einops import rearrange
from tqdm import tqdm

from accelerate import Accelerator
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from InstructorEmbedding import INSTRUCTOR


class Panda70MTextDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 max_length: int = -1,
                 instruct_text: str = "Represent the video caption for clustering: ",
                 ):
        self.data_path = data_path
        self.instruct_text = instruct_text

        csv.field_size_limit(9000000)
        with open(data_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            data = [x for x in reader]
        captions = []
        for i in range(1, len(data)):
            line: str = data[i][3]  # 3rd column is 'caption'
            line = line.replace("\"", "\'")
            captions.extend(line[2:-2].split("', '"))
        self.texts = captions

        if max_length == -1:
            max_length = len(self.texts)
        self.length = min(max_length, len(self.texts))
        print(f"[Panda70MTextDataset] Loaded from {data_path}, total_len={len(self.texts)}, "
              f"max_len={self.length}.")

    def __getitem__(self, index):
        sample = [
            self.instruct_text,
            self.texts[index],
        ]
        return {
            "sample": sample,
            "index": index,
        }

    def __len__(self):
        return self.length


def panda70m_collate_fn(examples):
    texts = [example["sample"] for example in examples]
    indices = [example["index"] for example in examples]
    return {
        "text": texts,
        "index": indices,
    }


def main(opts):
    """ """
    batch_size = 1024
    max_length = 100000  # 10w:586MB

    accelerator = Accelerator()
    local_rank = accelerator.process_index

    ''' load data '''
    dataset = Panda70MTextDataset(
        data_path="/mnt/dongxu-fs1/data-hdd/mingyang/datasets/Panda-70M/panda70m_training_full.csv",
        max_length=max_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=panda70m_collate_fn,
    )
    data_loader = accelerator.prepare_data_loader(data_loader)

    ''' load model '''
    model = INSTRUCTOR(
        'hkunlp/instructor-large',
        cache_folder='/home/geyuan/pretrained/instructor'
    )
    model = model.to(accelerator.device)
    model = model.eval()

    ''' encode '''
    cache_embeddings = np.zeros((max_length, 768 + 1))
    for idx, batch in enumerate(tqdm(data_loader, desc=f"Total", position=1)):
        sentences = batch["text"]
        indices = batch["index"]
        embeddings = model.encode(
            sentences,
            show_progress_bar=False,
        )
        b, c = embeddings.shape
        assert indices.shape == (b,)
        cache_embeddings[batch_size * idx: batch_size * idx + b] = np.concatenate(
            [embeddings, indices[:, np.newaxis]], axis=1
        )

    np.save(f"tmp_embeddings_rank{local_rank}.npy", cache_embeddings)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
        --multi_gpu \
        --machine_rank 0 \
        --main_process_ip 127.0.0.1  \
        --main_process_port 7233  \
        --num_machines 1 \
        --num_processes 4 \
        text_clustering.py
    """
    opts = parse_args()
    main(opts)
