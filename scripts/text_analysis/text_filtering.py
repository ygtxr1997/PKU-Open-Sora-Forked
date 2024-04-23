import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import json
import csv
import sklearn.cluster
from tqdm import tqdm


def filter_csv_row(row: list) -> bool:
    caption_list_str: str = row[3]  # 3rd column is 'caption'
    caption_list_str = caption_list_str.replace("\"", "\'")
    caption_list = caption_list_str[2:-2].split("', '")

    found_flag = False
    animal_keys = [
        "cat", "dog", "horse", "lion", "pig", "wolf",
        "elephant", "bear", "bird", "monkey", "rabbit",
        "fish", "tiger", "dolphin", "giraffe", "penguin",
        "puppy",
    ]
    for animal in animal_keys:
        for i in range(len(caption_list)):
            if f" {animal} " in caption_list[i] or f" {animal}s " in caption_list[i]:
                found_flag = True
                return True
    return False


def main(args):
    """ """
    ''' get data '''
    # # 1. InternVid
    # data_path = "/mnt/dongxu-fs1/data-hdd/mingyang/datasets/InternVid-10M-FLT-clip-metas.jsonl"
    # captions = []
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         captions.append(data["Caption"])

    # 2. Panda70M
    data_path = "/mnt/dongxu-fs1/data-hdd/mingyang/datasets/Panda-70M/panda70m_training_full.csv"
    csv.field_size_limit(9000000)
    print(f"Reading csv file: {data_path}")
    with open(data_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = [x for x in reader]

    good_data = [data[0]]  # save 1st row
    args.max_len = len(data) if args.max_len == -1 else args.max_len
    for i in tqdm(range(1, min(len(data), args.max_len)), desc="Total"):  # skip 1st row
        row = data[i]
        if filter_csv_row(row):
            good_data.append(row)
        else:
            pass
    print(f"Filter good data count: {len(good_data)}")

    ''' Save and check '''
    with open(args.save_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(good_data)
    print(f"Data saved to csv: {args.save_path}")
    with open(args.save_path, "r") as f:
        reader = csv.reader(f, delimiter=',')
        data = [x for x in reader]
        print(f"[Check saved csv]: {args.save_path}")
        print("row[0]:", data[0])
        print("row[1]:", data[1])
        print("row[-1]:", data[-1])
        print(f"len={len(data)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=1000, help="-1 means all;")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    python text_filtering.py \
      --save_path /mnt/dongxu-fs1/data-ssd/geyuan/datasets/Panda-70M/panda70m_training_animals.csv \
      --max_len -1
    """
    opts = parse_args()
    main(opts)
