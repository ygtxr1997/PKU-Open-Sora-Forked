import os, json, argparse
from datetime import datetime
import subprocess
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video list")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_txt", type=str, default=None)
    args = parser.parse_args()

    print(os.path.dirname(args.dataset_dir))

    for root, dirs, files in os.walk(args.dataset_dir, topdown=False):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
        exit()
