import os, json, argparse
from datetime import datetime
import subprocess
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video list")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_txt", type=str, default=None)
    args = parser.parse_args()

    dataset_name = os.path.basename(args.dataset_dir)
    root = os.path.dirname(args.dataset_dir)
    if args.output_txt is None:
        args.output_txt = os.path.join(root, f"{dataset_name}.txt")

    file_names = []
    for root, dirs, files in os.walk(args.dataset_dir, topdown=False):
        for name in files:
            file_names.append(os.path.join(root[len(args.dataset_dir) + 1:], name) + "\n")

    with open(args.output_txt, "w") as f:
        f.writelines(file_names)

    print(f"[GenVideoList] video-list file saved to: {args.output_txt} ; len={len(file_names)}")
