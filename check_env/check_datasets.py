import os
from datasets import load_dataset, load_from_disk, Dataset


target = "imdb"  # in ["opensora", "imdb"]
download = False  # in [True, False]

if target == "opensora":
    hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
    cache_dir = "/public/home/201810101923/datasets/opensora/dataset_v1.0.0"
    save_dir = cache_dir + "_sorted"
elif target == "imdb":
    hf_repo_name = "stanfordnlp/imdb"
    cache_dir = "/public/home/201810101923/datasets/opensora/imdb"
    save_dir = cache_dir + "_sorted"
else:
    raise NotImplementedError(f"Target ({target}) not supported!")

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

if download:
    # 1. Download
    target_dataset = load_dataset(
        path=hf_repo_name,
        cache_dir=cache_dir,
        num_proc=20
    )
    # 2. Reorder
    target_dataset.save_to_disk(
        dataset_dict_path=save_dir,
        max_shard_size="40GB"
    )
else:
    # 3. Check loading offline
    target_dataset = load_from_disk(
        dataset_path=save_dir
    )

suffix = "online" if download else "offline"
print(f"Dataset ({target}) loaded! mode={suffix}")
