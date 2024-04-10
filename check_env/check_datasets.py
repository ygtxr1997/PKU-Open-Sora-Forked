import os
from datasets import load_dataset, load_from_disk, Dataset


target = "imdb"  # in ["opensora", "imdb"]
download = True  # in [True, False]

if target == "opensora":
    hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
    cache_dir = "/public/home/201810101923/datasets/opensora/dataset_v1.0.0_tmptest"
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

    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=hf_repo_name,
        repo_type="dataset",
        # filename="mixkit.tar.gz",
        cache_dir=cache_dir
    )
    target_dataset = load_dataset(
        path=hf_repo_name,
        cache_dir=cache_dir,
    )

    # 1. Download, error: 8,9,10,12,13,15,18
    # data_files = ["pixabay.tar.gz.%02d" % x for x in [18]]
    # data_files = ["pexels.tar.gz.%02d" % x for x in range(6)]
    # target_dataset = load_dataset(
    #     path=hf_repo_name,
    #     cache_dir=cache_dir,
    #     # num_proc=8,
    #     # token="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",
    #     # data_files=data_files,
    #     # download_mode="force_redownload",
    #     # verification_mode="all_checks"
    # )
    # print(f"[Check Datasets] Dataset loaded to cache={cache_dir}.")
    # exit()

    # 2. Reorder
    target_dataset.save_to_disk(
        dataset_dict_path=save_dir,
        max_shard_size="40GB"
    )
    print(f"[Check Datasets] Dataset saved to {save_dir}.")
else:
    # 3. Check loading offline
    target_dataset = load_from_disk(
        dataset_path=save_dir
    )
    print(f"[Check Datasets] Dataset loaded from {save_dir}.")

suffix = "online" if download else "offline"
print(f"Dataset ({target}) loaded! mode={suffix}")
