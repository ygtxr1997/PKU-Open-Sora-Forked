import os
import argparse

from transformers import T5Tokenizer, T5EncoderModel

from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.ae import ae_stride_config, getae, getae_wrapper


def main():
    TARGETS_LIST = ["opensora_stage1", "opensora_stage2", "opensora_stage3",
                    "casual_vae",
                    "t5",
                    ]

    args = argparse.ArgumentParser("Check models.")
    args.add_argument("--targets", default="all", type=str, choices=TARGETS_LIST + ["all"])
    args.add_argument("--download", action="store_true", default=False)
    opts = args.parse_args()

    if opts.targets == "all":
        targets = TARGETS_LIST
    else:
        targets = [opts.targets]

    download = opts.download
    # If True, download to local
    # Else, load from local
    print(f"[Check Models] ready to check models, download={download}")
    for target in targets:
        load_model(target, need_download=download)


def load_model(target_name: str,
               need_download: bool,
               ):
    if "opensora" in target_name:
        hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
        cache_dir = "/public/home/201810101923/models/opensora/v1.0.0"
        save_dir = cache_dir + "_sorted"
        if "stage1" in target_name:
            sub_dir = "17x256x256"
        elif "stage2" in target_name:
            sub_dir = "65x256x256"
        elif "stage3" in target_name:
            sub_dir = "65x512x512"
        else:
            raise NotImplementedError(f"Target stage ({target_name}) not supported!")
    elif target_name == "casual_vae":
        hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
        cache_dir = "/public/home/201810101923/models/opensora/v1.0.0"
        save_dir = cache_dir + "_sorted"
        sub_dir = "vae"
    elif target_name == "t5":
        hf_repo_name = "DeepFloyd/t5-v1_1-xxl"
        cache_dir = "/public/home/201810101923/models/opensora/v1.0.0"
        save_dir = cache_dir + "_sorted"
        sub_dir = None
    else:
        raise NotImplementedError(f"Target ({target_name}) not supported!")

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if "opensora" in target_name:
        if need_download:
            # 1. Download
            target_model = LatteT2V.from_pretrained(
                pretrained_model_name_or_path=hf_repo_name,
                cache_dir=cache_dir,
                subfolder=sub_dir,
                num_proc=20
            )
        else:
            # 2. Check loading offline
            target_model = LatteT2V.from_pretrained(
                pretrained_model_name_or_path=hf_repo_name,
                cache_dir=cache_dir,
                subfolder=sub_dir,
            )
    elif "casual_vae" in target_name:
        if need_download:
            # 1. Download
            target_model = getae_wrapper("CausalVAEModel_4x8x8")(
                model_path=hf_repo_name,
                cache_dir=cache_dir,
                subfolder=sub_dir,
            )
        else:
            # 2. Check loading offline
            target_model = getae_wrapper("CausalVAEModel_4x8x8")(
                model_path=hf_repo_name,
                cache_dir=cache_dir,
                subfolder=sub_dir,
            )
    elif "t5" in target_name:
        if need_download:
            tokenizer = T5Tokenizer.from_pretrained(
                hf_repo_name,
                cache_dir="cache_dir"
            )
            text_encoder = T5EncoderModel.from_pretrained(
                hf_repo_name,
                cache_dir="cache_dir",
            )
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                hf_repo_name,
                cache_dir="cache_dir"
            )
            text_encoder = T5EncoderModel.from_pretrained(
                hf_repo_name,
                cache_dir="cache_dir",
            )

    suffix = "online" if need_download else "offline"
    print(f"[Check Models] Model ({target_name}) loaded! mode={suffix}")


if __name__ == "__main__":
    main()
