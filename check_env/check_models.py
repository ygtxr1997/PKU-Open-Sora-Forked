import os

from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.ae import ae_stride_config, getae, getae_wrapper


target = "opensora_stage3"  # in ["opensora_stage1", "opensora_stage2", "opensora_stage3", "casual_vae"]
download = True  # in [True, False]


if "opensora" in target:
    hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
    cache_dir = "/public/home/201810101923/models/opensora/v1.0.0"
    save_dir = cache_dir + "_sorted"
    if "stage1" in target:
        sub_dir = "17x256x256"
    elif "stage2" in target:
        sub_dir = "65x256x256"
    elif "stage3" in target:
        sub_dir = "65x512x512"
    else:
        raise NotImplementedError(f"Target stage ({target}) not supported!")
elif target == "casual_vae":
    hf_repo_name = "LanguageBind/Open-Sora-Plan-v1.0.0"
    cache_dir = "/public/home/201810101923/models/opensora/v1.0.0"
    save_dir = cache_dir + "_sorted"
    sub_dir = "vae"
else:
    raise NotImplementedError(f"Target ({target}) not supported!")

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

if "opensora" in target:
    if download:
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
elif "casual_vae" in target:
    if download:
        # 1. Download
        target_model = getae_wrapper("CausalVAEModel_4x8x8")(
            model_path=hf_repo_name,
            cache_dir=cache_dir,
            subfolder=sub_dir,
        )
    else:
        # 2. Check loading offline
        target_model = getae_wrapper("CausalVAEModel_4x8x8")(
            model_path=cache_dir,
            cache_dir=cache_dir,
            subfolder=sub_dir,
        )

suffix = "online" if download else "offline"
print(f"Model ({target}) loaded! mode={suffix}")
