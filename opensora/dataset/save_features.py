# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal extracting latents script for DiT using PyTorch DDP.
"""
import argparse
import logging
import math
import os
import datetime
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from einops import rearrange
from tqdm import tqdm
from dataclasses import field, dataclass
from torch.utils.data import DataLoader
from copy import deepcopy

import accelerate
import torch
from torch.nn import functional as F
import transformers
from safetensors.torch import load_file
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers.utils import check_min_version, is_wandb_available

from opensora.dataset import getdataset, ae_denorm
from opensora.models.ae import getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.text_encoder import get_text_enc
from opensora.utils.dataset_utils import Collate
from opensora.models.ae import ae_stride_config, ae_channel_config
from opensora.models.diffusion import Diffusion_models

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
os.environ["WANDB__SERVICE_WAIT"] = "300"
logger = get_logger(__name__)


#################################################################################
#                                  Extracting Loop                              #
#################################################################################

def main(args):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    os.makedirs(args.logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during extracting.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the extracting seed now.
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(42)  # should keep same

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Create model:
    # ae = getae_wrapper(args.ae)(args.ae_path).eval()
    ae = getae_wrapper(args.ae)("LanguageBind/Open-Sora-Plan-v1.0.0", subfolder="vae", cache_dir=args.cache_dir, is_training=False)
    if args.enable_tiling:
        ae.vae.enable_tiling()
        ae.vae.tile_overlap_factor = args.tile_overlap_factor
    text_enc = get_text_enc(args).eval()

    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert ae_stride_h == ae_stride_w, f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    assert patch_size_h == patch_size_w, f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    # assert args.num_frames % ae_stride_t == 0, f"Num_frames must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert args.max_image_size % ae_stride_h == 0, f"Image size must be divisible by ae_stride_h, but found max_image_size ({args.max_image_size}),  ae_stride_h ({ae_stride_h})."

    w_ratio, h_ratio = [int(x) for x in args.wh_ratio.split(":")]
    target_hw = (args.max_image_size // w_ratio * h_ratio, args.max_image_size)
    assert target_hw[0] / target_hw[1] == h_ratio / w_ratio
    latent_size = (target_hw[0] // ae_stride_h, target_hw[1] // ae_stride_w)

    if getae_wrapper(args.ae) == CausalVQVAEModelWrapper or getae_wrapper(args.ae) == CausalVAEModelWrapper:
        args.video_length = video_length = args.num_frames // ae_stride_t + 1
    else:
        video_length = args.num_frames // ae_stride_t

    # Freeze vae and text encoders.
    ae.requires_grad_(False)
    text_enc.requires_grad_(False)

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    ae.to(accelerator.device, dtype=torch.float32)

    # Setup data:
    extract_dataset = getdataset(args, logger=logger)
    extract_dataloader = torch.utils.data.DataLoader(
        extract_dataset,
        shuffle=False,  # Should be False here
        batch_size=args.extract_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare everything with our `accelerator`.
    extract_dataloader = accelerator.prepare(
        extract_dataloader,
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if accelerator.is_main_process:
        project_name = args.output_dir if args.tracker_project_name is None else args.tracker_project_name
        project_name = f"({now}){project_name}"
        accelerator.init_trackers(
            project_name,
            config=vars(args),
            init_kwargs={
                "wandb":
                    {
                        "dir": args.logging_dir,
                        "entity": args.tracker_entity,
                        "name": args.tracker_run_name,
                     }
            },
        )

    # Extract!
    total_batch_size = args.extract_batch_size * accelerator.num_processes

    logger.info("***** Running extracting *****")
    logger.info(f"  Instantaneous batch size per device = {args.extract_batch_size}")
    logger.info(f"  Total extract batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total extracting steps = {args.max_extract_steps}")
    logger.info(f"  Resume from step? = {args.resume_from_step}")
    global_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_step is not None:
        initial_global_step = args.resume_from_step
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_extract_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def print_env():
        line = ""
        for key in sorted(os.environ.keys()):
            if not (
                    key.startswith(("SLURM_", "SUBMITIT_"))
                    or key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
            ):
                continue
            if ("SLURM" in key) and ("PROCID" not in key):
                continue
            value = os.environ[key]
            line += f"{key}={value}, "
        print(line)
    print_env()

    tokenizer = extract_dataset.tokenizer
    assert args.latent_cache_size > args.extract_batch_size
    cache_tensors = torch.zeros(
        (args.latent_cache_size, ae.vae.config.z_channels,
         video_length + args.use_image_num, latent_size[0], latent_size[1]),
        dtype=torch.float32, device=accelerator.device, requires_grad=False)
    cache_ids = torch.zeros(
        (args.latent_cache_size),
        dtype=torch.long, device="cpu", requires_grad=False)
    cache_cnt = 0
    for step, (video_ids, x, text_ids, conda_mask) in enumerate(extract_dataloader):
        if global_step < initial_global_step:
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(status="Resuming, still skipping")
            continue

        # Sample noise that we'll add to the latents
        x = x.to(accelerator.device)  # B C T+num_images H W, 16 + 4

        with torch.no_grad():
            # Map input images to latent space + normalize latents
            if args.use_image_num == 0:
                x = ae.encode(x)  # B C T H W
                # cond = text_enc(input_ids, cond_mask)  # B L -> B L D
            else:
                videos, images = x[:, :, :-args.use_image_num], x[:, :, -args.use_image_num:]
                videos = ae.encode(videos)  # B C T H W
                images = rearrange(images, 'b c t h w -> (b t) c 1 h w')
                images = ae.encode(images)
                images = rearrange(images, '(b t) c 1 h w -> b c t h w', t=args.use_image_num)
                x = torch.cat([videos, images], dim=2)   #  b c 16+4, h, w

                # use for loop to avoid OOM, because T5 is too huge...
                # B, _, _ = input_ids.shape  # B T+num_images L  b 1+4, L
                # cond = torch.stack([text_enc(input_ids[i], cond_mask[i]) for i in range(B)])  # B 1+num_images L D

            # Save latents
            b, c, f, h, w = x.shape
            if cache_cnt + b > args.latent_cache_size:  # cache is full, save
                save_latents(cache_tensors, cache_ids, args.output_dir, max_cnt=cache_cnt)
                cache_cnt = 0
            cache_tensors[cache_cnt: cache_cnt + b] = x.detach()
            cache_ids[cache_cnt: cache_cnt + b] = video_ids.detach()
            cache_cnt = (cache_cnt + b) % args.latent_cache_size

        # Validation and log
        if accelerator.is_main_process and global_step % args.validation_steps == 0 and \
                0 == int(os.environ["SLURM_PROCID"]):
            with torch.no_grad():
                validation_prompt = tokenizer.decode(text_ids[0], skip_special_tokens=True)
                validation_latent = x[0].unsqueeze(0)
                logger.info(f"Running validation... \n"
                            f"Generating a video from the latent with caption: {validation_prompt}")
                val_output = ae.decode(validation_latent)
                val_output = (ae_denorm[args.ae](val_output[0]) * 255).add_(0.5).clamp_(0, 255).to(
                    dtype=torch.uint8).cpu().contiguous()  # t c h w
            videos = torch.stack([val_output]).numpy()
            if args.enable_tracker:
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_videos = np.stack([np.asarray(vid) for vid in videos])
                        tracker.writer.add_video("validation", np_videos, global_step, fps=10)
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Video(video, caption=f"{i}: {validation_prompt}", fps=10)
                                    for i, video in enumerate(videos)
                                ]
                            }
                        )
            torch.cuda.empty_cache()

        # Update tqdm and tracker log
        progress_bar.update(1)
        global_step += 1
        accelerator.log({"cache_cnt": cache_cnt}, step=global_step)
        logs = {"cache_cnt": cache_cnt}
        progress_bar.set_postfix(**logs)

        if global_step >= args.max_extract_steps:
            break

    # Save the rest latents
    if cache_cnt > 0:
        save_latents(cache_tensors, cache_ids, args.output_dir, max_cnt=cache_cnt)
        cache_cnt = 0

    accelerator.wait_for_everyone()
    accelerator.end_training()


def save_latents(latents: torch.Tensor, vids: torch.Tensor, save_root: str, max_cnt: int = -1):
    print("[DEBUG] ready to save latents.")
    b = latents.shape[0]
    if max_cnt == -1:
        max_cnt = b
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)
    latents = latents.cpu().numpy()
    vids = vids.cpu().numpy()
    for i in range(b):
        if i >= max_cnt:
            break
        latent = latents[i]
        video_id = vids[i]
        save_fn = os.path.join(save_root, f"{video_id}.npy")
        np.save(save_fn, latent)
    print("[DEBUG] npy saved.")


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--replace_root", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="DiT-XL/122")

    parser.add_argument("--ae", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--ae_path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sample_rate", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_image_size", type=int, default=128)
    parser.add_argument("--wh_ratio", type=str, default="1:1")

    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--latent_cache_size', type=int, default=64)

    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--model_max_length", type=int, default=120)

    parser.add_argument("--enable_tracker", action="store_true")
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible extracting.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--extract_batch_size", type=int, default=16, help="Batch size (per device) for the extracting dataloader."
    )
    parser.add_argument(
        "--max_extract_steps",
        type=int,
        default=None,
        help="Total number of extracting steps to perform.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help=(
            "Validate every X steps."
        ),
    )
    parser.add_argument(
        "--resume_from_step",
        type=int,
        default=None,
        help=("Resume from which global step."),
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed extracting: local_rank")

    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache dir. Default is ~/.cache/huggingface/hub/")
    parser.add_argument("--internvid_dir", type=str, default=None,
                        help="InternVid video root folder")
    parser.add_argument("--internvid_meta", type=str, default=None,
                        help="InternVid .jsonl file")
    parser.add_argument("--panda70m_dir", type=str, default=None,
                        help="Panda70M video root folder")
    parser.add_argument("--panda70m_meta", type=str, default=None,
                        help="Panda70M .csv file")
    parser.add_argument("--webvid_dir", type=str, default=None,
                        help="WebVid video root folder")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default=None,
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers"
        ),
    )
    parser.add_argument(
        "--tracker_entity",
        type=str,
        default=None,
        help=(
            "The `entity` argument passed to Accelerator.init_trackers Wandb."
            "E.g. `Text-to-Video-hg` "
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None,
        help=(
            "The `name` argument passed to Accelerator.init_trackers Wandb"
        ),
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    main(args)
