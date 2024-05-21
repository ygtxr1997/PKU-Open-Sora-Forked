import json

import math
import os
import torch
import argparse
import torchvision

from accelerate import Accelerator
from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.configuration_utils import ConfigMixin
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer
from safetensors.torch import load_file

import os, sys

from opensora.models.ae import ae_stride_config, getae, getae_wrapper
from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.models.text_encoder import get_text_enc
from opensora.utils.utils import save_video_grid

sys.path.append(os.path.split(sys.path[0])[0])
from pipeline_videogen import VideoGenPipeline

import imageio


@torch.no_grad()
def main(args):
    accelerator = Accelerator()

    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = accelerator.device

    vae = getae_wrapper(args.ae)(args.model_path, subfolder="vae", cache_dir=args.cache_dir, is_training=False).to(device, dtype=torch.float16)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    # Load model:
    if args.ckpt_path is None:
        print(f"Load transformer from cache dir: {args.cache_dir}")
        transformer_model = LatteT2V.from_pretrained(args.model_path, subfolder=args.version, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)
    else:
        if os.path.splitext(args.ckpt_path)[-1] == ".safetensors":
            config_json_path = os.path.join(os.path.dirname(args.ckpt_path), "config.json")
            print(f"Load transformer from safetensors: {args.ckpt_path}, config_path={config_json_path}")
            with open(config_json_path, "r") as f:
                config = json.load(f)
            transformer_model = LatteT2V.from_config(config, torch_dtype=torch.float16).to(device)
            transformer_model.load_state_dict(load_file(args.ckpt_path, device="cpu"))
            transformer_model = transformer_model.to(torch.float16)
        elif os.path.splitext(args.ckpt_path)[-1] == ".pt":
            print(f"Load transformer from pt: {args.ckpt_path}")
            opensora_version = "65x512x512"  # different versions share the same model architecture
            transformer_model = LatteT2V.from_config(args.model_path, subfolder=opensora_version, cache_dir=args.cache_dir,
                                                     torch_dtype=torch.float16).to(device)
            weight = torch.load(args.ckpt_path, map_location='cpu')
            if "model" in weight.keys:
                weight = weight["model"]
            transformer_model.load_state_dict(weight)
            transformer_model = transformer_model.to(torch.float16)
        else:
            raise TypeError(f"Ckpt file type not supported: {args.ckpt_path}")
    transformer_model.force_images = args.force_images
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, torch_dtype=torch.float16).to(device)

    video_length = transformer_model.config.video_length
    sample_size: str = args.version if args.sample_size is None else args.sample_size
    train_frames, image_size_h, image_size_w = [int(x) for x in sample_size.split('x')]  # e.g. 65x512x512
    video_length = train_frames // ae_stride_config[args.ae][0] + 1
    latent_size = (image_size_h // ae_stride_config[args.ae][1], image_size_w // ae_stride_config[args.ae][2])
    vae.latent_size = latent_size
    if args.force_images:
        video_length = 1
        ext = 'jpg'
    else:
        ext = 'mp4'

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler()
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler()
    elif args.sample_method == 'DPMSolverMultistep':
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    print('videogen_pipeline', device)
    videogen_pipeline = VideoGenPipeline(vae=vae,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         scheduler=scheduler,
                                         transformer=transformer_model).to(device=device)
    # videogen_pipeline.enable_xformers_memory_efficient_attention()

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path, exist_ok=True)

    video_grids = []
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        args.text_prompt = [i.strip() for i in text_prompt]
    args.text_prompt = args.text_prompt[accelerator.process_index::accelerator.num_processes]
    for prompt in args.text_prompt:
        print('Processing the ({}) prompt'.format(prompt))
        videos = videogen_pipeline(prompt,
                                   video_length=video_length,
                                   height=image_size_h,
                                   width=image_size_w,
                                   num_inference_steps=args.num_sampling_steps,
                                   guidance_scale=args.guidance_scale,
                                   enable_temporal_attentions=not args.force_images,
                                   num_images_per_prompt=1,
                                   mask_feature=True,
                                   ).video
        try:
            if args.force_images:
                videos = videos[:, 0].permute(0, 3, 1, 2)  # b t h w c -> b c h w
                save_image(videos / 255.0, os.path.join(args.save_img_path,
                                                     prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'),
                           nrow=1, normalize=True, value_range=(0, 1))  # t c h w

            else:
                imageio.mimwrite(
                    os.path.join(
                        args.save_img_path,
                        prompt.replace(' ', '_')[:100] + f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'
                    ), videos[0],
                    fps=args.fps, quality=9)  # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))
        video_grids.append(videos)
    video_grids = torch.cat(video_grids, dim=0)


    # torchvision.io.write_video(args.save_img_path + '_%04d' % args.run_time + '-.mp4', video_grids, fps=6)
    if args.force_images:
        save_image(video_grids / 255.0, os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}_rank{accelerator.process_index}.{ext}'),
                   nrow=math.ceil(math.sqrt(len(video_grids))), normalize=True, value_range=(0, 1))
    else:
        video_grids = save_video_grid(video_grids)
        imageio.mimwrite(os.path.join(args.save_img_path, f'{args.sample_method}_gs{args.guidance_scale}_s{args.num_sampling_steps}.{ext}'), video_grids, fps=args.fps, quality=9)

    print('save path {}'.format(args.save_img_path))

    # save_videos_grid(video, f"./{prompt}.gif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="/public/home/201810101923/models/opensora/v1.0.0")
    parser.add_argument("--version", type=str, default='65x512x512',
                        choices=['65x512x512', '65x256x256', '17x256x256',
                                 '17x288x512', '17x144x256', '17x72x128',
                                 '129x80x128', '257x80x128',
                                 '129x144x256',
                                 '129x288x512',
                                 'v257x288x512',
                                 ])
    parser.add_argument("--sample_size", type=str, default=None)
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--force_images', action='store_true')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.25)
    parser.add_argument('--enable_tiling', action='store_true')
    args = parser.parse_args()

    main(args)