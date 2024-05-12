import os
from torchvision.transforms import Compose
from transformers import AutoTokenizer

from .feature_datasets import T2V_Feature_dataset, T2V_T5_Feature_dataset
from torchvision import transforms
from torchvision.transforms import Lambda
from datasets.distributed import split_dataset_by_node

from .landscope import Landscope
from .t2v_datasets import T2V_dataset
from .internvid_datasets import InternVidDataset
from .panda70m_datasets import Panda70MPytorchDataset
from .webvid_datasets import WebVidHFWebDataset, WebVidLatentDataset
from .transform import ToTensorVideo, TemporalRandomCrop, RandomHorizontalFlipVideo, CenterCropResizeVideo
from .ucf101 import UCF101
from .sky_datasets import Sky

ae_norm = {
    'CausalVAEModel_4x8x8': Lambda(lambda x: 2. * x - 1.),
    'CausalVQVAEModel_4x4x4': Lambda(lambda x: x - 0.5),
    'CausalVQVAEModel_4x8x8': Lambda(lambda x: x - 0.5),
    'VQVAEModel_4x4x4': Lambda(lambda x: x - 0.5),
    'VQVAEModel_4x8x8': Lambda(lambda x: x - 0.5),
    "bair_stride4x2x2": Lambda(lambda x: x - 0.5),
    "ucf101_stride4x4x4": Lambda(lambda x: x - 0.5),
    "kinetics_stride4x4x4": Lambda(lambda x: x - 0.5),
    "kinetics_stride2x4x4": Lambda(lambda x: x - 0.5),
    'stabilityai/sd-vae-ft-mse': transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    'stabilityai/sd-vae-ft-ema': transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    'vqgan_imagenet_f16_1024': Lambda(lambda x: 2. * x - 1.),
    'vqgan_imagenet_f16_16384': Lambda(lambda x: 2. * x - 1.),
    'vqgan_gumbel_f8': Lambda(lambda x: 2. * x - 1.),

}
ae_denorm = {
    'CausalVAEModel_4x8x8': lambda x: (x + 1.) / 2.,
    'CausalVQVAEModel_4x4x4': lambda x: x + 0.5,
    'CausalVQVAEModel_4x8x8': lambda x: x + 0.5,
    'VQVAEModel_4x4x4': lambda x: x + 0.5,
    'VQVAEModel_4x8x8': lambda x: x + 0.5,
    "bair_stride4x2x2": lambda x: x + 0.5,
    "ucf101_stride4x4x4": lambda x: x + 0.5,
    "kinetics_stride4x4x4": lambda x: x + 0.5,
    "kinetics_stride2x4x4": lambda x: x + 0.5,
    'stabilityai/sd-vae-ft-mse': lambda x: 0.5 * x + 0.5,
    'stabilityai/sd-vae-ft-ema': lambda x: 0.5 * x + 0.5,
    'vqgan_imagenet_f16_1024': lambda x: (x + 1.) / 2.,
    'vqgan_imagenet_f16_16384': lambda x: (x + 1.) / 2.,
    'vqgan_gumbel_f8': lambda x: (x + 1.) / 2.,
}

def getdataset(args, logger=None):
    temporal_sample = TemporalRandomCrop(args.num_frames * args.sample_rate)  # 16 x
    norm_fun = ae_norm[args.ae]
    nnodes = int(os.environ["SLURM_NNODES"])
    global_gpus = int(os.environ["WORLD_SIZE"])
    node_id = int(os.environ["SLURM_PROCID"])
    global_rank = int(os.environ["RANK"])
    if args.dataset == 'ucf101':
        transform = Compose(
            [
                ToTensorVideo(),  # TCHW
                CenterCropResizeVideo(size=args.max_image_size),
                RandomHorizontalFlipVideo(p=0.5),
                norm_fun,
            ]
        )
        return UCF101(args, transform=transform, temporal_sample=temporal_sample)
    if args.dataset == 'landscope':
        transform = Compose(
            [
                ToTensorVideo(),  # TCHW
                CenterCropResizeVideo(size=args.max_image_size),
                RandomHorizontalFlipVideo(p=0.5),
                norm_fun,
            ]
        )
        return Landscope(args, transform=transform, temporal_sample=temporal_sample)
    elif args.dataset == 'sky':
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(args.max_image_size),
            RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        return Sky(args, transform=transform, temporal_sample=temporal_sample)
    elif args.dataset == 't2v':
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(args.max_image_size),
            RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir)
        return T2V_dataset(args, transform=transform, temporal_sample=temporal_sample, tokenizer=tokenizer)
    elif args.dataset == 'internvid':
        w_ratio, h_ratio = [int(x) for x in args.wh_ratio.split(":")]
        target_hw = (args.max_image_size // w_ratio * h_ratio, args.max_image_size)
        assert target_hw[0] / target_hw[1] == h_ratio / w_ratio
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(target_hw, h_ratio=h_ratio, w_ratio=w_ratio),
            RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir)
        return InternVidDataset(
            args.internvid_meta,
            args.internvid_dir,
            logger=logger,
            tokenizer=tokenizer,
            transform=transform,
            norm_fun=norm_fun,
            num_frames=args.num_frames,
            max_frame_stride=args.sample_rate,
        )
    elif args.dataset == 'panda70m':
        w_ratio, h_ratio = [int(x) for x in args.wh_ratio.split(":")]
        target_hw = (args.max_image_size // w_ratio * h_ratio, args.max_image_size)
        assert target_hw[0] / target_hw[1] == h_ratio / w_ratio
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(target_hw, h_ratio=h_ratio, w_ratio=w_ratio),
            RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir)
        all_dataset = Panda70MPytorchDataset(
            args.panda70m_meta,
            args.panda70m_dir,
            logger=logger,
            tokenizer=tokenizer,
            transform=transform,
            norm_fun=norm_fun,
            num_frames=args.num_frames,
            max_frame_stride=args.sample_rate,
        )
        return all_dataset
    elif args.dataset == 'webvid':
        w_ratio, h_ratio = [int(x) for x in args.wh_ratio.split(":")]
        target_hw = (args.max_image_size // w_ratio * h_ratio, args.max_image_size)
        assert target_hw[0] / target_hw[1] == h_ratio / w_ratio
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(target_hw, h_ratio=h_ratio, w_ratio=w_ratio),
            RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir)
        return WebVidHFWebDataset(
            args.webvid_dir,
            logger=logger,
            tokenizer=tokenizer,
            transform=transform,
            norm_fun=norm_fun,
            num_frames=args.num_frames,
            max_frame_stride=args.sample_rate,
            rank=global_rank,
            world_size=global_gpus,
        )
    elif args.dataset == 'webvid_latent':
        w_ratio, h_ratio = [int(x) for x in args.wh_ratio.split(":")]
        target_hw = (args.max_image_size // w_ratio * h_ratio, args.max_image_size)
        assert target_hw[0] / target_hw[1] == h_ratio / w_ratio
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_encoder_name, cache_dir=args.cache_dir)
        return WebVidLatentDataset(
            args.webvid_meta,
            args.webvid_dir,
            logger=logger,
            tokenizer=tokenizer,
        )
    elif args.dataset == 't2v_feature':
        return T2V_Feature_dataset(args, temporal_sample)
    elif args.dataset == 't2v_t5_feature':
        transform = transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(args.max_image_size),
            RandomHorizontalFlipVideo(p=0.5),
            norm_fun
        ])
        return T2V_T5_Feature_dataset(args, transform, temporal_sample)
    else:
        raise NotImplementedError(args.dataset)
