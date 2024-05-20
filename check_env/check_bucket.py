import time
import numpy as np

import torch

from opensora.dataset.bucket import Bucket
from opensora.dataset.bucket_configs import bucket_webvid_latent_v257x288x512


def main():
    bucket_config = bucket_webvid_latent_v257x288x512
    bucket = Bucket(bucket_config)

    num_frames = [1, 2, 4, 8, 16, 32, 33, 64]
    heights = [36] * len(num_frames)
    widths = [64] * len(num_frames)

    for i in range(len(num_frames)):
        t, h, w = num_frames[i], heights[i], widths[i]
        bucket_id = bucket.get_bucket_id(t, h, w, seed=int(time.time()))
        print(f"({t}, {h}, {w}): {bucket_id}")


if __name__ == "__main__":
    main()
