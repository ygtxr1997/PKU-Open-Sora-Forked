from collections import OrderedDict

import numpy as np

from .aspect import ASPECT_RATIOS, get_closest_ratio


def find_approximate_hw(hw, hw_dict, approx=0.8):
    for k, v in hw_dict.items():
        if hw >= v * approx:
            return k
    return None


def find_closet_smaller_bucket(t, t_dict, frame_interval):
    # process image
    if t == 1:
        if 1 in t_dict:
            return 1
        else:
            return None
    # process video
    for k, v in t_dict.items():
        if t >= v * frame_interval and v != 1:
            return k
    return None


class Bucket:
    def __init__(self, bucket_config):
        """
        Args:
            bucket_config: "ID_PIXELS": {NUM_FRAMES: (PROB_KEEP_PIXELS, BATCH_SIZE, Optional[PROB_KEEP_FRAMES]), ...}
        """
        for key in bucket_config:
            assert key in ASPECT_RATIOS, f"Aspect ratio {key} not found."
        # wrap config with OrderedDict
        bucket_pixels_probs = OrderedDict()
        bucket_bs = OrderedDict()
        bucket_frames_probs = OrderedDict()
        bucket_names = sorted(bucket_config.keys(), key=lambda x: ASPECT_RATIOS[x][0], reverse=True)  # by num_pixels

        def func_get_frames_prob(t: tuple) -> float:
            return t[2] if len(t) > 2 else 1.0

        for key in bucket_names:
            bucket_time_names = sorted(bucket_config[key].keys(), key=lambda x: x, reverse=True)  # big to small
            bucket_pixels_probs[key] = OrderedDict({k: bucket_config[key][k][0] for k in bucket_time_names})
            bucket_bs[key] = OrderedDict({k: bucket_config[key][k][1] for k in bucket_time_names})
            bucket_frames_probs[key] = OrderedDict({k: func_get_frames_prob(bucket_config[key][k])
                                                    for k in bucket_time_names})

        # first level: HW
        num_bucket = 0
        hw_criteria = dict()
        t_criteria = dict()
        ar_criteria = dict()
        bucket_index = OrderedDict()
        bucket_index_cnt = 0
        for k1, v1 in bucket_pixels_probs.items():  # k1:ID_PIXELS
            hw_criteria[k1] = ASPECT_RATIOS[k1][0]
            t_criteria[k1] = dict()
            ar_criteria[k1] = dict()
            bucket_index[k1] = dict()
            for k2, _ in v1.items():  # k2:NUM_FRAMES
                t_criteria[k1][k2] = k2
                bucket_index[k1][k2] = bucket_index_cnt
                bucket_index_cnt += 1
                ar_criteria[k1][k2] = dict()
                for k3, v3 in ASPECT_RATIOS[k1][1].items():  # k3:AR_STR, v3:(H,W)
                    ar_criteria[k1][k2][k3] = v3
                    num_bucket += 1

        self.bucket_pixels_probs = bucket_pixels_probs  # {ID_PIXELS: {NUM_FRAMES: PROB_KEEP_PIXELS}, ...}
        self.bucket_bs = bucket_bs                      # {ID_PIXELS: {NUM_FRAMES: BATCH_SIZE}, ...}
        self.bucket_frames_probs = bucket_frames_probs  # {ID_PIXELS: {NUM_FRAMES: PROB_KEEP_FRAMES}, ...}
        self.bucket_index = bucket_index                # {ID_PIXELS: {NUM_FRAMES: BUCKET_ID}, ...}
        self.hw_criteria = hw_criteria                  # {ID_PIXELS: NUM_PIXELS}
        self.t_criteria = t_criteria                    # {ID_PIXELS: {NUM_FRAMES: NUM_FRAMES}, ...}
        self.ar_criteria = ar_criteria                  # {ID_PIXELS: {NUM_FRAMES: {AR_STR: (H,W), ...}}, ...}
        self.num_bucket = num_bucket
        print(f"Number of buckets: {num_bucket}")

    def get_bucket_id(self, T, H, W, frame_interval=1, seed=42):
        assert T > 0
        resolution = H * W
        approx = 0.8

        fail = True
        for hw_id, t_criteria in self.bucket_pixels_probs.items():  # NUM_PIXELS: big to small
            ''' 1. Check resolution '''
            if resolution < self.hw_criteria[hw_id] * approx:
                continue

            # if sample is an image
            if T == 1:
                ''' 2. Check num_frames '''
                if 1 in t_criteria:
                    ''' 4. Check resolution_drop_rate '''
                    rng = np.random.default_rng(seed + self.bucket_index[hw_id][1])
                    if rng.random() < t_criteria[1]:  # 1 is NUM_FRAMES
                        fail = False
                        t_id = 1
                        break  # frames and pixels all satisfied
                else:
                    continue

            # otherwise, find suitable t_id for video
            t_fail = True
            for t_id, prob_keep_pixels in t_criteria.items():  # NUM_FRAMES: big to small
                ''' 2. Check num_frames '''
                if (T > t_id * frame_interval) or not t_fail:  # enough frames or already founded
                    t_fail = False  # enough frames, but still need to check frames drop rate
                    ''' 3. Check frames_drop_rate '''
                    rng = np.random.default_rng(seed + self.bucket_index[hw_id][1])
                    prob_keep_frames = self.bucket_frames_probs[hw_id][t_id]
                    if prob_keep_frames == 1 or rng.random() < prob_keep_frames:  # check frames drop rate
                        break  # frames satisfied
            if t_fail:
                continue

            ''' 4. Check resolution_drop_rate '''
            # leave the loop if prob is high enough
            rng = np.random.default_rng(seed + self.bucket_index[hw_id][t_id])
            if prob_keep_pixels == 1 or rng.random() < prob_keep_pixels:
                fail = False
                break  # frames and pixels all satisfied

        if fail:
            return None

        ''' 5. Check the nearest aspect ratio (bound to resolution) '''
        # get aspect ratio id
        ar_criteria = self.ar_criteria[hw_id][t_id]
        ar_id = get_closest_ratio(H, W, ar_criteria)
        return hw_id, t_id, ar_id  # ID_PIXELS, NUM_FRAMES, ID_ASPECT_RATIO

    def get_thw(self, bucket_id):
        assert len(bucket_id) == 3
        T = self.t_criteria[bucket_id[0]][bucket_id[1]]
        H, W = self.ar_criteria[bucket_id[0]][bucket_id[1]][bucket_id[2]]
        return T, H, W

    def get_prob(self, bucket_id):
        return self.bucket_pixels_probs[bucket_id[0]][bucket_id[1]]

    def get_batch_size(self, bucket_id):
        return self.bucket_bs[bucket_id[0]][bucket_id[1]]

    def __len__(self):
        return self.num_bucket


def closet_smaller_bucket(value, bucket):
    for i in range(1, len(bucket)):
        if value < bucket[i]:
            return bucket[i - 1]
    return bucket[-1]
