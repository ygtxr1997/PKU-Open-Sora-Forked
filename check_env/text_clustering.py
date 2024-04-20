import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import csv
import sklearn.cluster
from tqdm import tqdm

from InstructorEmbedding import INSTRUCTOR


model = INSTRUCTOR(
    'hkunlp/instructor-large',
    cache_folder='/home/geyuan/pretrained/instructor'
)
model = model.cuda()
model = model.eval()

''' get data '''
# # 1. InternVid
# data_path = "/mnt/dongxu-fs1/data-hdd/mingyang/datasets/InternVid-10M-FLT-clip-metas.jsonl"
# captions = []
# with open(data_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line)
#         captions.append(data["Caption"])

# 2. Panda70M
data_path = "/mnt/dongxu-fs1/data-hdd/mingyang/datasets/Panda-70M/panda70m_training_full.csv"
csv.field_size_limit(9000000)
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = [x for x in reader]
captions = []
for i in range(1, len(data)):
    line: str = data[i][3]  # 3rd column is 'caption'
    line = line.replace("\"", "\'")
    captions.extend(line[2:-2].split("', '"))

print(len(captions))
print(captions[0])
print(captions[-1])

animal_keys = [
    "cat", "dog", "horse", "lion", "pig", "wolf",
    "elephant", "bear", "bird", "monkey", "rabbit",
    "fish", "tiger", "dolphin", "giraffe", "penguin",
]
frequency_dict = {}
for animal in animal_keys:
    frequency_dict[animal] = 0
for animal in animal_keys:
    cnt = 0
    for i in tqdm(range(len(captions)), desc=animal):
        if f" {animal}" in captions[i]:
            cnt += 1
    frequency_dict[animal] = cnt
for animal in animal_keys:
    print(f"({animal}):{frequency_dict[animal]}")

max_len = 200000
captions = captions[:max_len]

instruct_text = "Represent the video caption for clustering: "
sentences = []
for caption in captions:
    sentences.append([instruct_text, caption])
embeddings = model.encode(
    sentences,
    show_progress_bar=True,
)

num_clusters = 17
clustering_model = sklearn.cluster.MiniBatchKMeans(
    n_clusters=num_clusters, max_iter=300, batch_size=32,
    verbose=False,
)

num_fit_sub_samples = 200000
num_iterations = max_len // num_fit_sub_samples + (1 if max_len % num_fit_sub_samples > 0 else 0)
# for i in range(num_iterations):
#     l, r = i * num_fit_sub_samples, min((i + 1) * num_fit_sub_samples, max_len)
#     clustering_model.partial_fit(embeddings[l:r])
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

for label in range(num_clusters):
    i, cnt = 0, 0
    while cnt < 20:
        if label == cluster_assignment[i]:
            print(label, captions[i])
            cnt += 1
        else:
            pass
        i += 1
