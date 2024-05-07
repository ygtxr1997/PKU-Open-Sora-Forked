import csv
import os
from tqdm import tqdm


csv_path = "/public/home/201810101923/datasets/panda70m/panda70m_training_clips_0.csv"
max_len = 100
save_path = "/public/home/201810101923/datasets/panda70m/panda70m_training_download_test.csv"

with open(csv_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = [x for x in reader]

good_data = [data[0]]  # save 1st row, ['videoID', 'url', 'timestamp', 'caption', 'matching_score']
for i in tqdm(range(1, len(data)), desc="TaskFilterCSV Walk CSV File"):  # skip 1st row
    if len(good_data) >= max_len:
        break
    row = data[i]
    good_data.append(row)

with open(save_path, "w") as fid:
    writer = csv.writer(fid)
    writer.writerows(good_data)
print(f"[TaskFilterCSV] Data saved to csv: {save_path}")
with open(save_path, "r") as fid:
    tmp_reader = csv.reader(fid, delimiter=',')
    tmp_data = [x for x in tmp_reader]
    print(f"[TaskFilterCSV] [Check saved csv]: {save_path}")
    print("row[0]:", tmp_data[0])
    print("row[1]:", tmp_data[1])
    print("row[-1]:", tmp_data[-1])
    print(f"len={len(tmp_data)}")


