import os
import json

def split_jsonl(input_file, output_dir, num_splits=7200):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计输入文件的总行数
    total_lines = 0
    with open(input_file, 'r', encoding='utf-8') as infile:
        for _ in infile:
            total_lines += 1

    # 计算每个小文件的行数
    lines_per_split = total_lines // num_splits

    # 初始化变量
    current_lines = 0
    data_list = []
    split_count = 0

    # 逐行读取输入文件
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # 解析JSON对象
            data = json.loads(line.strip())
            data_list.append(data)
            current_lines += 1

            # 如果达到了每个小文件的行数，保存小文件
            if current_lines >= lines_per_split:
                split_count += 1
                output_file = os.path.join(output_dir, f'data_{split_count}.jsonl')
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    for item in data_list:
                        outfile.write(json.dumps(item) + '\n')
                data_list = []
                current_lines = 0

    # 保存最后一个小文件
    if data_list:
        split_count += 1
        output_file = os.path.join(output_dir, f'data_{split_count}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in data_list:
                outfile.write(json.dumps(item) + '\n')

    print(f"成功将文件拆分成 {split_count} 个小文件，保存在目录 {output_dir} 中。")