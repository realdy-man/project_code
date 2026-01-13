import csv

file_path = 'F:\AAA_JIQIXUEXI\project_code\output.csv'

# 检查CSV文件的字段数量
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, 1):
        if i >= 125 and i <= 135:
            print(f"第{i}行: {row}")
            print(f"字段数量: {len(row)}")
            print()

# 尝试用pandas读取，看能否提供更详细的错误信息
try:
    import pandas as pd
    df = pd.read_csv(file_path)
    print(f"CSV文件总共有 {len(df)} 行，{len(df.columns)} 列")
except Exception as e:
    print(f"pandas读取错误: {e}")