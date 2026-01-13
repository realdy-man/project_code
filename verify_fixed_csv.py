import pandas as pd

# 尝试读取修复后的文件
try:
    df = pd.read_csv('F:\AAA_JIQIXUEXI\project_code\output_fixed.csv')
    print(f"修复后的CSV文件读取成功！")
    print(f"总共有 {len(df)} 行数据")
    print(f"列名: {df.columns.tolist()}")
    print(f"前5行数据:")
    print(df.head())
    print(f"数据类型:")
    print(df.dtypes)
except Exception as e:
    print(f"读取修复后的文件时出错: {e}")