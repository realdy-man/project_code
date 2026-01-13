import pandas as pd

# 读取修复后的CSV文件
df = pd.read_csv('F:\AAA_JIQIXUEXI\project_code\output_fixed.csv')

# 查看当前的列名
print(f"当前列名: {df.columns.tolist()}")

# 重新设置列名
new_df = df.copy()
new_df.columns = ['text', 'label']

# 保存带正确列名的新文件
final_output = 'F:\AAA_JIQIXUEXI\project_code\output_final.csv'
new_df.to_csv(final_output, index=False)

print(f"已添加正确的列名'text'和'label'")
print(f"新文件已保存为: {final_output}")
print(f"文件包含 {len(new_df)} 行数据")