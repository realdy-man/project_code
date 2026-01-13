import pandas as pd
import random

# 读取完整数据集
df = pd.read_csv('F:\AAA_JIQIXUEXI\project_code\output_final.csv')

# 设置随机种子，保证结果可复现
random.seed(42)

# 目标样本数量
target_samples = 500

# 获取所有唯一标签
unique_labels = df['label'].unique().tolist()
print(f"总共有 {len(unique_labels)} 种不同的标签")

# 根据标签分布进行分层抽样
# 计算每种标签应抽取的样本数量
label_counts = df['label'].value_counts()
total_samples = len(df)
samples_per_label = {label: int((count / total_samples) * target_samples) for label, count in label_counts.items()}

# 确保所有标签至少有1个样本
for label in unique_labels:
    if label not in samples_per_label or samples_per_label[label] < 1:
        samples_per_label[label] = 1

# 调整样本数量，确保总和接近目标
current_total = sum(samples_per_label.values())
if current_total != target_samples:
    # 按比例调整样本数量
    ratio = target_samples / current_total
    for label in samples_per_label:
        samples_per_label[label] = max(1, int(samples_per_label[label] * ratio))
    
    # 微调以达到精确的目标数量
    current_total = sum(samples_per_label.values())
    if current_total < target_samples:
        # 为样本最少的标签添加额外样本
        sorted_labels = sorted(samples_per_label.items(), key=lambda x: x[1])
        for i in range(target_samples - current_total):
            samples_per_label[sorted_labels[i][0]] += 1
    elif current_total > target_samples:
        # 从样本最多的标签中减少样本
        sorted_labels = sorted(samples_per_label.items(), key=lambda x: x[1], reverse=True)
        for i in range(current_total - target_samples):
            if samples_per_label[sorted_labels[i][0]] > 1:
                samples_per_label[sorted_labels[i][0]] -= 1

# 执行抽样
sampled_df = pd.DataFrame()
for label, count in samples_per_label.items():
    label_data = df[df['label'] == label]
    if len(label_data) > count:
        # 随机抽取指定数量的样本
        sampled_label_data = label_data.sample(n=count, random_state=42)
    else:
        # 如果标签样本数量不足，取全部
        sampled_label_data = label_data
    sampled_df = pd.concat([sampled_df, sampled_label_data])

# 打乱顺序
sampled_df = sampled_df.sample(frac=1, random_state=42)

# 保存为新文件
sampled_path = 'F:\AAA_JIQIXUEXI\project_code\output_sample.csv'
sampled_df.to_csv(sampled_path, index=False)

# 输出抽样结果
print(f"=== 抽样结果 ===")
print(f"原始数据集大小: {len(df)} 行")
print(f"抽样后数据集大小: {len(sampled_df)} 行")
print(f"抽样文件保存为: {sampled_path}")
print()
print(f"抽样后的标签分布:")
sampled_label_counts = sampled_df['label'].value_counts()
for label, count in sampled_label_counts.items():
    print(f"  {label}: {count} 个样本")

print(f"\n不同标签数量: {len(sampled_label_counts)}")