# train_model.py
import os
import sys
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
from utils import tokenize_for_tfidf  # ← 关键：从 utils 导入

# 配置路径 - 使用新的头条新闻数据集
# 配置路径
DATA_PATH = "F:/AAA_JIQIXUEXI/project_code/tools/data/toutiao_cat_data.txt"
MODEL_DIR = "../models"  # 保存到上一级目录的models文件夹
os.makedirs(MODEL_DIR, exist_ok=True)

# 加载和解析头条新闻数据集
def load_toutiao_data(file_path):
    """加载并解析头条新闻数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 使用 _!_ 分割字段
            fields = line.split('_!_')
            if len(fields) == 5:
                news_id, cat_code, cat_name, title, keywords = fields
                data.append({
                    'news_id': news_id,
                    'cat_code': cat_code,
                    'cat_name': cat_name,
                    'title': title,
                    'keywords': keywords
                })
    return data

# 加载数据
print("📂 加载头条新闻数据集...")
all_data = load_toutiao_data(DATA_PATH)
print(f"📊 数据集加载完成，共 {len(all_data)} 条样本")

# 划分训练集和验证集 (80% 训练, 20% 验证)
random.seed(42)
train_data, dev_data = train_test_split(all_data, test_size=0.2, random_state=42)

# 提取特征和标签
X_train = [item['title'] for item in train_data]
y_train = [item['cat_code'] for item in train_data]

X_dev = [item['title'] for item in dev_data]
y_dev = [item['cat_code'] for item in dev_data]

print(f"📊 训练集: {len(X_train)} 条样本")
print(f"📊 验证集: {len(X_dev)} 条样本")
print(f"🏷️  训练集标签类别数: {len(set(y_train))}")
print(f"🏷️  验证集标签类别数: {len(set(y_dev))}")

# 打印类别分布
from collections import Counter
print("\n类别分布:")
category_counts = Counter(y_train)
for cat_code, count in category_counts.most_common():
    print(f"  类别 {cat_code}: {count} 个样本")

# TF-IDF 向量化（针对头条新闻标题的优化设置）
print("🔄 正在向量化...")
vectorizer = TfidfVectorizer(
    tokenizer=tokenize_for_tfidf,
    lowercase=False,
    token_pattern=None,
    max_features=3000,  # 头条标题较短，减少特征维度
    ngram_range=(1, 2),  # 加入 bigram 捕捉短语特征
    min_df=1,  # 不过滤低频词
    max_df=1.0  # 不过滤高频词
)

X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)

# 训练模型（针对头条新闻数据集的优化设置）
print("🧠 训练模型...")
model = LogisticRegression(
    max_iter=1000,  # 增加迭代次数以确保收敛
    random_state=42,
    C=1.0,  # 默认正则化强度
    solver='lbfgs',  # 使用支持多分类的求解器
    n_jobs=-1  # 使用所有可用的CPU核心加速训练
)
model.fit(X_train_vec, y_train)

# 定义分类code与名称映射
CAT_CODE_TO_NAME = {
    '100': '民生故事',
    '101': '文化',
    '102': '娱乐',
    '103': '体育',
    '104': '财经',
    '106': '房产',
    '107': '汽车',
    '108': '教育',
    '109': '科技',
    '110': '军事',
    '112': '旅游',
    '113': '国际',
    '114': '股票',
    '115': '三农',
    '116': '游戏'
}

# 在验证集上评估（增加详细分类报告）
y_pred = model.predict(X_dev_vec)
acc = accuracy_score(y_dev, y_pred)
f1 = f1_score(y_dev, y_pred, average='weighted')
precision = precision_score(y_dev, y_pred, average='weighted')
recall = recall_score(y_dev, y_pred, average='weighted')

print(f"✅ 验证集准确率: {acc:.4f}")
print(f"✅ 验证集加权F1分数: {f1:.4f}")
print(f"✅ 验证集加权精确率: {precision:.4f}")
print(f"✅ 验证集加权召回率: {recall:.4f}")

# 打印类别分布
print("\n类别分布分析：")
from collections import Counter
train_dist = Counter(y_train).most_common(10)
pred_dist = Counter(y_pred).most_common(10)

print("训练集类别分布（最多10个）:")
for cat_code, count in train_dist:
    print(f"  {cat_code} ({CAT_CODE_TO_NAME.get(cat_code, cat_code)}): {count} 个样本")

print("预测结果分布（最多10个）:")
for cat_code, count in pred_dist:
    print(f"  {cat_code} ({CAT_CODE_TO_NAME.get(cat_code, cat_code)}): {count} 个样本")

# 打印几个具体样本的预测结果
print("\n样本预测示例：")
for i in range(5):
    print(f"样本 {i+1}:")
    print(f"  标题: {X_dev[i]}")
    print(f"  真实分类: {y_dev[i]} ({CAT_CODE_TO_NAME.get(y_dev[i], y_dev[i])})")
    print(f"  预测分类: {y_pred[i]} ({CAT_CODE_TO_NAME.get(y_pred[i], y_pred[i])})")
    print(f"  是否正确: {'✅' if y_dev[i] == y_pred[i] else '❌'}")
    print()


# 保存模型和向量化器
with open(os.path.join(MODEL_DIR, "text_classifier.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)
print("💾 模型已保存至 models/ 目录")