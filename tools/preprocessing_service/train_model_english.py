# train_model_english.py
import os
import sys
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter

# 确保可以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)



import pandas as pd
# 配置路径
TRAIN_DATA_PATH = "dict/news-categories/data/news_data.csv"
MODEL_DIR = "models/english_news"
os.makedirs(MODEL_DIR, exist_ok=True)

# 加载英文新闻数据集
print("📂 加载英文新闻训练数据...")
try:
    df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"✅ 成功加载数据集，共 {len(df)} 条样本")
    
    # 提取特征和标签
    X = df['title'].astype(str).tolist()  # 使用标题作为特征
    y = df['topic'].tolist()  # 使用topic作为分类标签
    
    print(f"🏷️  标签类别数: {len(set(y))}")
    print("🔍 训练集标签分布:", Counter(y).most_common(10))
    
    # 分割训练集和验证集
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 训练集: {len(X_train)} 条样本，验证集: {len(X_dev)} 条样本")
    print("🔍 验证集标签分布:", Counter(y_dev).most_common(10))
    
except Exception as e:
    print(f"❌ 加载数据集失败: {e}")
    sys.exit(1)

# 使用 sklearn 内置英文处理
print("🔄 正在向量化英文文本...")
vectorizer = TfidfVectorizer(
    lowercase=True,                # 转小写
    stop_words='english',          # 使用内置英文停用词
    ngram_range=(1, 2),            # 加入 bigram 提升语义捕捉
    max_features=15000,            # 增加特征维度
    strip_accents='unicode',       # 移除重音符号
    token_pattern=r'\b[a-zA-Z]{2,}\b'  # 只保留长度>=2的纯字母词
)

X_train_vec = vectorizer.fit_transform(X_train)
X_dev_vec = vectorizer.transform(X_dev)
print(f"📐 特征向量维度: {X_train_vec.shape}")

# 训练模型（启用类别平衡）
print("🧠 训练英文文本分类模型...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_vec, y_train)

# 在验证集上评估
acc = accuracy_score(y_dev, model.predict(X_dev_vec))
print(f"✅ 验证集准确率: {acc:.4f}")

# 保存模型和向量化器
model_path = os.path.join(MODEL_DIR, "english_news_classifier.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "english_news_vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"💾 英文新闻模型已保存至: {MODEL_DIR}/")
print(f"   - 分类器: {model_path}")
print(f"   - 向量化器: {vectorizer_path}")