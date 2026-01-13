# train_classic.py
"""
模型训练服务 - 经典机器学习模型（LR / SVM / MLP）
支持：逻辑回归、SVM、多层感知机
输入：CSV 格式文本数据（列: text, label）
输出：训练好的模型 + 向量化器 + 性能报告
"""

import sys
import os

# === 关键：将项目根目录加入 Python 路径 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_D_ROOT = os.path.dirname(CURRENT_DIR)
if SERVICE_D_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_D_ROOT)

# === 导入公共工具 ===
from model_training.utils import tokenize_for_tfidf

# === 其他依赖 ===
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# === 配置路径（使用相对路径，确保本地和容器内一致） ===
MODEL_SAVE_DIR = os.path.join(SERVICE_D_ROOT, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

def main(data_path=None, hyperparameters=None):
    print("🚀 开始经典模型训练服务...")
    
    # === 1. 加载数据 ===
    if data_path is None:
        raise ValueError("必须提供数据路径")
    else:
        print(f"📂 加载自定义数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    print(f"✅ 加载 {len(texts)} 条样本，{len(set(labels))} 个类别")

    # === 2. 划分训练/测试集 ===
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    print(f"📊 训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # === 3. TF-IDF 向量化 ===
    print("🔄 执行 TF-IDF 向量化...")
    
    # === 4. 超参数处理 ===
    hyperparameters = hyperparameters or {}
    lr_hyperparams = hyperparameters.get('lr', {}) if isinstance(hyperparameters.get('lr'), dict) else hyperparameters
    svm_hyperparams = hyperparameters.get('svm', {}) if isinstance(hyperparameters.get('svm'), dict) else hyperparameters
    mlp_hyperparams = hyperparameters.get('mlp', {}) if isinstance(hyperparameters.get('mlp'), dict) else hyperparameters
    
    # 定义默认超参数
    default_lr_params = {"max_iter": 1000, "random_state": 42, "C": 1.0}
    default_svm_params = {"kernel": 'linear', "random_state": 42, "probability": True}
    default_mlp_params = {"hidden_layer_sizes": (128,), "max_iter": 500, "random_state": 42, "early_stopping": True}
    
    # 合并默认超参数和用户提供的超参数
    lr_params = {**default_lr_params, **lr_hyperparams}
    svm_params = {**default_svm_params, **svm_hyperparams}
    mlp_params = {**default_mlp_params, **mlp_hyperparams}
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_for_tfidf,
        lowercase=False,
        token_pattern=None,  # 必须设置为 None，否则 tokenizer 不生效
        max_features=5000,
        dtype=np.float32
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"✅ 向量维度: {X_train_vec.shape[1]}")

    # 保存向量化器（供预测服务使用）
    vectorizer_path = os.path.join(MODEL_SAVE_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"💾 向量化器已保存: {vectorizer_path}")

    # === 4. 定义模型 ===
    models = {
        "LogisticRegression": LogisticRegression(**lr_params),
        "SVM": SVC(**svm_params),
        "MLP": MLPClassifier(**mlp_params)
    }

    # === 5. 训练与评估 ===
    results = {}
    for name, model in models.items():
        print(f"\n🧠 训练模型: {name}")
        model.fit(X_train_vec, y_train)
        
        # 预测
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        
        print(f"✅ {name} 准确率: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # 保存模型
        model_path = os.path.join(MODEL_SAVE_DIR, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 模型已保存: {model_path}")

    # === 6. 保存整体结果 ===
    results_path = os.path.join(MODEL_SAVE_DIR, "classic_results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n🏆 所有经典模型训练完成！结果已保存至: {results_path}")
    print("📊 最终准确率:")
    for name, acc in results.items():
        print(f"  - {name}: {acc:.4f}")

if __name__ == "__main__":
    main()