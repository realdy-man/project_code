# train_model_service.py
"""
模型训练服务 - 统一接口
根据参数选择训练不同的模型或全部模型
支持：经典模型（LR/SVM/MLP）、CNN模型
"""

import sys
import os
import argparse
import pickle
from typing import Dict, List, Optional, Union

# === 关键：将项目根目录加入 Python 路径 ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === 导入训练模块 ===
# 导入经典模型训练逻辑
from model_training_service.train_classic import main as train_classic
# 导入CNN模型训练逻辑
from model_training_service.train_cnn import train_model as train_cnn

# === 导入数据处理库 ===
import pandas as pd

# === 模型类型定义 ===
MODEL_TYPES = ["all", "classic", "cnn", "lr", "svm", "mlp"]

# === 导入CNN模型的超参数 ===
from model_training_service.train_cnn import (
    MAX_LEN, VOCAB_SIZE, EMBED_DIM, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, LEARNING_RATE
)

# === 导入numpy ===
import numpy as np


class ModelTrainingService:
    """模型训练服务类"""
    
    def __init__(self):
        self.results = {}
        self.model_save_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)
    
    def train_classic_models(self, model_names: Optional[List[str]] = None, data_path: Optional[str] = None) -> Dict[str, float]:
        """
        训练经典模型
        
        Args:
            model_names: 可选，指定要训练的经典模型列表，如 ["lr", "svm"]
                        如果为None，则训练所有经典模型
            data_path: 可选，自定义数据集路径。如果不提供，将使用默认数据集。
        
        Returns:
            模型名称到准确率的映射
        """
        print("[START] 开始训练经典模型...")
        
        # 导入经典模型训练所需的模块
        from model_training_service.train_classic import (
            pd, train_test_split, TfidfVectorizer, tokenize_for_tfidf,
            LogisticRegression, SVC, MLPClassifier, accuracy_score, classification_report
        )
        
        # 加载数据
        if data_path is None:
            data_path = os.path.join(PROJECT_ROOT, "data", "synthetic_thucnews.csv")
            print(f"📂 使用默认数据集: {data_path}")
        else:
            print(f"📂 使用自定义数据集: {data_path}")
        
        # 验证数据集存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # 验证数据集格式
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"数据集格式错误，必须包含'text'和'label'列: {data_path}")
        
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # 数据清洗：确保所有文本都是字符串类型，并且没有空值
        texts = [str(text) if pd.notna(text) else "" for text in texts]
        labels = [str(label) if pd.notna(label) else "unknown" for label in labels]
        
        # 移除空文本样本
        filtered_data = [(text, label) for text, label in zip(texts, labels) if text.strip()]
        texts, labels = zip(*filtered_data) if filtered_data else ([], [])
        
        print(f"[OK] 加载 {len(texts)} 条样本，{len(set(labels))} 个类别")
        
        # 将字符串标签转换为数值型标签（MLP需要数值型标签）
        label_to_id = {label: idx for idx, label in enumerate(sorted(set(labels)))}  # 按字母顺序排序确保一致性
        labels_numeric = [label_to_id[label] for label in labels]
        
        # 划分训练/测试集
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels_numeric, test_size=0.2, random_state=42, stratify=labels_numeric
        )
        
        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_for_tfidf,
            lowercase=False,
            token_pattern=None,
            max_features=5000,
            dtype=np.float32
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # 保存向量化器
        vectorizer_path = os.path.join(self.model_save_dir, "tfidf_vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        
        # 定义模型映射
        model_mapping = {
            "lr": ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
            "svm": ("SVM", SVC(kernel='linear', random_state=42, probability=True)),
            "mlp": ("MLP", MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42, early_stopping=True))
        }
        
        # 确定要训练的模型
        if model_names is None:
            models_to_train = model_mapping.items()
        else:
            models_to_train = [(name, model_mapping[name]) for name in model_names if name in model_mapping]
        
        # 训练模型
        classic_results = {}
        for model_key, (model_name, model) in models_to_train:
            print(f"\n[TRAIN] 训练模型: {model_name}")
            model.fit(X_train_vec, y_train)
            
            # 评估
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            classic_results[model_name] = acc
            
            print(f"[ACC] {model_name} 准确率: {acc:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))
            
            # 保存模型
            model_path = os.path.join(self.model_save_dir, f"{model_name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"[SAVE] 模型已保存: {model_path}")
        
        # 保存经典模型结果
        results_path = os.path.join(self.model_save_dir, "classic_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(classic_results, f)
        
        return classic_results
    
    def train_cnn_model(self, data_path: Optional[str] = None) -> float:
        """
        训练CNN模型
        
        Args:
            data_path: 可选，自定义数据集路径。如果不提供，将使用默认数据集。
        
        Returns:
            CNN模型的准确率
        """
        print("[START] 开始训练CNN模型...")
        
        # 导入CNN模型训练所需的模块
        import torch
        from model_training_service.train_cnn import (
            pd, train_test_split, build_vocab, TextDataset, TextCNN, DataLoader,
            nn, optim, tqdm
        )
        
        # 路径配置
        if data_path is None:
            data_path = os.path.join(PROJECT_ROOT, "data", "synthetic_thucnews.csv")
            print(f"📂 使用默认数据集: {data_path}")
        else:
            print(f"📂 使用自定义数据集: {data_path}")
        
        # 验证数据集存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 验证数据集格式
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"数据集格式错误，必须包含'text'和'label'列: {data_path}")
        
        label_to_id = {label: idx for idx, label in enumerate(df['label'].unique())}
        texts = df['text'].tolist()
        labels = [label_to_id[label] for label in df['label'].tolist()]
        
        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 构建词汇表
        vocab = build_vocab(X_train, max_size=VOCAB_SIZE)
        
        # 保存词汇表
        with open(os.path.join(self.model_save_dir, "cnn_vocab.pkl"), "wb") as f:
            pickle.dump((vocab, label_to_id), f)
        
        # 创建数据集
        train_dataset = TextDataset(X_train, y_train, vocab, MAX_LEN)
        test_dataset = TextDataset(X_test, y_test, vocab, MAX_LEN)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # 初始化模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextCNN(len(vocab), EMBED_DIM, NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 训练循环
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch_texts, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
                batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # 评估
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_texts, batch_labels in test_loader:
                    batch_texts, batch_labels = batch_texts.to(device), batch_labels.to(device)
                    outputs = model(batch_texts)
                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            acc = correct / total
            print(f"  Loss: {total_loss/len(train_loader):.4f}, Val Acc: {acc:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(self.model_save_dir, "cnn_model.pth"))
        print("💾 CNN模型已保存!")
        
        return acc
    
    def train(self, model_type: Union[str, int], data_path: Optional[str] = None) -> Dict[str, float]:
        """
        根据参数训练不同的模型
        
        Args:
            model_type: 模型类型，可选值：
                      - 字符串形式：
                        - "all": 训练所有模型（经典模型 + CNN）
                        - "classic": 训练所有经典模型（LR/SVM/MLP）
                        - "cnn": 训练CNN模型
                        - "lr": 只训练Logistic Regression模型
                        - "svm": 只训练SVM模型
                        - "mlp": 只训练MLP模型
                      - 数字形式：
                        - "1" 或 1: CNN模型
                        - "2" 或 2: MLP模型
                        - "3" 或 3: Logistic Regression模型
                        - "4" 或 4: SVM模型
                        - "5" 或 5: 所有经典模型（LR/SVM/MLP）
                        - "6" 或 6: 所有模型（经典模型 + CNN）
            data_path: 可选，自定义数据集路径。如果不提供，将使用默认数据集。
        
        Returns:
            所有训练模型的名称到准确率的映射
        """
        # 数字到字符串的映射
        number_to_model = {
            "1": "cnn",
            "2": "mlp", 
            "3": "lr",
            "4": "svm",
            "5": "classic",
            "6": "all"
        }
        
        # 转换数字参数为对应的字符串
        if model_type in number_to_model:
            model_type = number_to_model[model_type]
        elif str(model_type) in number_to_model:
            model_type = number_to_model[str(model_type)]
        
        if model_type not in MODEL_TYPES:
            raise ValueError(f"无效的模型类型: {model_type}。可选值: {', '.join(MODEL_TYPES)} 或数字 1-6")
        
        all_results = {}
        
        # 训练经典模型
        if model_type in ["all", "classic"]:
            classic_results = self.train_classic_models(data_path=data_path)
            all_results.update(classic_results)
        elif model_type in ["lr", "svm", "mlp"]:
            classic_results = self.train_classic_models([model_type], data_path=data_path)
            all_results.update(classic_results)
        
        # 训练CNN模型
        if model_type in ["all", "cnn"]:
            cnn_acc = self.train_cnn_model(data_path=data_path)
            all_results["CNN"] = cnn_acc
        
        # 保存所有结果
        all_results_path = os.path.join(self.model_save_dir, "all_model_results.pkl")
        with open(all_results_path, "wb") as f:
            pickle.dump(all_results, f)
        
        # 展示结果
        self.display_results(all_results)
        
        return all_results
    
    def display_results(self, results: Dict[str, float]):
        """
        展示训练结果
        
        Args:
            results: 模型名称到准确率的映射
        """
        print("\n[RESULT] 训练结果汇总:")
        print("=" * 40)
        for model_name, acc in results.items():
            print(f"{model_name:<20} | 准确率: {acc:.4f}")
        print("=" * 40)
        
        # 找出最佳模型
        if results:
            best_model = max(results.items(), key=lambda x: x[1])
            print(f"[BEST] 最佳模型: {best_model[0]}，准确率: {best_model[1]:.4f}")


def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="模型训练服务")
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        choices=MODEL_TYPES, 
        default="all",
        help="选择要训练的模型类型"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="指定自定义数据集路径，必须包含'text'和'label'列的CSV文件"
    )
    
    args = parser.parse_args()
    
    # 创建训练服务实例
    training_service = ModelTrainingService()
    
    # 开始训练
    try:
        results = training_service.train(args.model, data_path=args.data)
        return 0
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 添加train_cnn.py中的超参数定义
    from model_training_service.train_cnn import (
        MAX_LEN, VOCAB_SIZE, EMBED_DIM, NUM_CLASSES,
        BATCH_SIZE, EPOCHS, LEARNING_RATE
    )
    # 添加numpy导入
    import numpy as np
    
    sys.exit(main())
