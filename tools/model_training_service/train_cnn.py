# model_training_service/train_cnn.py

import sys
import os

# === 关键：添加项目根目录到路径 ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === 导入公共工具 ===
from utils import tokenize_for_tfidf

# === 其他依赖 ===
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# === 路径配置 ===
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "synthetic_thucnews.csv")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# === 超参数 ===
MAX_LEN = 32
VOCAB_SIZE = 5000
EMBED_DIM = 128
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# === 构建词汇表 ===
def build_vocab(texts, max_size=VOCAB_SIZE):
    word_counts = Counter()
    for text in texts:
        tokens = tokenize_for_tfidf(text)
        word_counts.update(tokens)
    
    # 保留高频词 + 特殊 token
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

# === 数据集类 ===
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = tokenize_for_tfidf(self.texts[idx])
        # 转为索引
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        # 截断或填充
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# === CNN 模型 ===
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters=100, kernel_sizes=[3,4,5]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.unsqueeze(1)     # [batch, 1, seq_len, embed_dim]
        xs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # [batch, num_filters, H, 1]
            pooled = torch.max(conv_out.squeeze(-1), dim=2)[0]  # [batch, num_filters]
            xs.append(pooled)
        x = torch.cat(xs, dim=1)  # [batch, num_filters * len(kernel_sizes)]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# === 训练函数 ===
def train_model(data_path=None):
    print("🚀 开始 CNN 模型训练...")
    
    # 加载数据
    if data_path is None:
        data_path = DATA_PATH
        print(f"📂 加载默认数据: {data_path}")
    else:
        print(f"📂 加载自定义数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    df = pd.read_csv(data_path)
    label_to_id = {label: idx for idx, label in enumerate(df['label'].unique())}
    texts = df['text'].tolist()
    labels = [label_to_id[label] for label in df['label'].tolist()]
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 构建词汇表
    vocab = build_vocab(X_train, max_size=VOCAB_SIZE)
    print(f"✅ 词汇表大小: {len(vocab)}")
    
    # 保存词汇表（供预测使用）
    with open(os.path.join(MODEL_SAVE_DIR, "cnn_vocab.pkl"), "wb") as f:
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
        
        # 简单评估
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
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "cnn_model.pth"))
    print("💾 CNN 模型已保存!")

if __name__ == "__main__":
    train_model()