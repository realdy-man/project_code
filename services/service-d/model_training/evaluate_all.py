# evaluate_all.py
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_DIR = "models"

# 加载经典模型结果
with open(os.path.join(MODEL_DIR, "classic_results.pkl"), "rb") as f:
    classic_results = pickle.load(f)

# 加载 CNN 结果（从 train_cnn.py 的输出中提取，或手动记录）
cnn_acc = 0.985  # 替换为你实际运行得到的值
all_results = {**classic_results, "CNN (MLP)": cnn_acc}


# 绘图
plt.figure(figsize=(10, 6))
sns.barplot(x=list(all_results.keys()), y=list(all_results.values()))
plt.title("模型准确率对比")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(all_results.values()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"))
plt.show()

print("📊 模型性能对比图已保存至 models/model_comparison.png")