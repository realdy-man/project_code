import pandas as pd
import numpy as np
import os

# --------------------------
# 步骤1：加载并分析数据集特性（修复核心）
# --------------------------
def analyze_dataset(data_path):
    """
    分析数据集的关键特性：文本平均长度、类别数、语言
    增加异常处理，兼容不规范的文本行
    """
    # 读取数据集（你的数据格式：文本+类别，用“。”分隔）
    df = pd.read_csv(data_path, header=None, names=["text_with_label"])

    # 定义安全拆分文本和标签的函数（核心修复）
    def split_text_label(row):
        """安全拆分文本和标签，处理无分隔符/空值的情况"""
        # 先去除空值和空白字符
        if pd.isna(row) or row.strip() == "":
            return "", "未知类别"

        # 按最后一个“。”分割
        parts = row.rsplit("。", 1)
        if len(parts) == 2:  # 有分隔符，正常拆分
            text = parts[0].strip()
            label = parts[1].strip()
        else:  # 无分隔符，全部视为文本，标签设为未知
            text = row.strip()
            label = "未知类别"

        # 兜底：如果文本为空，标签也为空的情况
        if text == "":
            text = "空文本"
        if label == "":
            label = "未知类别"

        return text, label

    # 应用拆分函数，避免索引越界
    df[["text", "label"]] = df["text_with_label"].apply(
        lambda x: pd.Series(split_text_label(x))
    )

    # 过滤掉纯空的文本行（可选，提升分析准确性）
    df = df[df["text"] != "空文本"].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("数据集无有效文本内容，请检查数据格式！")

    # 计算文本平均长度
    avg_text_length = np.mean(df["text"].apply(lambda x: len(x)))

    # 统计类别数量（排除“未知类别”，如果需要包含则去掉这个过滤）
    valid_labels = df[df["label"] != "未知类别"]["label"]
    label_count = valid_labels.nunique() if len(valid_labels) > 0 else 1

    # 判断语言（根据文本中是否包含中文字符）
    def is_chinese(text):
        return any("\u4e00" <= c <= "\u9fff" for c in text)

    # 取第一条有效文本判断语言
    first_text = df["text"].iloc[0]
    language = "中文" if is_chinese(first_text) else "英文"

    # 返回数据集特性
    dataset_features = {
        "avg_text_length": round(avg_text_length, 2),
        "label_count": label_count,
        "language": language,
        "total_valid_samples": len(df),  # 新增：有效样本数，辅助判断
        "unknown_label_count": len(df[df["label"] == "未知类别"])  # 新增：未知类别数
    }
    print("数据集特性分析结果：", dataset_features)
    return dataset_features


# --------------------------
# 步骤2：模型推荐规则
# --------------------------
def recommend_model(dataset_features):
    """
    根据数据集特性推荐最优模型
    规则基于：文本平均长度、类别数、语言
    """
    avg_len = dataset_features["avg_text_length"]
    label_num = dataset_features["label_count"]
    lang = dataset_features["language"]
    total_samples = dataset_features["total_valid_samples"]

    # 预定义模型适用规则（可根据实际训练效果调整）
    rules = [
        # 规则1：样本少+短文本+少类别 → 逻辑回归/SVM（泛化性好，不易过拟合）
        {
            "condition": total_samples < 1000 and avg_len < 50 and label_num <= 5,
            "models": ["逻辑回归", "SVM"],
            "reason": "样本量少+短文本+少类别，传统机器学习模型高效且不易过拟合"
        },
        # 规则2：样本多+长文本+多类别 → CNN（捕捉局部特征）
        {
            "condition": total_samples >= 1000 and avg_len >= 50 and label_num > 5,
            "models": ["CNN"],
            "reason": "大样本+长文本+多类别，CNN更擅长捕捉文本局部特征"
        },
        # 规则3：中文文本 → 优先适配中文预训练的模型（这里简化为CNN/MLP）
        {
            "condition": lang == "中文" and label_num > 0,
            "models": ["CNN", "MLP"],
            "reason": "中文文本适配深度学习模型的字符/词嵌入，效果优于传统模型"
        },
        # 默认规则
        {
            "condition": True,
            "models": ["MLP"],
            "reason": "通用场景下MLP兼容性较好，适配各类文本长度和类别数"
        }
    ]

    # 匹配规则，返回第一个满足条件的推荐
    for rule in rules:
        if rule["condition"]:
            return {
                "recommended_models": rule["models"],
                "reason": rule["reason"],
                "dataset_features": dataset_features
            }

def main(data_path):
    try:
        # 1. 分析数据集特性
        dataset_features = analyze_dataset(data_path)

        # 2. 推荐模型
        recommendation = recommend_model(dataset_features)

        # 3. 输出结果
        print("\n===== 模型推荐结果 =====")
        print(f"推荐模型：{recommendation['recommended_models']}")
        print(f"推荐理由：{recommendation['reason']}")
    except FileNotFoundError:
        print(f"错误：找不到数据集文件，请检查路径是否正确！路径：{DATA_PATH}")
    except ValueError as e:
        print(f"数据错误：{e}")
    except Exception as e:
        print(f"运行出错：{type(e).__name__} - {e}")

# --------------------------
# 步骤3：服务入口（可封装为API）
# --------------------------
if __name__ == "__main__":
    # 你的数据集路径（对应截图中的synthetic_thucnews.csv）
    DATA_PATH = r"f:\AAA_JIQIXUEXI\my_nlp_project\dict\news-categories\data\news_data.csv"
    # 运行主函数
    main(DATA_PATH)