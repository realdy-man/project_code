# test_inference.py
# test_inference.py
import pickle
import sys
import os

# 确保项目根目录在 Python 路径中（便于 pickle 找到自定义函数）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 显式导入 utils 中的分词函数（pickle 反序列化时需要）
from utils import tokenize_for_tfidf, chinese_tokenize


def predict_text_category(
    text: str,
    model_path: str=os.path.join(os.path.dirname(__file__), './models/text_classifier.pkl'),
    vectorizer_path: str=os.path.join(os.path.dirname(__file__), './models/tfidf_vectorizer.pkl'),
    use_toutiao_map: bool = True
) -> str:
    """
    使用指定的模型和向量化器对单条文本进行分类预测。
    
    参数:
        text (str): 待预测的输入文本
        model_path (str): 分类模型 .pkl 文件的完整路径
        vectorizer_path (str): TF-IDF 向量化器 .pkl 文件的完整路径
        use_toutiao_map (bool): 是否使用头条新闻数据集的分类映射
        
    返回:
        str: 预测的类别标签（文字描述）
    """
    # 头条新闻数据集的分类映射
    toutiao_label_map = {
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
    
    # 加载模型和向量化器
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    # 向量化（注意：transform 接受列表）
    X = vectorizer.transform([text])
    
    # 预测并返回结果（取第一个也是唯一一个）
    pred = model.predict(X)[0]
    
    # 根据参数选择使用哪种映射
    if use_toutiao_map:
        return toutiao_label_map.get(str(pred), str(pred))
    else:
        # 兼容旧的iflytek映射
        import json
        from utils import load_stopwords
        labels_path = os.path.join(project_root, "dict", "iflytek_public", "labels.json")
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [json.loads(line) for line in f]
        label_map = {label["label"]: label["label_des"] for label in labels}
        return label_map.get(str(pred), str(pred))


# ===== 示例用法 =====
if __name__ == "__main__":
    # 构建默认模型路径（可根据需要替换为其他路径）
    model_path = os.path.join(project_root, "models", "text_classifier.pkl")
    vectorizer_path = os.path.join(project_root, "models", "tfidf_vectorizer.pkl")

    test_texts = [
        "八百标兵奔北坡，北坡炮兵并排跑",
        "hello is a good man",
        "华为发布新一代AI芯片，性能提升200%"
    ]

    for text in test_texts:
        try:
            pred = predict_text_category(text, model_path, vectorizer_path)
            tokens = chinese_tokenize(text)
            print(f"文本: {text}")
            print(f"分词结果: {tokens}")
            print(f"预测类别: {pred}\n")
        except Exception as e:
            print(f"处理文本 '{text}' 时出错: {e}\n")