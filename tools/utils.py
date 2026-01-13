# utils.py
import os
import re
import jieba

jieba.setLogLevel("ERROR")

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# === 加载用户词典 ===
USER_DICT_PATH = os.path.join(current_dir, "dict/user_dict.txt")
if os.path.exists(USER_DICT_PATH):
    jieba.load_userdict(USER_DICT_PATH)
    print(f"[OK] 已加载用户词典: {USER_DICT_PATH}")

# === 加载停用词表 ===
def load_stopwords():
    candidate_paths = [
        os.path.join(current_dir, "dict/hit_stopwords.txt"),
        os.path.join(current_dir, "data/hit_stopwords.txt"),
        os.path.join(current_dir, "hit_stopwords.txt")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return {line.strip() for line in f if line.strip()}
            except Exception as e:
                print(f"⚠️ 读取停用词失败 ({path}): {e}")
    print("⚠️ 使用内置精简停用词")
    return {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一'}

HIT_STOPWORDS = load_stopwords()

# === 文本清洗与分词 ===
def clean_chinese_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def chinese_tokenize(text):
    cleaned = clean_chinese_text(text)
    if not cleaned:
        return []
    words = jieba.lcut(cleaned, cut_all=False)
    return [w for w in words if len(w) >= 2 and w not in HIT_STOPWORDS]

# === 供 TfidfVectorizer 使用的 tokenizer（关键！必须在此定义）===
def tokenize_for_tfidf(text):
    return chinese_tokenize(text)