from flask import Flask, request, jsonify
import os
import sys

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入tools.utils模块
try:
    from tools import utils
    sys.modules['utils'] = utils
except ImportError:
    print("Warning: Failed to import tools.utils, will use local implementation")
import atexit
import signal
import re
import pickle
import jieba
from dotenv import load_dotenv
from nacos import NacosClient, DEFAULT_GROUP_NAME
import socket


# 加载环境变量
load_dotenv()

# 设置jieba日志级别
jieba.setLogLevel(20)

# 语言检测函数
def detect_language(text):
    """
    检测文本语言（中文/英文）
    
    参数:
        text (str): 待检测的文本
    
    返回:
        str: 'zh' 表示中文，'en' 表示英文
    """
    if not text or not isinstance(text, str):
        return 'zh'  # 默认返回中文
    
    # 直接检查文本中是否包含中文字符
    if any('\u4e00' <= char <= '\u9fa5' for char in text):
        return 'zh'
    else:
        return 'en'

# 中文类别映射表
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

# 创建Flask应用
app = Flask(__name__)

# 从环境变量获取配置
SERVICE_NAME = os.getenv('SERVICE_NAME', 'service-a')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '5000'))
NACOS_SERVER_ADDR = os.getenv('NACOS_SERVER_ADDR', 'localhost:8848')
NACOS_NAMESPACE = os.getenv('NACOS_NAMESPACE', '')
# 新增环境变量，用于指定注册到Nacos的IP地址
# 如果未指定，默认使用127.0.0.1，解决Docker容器IP隔离问题
REGISTER_IP = os.getenv('REGISTER_IP', '127.0.0.1')

# 获取当前容器IP
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((NACOS_SERVER_ADDR.split(':')[0], int(NACOS_SERVER_ADDR.split(':')[1])))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return '127.0.0.1'

# 使用REGISTER_IP作为注册IP，如果未配置则使用127.0.0.1
HOST_IP = REGISTER_IP

# 全局变量保存Nacos客户端实例
naming_client = None

# 文本清洗与分词函数
def clean_chinese_text(text):
    """
    清洗中文文本
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+|\S+@\S+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def chinese_tokenize(text):
    """
    中文分词
    """
    cleaned = clean_chinese_text(text)
    if not cleaned:
        return []
    words = jieba.lcut(cleaned, cut_all=False)
    # 简单停用词过滤
    stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一'}
    return [w for w in words if len(w) >= 2 and w not in stopwords]

# 供TfidfVectorizer使用的tokenizer
def tokenize_for_tfidf(text):
    """
    供TfidfVectorizer使用的tokenizer
    """
    return chinese_tokenize(text)

# 文本预测函数
def predict_text(
    text: str,
    model_path: str = None,
    vectorizer_path: str = None
) -> dict:
    """
    使用指定的模型和向量化器对单条文本进行分类预测
    
    参数:
        text (str): 待预测的输入文本
        model_path (str): 分类模型.pkl文件的完整路径
        vectorizer_path (str): TF-IDF向量化器.pkl文件的完整路径
    
    返回:
        dict: 包含预测结果的字典
    """
    # 检测文本语言
    lang = detect_language(text)
    
    # 设置默认模型和向量器路径
    if model_path is None or vectorizer_path is None:
        # 检测是否在容器内运行（通过检查/app目录是否存在）
        if os.path.exists('/app'):
            # 容器环境
            base_path = '/app/models'
        else:
            # 本地测试环境
            base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
        
        if lang == 'zh':
            model_path = model_path or f'{base_path}/text_classifier.pkl'
            vectorizer_path = vectorizer_path or f'{base_path}/tfidf_vectorizer.pkl'
        else:  # en
            model_path = model_path or f'{base_path}/english_news_classifier.pkl'
            vectorizer_path = vectorizer_path or f'{base_path}/english_news_vectorizer.pkl'
    
    try:
        # 加载模型和向量化器
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        
        # 向量化（注意：transform接受列表）
        X = vectorizer.transform([text])
        
        # 预测并返回结果
        pred = model.predict(X)[0]
        
        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0].tolist()
            classes = model.classes_.tolist()
        else:
            proba = None
            classes = model.classes_.tolist()
        
        # 获取中文类别名称（如果是中文分类）
        prediction_name = None
        if lang == 'zh' and str(pred) in toutiao_label_map:
            prediction_name = toutiao_label_map[str(pred)]
        
        # 返回结果
        return {
            'success': True,
            'language': lang,
            'prediction': str(pred),
            'prediction_name': prediction_name,
            'probability': proba,
            'classes': classes,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'language': lang,
            'prediction': None,
            'prediction_name': None,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path
        }

# 初始化Nacos客户端并注册服务
def register_to_nacos():
    global naming_client
    try:
        # 创建同步Nacos客户端
        naming_client = NacosClient(NACOS_SERVER_ADDR, namespace=NACOS_NAMESPACE)
        
        # 注册服务实例（ephemeral=False实现持久化）
        naming_client.add_naming_instance(
            service_name=SERVICE_NAME,
            ip=HOST_IP,
            port=SERVICE_PORT,
            weight=1.0,
            cluster_name='DEFAULT',
            metadata={'version': '1.0'},
            enable=True,
            healthy=True,
            ephemeral=False,  # 设置为False实现持久化
            group_name='DEFAULT_GROUP'
        )
        print(f'Successfully registered {SERVICE_NAME} to Nacos')
        
        return naming_client
    except Exception as e:
        print(f'Failed to register {SERVICE_NAME} to Nacos: {e}')
        return None

# 健康检查端点
@app.route('/health')
def health_check():
    return {
        'status': 'UP',
        'service': SERVICE_NAME,
        'port': SERVICE_PORT
    }

# 服务发现端点
@app.route('/discover/<service_name>')
def discover_service(service_name):
    naming_client = register_to_nacos()
    if naming_client:
        try:
            result = naming_client.list_naming_instance(
                service_name=service_name,
                group_name='DEFAULT_GROUP',
                healthy_only=False
            )
            return {
                'service_name': service_name,
                'instances': [{"ip": ins["ip"], "port": ins["port"]} for ins in result["hosts"]]
            }
        except Exception as e:
            return {"error": str(e)}, 500
    return {"error": "Failed to connect to Nacos"}, 500

# 测试路由：加法运算
@app.route('/add')
def add():
    a = request.args.get('a', type=int)
    b = request.args.get('b', type=int)
    if a is None or b is None:
        return {'error': 'Missing parameters a or b'}, 400
    return {'result': a + b}

# 测试语言检测
@app.route('/test_language')
def test_language():
    text = request.args.get('text', '')
    result = detect_language(text)
    has_chinese = any('\u4e00' <= char <= '\u9fa5' for char in text)
    return {
        'text': text,
        'detected_language': result,
        'has_chinese_characters': has_chinese
    }

# 文本预测API端点
@app.route('/predict_text', methods=['POST'])
def predict_text_endpoint():
    """
    文本预测API端点
    
    请求体格式：
    {
        "text": "待预测的文本",
        "model_path": "可选，模型文件路径",
        "vectorizer_path": "可选，向量器文件路径"
    }
    
    返回格式：
    {
        "success": true/false,
        "language": "zh/en",
        "prediction": "预测结果",
        "probability": [预测概率列表],
        "classes": [类别列表],
        "model_path": "使用的模型路径",
        "vectorizer_path": "使用的向量器路径",
        "error": "错误信息（如果有）"
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        # 获取文本参数
        text = data.get('text')
        if not text or not isinstance(text, str):
            return jsonify({'success': False, 'error': 'Invalid or missing text parameter'}), 400
        
        # 获取模型和向量器路径参数（可选）
        model_path = data.get('model_path')
        vectorizer_path = data.get('vectorizer_path')
        
        # 调用预测函数
        result = predict_text(text, model_path, vectorizer_path)
        
        # 返回结果
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# 清理函数：注销服务

def deregister_service():
    global naming_client
    if naming_client:
        try:
            naming_client.remove_naming_instance(
                service_name=SERVICE_NAME,
                ip=HOST_IP,
                port=SERVICE_PORT,
                cluster_name='DEFAULT',
                ephemeral=False,
                group_name='DEFAULT_GROUP'
            )
            print(f'Successfully deregistered {SERVICE_NAME} from Nacos')
        except Exception as e:
            print(f'Failed to deregister {SERVICE_NAME} from Nacos: {e}')

# 信号处理函数
def signal_handler(signum, frame):
    print(f'Received signal {signum}, shutting down...')
    deregister_service()
    exit(0)

if __name__ == '__main__':
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # 注册退出清理函数
    atexit.register(deregister_service)
    
    # 启动时注册到Nacos
    register_to_nacos()
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=SERVICE_PORT, debug=False)
