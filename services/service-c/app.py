from flask import Flask, request, jsonify
import os
import atexit
import signal
from dotenv import load_dotenv
from nacos import NacosClient, DEFAULT_GROUP_NAME
import socket
import pandas as pd
import numpy as np
import tempfile
from werkzeug.utils import secure_filename

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)

# 从环境变量获取配置
SERVICE_NAME = os.getenv('SERVICE_NAME', 'service-c')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '5002'))
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

# 数据集分析函数
def analyze_dataset(df):
    """
    分析数据集的关键特性：文本平均长度、类别数、语言
    兼容不同格式的CSV文件
    """
    # 处理不同格式的CSV文件
    if len(df.columns) == 1:  # 单列格式，文本+类别用。分隔
        # 定义安全拆分文本和标签的函数
        def split_text_label(row):
            """安全拆分文本和标签，处理无分隔符/空值的情况"""
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
            
            # 兜底处理
            if text == "":
                text = "空文本"
            if label == "":
                label = "未知类别"
            
            return text, label
        
        # 应用拆分函数
        df[["text", "label"]] = df.iloc[:, 0].apply(
            lambda x: pd.Series(split_text_label(x))
        )
    elif len(df.columns) >= 2:  # 多列格式，第一列为文本，第二列为标签
        df = df.rename(columns={df.columns[0]: "text", df.columns[1]: "label"})
    else:
        raise ValueError("数据集格式不正确，无法解析")
    
    # 过滤掉纯空的文本行
    df = df[df["text"] != "空文本"].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("数据集无有效文本内容，请检查数据格式！")
    
    # 计算文本平均长度
    avg_text_length = np.mean(df["text"].apply(lambda x: len(x) if pd.notna(x) else 0))
    
    # 统计类别数量（排除“未知类别”）
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
        "total_valid_samples": len(df),
        "unknown_label_count": len(df[df["label"] == "未知类别"])
    }
    return dataset_features

# 模型推荐函数
def recommend_model(dataset_features):
    """
    根据数据集特性推荐最优模型
    规则基于：文本平均长度、类别数、语言
    """
    avg_len = dataset_features["avg_text_length"]
    label_num = dataset_features["label_count"]
    lang = dataset_features["language"]
    total_samples = dataset_features["total_valid_samples"]
    
    # 预定义模型适用规则
    rules = [
        # 规则1：样本少+短文本+少类别 → 逻辑回归/SVM
        {
            "condition": total_samples < 1000 and avg_len < 50 and label_num <= 5,
            "models": ["逻辑回归", "SVM"],
            "reason": "样本量少+短文本+少类别，传统机器学习模型高效且不易过拟合"
        },
        # 规则2：样本多+长文本+多类别 → CNN
        {
            "condition": total_samples >= 1000 and avg_len >= 50 and label_num > 5,
            "models": ["CNN"],
            "reason": "大样本+长文本+多类别，CNN更擅长捕捉文本局部特征"
        },
        # 规则3：中文文本 → CNN/MLP
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

# 文件上传和模型推荐API
@app.route('/recommend_model', methods=['POST'])
def recommend_model_api():
    """
    接收CSV文件，分析数据集特性并推荐模型
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return jsonify({"error": "请上传CSV格式的数据集文件"}), 400
        
        file = request.files['file']
        
        # 检查文件是否为空
        if file.filename == '':
            return jsonify({"error": "请选择要上传的文件"}), 400
        
        # 检查文件格式（更宽容的判断方式，处理网关转发的情况）
        filename = file.filename
        if not (filename and (filename.lower().endswith('.csv') or '.csv' in filename.lower())):
            return jsonify({"error": "请上传CSV格式的文件"}), 400
        
        # 保存临时文件
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp:
            file.save(temp.name)
            temp_path = temp.name
        
        try:
            # 读取CSV文件
            df = pd.read_csv(temp_path)
            
            # 分析数据集特性
            dataset_features = analyze_dataset(df)
            
            # 推荐模型
            recommendation = recommend_model(dataset_features)
            
            return jsonify({
                "status": "success",
                "recommendation": recommendation
            })
        finally:
            # 删除临时文件
            os.unlink(temp_path)
            
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"处理失败: {type(e).__name__} - {str(e)}"}), 500

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
