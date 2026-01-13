from flask import Flask, request
import os
import atexit
import signal
import base64
import tempfile
import time
from dotenv import load_dotenv
from nacos import NacosClient, DEFAULT_GROUP_NAME
import socket
import torch

# 添加项目根目录到Python路径
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 导入模型训练服务和数据库操作
from model_training.train_model_service import ModelTrainingService
from model_training.database import ModelTrainingDatabase

# 初始化数据库
db = ModelTrainingDatabase()

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)

# 从环境变量获取配置
SERVICE_NAME = os.getenv('SERVICE_NAME', 'service-d')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '5003'))
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

# 模型训练端点
@app.route('/train', methods=['POST'])
def train():
    try:
        # 获取请求数据
        data = request.json
        if not data:
            return {'error': 'No data provided'}, 400
        
        # 提取参数
        model_type = data.get('model_type', 'all')
        hyperparameters = data.get('hyperparameters', {})
        dataset_base64 = data.get('dataset')
        
        if not dataset_base64:
            return {'error': 'No dataset provided'}, 400
        
        # 解码base64数据集并保存到临时文件
        dataset_bytes = base64.b64decode(dataset_base64)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            f.write(dataset_bytes)
            temp_file_path = f.name
        
        try:
            # 创建训练服务实例
            training_service = ModelTrainingService()
            
            # 记录训练开始时间
            start_time = time.time()
            
            # 开始训练
            results = training_service.train(model_type, data_path=temp_file_path, hyperparameters=hyperparameters)
            
            # 记录训练结束时间
            training_time = time.time() - start_time
            
            # 获取设备信息
            device_used = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # 保存训练信息到数据库
            for model_name, accuracy in results.items():
                # 转换模型名称为模型类型
                if model_name == 'CNN':
                    model_type_db = 'cnn'
                elif model_name == 'LogisticRegression':
                    model_type_db = 'lr'
                elif model_name == 'SVM':
                    model_type_db = 'svm'
                elif model_name == 'MLP':
                    model_type_db = 'mlp'
                else:
                    model_type_db = model_type
                
                # 获取模型文件路径和大小
                model_path = os.path.join(training_service.model_save_dir, f"{model_name}.pkl")
                if model_name == 'CNN':
                    model_path = os.path.join(training_service.model_save_dir, "cnn_model.pth")
                
                model_size = os.path.getsize(model_path) if os.path.exists(model_path) else None
                
                # 保存到数据库
                db.save_training_info(
                    model_type=model_type_db,
                    model_name=model_name,
                    accuracy=accuracy,
                    hyperparameters=hyperparameters,
                    model_size=model_size,
                    device_used=device_used,
                    training_time=training_time
                )
            
            # 返回训练结果
            return {
                'status': 'success',
                'model_type': model_type,
                'accuracy': results,
                'training_time': round(training_time, 2),
                'device_used': device_used,
                'model_info': f"Trained {len(results)} models"
            }
        finally:
            # 删除临时文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    except Exception as e:
        return {'error': str(e)}, 500

# 查询模型训练信息端点
@app.route('/model-info', methods=['GET'])
def get_model_info():
    try:
        # 获取查询参数
        model_type = request.args.get('model_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 查询数据库
        results = db.get_training_info(
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            'status': 'success',
            'count': len(results),
            'data': results
        }
    except Exception as e:
        return {'error': str(e)}, 500

# 查询所有模型类型端点
@app.route('/model-types', methods=['GET'])
def get_model_types():
    try:
        # 查询数据库获取所有模型类型
        model_types = db.get_all_model_types()
        
        return {
            'status': 'success',
            'model_types': model_types
        }
    except Exception as e:
        return {'error': str(e)}, 500

# 查询最新模型训练信息端点
@app.route('/latest-model-info', methods=['GET'])
def get_latest_model_info():
    try:
        # 获取查询参数
        model_type = request.args.get('model_type')
        
        # 查询数据库获取最新训练信息
        latest_info = db.get_latest_training_info(model_type=model_type)
        
        return {
            'status': 'success',
            'data': latest_info
        }
    except Exception as e:
        return {'error': str(e)}, 500

# 导出训练记录端点 - 文本格式
@app.route('/export-records', methods=['GET'])
def export_records():
    try:
        # 获取查询参数（可选过滤条件）
        model_type = request.args.get('model_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 查询数据库获取所有训练记录
        records = db.get_training_info(
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        
        # 格式化文本
        text_lines = []
        text_lines.append("=" * 100)
        text_lines.append("模型训练记录导出")
        text_lines.append("=" * 100)
        text_lines.append(f"记录总数: {len(records)}")
        text_lines.append("=" * 100)
        text_lines.append("")
        
        for i, record in enumerate(records, 1):
            text_lines.append(f"记录 #{i}")
            text_lines.append("-" * 50)
            text_lines.append(f"模型类型: {record.get('model_type')}")
            text_lines.append(f"模型名称: {record.get('model_name')}")
            text_lines.append(f"训练日期: {record.get('training_date')}")
            text_lines.append(f"准确率: {record.get('accuracy'):.4f}")
            
            # 处理超参数
            hyperparams = record.get('hyperparameters')
            if hyperparams:
                text_lines.append("超参数:")
                if isinstance(hyperparams, dict):
                    for key, value in hyperparams.items():
                        text_lines.append(f"  - {key}: {value}")
                else:
                    text_lines.append(f"  {hyperparams}")
            else:
                text_lines.append("超参数: 无")
            
            # 其他字段
            if record.get('model_size') is not None:
                text_lines.append(f"模型大小: {record.get('model_size')} 字节")
            if record.get('device_used'):
                text_lines.append(f"使用设备: {record.get('device_used')}")
            if record.get('training_time') is not None:
                text_lines.append(f"训练时间: {record.get('training_time'):.2f} 秒")
            
            text_lines.append("")
        
        # 连接所有行
        text_content = "\n".join(text_lines)
        
        # 返回文本响应
        return app.response_class(
            response=text_content,
            status=200,
            mimetype='text/plain'
        )
    except Exception as e:
        return {'error': str(e)}, 500

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
