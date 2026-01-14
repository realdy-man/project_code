from flask import Flask, request, Response
import os
import atexit
import signal
from dotenv import load_dotenv
from nacos import NacosClient, DEFAULT_GROUP_NAME
import socket
import requests
import json
import csv
import io
import random
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 创建Flask应用
app = Flask(__name__)

# 配置文件上传
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB
app.config['UPLOAD_EXTENSIONS'] = ['.txt']  # 允许的文件扩展名

# 从环境变量获取配置
SERVICE_NAME = os.getenv('SERVICE_NAME', 'service-b')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', '5001'))
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

# 服务发现和负载均衡函数
def get_service_instance_url(service_name):
    """
    通过Nacos发现指定服务的实例，并随机选择一个（简单负载均衡）
    
    参数:
        service_name (str): 服务名称
    
    返回:
        str: 选中实例的完整URL（格式：http://ip:port）；如果发现失败则返回None
    """
    naming_client = register_to_nacos()
    if naming_client:
        try:
            # 发现服务实例
            result = naming_client.list_naming_instance(
                service_name=service_name,
                group_name='DEFAULT_GROUP',
                healthy_only=True  # 只获取健康实例
            )
            
            # 检查是否有可用实例
            if result and result.get("hosts"):
                # 随机选择一个实例（简单负载均衡）
                instance = random.choice(result["hosts"])
                ip = instance.get("ip")
                port = instance.get("port")
                
                if ip and port:
                    return f"http://{ip}:{port}"
        except Exception as e:
            print(f"Failed to discover service {service_name}: {e}")
    
    return None

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

# 从文件生成文本分类数据集API端点
@app.route('/generate_dataset_from_file', methods=['POST'])
def generate_dataset_from_file():
    """
    从上传的TXT文件生成文本分类数据集API端点
    
    请求：
    multipart/form-data格式，包含file字段（上传的TXT文件）和可选的service_a_url字段
    
    返回：
    CSV文件，包含text和label两列
    """
    try:
        # 检查是否有文件上传
        if 'file' not in request.files:
            return Response(
                response=json.dumps({'success': False, 'error': 'No file uploaded'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        file = request.files['file']
        if file.filename == '':
            return Response(
                response=json.dumps({'success': False, 'error': 'No file selected'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        # 检查文件扩展名
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            return Response(
                response=json.dumps({'success': False, 'error': 'Invalid file type, only TXT files are allowed'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        # 读取文件内容
        try:
            file_content = file.read().decode('utf-8')
            # 按行分割文本
            texts = [line.strip() for line in file_content.split('\n') if line.strip()]
        except Exception as e:
            return Response(
                response=json.dumps({'success': False, 'error': f'Failed to read file: {str(e)}'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        if not texts:
            return Response(
                response=json.dumps({'success': False, 'error': 'File is empty or contains only empty lines'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        # 获取service-a的URL
        service_a_url = request.form.get('service_a_url')
        
        # 始终通过网关连接service-a
        # 使用环境变量或默认值(host.docker.internal适用于Docker Desktop，172.17.0.1是Linux Docker默认网关)
        gateway_host = os.environ.get('GATEWAY_HOST', 'host.docker.internal')
        gateway_port = os.environ.get('GATEWAY_PORT', '8080')
        gateway_url = f'http://{gateway_host}:{gateway_port}'
        service_a_url = f'{gateway_url}/api/service-a/predict_text'
        
        # 记录实际使用的service_a_url
        logger.debug(f"Using service_a_url: {service_a_url}")
        
        # 创建CSV文件
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['text', 'label'])
        
        # 处理每个文本
        for text in texts:
            # 调用service-a的预测API
            try:
                print(f"Calling gateway at {service_a_url} with text: {text}")
                response = requests.post(
                    service_a_url,
                    json={'text': text},
                    headers={'Content-Type': 'application/json'},
                    timeout=10  # 添加超时设置
                )
                print(f"Gateway response status: {response.status_code}")
                print(f"Gateway response content: {response.text}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"Gateway response JSON: {result}")
                    if result.get('success'):
                        label = result.get('prediction_name', '')
                    else:
                        label = 'error'
                else:
                    label = 'error'
            except Exception as e:
                print(f"Error calling gateway: {e}")
                label = 'error'
            
            # 写入CSV
            writer.writerow([text, label])
        
        # 返回CSV文件
        output.seek(0)
        return Response(
            output.getvalue(),
            status=200,
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename="text_dataset.csv"'
            }
        )
        
    except Exception as e:
        error_json = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
        return Response(error_json, status=500, mimetype='application/json')

# 生成文本分类数据集API端点（原有的JSON文本列表方式）
@app.route('/generate_dataset', methods=['POST'])
def generate_dataset():
    """
    生成文本分类数据集API端点
    
    请求体格式：
    {
        "texts": ["文本1", "文本2", "文本3"],  # 待处理的文本列表
        "service_a_url": "http://localhost:5000/predict_text"  # service-a的预测API地址（可选）
    }
    
    返回：
    CSV文件，包含text和label两列
    """
    try:
        # 获取请求数据
        data = request.get_json()
        if not data:
            return Response(
                response=json.dumps({'success': False, 'error': 'No JSON data provided'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        texts = data.get('texts')
        if not texts or not isinstance(texts, list) or len(texts) == 0:
            return Response(
                response=json.dumps({'success': False, 'error': 'Invalid or missing texts parameter'}, ensure_ascii=False),
                status=400,
                mimetype='application/json'
            )
        
        # 获取service-a的URL
        service_a_url = data.get('service_a_url')
        
        # 始终通过网关连接service-a
        # 使用环境变量或默认值(host.docker.internal适用于Docker Desktop，172.17.0.1是Linux Docker默认网关)
        gateway_host = os.environ.get('GATEWAY_HOST', 'host.docker.internal')
        gateway_port = os.environ.get('GATEWAY_PORT', '8080')
        gateway_url = f'http://{gateway_host}:{gateway_port}'
        service_a_url = f'{gateway_url}/api/service-a/predict_text'
        
        # 创建CSV文件
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['text', 'label'])
        
        # 处理每个文本
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue  # 跳过空文本或非字符串
            
            # 调用service-a的预测API
            try:
                response = requests.post(
                    service_a_url,
                    json={'text': text},
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        label = result.get('prediction_name', '')
                    else:
                        label = 'error'
                else:
                    label = 'error'
            except Exception as e:
                label = 'error'
            
            # 写入CSV
            writer.writerow([text, label])
        
        # 返回CSV文件
        output.seek(0)
        return Response(
            output.getvalue(),
            status=200,
            mimetype='text/csv',
            headers={
                'Content-Disposition': 'attachment; filename="text_dataset.csv"'
            }
        )
        
    except Exception as e:
        error_json = json.dumps({'success': False, 'error': str(e)}, ensure_ascii=False)
        return Response(error_json, status=500, mimetype='application/json')

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
