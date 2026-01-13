from flask import Flask, request
import os
import atexit
import signal
from dotenv import load_dotenv
from nacos import NacosClient, DEFAULT_GROUP_NAME
import socket

# 加载环境变量
load_dotenv()

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
