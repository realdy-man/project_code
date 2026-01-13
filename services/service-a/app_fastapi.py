from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import asyncio
from v2.nacos import NacosNamingService, ClientConfigBuilder, RegisterInstanceParam
import socket

# 加载环境变量
load_dotenv()

# 创建FastAPI应用
app = FastAPI(title="Service A", version="1.0")

# 从环境变量获取配置
SERVICE_NAME = os.getenv('SERVICE_NAME')
SERVICE_PORT = int(os.getenv('SERVICE_PORT'))
NACOS_SERVER_ADDR = os.getenv('NACOS_SERVER_ADDR')
NACOS_NAMESPACE = os.getenv('NACOS_NAMESPACE')

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

HOST_IP = get_host_ip()

# 全局Nacos客户端实例
naming_client = None

# 初始化Nacos客户端并注册服务
async def register_to_nacos():
    try:
        # 构建客户端配置
        client_config = (ClientConfigBuilder()
                         .server_address(NACOS_SERVER_ADDR)
                         .log_level('INFO')
                         .build())
        
        # 创建命名服务客户端
        client = await NacosNamingService.create_naming_service(client_config)
        
        # 注册服务实例（ephemeral=False实现持久化）
        register_param = RegisterInstanceParam(
            service_name=SERVICE_NAME,
            group_name='DEFAULT_GROUP',
            ip=HOST_IP,
            port=SERVICE_PORT,
            weight=1.0,
            cluster_name='DEFAULT',
            metadata={'version': '1.0'},
            enabled=True,
            healthy=True,
            ephemeral=False  # 设置为False实现持久化
        )
        
        await client.register_instance(request=register_param)
        print(f'Successfully registered {SERVICE_NAME} to Nacos')
        
        return client
    except Exception as e:
        print(f'Failed to register {SERVICE_NAME} to Nacos: {e}')
        return None

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    global naming_client
    naming_client = await register_to_nacos()

# 健康检查端点
@app.get("/health")
async def health_check():
    return {
        'status': 'UP',
        'service': SERVICE_NAME,
        'port': SERVICE_PORT
    }

# 服务发现端点
@app.get("/discover/{service_name}")
async def discover_service(service_name: str):
    global naming_client
    if naming_client:
        try:
            instances = await naming_client.list_instances(
                service_name=service_name,
                group_name='DEFAULT_GROUP',
                healthy_only=True
            )
            return {
                'service_name': service_name,
                'instances': [{"ip": ins.ip, "port": ins.port} for ins in instances]
            }
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
    return JSONResponse(content={"error": "Failed to connect to Nacos"}, status_code=500)

# 测试路由：加法运算
@app.get("/add")
async def add(a: int, b: int):
    return {'result': a + b}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=SERVICE_PORT)