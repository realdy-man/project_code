import requests
import sys

# Spring网关的地址和端口，默认假设为localhost:8080
GATEWAY_HOST = "localhost"
GATEWAY_PORT = 9878

# 构建网关URL
gateway_url = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

# 要调用的service-b的add接口路径（通过网关）
add_endpoint = f"{gateway_url}/api/service-b/add"

def call_add_service(a, b):
    """
    通过Spring网关调用service-b的add接口
    
    参数:
    a (int): 第一个加数
    b (int): 第二个加数
    
    返回:
    dict: 包含结果或错误信息的字典
    """
    try:
        # 手动构造包含查询参数的完整URL
        import urllib.parse
        params = {
            "a": a,
            "b": b
        }
        full_url = f"{add_endpoint}?{urllib.parse.urlencode(params)}"
        
        # 发送GET请求到网关
        response = requests.get(full_url)
        
        # 解析响应
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            return {
                "error": f"请求失败，状态码: {response.status_code}",
                "message": response.text
            }
    except requests.exceptions.RequestException as e:
        return {
            "error": "网络请求错误",
            "message": str(e)
        }
    except ValueError as e:
        return {
            "error": "响应解析错误",
            "message": str(e)
        }

if __name__ == "__main__":
    # 从命令行参数获取a和b的值
    if len(sys.argv) != 3:
        print("用法: python call_service_b.py <a> <b>")
        print("示例: python call_service_b.py 10 20")
        sys.exit(1)
    
    try:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
    except ValueError:
        print("错误: a和b必须是整数")
        sys.exit(1)
    
    # 调用add服务
    result = call_add_service(a, b)
    
    # 输出结果
    print("调用结果:")
    if "result" in result:
        print(f"{a} + {b} = {result['result']}")
    else:
        print(f"错误: {result.get('message', '未知错误')}")