import requests
import sys
import urllib.parse

# Spring网关的地址和端口
GATEWAY_HOST = "localhost"
GATEWAY_PORT = 9878

# 构建网关URL
gateway_url = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"

# 要调用的service-b的add接口路径（通过网关）
add_endpoint = f"{gateway_url}/api/service-b/add"

def test_url_construction_and_forward(a, b):
    """
    测试URL构造并发送实际请求
    """
    print(f"原始参数: a={a}, b={b}")
    
    # 构造查询参数
    params = {
        "a": a,
        "b": b
    }
    
    # 1. 使用requests的params参数
    print("\n1. 使用requests的params参数:")
    print(f"基础URL: {add_endpoint}")
    print(f"参数: {params}")
    
    try:
        response = requests.get(add_endpoint, params=params)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"请求错误: {str(e)}")
    
    # 2. 手动构造完整URL
    full_url = f"{add_endpoint}?{urllib.parse.urlencode(params)}"
    print(f"\n2. 手动构造完整URL:")
    print(f"完整URL: {full_url}")
    
    try:
        response = requests.get(full_url)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
    except Exception as e:
        print(f"请求错误: {str(e)}")
    
    return full_url

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python test_url_construction.py <a> <b>")
        sys.exit(1)
    
    try:
        a = int(sys.argv[1])
        b = int(sys.argv[2])
    except ValueError:
        print("错误: a和b必须是整数")
        sys.exit(1)
    
    test_url_construction_and_forward(a, b)