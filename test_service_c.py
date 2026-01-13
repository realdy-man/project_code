import requests
import os
import argparse
import json

# 测试文件路径
TEST_FILE_PATH = "f:\AAA_JIQIXUEXI\project_code\output_sample.csv"

# 直接访问的API端点
API_URL = "http://127.0.0.1:5002/recommend_model"

# Spring Cloud Gateway 的地址（参考test_service-a.py）
SPRING_GATEWAY_URL = "http://localhost:8080"
# 通过网关访问的API端点
GATEWAY_API_URL = f"{SPRING_GATEWAY_URL}/api/service-c/recommend_model"

# 检查测试文件是否存在
if not os.path.exists(TEST_FILE_PATH):
    print(f"测试文件不存在: {TEST_FILE_PATH}")
    exit(1)

def test_direct_connection():
    """直接连接service-c进行测试"""
    print("\n=== 开始直接连接测试 ===")
    
    # 构造文件上传请求
    try:
        with open(TEST_FILE_PATH, 'rb') as file:
            files = {'file': (os.path.basename(TEST_FILE_PATH), file, 'text/csv')}
            response = requests.post(API_URL, files=files)
        
        # 输出详细的响应信息
        print(f"HTTP状态码: {response.status_code}")
        print(f"响应头:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        print(f"响应内容:")
        
        # 尝试解析JSON响应，只在状态码为200且内容类型为JSON时才解析
        try:
            if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
                response_json = response.json()
                print(json.dumps(response_json, indent=2, ensure_ascii=False))
            else:
                # 打印原始响应内容
                print(response.text[:500] + '...' if len(response.text) > 500 else response.text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，打印原始响应内容
            print(response.text[:500] + '...' if len(response.text) > 500 else response.text)
        
        # 添加额外的调试信息
        print(f"请求URL: {response.url}")
        print(f"请求方法: POST")
        
        if response.status_code == 200:
            print("✅ 直接连接测试成功！")
            return True
        else:
            print(f"❌ 直接连接测试失败，HTTP状态码: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ 直接连接请求异常: {e}")
        return False
    except Exception as e:
        print(f"❌ 直接连接测试过程中发生错误: {e}")
        return False

def test_gateway_connection():
    """通过网关连接service-c进行测试"""
    print("\n=== 开始网关连接测试 ===")
    
    # 构造文件上传请求
    try:
        print(f"网关API地址: {GATEWAY_API_URL}")
        print(f"测试文件路径: {TEST_FILE_PATH}")
        
        with open(TEST_FILE_PATH, 'rb') as file:
            # 构造文件上传数据（明确指定文件名和内容类型）
            files = {'file': (os.path.basename(TEST_FILE_PATH), file.read(), 'text/csv')}
            
            # 添加调试信息
            print(f"上传文件名: {os.path.basename(TEST_FILE_PATH)}")
            
            # 发送请求
            response = requests.post(
                GATEWAY_API_URL, 
                files=files, 
                timeout=10
            )
        
        # 输出详细的响应信息
        print(f"HTTP状态码: {response.status_code}")
        print(f"响应头:")
        for key, value in response.headers.items():
            print(f"  {key}: {value}")
        
        print(f"响应内容:")
        
        # 尝试解析JSON响应，只在状态码为200且内容类型为JSON时才解析
        try:
            if response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', ''):
                response_json = response.json()
                print(json.dumps(response_json, indent=2, ensure_ascii=False))
            else:
                # 打印原始响应内容
                print(response.text[:500] + '...' if len(response.text) > 500 else response.text)
        except json.JSONDecodeError:
            # 如果JSON解析失败，打印原始响应内容
            print(response.text[:500] + '...' if len(response.text) > 500 else response.text)
        
        # 添加额外的调试信息
        print(f"请求URL: {response.url}")
        print(f"请求方法: POST")
        
        if response.status_code == 200:
            print("✅ 网关连接测试成功！")
            return True
        else:
            print(f"❌ 网关连接测试失败，HTTP状态码: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ 网关连接请求异常: {e}")
        print("可能原因：网关服务未启动或配置不正确")
        return False
    except Exception as e:
        print(f"❌ 网关连接测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="测试service-c的模型推荐功能")
    parser.add_argument('--mode', choices=['direct', 'gateway', 'both'], 
                      default='both', help="测试模式: direct(直接连接), gateway(网关连接), both(两种方式都测试)")
    
    args = parser.parse_args()
    
    print("开始测试service-c的模型推荐功能...")
    
    # 根据选择的模式执行测试
    if args.mode in ['direct', 'both']:
        test_direct_connection()
    
    if args.mode in ['gateway', 'both']:
        test_gateway_connection()
    
    print("\n=== 测试完成 ===")
