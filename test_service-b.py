#!/usr/bin/env python3
import requests
import os
import sys

# 配置
FILE_PATH = r'f:\AAA_JIQIXUEXI\project_code\test_texts.txt'  # 使用原始字符串避免反斜杠转义问题
# 通过网关连接service-b服务
GATEWAY_URL = 'http://localhost:8080'  # 网关地址（与test_service-a.py中使用的保持一致）
SERVICE_B_URL = f'{GATEWAY_URL}/api/service-b/generate_dataset_from_file'
OUTPUT_CSV = 'output_final.csv'

def main():
    # 检查文件是否存在
    if not os.path.exists(FILE_PATH):
        print(f"错误: 文件 {FILE_PATH} 不存在")
        sys.exit(1)
    
    # 检查文件是否为空
    if os.path.getsize(FILE_PATH) == 0:
        print(f"错误: 文件 {FILE_PATH} 为空")
        sys.exit(1)
    
    print(f"正在上传文件: {FILE_PATH}")
    print(f"目标URL: {SERVICE_B_URL}")
    
    try:
        # 打开文件并发送请求
        with open(FILE_PATH, 'rb') as f:
            # 简化文件上传方式，让requests自动处理Content-Type
            files = {'file': f}
            
            response = requests.post(SERVICE_B_URL, files=files)
        
        # 检查响应状态
        if response.status_code == 200:
            # 保存CSV文件
            with open(OUTPUT_CSV, 'wb') as f:
                f.write(response.content)
            
            print(f"成功! CSV文件已保存为: {OUTPUT_CSV}")
            print(f"文件大小: {os.path.getsize(OUTPUT_CSV)} 字节")
            
            # 显示CSV内容（可选）
            print("\nCSV文件内容:")
            with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
                print(f.read())
        else:
            print(f"错误: 请求失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()