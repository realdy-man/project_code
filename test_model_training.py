import requests
import base64
import json
import os

# 读取CSV文件并转换为base64
dataset_path = 'F:\AAA_JIQIXUEXI\project_code\output_sample.csv'
with open(dataset_path, 'rb') as f:
    dataset_bytes = f.read()
dataset_base64 = base64.b64encode(dataset_bytes).decode('utf-8')

# API端点
url = 'http://localhost:5003/train'

# 测试不同的模型类型
model_types = ['cnn', 'mlp', 'svm', 'lr']

for model_type in model_types:
    print(f"\n=== 测试模型类型: {model_type} ===")
    
    # 构建请求数据
    data = {
        "model_type": model_type,
        "dataset": dataset_base64,
        "hyperparameters": {}  # 可选的超参数
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, json=data)
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            print(f"状态: {result['status']}")
            print(f"模型类型: {result['model_type']}")
            print("准确率:")
            for model_name, acc in result['accuracy'].items():
                print(f"  {model_name}: {acc:.4f}")
            print(f"训练时间: {result['training_time']}秒")
            print(f"使用设备: {result['device_used']}")
            print(f"模型信息: {result['model_info']}")
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"请求发生错误: {str(e)}")

print("\n=== 所有测试完成 ===")
