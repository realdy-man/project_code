import requests
import base64
import json

# 读取CSV文件并转换为base64
dataset_path = 'F:\AAA_JIQIXUEXI\project_code\output_sample.csv'
with open(dataset_path, 'rb') as f:
    dataset_bytes = f.read()
dataset_base64 = base64.b64encode(dataset_bytes).decode('utf-8')

# API端点
url = 'http://localhost:5003/train'

# 测试MLP超参数
print("=== 测试MLP超参数 ===")

test_cases = [
    {
        "name": "默认参数",
        "hyperparameters": {}
    },
    {
        "name": "大型网络",
        "hyperparameters": {
            "hidden_layer_sizes": [256, 128],
            "max_iter": 500,
            "learning_rate_init": 0.0001
        }
    },
    {
        "name": "学习率调整",
        "hyperparameters": {
            "learning_rate_init": 0.1
        }
    }
]

for test_case in test_cases:
    print(f"\n测试: {test_case['name']}")
    print(f"超参数: {json.dumps(test_case['hyperparameters'], indent=2)}")
    
    data = {
        "model_type": "mlp",
        "dataset": dataset_base64,
        "hyperparameters": test_case['hyperparameters']
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ 状态: {result['status']}")
            print(f"准确率: {result['accuracy'].get('MLP', 'N/A'):.4f}")
            print(f"训练时间: {result['training_time']}秒")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
