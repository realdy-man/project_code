import requests
import base64
import json
import os
import argparse

# 读取CSV文件并转换为base64
dataset_path = 'F:\AAA_JIQIXUEXI\project_code\output_sample.csv'
with open(dataset_path, 'rb') as f:
    dataset_bytes = f.read()
dataset_base64 = base64.b64encode(dataset_bytes).decode('utf-8')

# 直接访问的API端点
DIRECT_URL = 'http://localhost:5003/train'

# Spring Cloud Gateway 的地址
SPRING_GATEWAY_URL = "http://localhost:8080"
# 通过网关访问的API端点
GATEWAY_URL = f"{SPRING_GATEWAY_URL}/api/service-d/train"

print("=== 测试模型训练超参数 ===")

# 定义不同模型类型的超参数组合
hyperparameter_tests = [
    {
        "model_type": "cnn",
        "description": "CNN模型 - 高学习率",
        "hyperparameters": {
            "max_len": 64,
            "vocab_size": 10000,
            "embed_dim": 256,
            "batch_size": 32,
            "epochs": 5,
            "learning_rate": 0.01
        }
    },
    {
        "model_type": "cnn",
        "description": "CNN模型 - 低学习率",
        "hyperparameters": {
            "max_len": 32,
            "vocab_size": 5000,
            "embed_dim": 128,
            "batch_size": 64,
            "epochs": 10,
            "learning_rate": 0.0001
        }
    },
    {
        "model_type": "lr",
        "description": "逻辑回归 - 正则化参数C=0.1",
        "hyperparameters": {
            "C": 0.1,
            "max_iter": 500
        }
    },
    {
        "model_type": "lr",
        "description": "逻辑回归 - 正则化参数C=10",
        "hyperparameters": {
            "C": 10,
            "max_iter": 2000
        }
    },
    {
        "model_type": "svm",
        "description": "SVM - 线性核",
        "hyperparameters": {
            "kernel": "linear",
            "C": 1.0
        }
    },
    {
        "model_type": "svm",
        "description": "SVM - RBF核",
        "hyperparameters": {
            "kernel": "rbf",
            "C": 10.0,
            "gamma": 0.1
        }
    },
    {
        "model_type": "mlp",
        "description": "MLP - 小型网络",
        "hyperparameters": {
            "hidden_layer_sizes": [64],
            "max_iter": 200,
            "learning_rate_init": 0.001
        }
    },
    {
        "model_type": "mlp",
        "description": "MLP - 大型网络",
        "hyperparameters": {
            "hidden_layer_sizes": [256, 128],
            "max_iter": 500,
            "learning_rate_init": 0.0001
        }
    }
]

def run_test(url, mode):
    """执行超参数测试"""
    print(f"\n使用{mode}模式调用API: {url}")
    
    # 执行测试
    for i, test_case in enumerate(hyperparameter_tests, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   模型类型: {test_case['model_type']}")
        print(f"   超参数: {json.dumps(test_case['hyperparameters'], indent=2)}")
        
        # 构建请求数据
        data = {
            "model_type": test_case['model_type'],
            "dataset": dataset_base64,
            "hyperparameters": test_case['hyperparameters']
        }
        
        try:
            # 发送POST请求
            response = requests.post(url, json=data)
            
            # 检查响应状态
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 状态: {result['status']}")
                print(f"   模型类型: {result['model_type']}")
                print("   准确率:")
                for model_name, acc in result['accuracy'].items():
                    print(f"     {model_name}: {acc:.4f}")
                print(f"   训练时间: {result['training_time']}秒")
                print(f"   使用设备: {result['device_used']}")
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(f"   错误信息: {response.text}")
        except Exception as e:
            print(f"❌ 请求发生错误: {str(e)}")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="测试模型训练超参数")
    parser.add_argument('--mode', choices=['direct', 'gateway', 'both'], 
                      default='both', help="测试模式: direct(直接连接), gateway(网关连接), both(两种方式都测试)")
    
    args = parser.parse_args()
    
    # 根据选择的模式执行测试
    if args.mode in ['direct', 'both']:
        run_test(DIRECT_URL, "直接连接")
    
    if args.mode in ['gateway', 'both']:
        run_test(GATEWAY_URL, "网关连接")
    
    print("\n=== 所有测试完成 ===")
