import requests

# API端点
url = 'http://localhost:5003/model-info'

print("=== 测试模型训练信息查询 ===")

# 1. 查询所有模型训练信息
print("\n1. 查询所有模型训练信息:")
try:
    response = requests.get(url)
    if response.status_code == 200:
        result = response.json()
        print(f"状态: {result['status']}")
        print(f"总记录数: {result['count']}")
        for i, info in enumerate(result['data'][:5]):  # 只显示前5条
            print(f"\n  记录 {i+1}:")
            print(f"    模型类型: {info['model_type']}")
            print(f"    模型名称: {info['model_name']}")
            print(f"    训练日期: {info['training_date']}")
            print(f"    准确率: {info['accuracy']:.4f}")
            print(f"    设备: {info['device_used']}")
            print(f"    训练时间: {info['training_time']:.2f}秒")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")
except Exception as e:
    print(f"请求发生错误: {str(e)}")

# 2. 按模型类型查询
print("\n2. 按模型类型'cnn'查询:")
try:
    response = requests.get(f"{url}?model_type=cnn")
    if response.status_code == 200:
        result = response.json()
        print(f"状态: {result['status']}")
        print(f"记录数: {result['count']}")
        for info in result['data']:
            print(f"\n  模型名称: {info['model_name']}")
            print(f"    训练日期: {info['training_date']}")
            print(f"    准确率: {info['accuracy']:.4f}")
            print(f"    设备: {info['device_used']}")
except Exception as e:
    print(f"请求发生错误: {str(e)}")

# 3. 查询所有模型类型
print("\n3. 查询所有模型类型:")
try:
    response = requests.get('http://localhost:5003/model-types')
    if response.status_code == 200:
        result = response.json()
        print(f"状态: {result['status']}")
        print(f"模型类型列表: {result['model_types']}")
except Exception as e:
    print(f"请求发生错误: {str(e)}")

# 4. 查询最新模型训练信息
print("\n4. 查询最新模型训练信息:")
try:
    response = requests.get('http://localhost:5003/latest-model-info')
    if response.status_code == 200:
        result = response.json()
        print(f"状态: {result['status']}")
        if result['data']:
            info = result['data']
            print(f"最新模型信息:")
            print(f"  模型类型: {info['model_type']}")
            print(f"  模型名称: {info['model_name']}")
            print(f"  训练日期: {info['training_date']}")
            print(f"  准确率: {info['accuracy']:.4f}")
            print(f"  设备: {info['device_used']}")
            print(f"  训练时间: {info['training_time']:.2f}秒")
except Exception as e:
    print(f"请求发生错误: {str(e)}")

print("\n=== 所有测试完成 ===")