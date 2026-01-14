import requests
import base64
import json
import pandas as pd
import io

# 创建测试数据集
data = {
    'text': [
        '这是一个积极的评论', '这是一个消极的评论', 
        '这个产品很好用', '这个产品质量很差',
        '我很喜欢这个产品', '我不喜欢这个产品',
        '这个服务很周到', '这个服务很糟糕',
        '价格很合理', '价格太贵了'
    ],
    'label': [
        'positive', 'negative',
        'positive', 'negative',
        'positive', 'negative',
        'positive', 'negative',
        'positive', 'negative'
    ]
}

df = pd.DataFrame(data)

# 将DataFrame转换为CSV格式
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_content = csv_buffer.getvalue()

# 将CSV内容编码为base64
dataset_base64 = base64.b64encode(csv_content.encode('utf-8')).decode('utf-8')

# 准备请求数据
request_data = {
    'model_type': 'lr',  # 只训练逻辑回归模型
    'dataset_name': 'test_dataset',
    'dataset': dataset_base64
}

# 发送POST请求
try:
    response = requests.post('http://localhost:5003/train', json=request_data)
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应头: {response.headers}")
    print(f"响应内容: {response.text}")
    
    if response.status_code == 200:
        # 解析响应
        result = response.json()
        print("\n训练完成，响应结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 检查是否返回了模型文件
        if 'models' in result:
            print("\n返回的模型文件：")
            for filename, model_base64 in result['models'].items():
                print(f"- {filename}: {len(model_base64)} bytes")
                
            print("\n功能测试成功！模型文件已成功返回。")
        else:
            print("\n功能测试失败：响应中未包含模型文件。")
            
except requests.exceptions.RequestException as e:
    print(f"请求失败：{e}")
    import traceback
    traceback.print_exc()