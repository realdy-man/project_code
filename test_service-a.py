import requests
import json

# 配置 Spring Cloud Dispatcher 的地址（根据你的实际部署修改）
SPRING_GATEWAY_URL = "http://localhost:8080"  # 假设 Spring 应用运行在 8080 端口

# 要预测的文本
input_text = "近日，中国航天科技集团成功发射了一颗遥感卫星，用于国土资源监测。"

# 构造请求体
payload = {
    "text": input_text,
    # 可选：指定模型路径（一般不需要，除非你有多个模型）
    # "model_path": "/app/models/text_classifier.pkl",
    # "vectorizer_path": "/app/models/tfidf_vectorizer.pkl"
}

# 设置请求头
headers = {
    "Content-Type": "application/json"
}

# 完整的目标 URL（通过 Spring 转发到 service-a）
url = f"{SPRING_GATEWAY_URL}/api/service-a/predict_text"

try:
    print("正在向 Spring Gateway 发送请求...")
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)

    if response.status_code == 200:
        result = response.json()
        print("✅ 预测成功！")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"❌ 请求失败，状态码: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"⚠️ 请求异常: {e}")
except json.JSONDecodeError:
    print("⚠️ 响应不是有效的 JSON 格式")
    print(response.text)