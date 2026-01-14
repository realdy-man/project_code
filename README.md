# 课设项目:文本分类与模型推荐系统

## 项目简介
随着自然语言处理（NLP）技术的快速发展，文本分类已成为智能问答、新闻推荐、情感分析等系统的核心模块。传统的文本分类系统多采用单体架构，模型更新、服务扩展困难，难以适应多样化的业务需求。

本项目旨在设计并实现一个基于微服务架构的文本分类与模型推荐系统，集成多种经典机器学习与深度学习模型（如CNN、MLP、SVM、逻辑回归等），实现对中文与英文文本数据集的自动分类，并根据数据集特性推荐最优模型。

系统采用微服务架构，将模型训练、文本预处理、分类预测、结果评估等功能模块化，支持模型热更新、并发处理与弹性扩展，提升系统的可维护性与实用性。

## 项目结构

```
project_code/
├── muntils/                # 工具类库
│   ├── __init__.py
│   └── getHost.py
├── services/               # 微服务集合
│   ├── service-a/          # 文本预测服务
│   │   ├── app.py          # Flask应用
│   │   ├── app_fastapi.py  # FastAPI应用
│   │   └── test_import_service.py
│   ├── service-b/          # 未知功能服务
│   │   ├── app - b.py
│   │   └── app.py
│   ├── service-c/          # 未知功能服务
│   │   └── app.py
│   └── service-d/          # 模型训练服务
│       ├── app.py
│       └── model_training/ # 模型训练模块
│           ├── database.py
│           ├── evaluate_all.py
│           ├── train_classic.py
│           ├── train_cnn.py
│           ├── train_model_service.py
│           └── utils.py
├── tools/                  # 工具脚本和资源
│   ├── dict/               # 字典和数据集
│   │   ├── iflytek_public/
│   │   └── news-categories/
│   ├── model_training_service/ # 模型训练服务工具
│   ├── models/             # 模型存储目录
│   ├── preprocessing_service/  # 预处理服务
│   ├── __init__.py
│   ├── example_numeric_call.py
│   ├── mox.py
│   ├── test_custom_dataset.py
│   ├── test_training_service.py
│   └── utils.py
├── .gitignore              # Git忽略文件
├── docker-compose.yml      # Docker容器编排
├── README.md               # 项目说明文档
└── test_*.py               # 测试脚本
```

## 主要功能

1. **文本分类**：支持中文和英文文本的自动分类，根据文本内容智能检测语言类型
2. **多模型支持**：集成CNN、MLP、SVM、逻辑回归等多种机器学习与深度学习模型
3. **模型训练**：提供模型训练服务，支持自定义数据集的模型训练和评估
4. **微服务架构**：采用微服务设计，各功能模块独立部署，支持弹性扩展
5. **服务注册与发现**：基于Nacos实现服务注册与发现，支持服务动态管理
6. **Docker容器化**：提供Docker容器编排配置，支持快速部署和环境一致性

## 技术栈

- **开发语言**：Python
- **Web框架**：Flask、FastAPI
- **机器学习框架**：PyTorch、Scikit-learn
- **服务注册与发现**：Nacos
- **容器技术**：Docker、Docker Compose
- **其他工具**：Jieba（中文分词）、TF-IDF（文本向量化）

## 安装与运行

### 环境要求

- Python 3.7+
- Docker（可选，用于容器化部署）
- Nacos Server（用于服务注册与发现）

### 本地开发环境

1. **克隆项目**

```bash
git clone <仓库地址>
cd project_code
```

2. **创建虚拟环境**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **安装依赖**

```bash
# 安装服务依赖
pip install -r services/service-a/requirements.txt
pip install -r services/service-b/requirements.txt
pip install -r services/service-c/requirements.txt
pip install -r services/service-d/requirements.txt
```

4. **启动Nacos Server**

请参考[Nacos官方文档](https://nacos.io/zh-cn/docs/quick-start.html)启动Nacos Server。

5. **启动服务**

```bash
# 启动service-a
cd services/service-a
python app.py

# 启动service-b
cd services/service-b
python app.py

# 启动service-c
cd services/service-c
python app.py

# 启动service-d
cd services/service-d
python app.py
```

### Docker容器化部署

1. **构建并启动容器**

```bash
docker-compose up -d
```

2. **查看服务状态**

```bash
docker-compose ps
```

## API文档

### service-a - 文本预测服务

#### 健康检查

```
GET /health
```

**响应示例**：
```json
{
  "status": "UP",
  "service": "service-a",
  "port": 5000
}
```

#### 文本预测

```
POST /predict_text
```

**请求体**：
```json
{
  "text": "待预测的文本",
  "model_path": "可选，模型文件路径",
  "vectorizer_path": "可选，向量器文件路径"
}
```

**响应示例**：
```json
{
  "success": true,
  "language": "zh",
  "prediction": "102",
  "prediction_name": "娱乐",
  "probability": [0.1, 0.2, 0.7],
  "classes": ["100", "101", "102"],
  "model_path": "/app/models/text_classifier.pkl",
  "vectorizer_path": "/app/models/tfidf_vectorizer.pkl"
}
```

#### 语言检测

```
GET /test_language?text=待检测文本
```

**响应示例**：
```json
{
  "text": "待检测文本",
  "detected_language": "zh",
  "has_chinese_characters": true
}
```

## 开发指南

### 代码结构

- 每个服务独立位于`services/`目录下，包含自己的应用代码和依赖
- 共享工具类位于`muntils/`和`tools/`目录下
- 模型训练相关代码位于`services/service-d/model_training/`目录下

### 开发流程

1. 创建特性分支：`git checkout -b feature/xxx`
2. 开发功能并编写测试
3. 提交代码：`git commit -m "Add feature xxx"`
4. 推送到远程：`git push origin feature/xxx`
5. 创建Pull Request进行代码评审

### 测试

```bash
# 运行特定服务的测试
python test_service-a.py
python test_service-b.py
python test_service_c.py
python test_service_d.py

# 运行模型训练测试
python test_model_training.py
```

## 贡献说明

欢迎对本项目进行贡献！请遵循以下流程：

1. Fork项目仓库
2. 创建特性分支
3. 提交更改
4. 创建Pull Request
5. 等待代码评审和合并

## 许可证

本项目采用MIT许可证，详情请见LICENSE文件。
