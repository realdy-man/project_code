#!/usr/bin/env python3
"""
示例：使用数字参数调用模型训练服务
"""

import sys
import os

# 将项目根目录加入Python路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_training_service.train_model_service import ModelTrainingService

def example_numeric_call():
    """演示使用数字参数调用训练函数"""
    print("📚 示例：使用数字参数调用模型训练服务")
    print("=" * 50)
    
    # 创建训练服务实例
    training_service = ModelTrainingService()
    
    # 数字参数映射说明
    print("\n🔢 数字参数映射表:")
    print("-" * 40)
    print("| 数字 | 对应的模型类型 |")
    print("|------|----------------|")
    print("| 1    | CNN模型        |")
    print("| 2    | MLP模型        |")
    print("| 3    | Logistic Regression模型 |")
    print("| 4    | SVM模型        |")
    print("| 5    | 所有经典模型 (LR/SVM/MLP) |")
    print("| 6    | 所有模型 (经典模型 + CNN) |")
    print("-" * 40)
    
    # 示例1：使用数字1训练CNN模型
    print("\n📌 示例1：使用数字1训练CNN模型")
    print("调用代码：training_service.train(1)")
    print("(实际训练会耗时较长，这里仅显示参数处理)")
    
    # 示例2：使用数字2训练MLP模型
    print("\n📌 示例2：使用数字2训练MLP模型")
    print("调用代码：training_service.train(2)")
    print("(实际训练会耗时较长，这里仅显示参数处理)")
    
    # 示例3：使用数字6训练所有模型
    print("\n📌 示例3：使用数字6训练所有模型")
    print("调用代码：training_service.train(6)")
    print("(实际训练会耗时较长，这里仅显示参数处理)")
    
    # 示例4：支持字符串形式的数字
    print("\n📌 示例4：支持字符串形式的数字")
    print("调用代码：training_service.train('3')")
    print("(实际训练会耗时较长，这里仅显示参数处理)")
    
    print("\n" + "=" * 50)
    print("✅ 使用数字参数的方法已经实现完成！")
    print("🔧 你可以直接在代码中使用：")
    print("   from model_training_service.train_model_service import ModelTrainingService")
    print("   service = ModelTrainingService()")
    print("   results = service.train(1)  # 训练CNN模型")
    print("   print(results)")

if __name__ == "__main__":
    example_numeric_call()
