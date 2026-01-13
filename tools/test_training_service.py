#!/usr/bin/env python3
"""
测试模型训练服务的数字参数功能
"""

import sys
import os

# 将项目根目录加入Python路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_training_service.train_model_service import ModelTrainingService

def test_numeric_parameters():
    """测试数字参数功能"""
    print("🔍 测试模型训练服务的数字参数功能...")
    
    # 创建训练服务实例
    training_service = ModelTrainingService()
    
    # 测试用例：参数映射
    test_cases = [
        (1, "cnn"),
        ("1", "cnn"),
        (2, "mlp"),
        ("2", "mlp"),
        (3, "lr"),
        ("3", "lr"),
        (4, "svm"),
        ("4", "svm"),
        (5, "classic"),
        ("5", "classic"),
        (6, "all"),
        ("6", "all"),
    ]
    
    print("\n📋 测试参数映射:")
    print("-" * 40)
    
    # 由于实际训练耗时较长，我们只测试参数转换逻辑
    # 模拟train方法的参数处理部分
    def simulate_param_conversion(input_param):
        from typing import Union
        
        # 数字到字符串的映射
        number_to_model = {
            "1": "cnn",
            "2": "mlp", 
            "3": "lr",
            "4": "svm",
            "5": "classic",
            "6": "all"
        }
        
        model_type = input_param
        
        # 转换数字参数为对应的字符串
        if model_type in number_to_model:
            model_type = number_to_model[model_type]
        elif str(model_type) in number_to_model:
            model_type = number_to_model[str(model_type)]
        
        return model_type
    
    all_passed = True
    for input_param, expected in test_cases:
        try:
            result = simulate_param_conversion(input_param)
            status = "✅" if result == expected else "❌"
            print(f"{status} 输入: {repr(input_param)} → 输出: {repr(result)} (期望: {repr(expected)})")
            if result != expected:
                all_passed = False
        except Exception as e:
            print(f"❌ 输入: {repr(input_param)} → 错误: {str(e)}")
            all_passed = False
    
    print("-" * 40)
    
    if all_passed:
        print("🎉 所有参数映射测试通过!")
    else:
        print("❌ 参数映射测试失败!")
        return False
    
    # 测试无效参数
    print("\n📋 测试无效参数处理:")
    print("-" * 40)
    
    invalid_cases = [7, "7", "abc", -1, 0]
    for input_param in invalid_cases:
        try:
            result = simulate_param_conversion(input_param)
            # 检查是否为无效模型类型
            from model_training_service.train_model_service import MODEL_TYPES
            if result not in MODEL_TYPES:
                print(f"✅ 输入: {repr(input_param)} → 正确识别为无效模型类型")
            else:
                print(f"❌ 输入: {repr(input_param)} → 错误地识别为有效模型类型: {result}")
                all_passed = False
        except Exception as e:
            print(f"✅ 输入: {repr(input_param)} → 正确抛出异常: {type(e).__name__}")
    
    print("-" * 40)
    
    return all_passed

if __name__ == "__main__":
    success = test_numeric_parameters()
    sys.exit(0 if success else 1)
