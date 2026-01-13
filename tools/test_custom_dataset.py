#!/usr/bin/env python3
# test_custom_dataset.py
# 测试自定义数据集参数功能

import sys
import os
import subprocess

def test_custom_dataset():
    """测试自定义数据集参数功能"""
    print("=== 测试自定义数据集参数功能 ===")
    
    # 项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 测试1: 使用默认数据集训练LR模型
    print("\n📝 测试1: 使用默认数据集训练LR模型")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "model_training_service.train_model_service", "--model", "lr"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✅ 测试1通过: 可以使用默认数据集训练LR模型")
        else:
            print("❌ 测试1失败: 使用默认数据集训练LR模型失败")
            print("错误输出:", result.stderr)
    except Exception as e:
        print(f"❌ 测试1失败: {str(e)}")
    
    # 测试2: 使用不存在的数据集文件
    print("\n📝 测试2: 使用不存在的数据集文件")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "model_training_service.train_model_service", "--model", "lr", "--data", "nonexistent.csv"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode != 0 and "数据集文件不存在" in result.stderr:
            print("✅ 测试2通过: 正确检测到不存在的数据集文件")
        else:
            print("❌ 测试2失败: 没有正确检测到不存在的数据集文件")
            print("返回码:", result.returncode)
            print("错误输出:", result.stderr)
    except Exception as e:
        print(f"❌ 测试2失败: {str(e)}")
    
    # 测试3: 检查命令行参数帮助信息
    print("\n📝 测试3: 检查命令行参数帮助信息")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "model_training_service.train_model_service", "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=60
        )
        if "--data" in result.stdout or "-d" in result.stdout:
            print("✅ 测试3通过: 命令行参数帮助信息中包含--data参数")
        else:
            print("❌ 测试3失败: 命令行参数帮助信息中不包含--data参数")
            print("输出:", result.stdout)
    except Exception as e:
        print(f"❌ 测试3失败: {str(e)}")

if __name__ == "__main__":
    test_custom_dataset()
