import sys
import os

# 打印当前Python路径
print("Current sys.path:")
for path in sys.path:
    print(f"- {path}")

# 打印当前工作目录
print(f"\nCurrent working directory: {os.getcwd()}")

# 尝试导入tools.utils
try:
    print("\nTrying to import tools.utils...")
    from tools import utils
    print("✓ Import successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# 尝试使用app.py中的方式导入
try:
    print("\nTrying to import using app.py's method...")
    # 复制app.py中的修复后的导入逻辑
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    print(f"Added to sys.path: {project_root}")
    from tools import utils
    print("✓ Import successful with app.py's method!")
except ImportError as e:
    print(f"✗ Import failed with app.py's method: {e}")
