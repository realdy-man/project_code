import sqlite3
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

class ModelTrainingDatabase:
    """模型训练信息数据库操作类"""
    
    def __init__(self):
        # 数据库文件路径
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        SERVICE_D_ROOT = os.path.dirname(CURRENT_DIR)
        self.db_path = os.path.join(SERVICE_D_ROOT, "model_training.db")
        
        # 初始化数据库
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建模型训练信息表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_training_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            training_date TEXT NOT NULL,
            accuracy REAL NOT NULL,
            hyperparameters TEXT,
            model_size INTEGER,
            device_used TEXT,
            training_time REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_training_info(self, model_type: str, model_name: str, accuracy: float, 
                          hyperparameters: Optional[Dict] = None, model_size: Optional[int] = None,
                          device_used: Optional[str] = None, training_time: Optional[float] = None):
        """
        保存模型训练信息到数据库
        
        Args:
            model_type: 模型类型（如cnn, lr, svm, mlp）
            model_name: 模型名称（如CNN, LogisticRegression, SVM, MLP）
            accuracy: 模型准确率
            hyperparameters: 模型超参数（字典形式）
            model_size: 模型大小（字节）
            device_used: 使用的设备（如cpu, cuda）
            training_time: 训练时间（秒）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 格式化训练日期
        training_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 序列化超参数
        hyperparameters_str = json.dumps(hyperparameters) if hyperparameters else None
        
        # 插入数据
        cursor.execute('''
        INSERT INTO model_training_info 
        (model_type, model_name, training_date, accuracy, hyperparameters, model_size, device_used, training_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (model_type, model_name, training_date, accuracy, hyperparameters_str, 
              model_size, device_used, training_time))
        
        conn.commit()
        conn.close()
    
    def get_training_info(self, model_type: Optional[str] = None, 
                         start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """
        查询模型训练信息
        
        Args:
            model_type: 可选，模型类型过滤
            start_date: 可选，开始日期过滤（格式：YYYY-MM-DD）
            end_date: 可选，结束日期过滤（格式：YYYY-MM-DD）
        
        Returns:
            模型训练信息列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询语句
        query = "SELECT * FROM model_training_info WHERE 1=1"
        params = []
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        if start_date:
            query += " AND training_date >= ?"
            params.append(f"{start_date} 00:00:00")
        
        if end_date:
            query += " AND training_date <= ?"
            params.append(f"{end_date} 23:59:59")
        
        # 按训练日期倒序排列
        query += " ORDER BY training_date DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # 转换为字典列表
        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "model_type": row[1],
                "model_name": row[2],
                "training_date": row[3],
                "accuracy": row[4],
                "hyperparameters": json.loads(row[5]) if row[5] else None,
                "model_size": row[6],
                "device_used": row[7],
                "training_time": row[8]
            })
        
        conn.close()
        return result
    
    def get_all_model_types(self) -> List[str]:
        """
        获取所有模型类型
        
        Returns:
            模型类型列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT model_type FROM model_training_info")
        rows = cursor.fetchall()
        
        result = [row[0] for row in rows]
        conn.close()
        return result
    
    def get_latest_training_info(self, model_type: Optional[str] = None) -> Optional[Dict]:
        """
        获取最新的模型训练信息
        
        Args:
            model_type: 可选，模型类型过滤
        
        Returns:
            最新的模型训练信息，如果没有则返回None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM model_training_info WHERE 1=1"
        params = []
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        query += " ORDER BY training_date DESC LIMIT 1"
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row:
            result = {
                "id": row[0],
                "model_type": row[1],
                "model_name": row[2],
                "training_date": row[3],
                "accuracy": row[4],
                "hyperparameters": json.loads(row[5]) if row[5] else None,
                "model_size": row[6],
                "device_used": row[7],
                "training_time": row[8]
            }
        else:
            result = None
        
        conn.close()
        return result