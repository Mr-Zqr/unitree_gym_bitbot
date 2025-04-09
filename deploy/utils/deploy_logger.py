
# import numpy as np
# from collections import defaultdict
# from multiprocessing import Process, Value
import csv
import os
from datetime import datetime
    
class DeployLogger:
    def __init__(self):
        self.data = {}  # 用于存储数据的字典

    def record(self, key, value):
        """记录数据，类似字典的方式"""
        if key not in self.data:
            self.data[key] = []  # 如果键不存在，初始化一个空列表
        self.data[key].append(value)

    def save_to_csv(self):
        """将记录的数据保存为CSV文件"""
        # 获取所有键（列名）
        columns = list(self.data.keys())
        # 获取最大行数（以最长的列表为准）
        max_rows = max(len(self.data[key]) for key in columns)
         # 创建log目录（如果不存在）
        if not os.path.exists('log'):
            os.makedirs('log')

        # 生成文件名（如果未提供）
        filename = datetime.now().strftime('log/%Y%m%d-%H%M%S.csv')

        # 写入CSV文件
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入列名
            writer.writerow(columns)
            # 逐行写入数据
            for i in range(max_rows):
                row = [self.data[key][i] if i < len(self.data[key]) else "" for key in columns]
                writer.writerow(row)