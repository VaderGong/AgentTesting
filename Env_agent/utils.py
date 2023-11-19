import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.label = torch.from_numpy(data_df['label'].values)
        self.data = torch.from_numpy(data_df[data_df.columns[:-1]].values).to(torch.float32)

    # 每次迭代取出对应的data和author
    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_label = self.get_batch_label(idx)
        return batch_data, batch_label

    def classes(self):
        return self.label

    def __len__(self):
        return self.data.size(0)

    def get_batch_label(self, idx):
        return np.array(self.label[idx])

    def get_batch_data(self, idx):
        return self.data[idx]


# 存数据，加载数据用的
class Config:
    def __init__(self, data_path, name, batch_size, learning_rate, epoch):
        
        self.name = name
        self.data_path = data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.train_loader, self.dev_loader, self.test_loader = self.load_tdt()
        self.input_col, self.output_class = self.get_class()
    
    # 加载train, dev, test，把数据封装成Dataloader类
    def load_tdt(self):
        file = self.read_file()
        train_dev_test = self.cut_data(file)
        tdt_loader = [self.load_data(i) for i in train_dev_test]
        return tdt_loader[0], tdt_loader[1], tdt_loader[2]

    # 读文件，后面会把若干pkl,变成一个csv
    def read_file(self):
        file = pd.read_csv(self.data_path, encoding="utf-8", index_col=None)
        # 本例中最后一列应为crash，但为泛用性，这里用label代替
        file.columns.values[-1] = "label"
        self.if_nan(file)
        return file

    # 切7:1:2 => 训练:验证:测试
    def cut_data(self, data_df):
        try:
            train_df, test_dev_df = train_test_split(data_df, test_size=0.3, random_state=1129, stratify=data_df["label"])
            dev_df, test_df = train_test_split(test_dev_df, test_size=0.66, random_state=1129, stratify=test_dev_df["label"])
        except ValueError:
            train_df, test_dev_df = train_test_split(data_df, test_size=0.3, random_state=1129)
            dev_df, test_df = train_test_split(test_dev_df, test_size=0.66, random_state=1129)
        return [train_df, dev_df, test_df]

    # Dataloader 封装进去
    def load_data(self, data_df):
        dataset = Dataset(data_df)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    # 检验输入输出是否有空值
    def if_nan(self, data):
        if data.isnull().any().any():
            empty = data.isnull().any()
            print(empty[empty].index)
            print("Empty data exists")
            sys.exit(0)

    # 混淆矩阵不一定有用，先放着
    def get_class(self):
        file = self.read_file()
        label = file[file.columns[-1]]
        label = list(set(list(label)))
        return file.columns[:-1], label
    

