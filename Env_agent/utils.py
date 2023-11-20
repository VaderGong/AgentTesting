import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import time

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

    # 切训练:验证:测试 7:1:2 
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

    # 混淆矩阵不一定要用，先放着
    def get_class(self):
        file = self.read_file()
        label = file[file.columns[-1]]
        label = list(set(list(label)))
        return file.columns[:-1], label
    
class ENVModel:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self):
        dev_best_loss = float('inf')
        start_time = time.time()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        # 记录损失不下降的epoch数，防止过拟合
        break_epoch = 0
        for epoch in range(self.config.epoch):
            # 模型为训练模式
            self.model.train()
            print('Epoch [{}/{}]'.format(epoch + 1, self.config.epoch))
            # 训练模型
            for i, (batch_data, batch_label) in enumerate(self.config.train_loader):
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                batch_label = batch_label.long()
                loss = F.cross_entropy(outputs, batch_label)
                loss.backward()
                optimizer.step()
            
            true = batch_label.data.cpu()
            # 预测类别
            predict = torch.max(outputs.data, 1)[1].cpu()
            # 计算训练准确率
            train_acc = metrics.accuracy_score(true, predict)
            # 验证模型
            dev_acc, dev_loss = self.evaluate()
            print('Train Loss: {:.6f}, Acc: {:.6f}'.format(loss.item(), train_acc))
            print('Dev Loss: {:.6f}, Acc: {:.6f}'.format(dev_loss, dev_acc))
            print("Time: {:.1f}s".format(time.time() - start_time))
            # 如果验证损失下降，保存模型
            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(self.model.state_dict(), self.config.name)
                print("Save model!")
                break_epoch = 0
            else:
                break_epoch += 1
            # 如果验证损失不下降，停止训练
            if break_epoch >= 10:
                print("Early stop!")
                break

    def test(self):
        start_time = time.time()
        test_acc, test_loss = self.evaluate(test=True)
        print('Test Loss: {:.6f}, Acc: {:.6f}, Time: {:.1f}'.format(test_loss, test_acc, time.time() - start_time))

    def evaluate(self,test=False):
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        loader=self.config.test_loader if test else self.config.dev_loader
        with torch.no_grad():
            for (dev, dev_label) in loader:
                outputs = self.model(dev)
                dev_label = dev_label.long()
                loss = F.cross_entropy(outputs, dev_label)
                loss_total += loss
                dev_label = dev_label.data.cpu().numpy()
                predict = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, dev_label)
                predict_all = np.append(predict_all, predict)
        acc = metrics.accuracy_score(labels_all, predict_all)
        return acc, loss_total / len(loader)

    def load(self):
        self.model.load_state_dict(torch.load(self.config.name))
        print("Load model!")


                

