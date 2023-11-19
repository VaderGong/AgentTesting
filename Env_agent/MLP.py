import torch
from torch.nn import Linear, ReLU, ModuleList, Sequential, Dropout, Softmax, Tanh
import torch.nn.functional as F

class MLP(torch.nn.Module):
    # 一个简单的多层感知机(MLP)模型
    def __init__(self, input_n=10, output_n=256, num_layer=2, layer_list=[128, 256], dropout=0.2):
        super(MLP, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list

        # 输入层
        self.input_layer = Sequential(
            Linear(input_n, layer_list[0], bias=False),
            ReLU()
        )

        # 隐藏层
        self.hidden_layer = Sequential()

        for index in range(num_layer-1):
            self.hidden_layer.extend([Linear(layer_list[index], layer_list[index+1], bias=False), ReLU()])

        self.dropout = Dropout(dropout)

        # 输出层
        self.output_layer = Sequential(
            Linear(layer_list[-1], output_n, bias=False),
            Softmax(dim=1),
        )

    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output
    

