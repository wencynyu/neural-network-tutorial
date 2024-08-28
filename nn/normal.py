import torch
import torch.nn as nn
from torch import optim
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

class NormalNN(nn.Module):

    def __init__(self, in_dimension, out_dimension):
        super(NormalNN, self).__init__()

        self.hide_layer_func = nn.Linear(in_dimension, 10)
        self.out_layer_func = nn.Linear(10, out_dimension)
        self.activate_func = nn.ReLU()

    def forward(self, x):
        x = self.hide_layer_func(x)
        x = self.activate_func(x)
        x = self.out_layer_func(x)
        return x


if __name__ == '__main__':
    print('cuda available: %s' % torch.cuda.is_available())

    # 创建模型实例
    model = NormalNN(2, 1)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 定义优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 学习率为0.01

    # 输入数据
    inputs = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    outputs = torch.tensor([[3.0], [5.0], [7.0]])

    num_epochs = 500
    for epoch in range(num_epochs):
        # 前向传播
        predictions = model(inputs)
        # 计算损失
        loss = criterion(predictions, outputs)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    test_input = torch.tensor([[4.0, 5.0], [3.14, 9.29]])
    with torch.no_grad():
        prediction = model(test_input)
        print("Prediction:", prediction.flatten().tolist())
