import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os

os.makedirs('./mnist_result/torch/', exist_ok=True)

# 设置训练参数
n_epochs = 3  # 训练轮数
batch_size_train = 64  # 训练时的批次大小
batch_size_test = 1000  # 测试时的批次大小
learning_rate = 0.01  # 学习率
log_interval = 10  # 打印训练信息的间隔
random_seed = 1  # 设置随机种子以确保结果可复现
torch.manual_seed(random_seed)  # 手动设置随机种子

# 定义训练和测试数据加载器
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True,  # 导入MNIST训练集
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 转换图像为Tensor
                                   torchvision.transforms.Normalize(  # 标准化处理
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)  # 设置批次大小和是否打乱数据
len_train_loader = len(train_loader.dataset)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True,  # 导入MNIST测试集
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)
len_test_loader = len(test_loader.dataset)

# 取出测试数据集的前几个样本进行可视化
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)  # 设置子图位置
    plt.tight_layout()  # 调整布局
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')  # 显示图像
    plt.title("Ground Truth: {}".format(example_targets[i]))  # 设置标题为真实标签
    plt.xticks([])  # 隐藏x轴
    plt.yticks([])  # 隐藏y轴
plt.show()


# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 第一个卷积层
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 第二个卷积层
        self.conv2_drop = nn.Dropout2d()  # 用于dropout的层
        self.fc1 = nn.Linear(320, 50)  # 第一个全连接层
        self.fc2 = nn.Linear(50, 10)  # 第二个全连接层

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 卷积后激活函数和池化
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))  # 第二个卷积层后激活函数和池化
        x = x.view(-1, 320)  # 展平层
        x = F.relu(self.fc1(x))  # 全连接层后激活函数
        x = F.dropout(x, training=self.training)  # dropout
        x = self.fc2(x)  # 第二个全连接层
        return F.log_softmax(x, dim=1)  # 输出层


# 实例化网络和优化器
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate)

# 初始化训练和测试损失列表
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len_train_loader for i in range(1, n_epochs + 1)]


# 定义训练函数
def train(epoch):
    network.train()  # 设置网络为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 清空梯度
        output = network(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        if batch_idx % log_interval == 0:
            # 打印训练信息和保存模型
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len_train_loader,
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len_train_loader))
            torch.save(network.state_dict(), './model.pth')  # 保存模型参数
            torch.save(optimizer.state_dict(), './optimizer.pth')  # 保存优化器参数


# 定义测试函数
def test():
    network.eval()  # 设置网络为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            output = network(data)  # 前向传播
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算测试损失
            pred = output.data.max(1, keepdim=True)[1]  # 预测结果
            correct += pred.eq(target.data.view_as(pred)).sum()  # 计算正确数量
    test_loss /= len_test_loader  # 计算平均损失
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len_test_loader,
        100. * correct / len_test_loader))  # 打印测试信息


# 循环训练和测试
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# 绘制训练和测试损失曲线
plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')


# 再次取出测试数据集的前几个样本进行可视化
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = network(example_data)
plt.figure()
for i in range(96):
    plt.subplot(2, 3, i%6 + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
    if i % 6 == 0:
        plt.savefig(f'./mnist_result/torch/{i//6}.png')
        # 训练结果保存路径。
        plt.figure()


# 加载保存的模型和优化器状态，并继续训练
continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate)

network_state_dict = torch.load('model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

# 继续训练和测试
for i in range(4, 9):
    test_counter.append(i * len_train_loader)
    train(i)
    test()

# 再次绘制训练和测试损失曲线
plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()