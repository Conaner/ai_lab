import paddle
import paddle.nn as nn
from paddle.nn import functional as F
import paddle.optimizer as optim
import paddle.vision.transforms as transforms
from paddle.vision.datasets import MNIST
from paddle.io import DataLoader
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs('./mnist_result/paddle/', exist_ok=True)

# 定义训练的轮数
n_epochs = 3
# 定义训练时的批量大小
batch_size_train = 64
# 定义测试时的批量大小
batch_size_test = 1000
# 定义学习率
learning_rate = 0.01
# 定义日志输出的间隔
log_interval = 10
# 设置随机种子以保证结果的可复现性
random_seed = 1
# 使用PaddlePaddle设置随机种子
paddle.seed(random_seed)

# 加载MNIST训练数据集，并进行数据转换
train_dataset = MNIST(image_path='./MNIST/raw/train-images-idx3-ubyte.gz', label_path='./MNIST/raw/train-labels-idx1-ubyte.gz',
                      mode='train', transform=transforms.Compose([
    transforms.ToTensor(),  # 将数据转换为Tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化处理
]))
# 创建训练数据的加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
# 计算训练数据加载器中的样本总数
len_train_loader = len(train_loader.dataset)

# 加载MNIST测试数据集，并进行数据转换
test_dataset = MNIST(image_path='./MNIST/raw/t10k-images-idx3-ubyte.gz', label_path='./MNIST/raw/t10k-labels-idx1-ubyte.gz',
                     mode='test', transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
# 创建测试数据的加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)
# 计算测试数据加载器中的样本总数
len_test_loader = len(test_loader.dataset)

# 从测试数据加载器中取出一些样本用于可视化
examples = next(iter(test_loader))
# 分别获取样本数据和标签
example_data, example_targets = examples

# 使用matplotlib绘制一些测试样本及其真实标签
plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].numpy(), cmap='gray', interpolation='none')  # 显示图像
    plt.title("Ground Truth: {}".format(example_targets[i].item()))  # 显示真实标签
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
plt.show()


# 定义一个简单的卷积神经网络类
class Net(nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=10, kernel_size=5)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2D(in_channels=10, out_channels=20, kernel_size=5)
        # 定义一个用于dropout的层，以减少过拟合
        self.conv2_drop = nn.Dropout2D(p=0.5)
        # 定义第一个全连接层
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    # 定义前向传播过程
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))  # 卷积后接ReLU激活和最大池化
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))  # 卷积后接dropout和ReLU激活以及最大池化
        x = paddle.flatten(x, start_axis=1)  # 将多维数据展平
        x = F.relu(self.fc1(x))  # 全连接层后接ReLU激活
        x = F.dropout(x, p=0.5, training=self.training)  # dropout操作
        x = self.fc2(x)  # 最后的全连接层
        return F.log_softmax(x, axis=1)  # 应用log_softmax函数

# 实例化网络
network = Net()
# 创建优化器
optimizer = optim.SGD(parameters=network.parameters(), learning_rate=learning_rate)

# 初始化训练损失和测试损失的列表
train_losses = []
train_counter = []
test_losses = []
# 初始化测试损失对应的样本数量
test_counter = [i * len_train_loader for i in range(1, n_epochs + 1)]


# 定义训练过程的函数
def train(epoch):
    network.train()  # 设置网络为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):  # 遍历训练数据
        optimizer.clear_grad()  # 清除梯度
        output = network(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        # 每隔log_interval个batch输出一次训练信息
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len_train_loader,
                100. * batch_idx / len(train_loader), loss.numpy()))
            train_losses.append(loss.numpy())  # 记录训练损失
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len_train_loader))  # 记录样本数量
            # 保存模型和优化器的状态
            paddle.save(network.state_dict(), './model.pdparams')
            paddle.save(optimizer.state_dict(), './optimizer.pdopt')


# 定义测试过程的函数
def test():
    network.eval()  # 设置网络为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测的数量
    with paddle.no_grad():  # 不计算梯度，以加快计算速度
        for data, target in test_loader:  # 遍历测试数据
            output = network(data)  # 前向传播
            test_loss += F.nll_loss(output, target, reduction='sum').numpy()  # 计算测试损失
            pred = output.argmax(axis=1)  # 预测最大值所在的类别
            correct += (pred == target.data.view_as(pred)).astype('int').sum().numpy()  # 统计正确预测的数量
    test_loss /= len_test_loader  # 计算平均测试损失
    test_losses.append(test_loss)  # 记录测试损失
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len_test_loader,
        100. * correct / len_test_loader))  # 打印测试损失和准确率


# 进行训练和测试
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# 绘制训练和测试损失的图表
plt.figure()
plt.plot(train_counter, train_losses, color='blue')  # 绘制训练损失曲线
plt.scatter(test_counter, test_losses, color='red')  # 绘制测试损失散点图
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')  # 添加图例
plt.xlabel('number of training examples seen')  # x轴标签
plt.ylabel('negative log likelihood loss')  # y轴标签

# 再次从测试数据加载器中取出一些样本用于可视化
examples = next(iter(test_loader))
example_data, example_targets = examples
# 不计算梯度，以加快计算速度
with paddle.no_grad():
    output = network(example_data)  # 使用训练好的网络进行预测

# 绘制一些测试样本及其预测标签
plt.figure()
for i in range(96):
    plt.subplot(2, 3, i%6 + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0].numpy(), cmap='gray', interpolation='none')  # 显示图像
    plt.title("Prediction: {}".format(output.argmax(axis=1)[i].numpy()))  # 显示预测标签
    plt.xticks([])  # 不显示x轴
    plt.yticks([])  # 不显示y轴
    if i % 6 == 0:
        plt.savefig(f'./mnist_result/paddle/{i // 6}.png')
        plt.figure()

# 加载之前保存的模型和优化器状态，并继续训练
continued_network = Net()
continued_optimizer = optim.SGD(parameters=continued_network.parameters(), learning_rate=learning_rate)

# 加载保存的模型状态
network_state_dict = paddle.load('model.pdparams')
continued_network.set_state_dict(network_state_dict)
optimizer_state_dict = paddle.load('optimizer.pdopt')
continued_optimizer.set_state_dict(optimizer_state_dict)

for i in range(4, 9):
    test_counter.append(i * len_train_loader)
    train(i)
    test()

plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
