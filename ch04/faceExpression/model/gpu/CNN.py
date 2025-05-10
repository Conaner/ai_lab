import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import cv2

# 参数初始化
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

# 验证模型在验证集上的正确率
def validate(model, dataset, batch_size, device):
    val_loader = data.DataLoader(dataset, batch_size)
    result, num = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            pred = torch.argmax(pred, dim=1)
            result += torch.sum(pred == labels).item()
            num += len(images)
    acc = result / num
    return acc

# 我们通过继承Dataset类来创建我们自己的数据加载类，命名为FaceDataset
class FaceDataset(data.Dataset):
    def __init__(self, root):
        super(FaceDataset, self).__init__()
        self.root = root
        df_path = pd.read_csv(root + r'\image_emotion.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + r'\image_emotion.csv', header=None, usecols=[1])
        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    def __getitem__(self, item):
        face = cv2.imread(self.root + '\\' + self.path[item])
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_hist = cv2.equalizeHist(face_gray)
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0
        face_tensor = torch.from_numpy(face_normalized).type(torch.FloatTensor)
        label = self.label[item]
        return face_tensor, label

    def __len__(self):
        return self.path.shape[0]

class FaceCNN(nn.Module):
    def __init__(self):
        super(FaceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.RReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.conv3.apply(gaussian_weights_init)

        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.RReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=1024),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=256),
            nn.RReLU(inplace=True),
            nn.Linear(in_features=256, out_features=7),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

def train(train_dataset, val_dataset, batch_size, epochs, learning_rate, wt_decay, device):
    train_loader = data.DataLoader(train_dataset, batch_size)
    model = FaceCNN().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wt_decay)

    for epoch in range(epochs):
        loss_rate = 0
        model.train()
        for images, emotion in train_loader:
            images, emotion = images.to(device), emotion.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss_rate = loss_function(output, emotion)
            loss_rate.backward()
            optimizer.step()

        print('After {} epochs , the loss_rate is : '.format(epoch+1), loss_rate.item())
        if epoch % 5 == 0:
            acc_train = validate(model, train_dataset, batch_size, device)
            acc_val = validate(model, val_dataset, batch_size, device)
            print('After {} epochs , the acc_train is : '.format(epoch+1), acc_train)
            print('After {} epochs , the acc_val is : '.format(epoch+1), acc_val)

    return model

def main():
    device = torch.device('cuda')
    train_dataset = FaceDataset(root=r'C:\Users\17216\Desktop\ai_lab\ch04\faceExpression\data\pre_process\train_set')
    val_dataset = FaceDataset(root=r'C:\Users\17216\Desktop\ai_lab\ch04\faceExpression\data\pre_process\verify_set')
    model = train(train_dataset, val_dataset, batch_size=128, epochs=20, learning_rate=0.1, wt_decay=0, device=device)
    torch.save(model.state_dict(), r'C:\Users\17216\Desktop\ai_lab\ch04\faceExpression\model\gpu\model_cnn_gpu.pth')

if __name__ == '__main__':
    main()
