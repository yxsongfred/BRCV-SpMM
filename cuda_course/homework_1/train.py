import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import os


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载数据集
trainset = torchvision.datasets.FashionMNIST(os.path.join(
    script_dir, '../../data'), download=True, train=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(os.path.join(
    script_dir, '../../data'), download=True, train=False, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 创建模型
model = LeNet()
model = model.to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters())

# 训练模型
for epoch in range(80):
    print('epoch ', epoch)
    for inputs, labels in trainloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(correct/total)

# 导出模型参数，也可以自定义导出模型参数的文件格式，这里使用了最简单的方法，但请注意，如果改动了必须保证程序二能够正常读取
for name, param in model.named_parameters():
    np.savetxt(os.path.join(
        script_dir, f'./{name}.txt'), param.detach().cpu().numpy().flatten())
