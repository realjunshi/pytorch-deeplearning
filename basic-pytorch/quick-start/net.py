import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def getData():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    trainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    return [trainLoader, testloader]


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # 第一定义网络
    net = Net().to(device)

    # 第二定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 第三获取数据
    trainLoader, testLoader = getData()
    # print(trainLoader)
    trainIter = iter(trainLoader)
    images, labels = next(trainIter)

    # 第四 训练
    start = time.time()
    for epoch in range(2):

        runningLoss = 0
        for i, data in enumerate(trainLoader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 清空梯度缓存
            optimizer.zero_grad()

            outputs = net(inputs)
            # print(inputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 打印统计信息
            runningLoss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, runningLoss / 2000))
                runningLoss = 0
    print('Finish train! Total cost time:', time.time() - start)

    # 第五 测试
    dataiter = iter(testLoader)
    images, labels = next(dataiter)
    print('GroundTruth: ', ' '.join('%5s' % labels[j] for j in range(4)))

    # 网络输出
    outputs = net(images)

    # 预测结果
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % predicted[j] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
