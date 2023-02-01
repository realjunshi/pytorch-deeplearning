# 定义一个神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms


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


def flat_features(x):
    # 0 维是batch size
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像是单通道，输出是6通道，使用的卷积核是5 * 5
        self.conv1 = nn.Conv2d(1, 6, 5, 2)  # 1 * 32 * 32 ==> 6 * 14 * 14
        # 输入图像6通道，输出16通道，使用卷积核是5 * 5
        self.conv2 = nn.Conv2d(6, 16, 5, 2)  # 16 * 14 * 14 ==> 16 * 5 * 5
        # 全连接层
        self.fc = nn.Linear(16 * 5 * 5, 10)

    def forward(self, x):
        # 使用max-pooling 和 （2，2）的滑动窗口
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # kernel核的大小是方形的话，可定义一个数字（2， 2） =》 2
        # 简单一点不使用pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, flat_features(x))
        x = F.relu(self.fc(x))
        return x


def imshow(img):
    img = img / 2 + 0.5     # 非归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    net = Net()
    # print(net)
    # params = list(net.parameters())
    # print("参数量：", len(params))
    # print(params)
    # 第一个参数的大小
    # print('第一个参数大小:', params[0].size())
    x = torch.randn(1, 1, 32, 32)
    out = net(x)
    # print(out)
    # 损失函数  使用简单的均方误差 nn.MSELoss
    # 定义伪标签
    target = torch.randn(10)
    # print(out.shape)  1 * 10
    # print(target.shape) 10  ==> 转为二维 1 * 10
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    # loss = criterion(out, target)
    # print(out)
    # print(target)
    # print(loss)
    # print(loss.grad_fn)
    # print(loss.grad_fn.next_functions[0][0])
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

    # 清空所有参数的梯度缓存
    # net.zero_grad()
    # print(net.conv1.bias.grad)
    # loss.backward()
    # print(net.conv1.bias.grad)

    # optimizer = optim.SGD(net.parameters(), lr=0.01)
    #
    # # 清空梯度缓存
    # optimizer.zero_grad()
    # out = net(input)
    # loss = criterion(out, target)
    # # 反向传播
    # loss.backward()
    # # 更新权重
    # optimizer.step()
    trainLoader, testLoader = getData()
    # print(trainLoader)
    trainIter = iter(trainLoader)
    images, labels = next(trainIter)

    # 展示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印图片类别标签
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))