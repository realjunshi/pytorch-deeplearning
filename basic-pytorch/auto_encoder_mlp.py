import os.path

import torch
import time
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import dataloader


# 使用线性变换的自编码器
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)

        out = self.decoder(x)
        out = out.view(out.size(0), 1, 28, 28)

        return out


if __name__ == '__main__':
    timestamp = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # 定义超参数
    learning_rate = 0.0003
    batch_size = 64
    epochs = 30
    path = "./data"
    imgPath = "./image"
    checkpoint = 'auto_mlp_mnist.ckpt'

    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

    # 处理图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    trainData = datasets.MNIST(root=path, train=True, transform=transform, download=True)
    testData = datasets.MNIST(root=path, train=False, transform=transform, download=True)

    trainLoader = dataloader.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
    testLoader = dataloader.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

    # 模型
    ae = AutoEncoder().to(device)
    if os.path.exists(checkpoint):
        ae.load_state_dict(torch.load(checkpoint))

    # 损失函数 和 优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=learning_rate)

    print("start train....")
    for epoch in range(epochs):

        for idx, data in enumerate(trainLoader):
            # 获取图片
            image, _ = data
            image = image.to(device)

            # 计算
            fakeImage = ae(image)

            # 损失
            loss = criterion(image, fakeImage)

            # 梯度下降
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 300 == 0:
                print("epoch:{} / {},idx:{}, loss:{}".format(epoch, epochs, idx, loss))

        # 训练完一次就生成一次图像
        testImage, _ = next(iter(testLoader))
        fakeImage = ae(testImage)

        img = torch.cat([testImage, fakeImage], dim=0)

        # 保存图像
        save_image(img, os.path.join(imgPath, 'img{}-{}.png'.format(timestamp, epoch + 1)), nrow=8, normalize=True)

        torch.save(ae.state_dict(), checkpoint)
