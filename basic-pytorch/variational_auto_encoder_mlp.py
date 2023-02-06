import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, datasets
import torch.nn.functional as F
from torch.utils.data import dataloader
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 隐变量维度
latent_dim = 2
input_dim = 28 * 28
middle_dim = 256


class VariationalAutoEncoder(nn.Module):

    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.rand_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        org_size = x.size()
        x = x.view(org_size[0], -1)
        encode = self.encoder(x)

        mu, log_var = encode.chunk(2, dim=1)
        z = self.reparameterise(mu, log_var)
        out = self.decoder(z).view(size=org_size)

        return out, mu, log_var


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # 定义超参数
    batch_size = 128
    epochs = 10
    path = "./data"
    imgPath = "./image"
    checkpoint = 'auto_mlp_mnist.ckpt'

    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

    # 处理图像 变分自编码器不能先归一化图像 ！！！！
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainData = datasets.MNIST(root=path, train=True, transform=transform, download=False)
    testData = datasets.MNIST(root=path, train=False, transform=transform, download=False)

    trainLoader = dataloader.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
    testLoader = dataloader.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)

    # 模型
    model = VariationalAutoEncoder().to(device)
    # mse_loss = nn.MSELoss()
    # # 损失函数
    # kl_loss = lambda mu, log_var: -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    # reconstruct_loss = lambda out, x: mse_loss(out, x)
    kl_loss = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    reconstruct_loss = lambda recon_x, x: F.binary_cross_entropy(recon_x, x, size_average=False)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = 1e9
    best_epoch = 0

    valid_losses = []
    train_losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        model.train()

        train_loss = 0.
        train_num = len(trainLoader.dataset)

        for idx, (x, _) in enumerate(trainLoader):
            batch = x.size(0)
            x = x.to(device)
            out, mu, log_var = model(x)
            kl = kl_loss(mu, log_var)
            recon = reconstruct_loss(out, x)

            loss = kl + recon
            train_loss += loss.item()
            loss = loss / batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f"Training loss {loss: .3f} \t Recon {recon / batch: .3f} \t KL {kl / batch: .3f} in Step {idx}")

        train_losses.append(train_loss / train_num)

        valid_loss = 0.
        valid_kl = 0.
        valid_recon = 0.
        valid_num = len(testLoader.dataset)
        model.eval()
        with torch.no_grad():
            for idx, (x, _) in enumerate(testLoader):
                x = x.to(device)
                out, mu, log_var = model(x)
                kl = kl_loss(mu, log_var)
                recon = reconstruct_loss(out, x)
                loss = kl + recon

                valid_kl += kl.item()
                valid_recon += recon.item()
                valid_loss += loss.item()

            valid_losses.append(valid_loss / valid_num)

            if valid_loss < best_loss:
                best_epoch = valid_loss
                best_epoch = epoch

                torch.save(model.state_dict(), "best_vae_mlp.ckpt")
                print("Model save")

    plt.plot(train_losses, label='Train')
    plt.plot(valid_losses, label='Valid')
    plt.legend()
    plt.title('Learning Curve')


def test():
    state = torch.load("best_vae_mlp.ckpt")
    model = VariationalAutoEncoder()
    model.load_state_dict(state)

    n = 20
    digit_size = 28

    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    model.eval()
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            t = [xi, yi]
            z_sampled = torch.FloatTensor(t)
            with torch.no_grad():
                decode = model.decoder(z_sampled)
                digit = decode.view((digit_size, digit_size))
                figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size
                ] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # train()
    test()
