from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
# %run model_2.ipynb
from model_1 import *

if __name__ == '__main__':
    DOWNLOAD_MNIST = False
    device_str = "cuda" if torch.cuda.is_available() else "cpu"


    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='dataset/cifar10',
        train=True,
        transform=data_transform,  # 改成torch.可讀
        download=True,
    )

    trainloader = dset.DataLoader(trainset, batch_size=64, shuffle=True)

    x = next(iter(trainloader))[0]
    x = x.numpy().transpose((0, 2, 3, 1))
    x = np.clip(x, 0.0, 1.0)

    fig = plt.figure(figsize=(32, 32))
    num_fig = 8
    for j in range(64):
        plt.subplot(num_fig, num_fig, j + 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
        plt.imshow(x[j], )
        plt.axis("off")
        # plt.margins(0, 0)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(f"figs/fig64/true_cifar10")  # 显示窗口
    plt.clf()
    plt.close()