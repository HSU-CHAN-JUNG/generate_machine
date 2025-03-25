import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from model_1 import *

if __name__ == "__main__":
    batch = 128
    N = 1000
    epoch = 500
    DOWNLOAD_MNIST = False
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # device_str = "cpu"
    resize = 32
    save_name = "DDPM_1"
    device = torch.device(device_str)
    print(device)

    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,) ,(0.5,)),
    ]
    )

    # trainset = torchvision.datasets.MNIST(
    #     root = '../mnist',
    #     train = True,
    #     transform = data_transform, #改成torch可讀
    #     download = DOWNLOAD_MNIST,
    # )
    trainset = torchvision.datasets.CIFAR10(
        root='dataset/cifar10',
        train=True,
        transform=data_transform,  # 改成torch.可讀
        download=True,
    )

    beta = torch.linspace(0.0001, 0.02, N)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    trainLoader = dset.DataLoader(trainset, batch_size=batch, shuffle=True)
    lr = 1e-3
    # eps_theta = Unet().to(device)
    #
    # optimizer = torch.optim.Adam(eps_theta.parameters(), lr=lr)
    # loss_func = nn.MSELoss()
    #
    # loss_set = []
    # batch_loss = []


    ### Load model
    eps_theta = Unet().to(device)
    checkpoint = torch.load("weights/eps_theta_cifar10_6.pt", weights_only=False)
    eps_theta.load_state_dict(checkpoint['model_state_dict'])
    eps_theta = eps_theta.eval()
    loss_set = checkpoint['loss_set']
    optimizer = torch.optim.Adam(eps_theta.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    pbar = tqdm(range(epoch))
    for ep in pbar:
        # for ep in range(epoch):
        for X_data, _ in trainLoader:
            # with torch.autocast(device_type=device_str, dtype=torch.bfloat16):

            X_data = X_data.to(device)

            time = (torch.randint(N, size=(X_data.shape[0],)))
            # forward of DDPM
            noise = torch.randn_like(X_data)
            Xt = alpha_bar[time].sqrt()[:, None, None, None].to(device) * X_data \
                 + (1 - alpha_bar[time]).sqrt()[:, None, None, None].to(device) * noise

            prediction = eps_theta(Xt, time.to(device))  # 餵給net吃訓練數據x, 輸出預測值
            loss = loss_func(prediction, noise)
            # # 訓練網路三個最主要步驟
            optimizer.zero_grad()  # 梯度清0
            loss.backward()  # 誤差反向傳導
            optimizer.step()  # 神經網路參數更新
            pbar.set_description('Loss: {}'.format(loss.item()))  # 更新pbar
            # print('Loss{}: {}'.format(loss.item()))
            # lr = 8e-5 if loss<0.02 else 1e-3
            loss_set.append(loss.item())

    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    # plt.show()

    # try:
    #     checkpoint
    # except NameError:
    #     epoch_last = 0
    # else:
    #     epoch_last = checkpoint["epoch"]
    torch.save({
        # 'statement':
        # f"This model is U-net structure and DDPM process with linear schedule. \n
        #   It trained for {epoch_last + epoch} epoches with {N} steps. \n
        #   The dictionary save the model, trained parameter of model, Adam optimizer and loss. \n
        #   You can keep training for your goal with the dictionary."
        'epoch': epoch,
        'step': N,
        'model': eps_theta,
        'model_state_dict': eps_theta.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_set': loss_set
    }, f"weights/{save_name}.pt")
    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    plt.savefig(f"figs/{save_name}.png")
    plt.close()