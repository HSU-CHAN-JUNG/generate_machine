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
    batch = 128
    N = 1000
    epoch = 50
    DOWNLOAD_MNIST = False
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # device_str = "cpu"
    resize = 32
    save_name = "sm_1"
    beta_max = 20
    beta_min = 0.1
    T = 1
    dt = T / N
    device = torch.device(device_str)
    print(device)

    # beta(s) = s
    beta = lambda t: beta_min + (beta_max - beta_min) * t
    mu_t = lambda t: torch.exp(-0.1 * t / 2 - (20 - 0.1) * t ** 2 / 4)[:, None, None, None]
    sigma_t = lambda t: torch.sqrt(1 - torch.exp(-0.1 * t - (20 - 0.1) * t ** 2 / 2))[:, None, None, None]
    # mu_t = lambda t : torch.exp(-1*t**2/4)[:,None, None, None]
    # sigma_t = lambda t : torch.sqrt(1 - torch.exp(-1*t**2/2) )[:, None, None, None]

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

    trainLoader = dset.DataLoader(trainset, batch_size=batch, shuffle=True)
    lr = 1e-2
    f_theta = Unet().to(device)
    optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    loss_set = []
    batch_loss = []

    ### Load model
    # f_theta = Unet().to(device)
    # checkpoint = torch.load(f"weights/eps_theta_{save_name}.pt", weights_only=False)
    # f_theta.load_state_dict(checkpoint['model_state_dict'])
    # f_theta_mius = Unet().to(device)
    # f_theta_mius.load_state_dict(f_theta.state_dict())
    # loss_set = checkpoint['loss_set']
    # optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    # loss_func = nn.MSELoss()

    pbar = tqdm(range(epoch))
    # mu = 0.1
    # accumulation_steps = 10
    for ep in pbar:
        # for ep in range(epoch):
        for i, (X_data, _) in enumerate(trainLoader):
            # with torch.autocast(device_type=device_str, dtype=torch.bfloat16):

            X_data = X_data.to(device)
            time = dt * (torch.randint(N, size=(X_data.shape[0],))).to(device)
            noise = torch.randn_like(X_data).to(device)
            Xt = mu_t(time) * X_data + sigma_t(time) * noise

            f = f_theta(Xt, time)
            # 餵給net吃訓練數據x, 輸出預測值
            loss = loss_func(f, noise)

            # loss = loss / accumulation_steps


            loss.backward()  # 誤差反向傳導
            optimizer.step()  # 神經網路參數更新
            optimizer.zero_grad()  # 梯度清0
            # mu_k = mu(ep)
            # update_theta_mius(f_theta, f_theta_mius, mu_k)

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
        # 'step': N,
        'model': f_theta,
        'model_state_dict': f_theta.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_set': loss_set
    }, f"weights/eps_theta_{save_name}.pt")
    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    plt.savefig(f"figs/{save_name}.png")
    plt.close()
