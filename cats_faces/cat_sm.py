import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from model import *

if __name__ == "__main__":
    ### Set parameters ###
    batch = 16              # batch size
    N = 1000                # number of discrete steps
    epoch = 100             # number of epochs
    save_name = "sm_1"      # Name for saving the model
    device_str = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, otherwise CPU
    # device_str = "cpu"    # Uncomment to force CPU usage
    device = torch.device(device_str)
    print(device)

    ### Data preparation (image to pytorch tensor) ###
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,) ,(0.5,)),
    ]
    )

    ### input data directory ###
    traindir = "dataset/cat_face/cats-faces-64x64-for-generative-models/Cat-faces-dataset-master"
    trainset = torchvision.datasets.ImageFolder(traindir, transform=data_transform)
    trainLoader = dset.DataLoader(trainset, batch_size=batch, shuffle=False)

    ### Create a beta schedule and alpha for the score-based process ###
    beta_max = 20
    beta_min = 0.1
    T = 1
    dt = T / N
    beta = lambda t: beta_min + (beta_max - beta_min) * t
    mu_t = lambda t: torch.exp(-beta_min * t / 2 - (beta_max - beta_min) * t ** 2 / 4)[:, None, None, None]
    sigma_t = lambda t: torch.sqrt(1 - torch.exp(-beta_min * t - (beta_max - beta_min) * t ** 2 / 2))[:, None, None,
                        None]



    ### Crate model, optimizer, loss function and other parameters ###
    eps_theta = Unet().to(device)
    lr = 1e-4
    optimizer = torch.optim.Adam(eps_theta.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    loss_set = []
    batch_loss = []

    ## Load model
    # eps_theta = Unet().to(device)
    # checkpoint = torch.load(f"weights/{save_name}.pt", weights_only=False)
    # eps_theta.load_state_dict(checkpoint['model_state_dict'])
    # lr = 1e-4
    # optimizer = torch.optim.Adam(eps_theta.parameters(), lr=lr)
    # loss_set = checkpoint['loss_set']
    # loss_func = nn.MSELoss()

    pbar = tqdm(range(epoch))
    for ep in pbar:
        # for ep in range(epoch):
        for X_data, _ in trainLoader:
            X_data = X_data.to(device)
            index = (torch.randint(N, size=(X_data.shape[0],))+1).to(device)
            time = dt * index
            noise = torch.randn_like(X_data).to(device)
            Xt = mu_t(time) * X_data + sigma_t(time) * noise

            f = eps_theta(Xt, time)
            # 餵給net吃訓練數據x, 輸出預測值
            loss = loss_func(f, noise)
            ### 訓練網路三個最主要步驟
            optimizer.zero_grad()  # 梯度清0
            loss.backward()  # 誤差反向傳導
            optimizer.step()  # 神經網路參數更新
            pbar.set_description('Loss: {}'.format(loss.item()))  # 更新pbar
            # print('Loss{}: {}'.format(loss.item()))
            # lr = 8e-5 if loss<0.02 else 1e-3
            loss_set.append(loss.item())


    torch.save({
        'statement':
        "This model is U-net structure and score-based process "
        f"It trained for {epoch} epoches with {N} steps. "
        "The dictionary save the model['model'], "
        "trained parameter of model['model_state_dict'], "
        "Adam optimizer['optimizer_state_dict'] and loss['loss_set'].",
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