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
    batch = 16          # batch size
    epoch = 100         # number of epochs
    N = 1000            # number of discrete steps
    model_weight_name = "DDPM_1"    # Name for saving the model
    device_str = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available, otherwise CPU
    # device_str = "cpu"            # Uncomment to force CPU usage
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

    ### Create a beta schedule and alpha for the DDPM process ###
    beta = torch.linspace(0.0001, 0.02, N)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    ### Crate model, optimizer, loss function and other parameters ###
    eps_theta = Unet().to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(eps_theta.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    loss_set = []
    batch_loss = []


    ### Load model ###
    # eps_theta = Unet().to(device)
    # checkpoint = torch.load(f"weights/{model_weight_name}.pt", weights_only=False)
    # eps_theta.load_state_dict(checkpoint['model_state_dict'])
    # loss_set = checkpoint['loss_set']
    # lr = 1e-3
    # optimizer = torch.optim.Adam(eps_theta.parameters(), lr=lr)
    # loss_func = nn.MSELoss()

    pbar = tqdm(range(epoch)) # create a progress bar for epochs
    for ep in pbar:
        for i, (X_data, _) in enumerate(trainLoader):
            X_data = X_data.to(device)

            time = (torch.randint(N, size=(X_data.shape[0],)))+1
            # forward of DDPM
            noise = torch.randn_like(X_data)
            Xt = alpha_bar[time-1].sqrt()[:, None, None, None].to(device) * X_data \
                 + (1 - alpha_bar[time-1]).sqrt()[:, None, None, None].to(device) * noise

            prediction = eps_theta(Xt, time.to(device))  # 餵給net吃訓練數據x, 輸出預測值
            loss = loss_func(prediction, noise)

            ### 訓練網路三個最主要步驟 ###
            optimizer.zero_grad()  # 梯度清0
            loss.backward()  # 誤差反向傳導
            optimizer.step()  # 神經網路參數更新
            pbar.set_description('Loss: {}'.format(loss.item()))  # 更新pbar
            loss_set.append(loss.item())


    torch.save({
        'statement':
            "This model is U-net structure and DDPM process with linear schedule. "
            f"It trained for {epoch} epoches with {N} steps. "
            "The dictionary save the model, trained parameter of model, Adam optimizer and loss. ",
        'epoch': epoch,
        'step': N,
        # 'model': eps_theta,
        'model_state_dict': eps_theta.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_set': loss_set
    }, f"weights/{model_weight_name}.pt")
    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    plt.savefig(f"figs/{model_weight_name}.png")
    plt.close()