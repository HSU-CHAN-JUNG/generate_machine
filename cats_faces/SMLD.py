import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from PIL import Image
from tqdm import tqdm
from model import *

if __name__ == '__main__':
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    # device_str = "cpu"
    print(device)

    ### Set parameters ###
    batch = 16
    epoch = 50
    save_name = "SMLD_1"


    data_transform = transforms.Compose([
        # transforms.Resize((64, 64)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,) ,(0.5,)),
    ]
    )

    ### input data directory ###
    traindir = "dataset/cat_face/cats-faces-64x64-for-generative-models/Cat-faces-dataset-master"
    trainset = torchvision.datasets.ImageFolder(traindir, transform=data_transform)
    trainLoader = dset.DataLoader(trainset, batch_size=batch, shuffle=False)
    
    ### Create a beta schedule and alpha for the DDPM process ###
    N = 1000
    T = 1
    dt = T / N
    sigma_max = 50
    sigma_min = 0.01
    # discrete
    lin_t = torch.linspace(0, T, N).to(device)
    sigma_list = sigma_min * (sigma_max / sigma_min) ** (lin_t)
    
    ### Crate model, optimizer, loss function and other parameters ###
    lr = 1e-3
    f_theta = Unet().to(device)
    optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    loss_set = []
    batch_loss = []

    ### Load model ###
    # lr = 1e-3
    # f_theta = Unet().to(device)
    # checkpoint = torch.load(f"weights/{save_name}.pt", weights_only=False)
    # f_theta.load_state_dict(checkpoint['model_state_dict'])
    # loss_set = checkpoint['loss_set']
    # optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    # loss_func = nn.MSELoss()


    pbar = tqdm(range(epoch))
    for ep in pbar:
        for i, (X_data, _) in enumerate(trainLoader):
            with torch.autocast(device_type=device_str, dtype=torch.bfloat16):
                X_data = X_data.to(device)

                # Euler-Maruyama method
                time = (torch.randint(N, size=(X_data.shape[0],), device=device) + 1)
                noise = torch.randn_like(X_data)
                Xt = X_data + (sigma_list[time - 1, None, None, None] ** 2 - sigma_min ** 2).sqrt() * noise
                f = f_theta(Xt, time - 1)
                loss = loss_func(f, noise)

                optimizer.zero_grad()  # 梯度清0
                loss.backward()  # 誤差反向傳導
                optimizer.step()  # 神經網路參數更新

                pbar.set_description('Loss: {}'.format(loss.item()))  # 更新pbar
                loss_set.append(loss.item())

    torch.save({
        'epoch': epoch,
        'step': N,
        # 'model': f_theta,
        'model_state_dict': f_theta.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_set': loss_set
    }, f"weights/{save_name}.pt")
    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    plt.savefig(f"figs/loss/{save_name}.png")
    plt.close()