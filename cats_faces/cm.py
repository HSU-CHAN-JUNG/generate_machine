from itertools import accumulate

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torch import no_grad
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from model import *
# from model_1 import *

def update_theta_mius(f_theta, f_theta_mius, mu):
    with torch.no_grad():
        for para in zip(f_theta_mius.parameters(), f_theta.parameters()):
           para[0].data = mu*para[0].data + (1-mu)*para[1].data
    # return mu*theta_mius + (1-mu)*theta

if __name__ == '__main__':
    ### Set parameters ###
    batch = 16          # batch size
    epoch = 200         # number of epochs
    Total_data = 29842  # total number of data
    save_name = "cm_1"   # Name for saving the model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"     # Use GPU if available, otherwise CPU
    # device_str = "cpu"        # Uncomment to force CPU usage
    device = torch.device(device_str)
    print(device)

    ### cm parameters ###
    s1 = 150
    s0 = 2
    mu0 = 0.9
    K = epoch*(Total_data//batch + (Total_data % batch != 0))
    N = lambda k: np.ceil(np.sqrt((k / K * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2)) - 1) + 1
    # N = lambda k: 80
    mu = lambda k: np.exp(s0 * np.log(mu0) / N(k))
    # mu = lambda k: 0.95
    epsilon = 0.002
    T = 80
    ro = 7
    time_schedule = lambda i, k: (epsilon ** (1 / ro) + (i - 1) / (N(k) - 1) *(
                T ** (1 / ro) - epsilon ** (1 / ro))) ** ro
    c_skip = lambda t: sigma_data**2 / ((t - epsilon) ** 2 + sigma_data**2)
    c_out = lambda t: sigma_data * (t - epsilon) / (t ** 2 + sigma_data**2).sqrt()

    ### input data directory and data transform ###
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    )
    traindir = "dataset/cat_face/cats-faces-64x64-for-generative-models/Cat-faces-dataset-master"
    trainset = torchvision.datasets.ImageFolder(traindir, transform=data_transform)
    trainLoader = dset.DataLoader(trainset, batch_size=batch, shuffle=True)


    ### Crate model, optimizer, loss function and other parameters ###
    f_theta = Unet().to(device)
    f_theta_mius = Unet().to(device)
    f_theta_mius.load_state_dict(f_theta.state_dict())
    lr = 1e-4
    optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    loss_set = []
    batch_loss = []

    ### Load model ###
    # f_theta = Unet().to(device)
    # checkpoint = torch.load(f"weights/{save_name}.pt", weights_only=False)
    # f_theta.load_state_dict(checkpoint['model_state_dict'])
    # f_theta_mius = Unet().to(device)
    # f_theta_mius.load_state_dict(f_theta.state_dict())
    # loss_set = checkpoint['loss_set']
    # optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)

    # loss_func = nn.MSELoss()
    loss_func = lpips.LPIPS(net='vgg').to(device)  # closer to "traditional" perceptual loss, when used for optimization


    pbar = tqdm(range(epoch)) # create a progress bar for epoch
    for ep in pbar:
        for i, (X_data, _) in enumerate(trainLoader):
            X_data = X_data.to(device)
            
            K = ep*(Total_data//batch + (Total_data % batch != 0)) + i + 1
            # K = epoch
            index = torch.randint(int(N(K)), size=(X_data.shape[0],))

            # forward of CM
            noise = torch.randn_like(X_data)
            time = time_schedule(index, K)[:, None, None, None].to(device)
            time_next = time_schedule(index + 1, K)[:, None, None, None].to(device)
            Xt = X_data + time * noise
            Xt_next = X_data + time_next * noise
            sigma_data = 0.5

            f1 = c_skip(time_next) * Xt_next + c_out(time_next) * f_theta(Xt_next, time_schedule(index + 1, K).to(
                device))  
            with no_grad():
                f2 = c_skip(time) * Xt + c_out(time) * f_theta_mius(Xt, time_schedule(index, K).to(
                    device))  


            # loss = 1 * loss_func(f1, f2)              # computing l_2-loss
            loss = 1 * loss_func.forward(f1, f2).mean() # computing lpips-loss
            loss.backward()  # 誤差反向傳導
            optimizer.step()  # 神經網路參數更新
            optimizer.zero_grad()  # 梯度清0

            ### update theta_mius ###
            mu_k = mu(ep)
            update_theta_mius(f_theta, f_theta_mius, mu_k)

            pbar.set_description('Loss: {}'.format(loss.item()))  # 更新pbar
            loss_set.append(loss.item())


    torch.save({
        'statement':
        f"This model is U-net structure and DDPM process with linear schedule. "
        f"It trained for {len(loss_set)/np.ceil(29842/batch)} epoches with {N(0)} steps. "
        "The dictionary save the model, trained parameter of model, Adam optimizer and loss. ",
        'normal': f'True, mean :{mean}, std: {std}',
        'c_sigma': sigma_data,
        'step': N(0),
        'T': T,
        'epsilon': epsilon,
        # 'model': f_theta,
        'model_state_dict': f_theta.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_function':loss_func,
        'loss_set': loss_set
    }, f"weights/{save_name}.pt")
    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    plt.savefig(f"figs/{save_name}.png")
    plt.close()