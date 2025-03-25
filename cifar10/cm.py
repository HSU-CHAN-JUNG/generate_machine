from itertools import accumulate

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from mkl_random import normal
from torch import no_grad
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel

from model_2 import *
# from model_1 import *

def update_theta_mius(f_theta, f_theta_mius, mu):
    with torch.no_grad():
        for para in zip(f_theta_mius.parameters(), f_theta.parameters()):
           para[0].data = mu*para[0].data + (1-mu)*para[1].data
    # return mu*theta_mius + (1-mu)*theta

if __name__ == '__main__':
    batch = 128
    # N = 1000

    epoch = 40
    DOWNLOAD_MNIST = False
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    # device_str = "cpu"
    resize = 32
    save_name = "cm_11"
    device = torch.device(device_str)
    print(device)

    # cm para
    s1 = 150
    s0 = 2
    mu0 = 0.9
    K = epoch
    N = lambda k: np.ceil(np.sqrt(k / K * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2) - 1) + 1
    # N = lambda k: 80
    mu = lambda k: np.exp(s0 * np.log(mu0) / N(k))
    # mu = lambda k: 0.9

    epsilon = 0.002
    T = 80
    ro = 7
    # time_schedule = lambda i: (epsilon**(1/ro) + (i-1)/(N-1)*(T**(1/ro) - epsilon**(1/ro)))**ro
    time_schedule = lambda i, k: (epsilon ** (1 / ro) + (i - 1) / (N(k) - 1) *(
                T ** (1 / ro) - epsilon ** (1 / ro))) ** ro

    # mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
    # std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    data_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    )

    # trainset = torchvision.datasets.MNIST(
    #     root = 'dataset/mnist',
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

    trainLoader = dset.DataLoader(trainset, batch_size=batch, shuffle=True)
    lr = 4e-4
    f_theta = UNet2DModel(
        sample_size=batch,
        in_channels=3,
        out_channels=3,
        norm_num_groups= 4,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512,),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        ),
    ).to(device)
    f_theta_mius = UNet2DModel(
        sample_size=batch,
        in_channels=3,
        out_channels=3,
        norm_num_groups=4,
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512,),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        ),
    ).to(device)
    # f_theta_mius.load_state_dict(f_theta.state_dict())
    # optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    # # optimizer = torch.optim.RAdam(f_theta.parameters(), lr=lr)
    # loss_set = []
    # batch_loss = []


    # x = torch.randn((batch, 3, 32, 32)).to(device)
    # t = torch.randn(batch).to(device)
    # y = f_theta(x, t).sample
    # print(y.shape)
    # asd/2


    ## Load model
    # f_theta = Unet().to(device)
    checkpoint = torch.load(f"weights/{save_name}_2.pt", weights_only=False)
    f_theta.load_state_dict(checkpoint['model_state_dict'])
    # f_theta_mius = Unet().to(device)
    f_theta_mius.load_state_dict(f_theta.state_dict())
    loss_set = checkpoint['loss_set']
    # optimizer = torch.optim.Adam(f_theta.parameters(), lr=lr)
    optimizer = torch.optim.RAdam(f_theta.parameters(), lr=lr)


    # loss_func = nn.MSELoss()
    ### image should be RGB, IMPORTANT: normalized to [-1,1]
    loss_func = lpips.LPIPS(net='alex').to(device)  # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    pbar = tqdm(range(epoch))
    # mu = 0.1
    # accumulation_steps = 4
    for ep in pbar:
        # for ep in range(epoch):
        for i, (X_data, _) in enumerate(trainLoader):
            # with torch.autocast(device_type=device_str, dtype=torch.bfloat16):

            X_data = X_data.to(device)
            # X_data = 2 * X_data.to(device) - 1
            # index = torch.randint(N, size=(X_data.shape[0],))
            index = torch.randint(int(N(ep)), size=(X_data.shape[0],))

            # forward
            noise = torch.randn_like(X_data)
            time_next = time_schedule(index + 1, ep)[:, None, None, None].to(device)
            time = time_schedule(index, ep)[:, None, None, None].to(device)
            # Xt = X_data + (2 * time).sqrt() * noise

            Xt = X_data + time * noise
            Xt_next = X_data + time_next * noise
            sigma_data = 0.5
            c_skip = lambda t: sigma_data**2 / ((t - epsilon) ** 2 + sigma_data**2)
            c_out = lambda t: sigma_data * (t - epsilon) / (t ** 2 + sigma_data**2).sqrt()
            # f1 = (c_skip(time_next) * Xt_next + c_out(time_next)
            #       * f_theta(Xt_next, time_schedule(index + 1, ep).to(device)).sample)  # 餵給net吃訓練數據x, 輸出預測值
            f1 = (c_skip(time) * Xt + c_out(time)
                  * f_theta(Xt, time_schedule(index, ep).to(device)).sample)  # 餵給net吃訓練數據x, 輸出預測值
            with (no_grad()):
                f2 = (c_skip(time_next) * Xt_next + c_out(time_next)
                      * f_theta_mius(Xt_next, time_schedule(index+1, ep).to(device)).sample)  # 餵給net吃訓練數據x, 輸出預測值


                # f2 = (c_skip(time) * Xt + c_out(time)
                #     * f_theta_mius(Xt, time_schedule(index, ep).to(device)).sample)  # 餵給net吃訓練數據x, 輸出預測值

                    # loss = 1 * loss_func(f1, f2)
            loss = 1 * loss_func.forward(f1, f2).mean()

            # loss = loss /accumulation_steps

            loss.backward()  # 誤差反向傳導
            optimizer.step()  # 神經網路參數更新
            optimizer.zero_grad()  # 梯度清0
            mu_k = mu(ep)
            update_theta_mius(f_theta, f_theta_mius, mu_k)


            # if (i+1) % accumulation_steps == 0:
            #     optimizer.step()  # 神經網路參數更新
            #     optimizer.zero_grad()  # 梯度清0
            #     mu_k = mu(ep)
            #     update_theta_mius(f_theta, f_theta_mius, mu_k)
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
        'statement':
        f"This model is U-net structure and DDPM process with linear schedule. "
        f"It trained for {len(loss_set)/np.ceil(50000/batch)} epoches. "
        "The dictionary save the model, trained parameter of model, RAdam optimizer and loss. ",
        'normal': f'True, mean :{mean}, std: {std}' if isinstance(mean, list) else 'False',
        'c_sigma': sigma_data,
        'step': "lambda k: np.ceil(np.sqrt(k / K * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2) - 1) + 1"
                    if callable(N) else N(0),
        'T': T,
        'epsilon': epsilon,
        # 'model': f_theta,
        'model_state_dict': f_theta.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_set': loss_set
    }, f"weights/{save_name}_2.pt")
    iteration = np.arange(len(loss_set))
    fig = plt.figure()
    plt.loglog(iteration, loss_set, "-")
    plt.savefig(f"figs/{save_name}.png")
    plt.close()