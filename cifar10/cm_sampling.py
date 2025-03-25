from tkinter.font import names

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torch.distributed.tensor.parallel import loss_parallel
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel


# from model_1 import *
from model_2 import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything(1)
    torch.set_float32_matmul_precision('high')
    Total_data = 50000
    batch = 64#5000
    dtype = torch.float32
    Total_iter = Total_data // batch + (Total_data % batch != 0)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_weight_name = "cm_11_2"
    # mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
    # std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    device = torch.device(device_str)
    print(device)

    # f_theta = Unet()
    f_theta = UNet2DModel(
        sample_size=batch,
        in_channels=3,
        out_channels=3,
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
    checkpoint = torch.load(f"weights/{model_weight_name}.pt",weights_only=False)
    f_theta.load_state_dict(checkpoint['model_state_dict'])
    f_theta = f_theta.eval()
    f_theta = torch.compile(f_theta.to(device, dtype=dtype),fullgraph=True)
    loss_set = checkpoint['loss_set']
    print(len(loss_set))
    # N = checkpoint['step']
    N = 1000
    # print(checkpoint["statement"])
    # asd/2

    # cm para
    s1 = 150
    s0 = 2
    # mu0 = 0.01
    K = 1000
    N = lambda k: np.ceil(np.sqrt(k / K * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2) - 1) + 1
    # N = lambda k: 80

    epsilon = 0.002
    T = 76
    ro = 7
    # time_schedule = lambda i: (epsilon**(1/ro) + (i-1)/(N-1)*(T**(1/ro) - epsilon**(1/ro)))**ro
    time_schedule = lambda i, k: (epsilon ** (1 / ro) + (i - 1) / (N(k) - 1) *(
                T ** (1 / ro) - epsilon ** (1 / ro))) ** ro



    #X_data = torch.empty(batch, 3, 32, 32)#.to(device)
    # X_data, _ = next(iter(trainLoader))
    # X_data = torch.Tensor(X_data).to(device)
    results =[]

    # pbar = tqdm(range(Total_iter))
    pbar = tqdm(range(1))
    for k in pbar:
        ep = 1000
        stard_id = k * batch
        # index = torch.randint(int(N(ep)), size=(int(batch),))
        end_id = min((k + 1) * batch, Total_data)
        num_of_data = end_id - stard_id
        # backward process
        with torch.inference_mode(), torch.amp.autocast(enabled=True, device_type=device_str, dtype=torch.bfloat16):
            #with torch.amp.autocast():
            x_hat = torch.randn(num_of_data, 3, 32, 32,device=device, dtype=dtype) *T
            T_time = torch.ones(len(x_hat), device=device, dtype=dtype) * T
            sigma_data = 0.5
            c_skip = lambda t: sigma_data ** 2 / ((t - epsilon) ** 2 + sigma_data ** 2)
            c_out = lambda t: np.sqrt(sigma_data * (t - epsilon) / (t ** 2 + sigma_data ** 2))
            x = c_skip(T) * x_hat + c_out(T) * f_theta(x_hat, T_time).sample
            # sampling_step = 5
            # sampling_scheldule = [40, 20, 10, 2]
            sampling_scheldule = [epsilon]
            # for i in range(sampling_step-1, 0, -1):
            for i in sampling_scheldule:
                noise = torch.randn_like(x_hat)
                # time = time_schedule(i*N(0)/sampling_step, ep)
                # time = time_schedule(i*N(0)/sampling_step, ep)
                time = i
                time_tensor = time * \
                              torch.ones(len(x_hat), device=device, dtype=dtype)
                x = x + time_tensor[:, None, None, None] * noise
                x = c_skip(time) * x + c_out(time) * f_theta(x, time_tensor).sample

                # T_time = torch.ones(len(x_hat), device=device, dtype=dtype) * T
                # x = c_skip(T) * x + c_out(T) * f_theta(x_hat, T_time)

            x = (x + 1.)/2.0
            # x = torch.Tensor(std)[:,None,None].to(device) * x + torch.Tensor(mean)[:,None,None].to(device)

        # x_motion = torch.hstack([x_motion, x.unsqueeze(-1)])

        # result.append(x)   # Save the result
        # resize_f = transforms.Resize((28,28))
        # x_temp = resize_f(x).to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = np.clip(x_temp, 0.0, 1.0)
        # results.append((x_temp * 255).astype(np.uint8))




    #results = np.concatenate(results, axis=0)
    #print(results.dtype, results.shape)
    #np.save(f"result_2/{model_weight_name}_result.npy", results) # (batch, 32, 32, 3), dtype = uint8 range(0, 255)

        # # x_motion = resize(x_motion.squeeze(-1))
        # save per image
        # for j in range(num_of_data):
        #     plt.figure(figsize=(32, 32), dpi=1)
        #     plt.imshow(x_temp[j], cmap = "gray")
        #
        #     plt.axis("off")
        #     plt.margins(0, 0)
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #     plt.savefig(f"cm_result/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
        #     plt.clf()
        #     plt.close()


        # result = torch.cat(result, dim=0)
        # torch.save(result, f"result/x.pth")
        # print(result.shape)
        #         asd/2

        fig = plt.figure(figsize=(32, 32))
        num_fig = 8
        for j in range(num_of_data):
            plt.subplot(num_fig, num_fig, j + 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
            plt.imshow(x_temp[j],)
            plt.axis("off")
            # plt.margins(0, 0)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(f"figs/fig64/{model_weight_name}")  # 显示窗口
        plt.clf()
        plt.close()


