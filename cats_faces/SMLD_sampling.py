import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
from model import *


if __name__ == "__main__":
    t1_start = time.process_time()

    ### Set parameters ###
    torch.set_float32_matmul_precision('high')
    dtype = torch.float32

    Total_data = 10000              # Total number of cat faces = 29842
    batch = 64                      # batch size for sampling
    Total_iter = Total_data // batch + (Total_data % batch != 0)
    model_weight_name = "SMLD_1"

    ### Set device ###
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(device)

    ### Create a beta schedule and alpha for the SMLD process ###
    # N = checkpoint['step']
    sigma_max = 50
    sigma_min = 0.01
    T = 1
    N = 1000
    dt = T / N
    lin_t = torch.linspace(0, T, N).to(device)
    sigma_list = sigma_min * (sigma_max / sigma_min) ** (lin_t)

    ### construct model and load weights ###
    f_theta = Unet().to(device)
    checkpoint = torch.load(f"weights/{model_weight_name}.pt")  # , weights_only=False, map_location="cpu")
    f_theta.load_state_dict(checkpoint['model_state_dict'])
    f_theta = f_theta.eval()
    f_theta = torch.compile(f_theta.to(device, dtype=dtype), fullgraph=True)
    loss_set = checkpoint['loss_set']
    print(len(loss_set))


    # pbar = tqdm(range(Total_iter))
    pbar = tqdm(range(1))
    for k in pbar:
        stard_id = k * batch
        end_id = min((k + 1) * batch, Total_data)
        num_of_data = end_id - stard_id
        ### backward process
        with torch.inference_mode(), torch.amp.autocast(enabled=True, device_type=device_str, dtype=torch.bfloat16):
            
            ### exact score and exact epsilon ###
            # X_data = X_data.to(device)
            # score_exact = lambda x, t: -(x - mu_t(X_data,t) ) / sigma_t(t)**2
            # eps_exact   = lambda x, t: -sigma_t(t) * score_exact(x,t) #-beta*score_exact(x,t)
            # score_exact = lambda x, t: -(x - X_data) / (sigma_list[t]**2 - sigma_min**2)
            # eps_exact   = lambda x, t: -(sigma_list[t]**2 - sigma_min**2).sqrt() * score_exact(x,t) #-beta*score_exact(x,t)
            # x = torch.randn_like(X_data, device = device) * np.sqrt(sigma_max**2 - sigma_min**2) + X_data # 原X_T分布


            x = torch.randn(batch, 3, 64, 64, device=device) * np.sqrt(sigma_max ** 2 - sigma_min ** 2)  # 原X_T分布
            for i in range(N, 1, -1):
                idx = torch.ones(len(x)).to(device) * (i - 1)
                prediction = f_theta(x, idx)

                g_t = sigma_list[i - 1] * (
                            2 / T * torch.log(torch.Tensor([sigma_max / sigma_min]).to(device)) * dt).sqrt()

                x = x - g_t ** 2 * prediction / (
                            sigma_list[i - 1] ** 2 - sigma_min ** 2).sqrt() + g_t * torch.randn_like(x)

                # x_motion = torch.cat([x_motion, x.unsqueeze(-1)],dim = -1)
                pbar.set_description(f"current batch reverse process: {(N - i)} / {N} ")

            idx = torch.ones(len(x)).to(device) * 0
            prediction = f_theta(x, idx)
            g_t = sigma_list[i - 1] ** 2 * 2 / T * torch.log(torch.Tensor([sigma_max / sigma_min]).to(device)) * dt
            x = x - g_t ** 2 * prediction / (sigma_list[i - 1] ** 2 - sigma_min ** 2).sqrt()
            torch.cuda.empty_cache()

        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]

    ### save per image ###
    # for j in range(batch):
    #     plt.figure(figsize=(64,64), dpi=1)
    #     plt.imshow(x_temp[j])

    #     plt.axis("off")
    #     plt.margins(0, 0)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())

    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     plt.savefig(f"results/result_SMLD_2/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
    #     plt.clf()
    #     plt.close()

    ## save some images in one figure ###
    fig = plt.figure(figsize=(32, 32))
    num_fig = 8
    for j in range(num_fig**2):
        plt.subplot(num_fig, num_fig, j + 1)  
        plt.imshow(x_temp[j])
        plt.axis("off")
        # plt.margins(0, 0)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(f"{model_weight_name}")   #显示窗口
    # plt.clf()
    # plt.close()
    # plt.show()

    t1_end = time.process_time()

    # path = 'record_time.txt'
    # with open(path, 'a') as f:
    #     f.write(f"sm\n")
    #     f.write(f"time of the whole process {t1_end - t1_start}\n")