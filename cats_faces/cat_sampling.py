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
    torch.set_float32_matmul_precision('high')      # set float32 matmul precision to high
    dtype = torch.float32

    Total_data = 10000        # Total number of cat faces = 29842
    batch = 16                
    Total_iter = Total_data // batch + (Total_data % batch != 0)
    model_weight_name = "DDPM_6_3"
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(device)

    ### construct model and load weights ###
    eps_theta = Unet().to(device)
    checkpoint = torch.load(f"weights/{model_weight_name}.pt",weights_only=False, map_location=device)
    eps_theta.load_state_dict(checkpoint['model_state_dict'])
    eps_theta = eps_theta.eval()
    loss_set = checkpoint['loss_set']
    print(len(loss_set))
    # N = checkpoint['step']
    N = 1000

    ### Create a beta schedule and alpha for the DDPM process ###
    beta = torch.linspace(0.0001, 0.02, N,device=device, dtype=dtype)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # pbar = tqdm(range(Total_iter))  # Create a progress bar for the sampling process
    pbar = tqdm(range(1))
    for k in pbar:
        stard_id = k * batch
        end_id = min((k + 1) * batch, Total_data)
        num_of_data = end_id - stard_id
        # backward process
        with torch.inference_mode():
            x = torch.randn(num_of_data, 3, 64, 64,device=device, dtype=dtype)
            for i in range(N, 1, -1):
                idx = torch.ones(len(x),device=device, dtype=dtype) * (i-1) # Create a tensor of indices for the current step
                prediction = eps_theta(x, idx) # Get the model's prediction for the current step

                # DDPM reverse process
                x = (x - (1 - alpha[i - 1]) / torch.sqrt(1 - alpha_bar[i - 1]) * prediction.squeeze(
                    -1)) / torch.sqrt(alpha[i - 1]) \
                    + ((1 - alpha[i - 1]) * (1 - alpha_bar[i - 2]) / (
                            1 - alpha_bar[i - 1])).sqrt() * torch.randn_like(x, device=device, dtype=dtype)  #

                pbar.set_description(f"current batch reverse process: {(N - i)} / {N} ") # Update the progress bar description

            ### last step ###
            idx = torch.zeros(len(x),device=device, dtype=dtype)
            prediction = eps_theta(x, idx)
            x = (x  - (1 - alpha[0]) / torch.sqrt(1 - alpha_bar[0]) * prediction) / torch.sqrt(alpha[0])
            torch.cuda.empty_cache()

 
        ### turn tensor x to numpy and clip the values to [0, 1] ###
        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = np.clip(x_temp, 0.0, 1.0)



        ### save per image for compute fid score ###
        # for j in range(num_of_data):
        #     plt.figure(figsize=(64, 64), dpi=1)
        #     plt.imshow(x_temp[j])
        #
        #     plt.axis("off")
        #     plt.margins(0, 0)
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #     plt.savefig(f"results/cat_DDPM_result_2/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
        #     plt.clf()
        #     plt.close()


    t1_end = time.process_time()

    # path = 'record_time.txt'
    # with open(path, 'a') as f:
    #     f.write(f"consistency model\n")
    #     f.write(f"time of the whole process {t1_end - t1_start}\n")

    ## save some images in one figure ###
    fig = plt.figure(figsize=(32, 32))
    num_fig = 4
    for j in range(num_of_data):
        plt.subplot(num_fig, num_fig, j + 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
        plt.imshow(x_temp[j])
        plt.axis("off")
        # plt.margins(0, 0)

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.tight_layout()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig(f"{model_weight_name}")   #显示窗口
    plt.clf()
    plt.close()

