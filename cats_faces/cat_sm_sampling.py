import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from model import *

def seed_everything(seed = 0):
    """
    Set the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    t1_start = time.process_time()
    seed_everything(0)
    torch.set_float32_matmul_precision('high')
    dtype = torch.float32

    Total_data = 10000              # Total number of cat faces = 29842
    batch = 64#1500                 # batch size for sampling
    Total_iter = Total_data // batch + (Total_data % batch != 0) # Total number of iterations for sampling
    model_weight_name = "sm_1"

    ### Set device ###
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(device)

    # device = torch.device("cpu")  # Uncomment this line to run on CPU
    eps_theta = Unet().to(device)
    checkpoint = torch.load(f"weights/{model_weight_name}.pt",weights_only=False,map_location=device)
    eps_theta.load_state_dict(checkpoint['model_state_dict'])
    eps_theta = eps_theta.eval()
    eps_theta = torch.compile(eps_theta.to(device, dtype=dtype),fullgraph=True)
    loss_set = checkpoint['loss_set']
    print(len(loss_set))
    # N = checkpoint['step']
    N = 1000

    beta_max = 20
    beta_min = 1e-5
    T = 1
    dt = T / N
    beta = lambda t: beta_min + (beta_max - beta_min) * t
    mu_t = lambda t: np.exp(-beta_min * t / 2 - (beta_max - beta_min) * t ** 2 / 4)
    sigma_t = lambda t: np.sqrt(1 - np.exp(-beta_min * t - (beta_max - beta_min) * t ** 2 / 2))

    # exact_eps = lambda x, t: (x - mu_t(t) * X_data.to(device)) / sigma_t(t)

    results =[]

    # pbar = tqdm(range(Total_iter))
    pbar = tqdm(range(1))
    t2_start = time.process_time()
    for k in pbar:
        stard_id = k * batch
        end_id = min((k + 1) * batch, Total_data)
        num_of_data = end_id - stard_id
        # backward process
        with torch.inference_mode():
            t3_start = time.process_time()
            x = torch.randn(num_of_data, 3, 64, 64,device=device, dtype=dtype)
            for i in range(N, 1, -1):
                ti = i * dt
                # DDPM reverse
                noise = torch.randn_like(x).to(device)
                s_theta = eps_theta(x, ti * torch.ones(len(x), device=device))
                x = (1 + beta(ti) * dt / 2) * x - beta(ti) * s_theta * dt / sigma_t(ti) \
                    + np.sqrt(beta(ti) * dt) * noise

                
                pbar.set_description(f"current batch reverse process: {(N - i)} / {N} ")
            idx = torch.zeros(len(x),device=device, dtype=dtype)
            s_theta = eps_theta(x, idx)

            x = (1 + beta(ti) * dt / 2) * x - beta(ti) * s_theta * dt / sigma_t(ti)
            torch.cuda.empty_cache()
            t3_end = time.process_time()
        

        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = np.clip(x_temp, 0.0, 1.0)
        t2_end = time.process_time()


        # # x_motion = resize(x_motion.squeeze(-1))
        ### save per image
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
        #     plt.savefig(f"results/sm1/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
        #     plt.clf()
        #     plt.close()



        ### save some images in one figure ###
        fig = plt.figure(figsize=(32, 32))
        num_fig = 8
        for j in range(num_of_data):
            plt.subplot(num_fig, num_fig, j + 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
            plt.imshow(x_temp[j])
            plt.axis("off")
            # plt.margins(0, 0)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig(f"figs/fig64/{model_weight_name}")   #显示窗口
        plt.clf()
        plt.close()
    t1_end = time.process_time()

    # path = 'record_time.txt'
    # with open(path, 'a') as f:
    #     f.write(f"sm\n")
    #     f.write(f"time of the whole process {t1_end - t1_start}\n")