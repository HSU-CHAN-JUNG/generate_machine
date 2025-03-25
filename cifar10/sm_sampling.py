import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from model_1_64 import *

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    Total_data = 30000
    batch = 64#5000
    dtype = torch.float32
    Total_iter = Total_data // batch + (Total_data % batch != 0)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_weight_name = "sm_1"
    device = torch.device(device_str)
    print(device)

    eps_theta = Unet()
    # checkpoint = torch.load(f"weights/eps_theta_{model_weight_name}.pt",weights_only=False)
    checkpoint = torch.load(f"eps_theta_sm_1.pt",weights_only=False)
    eps_theta.load_state_dict(checkpoint['model_state_dict'])
    eps_theta = eps_theta.eval()
    eps_theta = torch.compile(eps_theta.to(device, dtype=dtype),fullgraph=True)
    loss_set = checkpoint['loss_set']
    print(len(loss_set))
    # N = checkpoint['step']
    N = 1000
    # asd/2
    beta_max = 20
    beta_min = 0.1
    T = 1
    dt = T / N
    beta = lambda t: beta_min + (beta_max - beta_min) * t
    mu_t = lambda t: np.exp(-beta_min * t / 2 - (beta_max - beta_min) * t ** 2 / 4)
    sigma_t = lambda t: np.sqrt(1 - np.exp(-beta_min * t - (beta_max - beta_min) * t ** 2 / 2))

    exact_eps = lambda x, t: (x - mu_t(t) * X_data.to(device)) / sigma_t(t)

    # X_data = torch.empty(batch, 3, 32, 32)#.to(device)
    # X_data, _ = next(iter(trainLoader))
    # X_data = torch.Tensor(X_data).to(device)
    results =[]

    # pbar = tqdm(range(Total_iter))
    pbar = tqdm(range(1))
    for k in pbar:
        stard_id = k * batch
        end_id = min((k + 1) * batch, Total_data)
        num_of_data = end_id - stard_id
        # backward process
        with torch.inference_mode(), torch.amp.autocast(enabled=True, device_type=device_str, dtype=torch.bfloat16):
            #with torch.amp.autocast():
            x = torch.randn(num_of_data, 3, 64, 64,device=device, dtype=dtype)
            for i in range(N, 1, -1):
                time = i * dt
                # DDPM reverse
                noise = torch.randn_like(x).to(device)
                s_theta = eps_theta(x, time * torch.ones(len(x), device=device))
                # x = (1 + beta(time)*dt/2) * x - beta(time) * s_theta/ sigma_t(time) * dt\
                # x = (1 + beta(time)*dt/2) * x - beta(time) * s_theta * dt/ sigma_t(time) / np.sqrt(1 - beta(time))\
                x = (1 + beta(time) * dt / 2) * x - beta(time) * s_theta * dt / sigma_t(time) \
                    + np.sqrt(beta(time) * dt) * noise

                # x = (1 + beta(time)*dt/2) * x - beta(time) * exact_eps(x, time) * dt / sigma_t(time)\
                #     + np.sqrt(beta(time) * dt) * noise


                # x_motion = torch.cat([x_motion, x.unsqueeze(-1)], dim=-1)
                pbar.set_description(f"current batch reverse process: {(N - i)} / {N} ")
            idx = torch.zeros(len(x),device=device, dtype=dtype)
            prediction = eps_theta(x, idx)

            x = (1 + beta(time) * dt / 2) * x - beta(time) * s_theta * dt / sigma_t(time)
            torch.cuda.empty_cache()
        # x_motion = torch.cat([x_motion, x.unsqueeze(-1)], dim=-1)

        # result.append(x)   # Save the result
        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = np.clip(x_temp, 0.0, 1.0)
        # results.append((x_temp * 255).astype(np.uint8))




    #results = np.concatenate(results, axis=0)
    #print(results.dtype, results.shape)
    #np.save(f"result_2/{model_weight_name}_result.npy", results) # (batch, 32, 32, 3), dtype = uint8 range(0, 255)

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
        #     plt.savefig(f"cat_sm_result/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
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
            plt.imshow(x_temp[j])
            plt.axis("off")
            # plt.margins(0, 0)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig.tight_layout()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.savefig("sm_cifar10_64_in_one_1")   #显示窗口
        plt.clf()
        plt.close()

