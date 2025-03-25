from tkinter.font import names

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from model_1 import *

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    Total_data = 50000
    batch = 64#5000
    dtype = torch.float32
    Total_iter = Total_data // batch + (Total_data % batch != 0)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model_weight_name = "cifar10_6"
    # model_weight_name = "cm_1"
    device = torch.device(device_str)
    print(device)





    eps_theta = Unet()
    checkpoint = torch.load(f"weights/eps_theta_{model_weight_name}.pt",weights_only=False)
    eps_theta.load_state_dict(checkpoint['model_state_dict'])
    eps_theta = eps_theta.eval()
    eps_theta = torch.compile(eps_theta.to(device, dtype=dtype),fullgraph=True)
    loss_set = checkpoint['loss_set']
    print(len(loss_set))
    # asd/2
    # N = checkpoint['step']
    N = 1000

    beta = torch.linspace(0.0001, 0.02, N,device=device, dtype=dtype)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)



    #X_data = torch.empty(batch, 3, 32, 32)#.to(device)
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
            x = torch.randn(num_of_data, 3, 32, 32,device=device, dtype=dtype)
            for i in range(N, 1, -1):
                idx = torch.ones(len(x),device=device, dtype=dtype) * (i-1)
                prediction = eps_theta(x, idx)

                # DDPM reverse
                x = (x - (1 - alpha[i - 1]) / torch.sqrt(1 - alpha_bar[i - 1]) * prediction.squeeze(
                    -1)) / torch.sqrt(alpha[i - 1]) \
                    + ((1 - alpha[i - 1]) * (1 - alpha_bar[i - 2]) / (
                            1 - alpha_bar[i - 1])).sqrt() * torch.randn_like(x, device=device, dtype=dtype)  #

                # x_motion = torch.hstack([x_motion, x.unsqueeze(-1)])
                pbar.set_description(f"current batch reverse process: {(N - i)} / {N} ")
            idx = torch.zeros(len(x),device=device, dtype=dtype)
            prediction = eps_theta(x, idx)

            x = (x  - (1 - alpha[0]) / torch.sqrt(1 - alpha_bar[0]) * prediction) / torch.sqrt(alpha[0])
            torch.cuda.empty_cache()
        # x_motion = torch.hstack([x_motion, x.unsqueeze(-1)])

        # result.append(x)   # Save the result
        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = np.clip(x_temp, 0.0, 1.0)
        results.append((x_temp * 255).astype(np.uint8))




    #results = np.concatenate(results, axis=0)
    #print(results.dtype, results.shape)
    #np.save(f"result_2/{model_weight_name}_result.npy", results) # (batch, 32, 32, 3), dtype = uint8 range(0, 255)

        # # x_motion = resize(x_motion.squeeze(-1))
        # save per image
        # for j in range(num_of_data):
        #     plt.figure(figsize=(32, 32), dpi=1)
        #     plt.imshow(x_temp[j])
        #
        #     plt.axis("off")
        #     plt.margins(0, 0)
        #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
        #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
        #
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #     plt.savefig(f"result_4/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
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
        plt.savefig("6_64_in_one")   #显示窗口
        plt.clf()
        plt.close()

