import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as dset
import torchvision
from torch.distributed.tensor.parallel import loss_parallel
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm
from model import *
import lpips
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sampling(time, num_of_data, f_theta):
    '''
    Sampling function for the consistency model.
    Args:
        time (list): List of time steps for the sampling process.
        num_of_data (int): Number of data samples to generate.
        f_theta (nn.Module): The consistency model to use for sampling.
    Returns:
        x (torch.Tensor): Generated samples of shape (num_of_data, 3, 64, 64).
    '''
    # backward process
    with torch.inference_mode():
        x_hat = torch.randn(num_of_data, 3, 64, 64, device=device, dtype=dtype) * T
        T_time = torch.ones(len(x_hat), device=device, dtype=dtype) * T
        sigma_data = 0.5
        c_skip = lambda t: sigma_data ** 2 / ((t - epsilon) ** 2 + sigma_data ** 2)
        c_out = lambda t: sigma_data * (t - epsilon) / np.sqrt(t ** 2 + sigma_data ** 2)
        x = c_skip(T) * x_hat + c_out(T) * f_theta(x_hat, T_time)
        sampling_scheldule = time
        for i in sampling_scheldule:
            noise = torch.randn_like(x_hat)
            time = i
            time_tensor = time * \
                          torch.ones(len(x_hat), device=device, dtype=dtype)
            x = x + time_tensor[:, None, None, None] * noise
            x = c_skip(time) * x + c_out(time) * f_theta(x, time_tensor)
        # x = (x + 1) / 2
    return x


def get_statistics(image, model, batch_size=1, dims=2048, device="cpu", num_workers=1):
    '''
    Calculate the mean and covariance of the features extracted from the images using the InceptionV3 model.
    Args:
        image (torch.Tensor): Input images of shape (N, C, H, W).
        model (nn.Module): Pre-trained InceptionV3 model.
        batch_size (int): Batch size for processing images.
        dims (int): Dimensionality of the features to extract.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_workers (int): Number of workers for data loading.
    '''
    with torch.no_grad():
       pred = model(image)[0]
    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred = pred.squeeze(3).squeeze(2).cpu().numpy()

    mu = np.mean(pred, axis=0)
    sigma = np.cov(pred, rowvar=False)
    return mu, sigma


def fid(time):
    ''' 
    Calculate the FID score for the generated samples at a specific time step.
    Args:
        time (list): List of time steps for the sampling process.
    Returns:
        fid_value (float): The calculated FID score.
    '''
    sample = sampling(time)
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    checkpoint = torch.load("mnist_act.pt")
    m1, s1 = checkpoint["mean"], checkpoint["std"]
    m2, s2 = get_statistics(sample,
                            model,
                            dims=dims,
                            device=device)
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    torch.cuda.empty_cache()
    print(time, fid_value)
    return fid_value

if __name__ == "__main__":
    t1_start = time.process_time()
    seed_everything(1)

    ### Set parameters ###
    torch.set_float32_matmul_precision('high')
    dtype = torch.float32

    Total_data = 10000   # Total number of cat faces = 29842
    batch = 32
    Total_iter = Total_data // batch + (Total_data % batch != 0)
    model_weight_name = "cm_1"
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(device)

    ### construct model and load weights ###
    f_theta = Unet().to(device)
    checkpoint = torch.load(f"weights/{model_weight_name}.pt",weights_only=False,map_location=device)
    f_theta.load_state_dict(checkpoint['model_state_dict'])
    f_theta = f_theta.eval()
    loss_set = checkpoint['loss_set']
    print(len(loss_set))
    
    N = lambda k: 80
    epsilon = 0.002
    T = 76
    ro = 7
    time_schedule = lambda i, k: (epsilon ** (1 / ro) + (i - 1) / (N(k) - 1) *(
                T ** (1 / ro) - epsilon ** (1 / ro))) ** ro

    pbar = tqdm(range(Total_iter))
    # pbar = tqdm(range(1))
    for k in pbar:

        stard_id = k * batch
        end_id = min((k + 1) * batch, Total_data)
        num_of_data = end_id - stard_id
        x = sampling([40], num_of_data, f_theta)
        x = (x+1.0)/2

        x_temp = x.to(device="cpu", dtype=torch.float32).numpy().transpose((0, 2, 3, 1))  # [0][999]
        x_temp = np.clip(x_temp, 0.0, 1.0)
  
        ### save per image
        for j in range(num_of_data):
            plt.figure(figsize=(64, 64), dpi=1)
            plt.imshow(x_temp[j])

            plt.axis("off")
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.savefig(f"results/cm6/fig{k*batch+j}",dpi = 1, pad_inches = 0)   #显示窗口
            plt.clf()
            plt.close()


    # t1_end = time.process_time()
    # path = 'record_time.txt'
    # with open(path, 'a') as f:
    #     f.write(f"consistency model\n")
    #     f.write(f"time of the whole process {t1_end - t1_start}\n")
    #     f.write(f"time of the whole loop {t2_end - t2_start}\n")
    #     f.write(f"time of the one loop {t3_end - t3_start}\n")


        # result = torch.cat(result, dim=0)
        # torch.save(result, f"result/x.pth")
        # print(result.shape)
        #         asd/2

        # fig = plt.figure(figsize=(32, 32))
        # num_fig = 8
        # for j in range(num_of_data):
        #     plt.subplot(num_fig, num_fig, j + 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
        #     plt.imshow(x_temp[j],)
        #     plt.axis("off")
        #     # plt.margins(0, 0)
        #
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # fig.tight_layout()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.savefig(f"figs/fig64/{model_weight_name}_2")  # 显示窗口
        # plt.clf()
        # plt.close()


