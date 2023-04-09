import torch
import numpy as np
from skimage.metrics import structural_similarity
#np.seterr(divide='ignore',invalid='ignore')


def calc_ergas(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = torch.mean((img_tgt - img_fus) ** 2)
    rmse = rmse ** 0.5
    mean = torch.mean(img_tgt)

    ergas = torch.mean((rmse / mean) ** 2)
    ergas = 100 / 4 * ergas ** 0.5

    return ergas.item()

def calc_psnr(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    mse = torch.mean(torch.square(img_tgt-img_fus))
    img_max = torch.max(img_tgt)
    #img_max = 1.0
    psnr = 10.0 * torch.log10(img_max**2/mse)

    return psnr.item()

def calc_rmse(img_tgt, img_fus):

    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = torch.sqrt(torch.mean((img_tgt-img_fus)**2))

    return rmse.item()

def calc_sam(img_tgt, img_fus):
    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[1], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[1], -1)
    img_tgt = img_tgt / torch.max(img_tgt)
    img_fus = img_fus / torch.max(img_fus)

    A = torch.sqrt(torch.sum(img_tgt**2))
    B = torch.sqrt(torch.sum(img_fus**2))
    AB = torch.sum(img_tgt*img_fus)

    sam = AB/(A*B)

    sam = torch.arccos(sam)
    sam = torch.mean(sam)*180/torch.pi

    return sam.item()

def calc_ssim(img_tgt, img_fus):
    '''
    :param reference:
    :param target:
    :return:
    '''

    img_tgt = torch.squeeze(img_tgt)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = torch.squeeze(img_fus)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt.cpu().numpy()
    img_fus = img_fus.cpu().numpy()

    ssim = structural_similarity(img_tgt, img_fus)

    return ssim