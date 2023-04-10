# Multiscale Spatial–spectral Transformer Network for Hyperspectral and Multispectral Image Fusion (INFFUS 2023)


[Sen Jia](https://scholar.google.com.hk/citations?hl=zh-CN&user=UxbDMKoAAAAJ), Zhichao Min, [Xiyou Fu](https://scholar.google.com.hk/citations?user=DFgGGCQAAAAJ&hl=zh-CN&oi=sra)



<hr />

> **Abstract:** *Fusing hyperspectral images (HSIs) and multispectral images (MSIs) is an economic and feasible way to obtain images with both high spectral resolution and spatial resolution. Due to the limited receptive field of convolution kernels, fusion methods based on convolutional neural networks (CNNs) fail to take advantage
of the global relationship in a feature map. In this paper, to exploit the powerful capability of Transformer to extract global information from the whole feature map for fusion, we propose a novel Multiscale Spatial–spectral Transformer Network (MSST-Net). The proposed network is a two-branch network that integrates the self-attention mechanism of the Transformer to extract spectral features from HSI and spatial features from MSI, respectively. Before feature extraction, cross-modality concatenations are performed to achieve cross-modality information interaction between the two branches. Then, we propose a spectral Transformer (SpeT) to extract spectral features and introduce multiscale band/patch embeddings to obtain multiscale features through SpeTs and spatial Transformers (SpaTs). To further improve the network’s performance and generalization, we proposed a self-supervised pre-training strategy, in which a masked bands autoencoder (MBAE) and a masked patches autoencoder (MPAE) are specially designed for self-supervised pre-training of the SpeTs and SpaTs. Extensive experiments on simulated and real datasets illustrate that the proposed network can achieve better performance when compared to other state-of-the-art fusion methods. The code of MSST-Net will be available at http://www.jiasen.tech/papers/ for the sake of reproducibility.* 
<hr />


## Network Architecture
<!-- ![Illustration of MSST-Net](figure/framework.png) -->
<div aligh=center witdh="200"><img src="figure/framework.png"></div>



<!-- ![Illustration of hsi pretrain](figure/pretrain_hsi.png) -->
<img src="figure/pretrain_hsi.png" aligh=center witdh="50px">

<!-- ![Illustration of msi pretrain](figure/pretrain_msi.png) -->
<img src="figure/pretrain_msi.png" aligh=center witdh="50px">



## 1. Create Envirement:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

  ```shell
  cd graph-in-graph
  pip install -r requirements.txt
  ```

## 2. Data Preparation:
- Download the data including raw `.mat` files and corresponding `.jpg` files used in superpixel segmentation from <a href="https://pan.baidu.com/s/1In_ySXoMG7DP5Q1hEyOzXA">here</a> (code: 4zyf) for a quick start and place them in `GiGCN/`.

- Before trainig, every data set is split by runing `trainTestSplit.py`, shown as follow:

  ```shell
  python trainTestSplit.py --name PaviaU (data set name)
  ```

## 3. Training

To train a model, run

```shell
# Training on PaviaU data set
python train.py --name PaviaU --block 100 --gpu 0
```
Here, `--block` denots the number of superpixel, which lies in `[50, 100, 150, 200]` in our ensemble setup.

The model with best accuracy will be saved.

Note: The `scikit-image` package in our experimental configuaration is of version 0.15.0 whose parameter `start_label` defaults to 0. However, in the lastest version, it defaults to 1. So when encountering the problem that indexes are out of the bounder at `Line 54` in `Trainer.py`, you should set `start_label` as 0 explicitly.

## 4. Prediction:

To test a trained model, run 

```shell
# Testing on PaviaU data set
python predict.py --name PaviaU --block 100 --gpu 0
```
The code will load the best model in the last phase automatically.


## Citation
If this repo helps you, please consider citing our works:


```
@ARTICLE{jia2023multiscale,
  title={Multiscale spatial-spectral transformer network for hyperspectral and multispectral image fusion},
  author={Jia, Sen and Min, Zhichao and Fu, Xiyou},
  journal={Information Fusion}, 
  year={2023},
  volume={96},
  pages={117-129},
}
```
