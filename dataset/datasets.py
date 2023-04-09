import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio
import torch
from utils import normalize


class CaveDataset(Dataset):
    def __init__(self, mat_save_path, patch_size=16*1, stride=6*1, ratio=8, type='train'):
        super(CaveDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.stride = stride
        self.rows = 64*1
        self.cols = 64*1
        self.patch_size = patch_size
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
            self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_size, self.ratio, num_star=1, num_end=23, s=9)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_size=64*1, ratio=ratio, num_star=23, num_end=28, s=1)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_size=16*1, ratio=ratio, num_star=28, num_end=33, s=4)


    def generateTrain(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, self.stride):
                for y in range(0, self.cols - patch_size + 1, self.stride):
                    # rotTimes = random.randint(0, 3)
                    # vFlip = random.randint(0, 1)
                    # hFlip = random.randint(0, 1)
                    # label_patch[count] = self.arguement(hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # hrmsi_patch[count] = self.arguement(hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :], rotTimes, vFlip, hFlip)
                    # lrhsi_patch[count] = self.arguement(lrhsi[x:x + patch_size, y:y + patch_size, :], rotTimes, vFlip, hFlip)
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, patch_size):
                for y in range(0, self.cols - patch_size + 1, patch_size):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_size, ratio, num_star, num_end, s):
        label_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s * s * (num_end - num_star), patch_size * ratio, patch_size * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s * s * (num_end - num_star), patch_size, patch_size, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']


            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_size + 1, patch_size):
                for y in range(0, self.cols - patch_size + 1, patch_size):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_size) * ratio, y * ratio:(y + patch_size) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_size, y:y + patch_size, :]
                    count += 1

        return lrhsi_patch, hrmsi_patch, label_patch

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(0, 1))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[::-1, :, :].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))

        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]

class HarvardDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=16*1, patch_w=16*1, h_stride=6*1, w_stride=12*1, ratio=8, type='train'):
        super(HarvardDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.rows = 130*1
        self.cols = 174*1
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio, num_star=1, num_end=35, s_h=20, s_w=14)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=128*1, patch_w=80*1, ratio=ratio, num_star=43, num_end=51, s_h=1, s_w=2)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=64*1, patch_w=64*1, ratio=ratio, num_star=35, num_end=43, s_h=2, s_w=2)


    def generateTrain(self, patch_h, patch_w, ratio, num_star, num_end, s_h, s_w):
        label_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h * ratio, patch_w * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h * ratio, patch_w * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h, patch_w, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_h + 1, self.h_stride):
                for y in range(0, self.cols - patch_w + 1, self.w_stride):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio, num_star, num_end, s_h, s_w):
        label_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h * ratio, patch_w * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h * ratio, patch_w * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h, patch_w, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_h + 1, patch_h):
                for y in range(0, self.cols - patch_w + 1, patch_w):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio, num_star, num_end, s_h, s_w):
        label_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h * ratio, patch_w * ratio, 31), dtype=np.float32)
        hrmsi_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h * ratio, patch_w * ratio, 3), dtype=np.float32)
        lrhsi_patch = np.zeros((s_h * s_w * (num_end - num_star), patch_h, patch_w, 31), dtype=np.float32)
        count = 0
        for i in range(num_star, num_end):
            mat = scio.loadmat(self.mat_save_path + '%d.mat' % i)
            hrhsi = mat['hrhsi']
            lrhsi = mat['lrhsi']
            hrmsi = mat['hrmsi']

            # Data type conversion
            if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
            if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
            if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

            for x in range(0, self.rows - patch_h + 1, patch_h):
                for y in range(0, self.cols - patch_w + 1, patch_w):
                    label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                    hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                    lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                    count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]

class WDCMDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=16*1, patch_w=16*1, h_stride=6*1, w_stride=3*1, ratio=8, type='train'):
        super(WDCMDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.rows = 144*1
        self.cols = 37*1
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio, s_h=25, s_w=8)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=16*1, patch_w=16*1, ratio=ratio)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=16*1, patch_w=16*1, ratio=ratio)

    def getData(self, ratio):
        hrhsi = scio.loadmat(self.mat_save_path + 'DC_HRHSI.mat')['S']
        lrhsi = scio.loadmat(self.mat_save_path + 'DC_LRHSI_ratio{}.mat'.format(ratio))['HSI']
        hrmsi = scio.loadmat(self.mat_save_path + 'DC_MSI_Band10.mat')['MSI']
        # Data normalization and scaling[0, 1]
        hrhsi = normalize(hrhsi)
        lrhsi = normalize(lrhsi)
        hrmsi = normalize(hrmsi)

        return hrhsi, lrhsi, hrmsi

    def generateTrain(self, patch_h, patch_w, ratio, s_h, s_w):
        label_patch = np.zeros((s_h * s_w, patch_h * ratio, patch_w * ratio, 191), dtype=np.float32)
        hrmsi_patch = np.zeros((s_h * s_w, patch_h * ratio, patch_w * ratio, 10), dtype=np.float32)
        lrhsi_patch = np.zeros((s_h * s_w, patch_h, patch_w, 191), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[:self.rows*ratio, :, :]
        lrhsi = lrhsi[:self.rows, :, :]
        hrhsi = hrhsi[:self.rows*ratio, :, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h + 1, self.h_stride):
            for y in range(0, self.cols - patch_w + 1, self.w_stride):
                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 191), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 10), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 191), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[self.rows * ratio:, :patch_w*ratio, :]
        lrhsi = lrhsi[self.rows:, :patch_w, :]
        hrmsi = hrmsi[self.rows * ratio:, :patch_w*ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def generateTest(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 191), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 10), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 191), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[self.rows * ratio:, patch_w*ratio:patch_w*ratio*2, :]
        lrhsi = lrhsi[self.rows:, patch_w:patch_w*2, :]
        hrmsi = hrmsi[self.rows * ratio:, patch_w*ratio:patch_w*ratio*2, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch

    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]

class YREDataset(Dataset):
    def __init__(self, mat_save_path, patch_h=32, patch_w=32, h_stride=12, w_stride=12, ratio=3, type='train'):
        super(YREDataset, self).__init__()
        self.mat_save_path = mat_save_path
        self.h_stride = h_stride
        self.w_stride = w_stride
        self.rows = 392
        self.cols = 452
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.ratio = ratio
        self.type = type
        # Generate samples and labels
        if self.type=='train':
           self.hsi_data, self.msi_data, self.label = self.generateTrain(self.patch_h, self.patch_w, self.ratio, s_h=31, s_w=36)
        if self.type=='eval':
            self.hsi_data, self.msi_data, self.label = self.generateEval(patch_h=64, patch_w=64, ratio=ratio)
        if self.type=='test':
            self.hsi_data, self.msi_data, self.label = self.generateTest(patch_h=64, patch_w=64, ratio=ratio)

    def getData(self, ratio):
        mat = scio.loadmat(self.mat_save_path + 'GF5.mat')
        hrhsi = mat['hrhsi']
        lrhsi = mat['lrhsi']
        hrmsi = mat['hrmsi']

        return hrhsi, lrhsi, hrmsi

    def generateTrain(self, patch_h, patch_w, ratio, s_h, s_w):
        label_patch = np.zeros((s_h * s_w, patch_h * ratio, patch_w * ratio, 280), dtype=np.float32)
        hrmsi_patch = np.zeros((s_h * s_w, patch_h * ratio, patch_w * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((s_h * s_w, patch_h, patch_w, 280), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[:self.rows * ratio, :self.cols * ratio, :]
        lrhsi = lrhsi[:self.rows, :self.cols, :]
        hrhsi = hrhsi[:self.rows * ratio, :self.cols * ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        for x in range(0, self.rows - patch_h + 1, self.h_stride):
            for y in range(0, self.cols - patch_w + 1, self.w_stride):
                label_patch[count] = hrhsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                hrmsi_patch[count] = hrmsi[x * ratio:(x + patch_h) * ratio, y * ratio:(y + patch_w) * ratio, :]
                lrhsi_patch[count] = lrhsi[x:x + patch_h, y:y + patch_w, :]
                count += 1
        return lrhsi_patch, hrmsi_patch, label_patch

    def generateEval(self, patch_h, patch_w, ratio):
        label_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 280), dtype=np.float32)
        hrmsi_patch = np.zeros((1, patch_h * ratio, patch_w * ratio, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((1, patch_h, patch_w, 280), dtype=np.float32)
        count = 0

        hrhsi, lrhsi, hrmsi = self.getData(ratio)
        hrhsi = hrhsi[396 * ratio:, :patch_w*ratio, :]
        lrhsi = lrhsi[396:, :patch_w, :]
        hrmsi = hrmsi[396 * ratio:, :patch_w*ratio, :]

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch


    def generateTest(self):
        label_patch = np.zeros((1, 576, 576, 4), dtype=np.float32)
        hrmsi_patch = np.zeros((1, 576, 576, 4), dtype=np.float32)
        lrhsi_patch = np.zeros((1, 192, 192, 280), dtype=np.float32)
        count = 0

        mat = scio.loadmat(self.mat_save_path + 'YRE_576_3.mat')
        hrhsi = mat['hrmsi']
        lrhsi = mat['lrhsi']
        hrmsi = mat['hrmsi']
        # Data normalization and scaling[0, 1]
        hrhsi = normalize(hrhsi)
        lrhsi = normalize(lrhsi)
        hrmsi = normalize(hrmsi)

        # Data type conversion
        if hrhsi.dtype != np.float32: hrhsi = hrhsi.astype(np.float32)
        if lrhsi.dtype != np.float32: lrhsi = lrhsi.astype(np.float32)
        if hrmsi.dtype != np.float32: hrmsi = hrmsi.astype(np.float32)

        label_patch[count] = hrhsi
        hrmsi_patch[count] = hrmsi
        lrhsi_patch[count] = lrhsi

        return lrhsi_patch, hrmsi_patch, label_patch


    def __getitem__(self, index):
        hrhsi = np.transpose(self.label[index], (2,0,1))
        hrmsi = np.transpose(self.msi_data[index], (2,0,1))
        lrhsi = np.transpose(self.hsi_data[index], (2,0,1))
        sample = {'hrhsi': torch.tensor(hrhsi, dtype=torch.float32),
                  'hrmsi': torch.tensor(hrmsi, dtype=torch.float32),
                  'lrhsi': torch.tensor(lrhsi, dtype=torch.float32)
                  }
        return sample

    def __len__(self):
        return self.label.shape[0]