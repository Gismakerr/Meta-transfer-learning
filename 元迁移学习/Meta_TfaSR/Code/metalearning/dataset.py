import numpy as np
import random
from imresize import imresize as resize
# !pip install h5py
#import h5py
import torch.utils.data as data
import torchvision.transforms as transforms
from gkernel import *
from imresize import imresize
from source_target_transforms import *
import os
from PIL import Image
import cv2
import torch
from osgeo import gdal
from myutils1 import *

META_BATCH_SIZE = 4
TASK_BATCH_SIZE = 8




class preTrainDataset(data.Dataset):
    def __init__(self, path, patch_size=64, scale=[2,4]):
        super(preTrainDataset, self).__init__()
        self.patch_size = patch_size
        self.scale = scale
        # path是所有文件的主路径
        self.path = path
        files = os.listdir(path)
        self.data_list = []
        LR_list = []
        # path下的分路径HR、LR
        HR_path = path + "\\HR"
        LR_path = path + "\\LR"
        self.data_list.append(os.listdir(HR_path))
        for j, LR_files in enumerate(os.listdir(LR_path)):
            LR_list.append(os.listdir(LR_path+'\\'+LR_files))
        self.data_list.append(LR_list)

        """
        h5f = h5py.File(path, 'r')
        self.hr = [v[:] for v in h5f["HR"].values()]
        if scale == 0:
            self.scale = [2, 3, 4]
            self.lr = [[v[:] for v in h5f["X{}".format(i)].values()] for i in self.scale]
        else:
            self.scale = [scale]
            self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]
        
        h5f.close()        
        """
        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(self.patch_size),
            ToTensor()
        ])



    def __getitem__(self, index):
        item = []
        HR_path = self.path + "\\HR\\"+self.data_list[0][index]
        HR_data = read_dem(HR_path)
        HR_max, HR_min = GetMaxMin(HR_data)
        HR_data_norm = GetNormalData(HR_data, HR_max, HR_min)
        for i in range(len(self.scale)):
            LR_data = Resize(HR_data, HR_data.shape[0] // self.scale[i], HR_data.shape[1] // self.scale[i], "LINEAR")
            LR_data_noise = add_noise(LR_data, noise_type="gaussian", noise_level = 3)
            LR_data_recover = Resize(LR_data_noise, HR_data.shape[0], HR_data.shape[1], "NEAREST")
            LR_max, LR_min = GetMaxMin(LR_data_recover)
            LR_data_norm = GetNormalData(LR_data_recover, LR_max, LR_min)
            item.append([HR_data_norm, LR_data_norm])
        

        """
        HR_path = self.path + "\\HR\\"+self.data_list[0][index]
        LR_path = []
        for i in range(self.scale-1):
            LR_path.append(self.path + "\\LR\\" + 'x'+ str(i+2) + '\\' + self.data_list[1][i][index])
        HR = np.array(Image.open(HR_path))
        LR = [np.array(Image.open(lr_path)) for lr_path in LR_path]
        item = [(HR, resize(LR[i], i+2, kernel='cubic').astype(np.uint8)) for i in range(self.scale-1)]
        """

        #item = [(self.hr[index], resize(self.lr[i][index], self.scale*100, interp='cubic')) for i, _ in enumerate(self.lr)]
        # return [(self.transform(hr), self.transform(imresize(lr, 400, interp='cubic'))) for hr, lr in item]
        return [self.transform((hr,lr)) for hr, lr in item]

    def __len__(self):
        return len(self.data_list[0])
    
def file_filter(f):
    if "icesat" in f:
        return False
    else:
        return True

class metaTrainDataset(data.Dataset):
    def __init__(self, path, patch_size=64, scale=[2,4]):
        super(metaTrainDataset, self).__init__()
        self.scale = scale
        self.size = patch_size

        # path是所有文件的主路径
        self.path = path
        self.data_list = []
        # path下的分路径HR、LR
        catergory_files = os.listdir(self.path)
        for catergory_file in catergory_files:
            catergory_data = os.listdir(os.path.join(self.path, catergory_file))
            catergory_data = list(filter(file_filter, catergory_data))
            catergory_data = random.sample(catergory_data, TASK_BATCH_SIZE*2)
            self.data_list.extend(catergory_data)
        #LR_path = path + "\\LR"
        #for j, LR_files in enumerate(os.listdir(LR_path)):
            #LR_list.append(os.listdir(LR_path+'\\'+LR_files))
        #self.data_list.append(LR_list)

        """
        h5f = h5py.File(path, 'r')
        self.hr = [v[:] for v in h5f["HR"].values()]
        self.hr = random.sample(self.hr, TASK_BATCH_SIZE*2*META_BATCH_SIZE)
        h5f.close()
        """

        # self.tansform = transforms.Compose([
        #     transforms.RandomCrop(64),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip()
        #     # transforms.ToTensor()
        # ])

    def __getitem__(self, index):
        item = []
        HR_path = os.path.join(self.path, self.data_list[index].split('_')[0][:-1], self.data_list[index])
        # HR_icesat = os.path.join(self.path, self.data_list[index].split('_')[0][:-1], self.data_list[index].split("_")[0]+"_icesat_"+self.data_list[index].split("_")[1]+"_"+self.data_list[index].split("_")[2])
        # HR_icesat_data = read_dem(HR_icesat)
        HR_data = read_dem(HR_path)
        item.append((HR_data))
        """归一化(先下采样再归一化，这里先不进行操作，在make_data_tensor中操作)"""
        """
        HR_max, HR_min = GetMaxMin(HR_data)
        HR_data_norm = GetNormalData(HR_data, HR_max, HR_min)
        for i in range(len(self.scale)):
            LR_data = Resize(HR_data, HR_data.shape[0] // self.scale[i], HR_data.shape[1] // self.scale[i], "LINEAR")
            LR_data_noise = add_noise(LR_data, noise_type="gaussian", noise_level = 3)
            LR_data_recover = Resize(LR_data_noise, HR_data.shape[0], HR_data.shape[1], "NEAREST")
            LR_max, LR_min = GetMaxMin(LR_data_recover)
            LR_data_norm = GetNormalData(LR_data_recover, LR_max, LR_min)
            item.append([HR_data_norm, LR_data_norm])
        """

        """
        HR_path = self.path + "\\HR\\"+self.data_list[0][index]
        #LR_path = []
        #for i in range(self.scale-1):
            #LR_path.append(self.path + "\\LR\\" + 'x'+ str(i+2) + '\\' + self.data_list[1][i][index])
        HR = np.array(Image.open(HR_path))
        #LR = [np.array(Image.open(lr_path)) for lr_path in LR_path]
        item = [HR/255.]
        item = [random_crop(hr, self.size) for hr in item]
        """

        """
        item = [self.hr[index]/255.]
        item = [random_crop(hr,self.size) for hr in item]
        """
        return [random_flip_and_rotate(hr) for hr in item]
    def __len__(self):
        return len(self.data_list)

def make_data_tensor(scale, noise_std=0.0):
    label_train = metaTrainDataset(r'D:\元迁移学习\To_黄涛\元迁移学习\Meta_TfaSR\Code\metalearning\meta_data')

    input_meta = []
    label_meta = []
    icesat_meta = []

    for t in range(META_BATCH_SIZE):
        input_task = []
        label_task = []
        input_icesat = []
        for idx in range(TASK_BATCH_SIZE*2):
            """归一化和生成低分辨率"""
            img_HR = label_train[t*TASK_BATCH_SIZE*2 + idx][-1]
            #HR_icesat_index = get_index(HR_icesat, "")
            img_LR = Resize(img_HR, img_HR.shape[0] // scale, img_HR.shape[1] // scale, "LINEAR")
            
            #info = read_tif(r'H:\meta_data\\'+label_train.data_list[t*TASK_BATCH_SIZE*2 + idx].split('_')[0][:-1]+'\\'+label_train.data_list[t*TASK_BATCH_SIZE*2 + idx])
            #write_tif(img_LR, info, r'H:\meta_data\\'+label_train.data_list[t*TASK_BATCH_SIZE*2 + idx].split('_')[0][:-1]+'\\test.tif')
            img_LR_noise = add_noise(img_LR, noise_type="gaussian", noise_level = 0)
            
            img_LR_recover = Resize(img_LR_noise, img_HR.shape[0], img_HR.shape[1], "NEAREST")
            LR_max, LR_min = GetMaxMin(img_LR)
            LR_data_norm = GetNormalData(img_LR_recover, LR_max, LR_min)
            
            HR_data_norm = GetNormalData(img_HR, LR_max, LR_min)
            #HR_icesat = GetNormalData(HR_icesat, LR_max, LR_min) * HR_icesat_index

            """
            img_HR = label_train[t*TASK_BATCH_SIZE*2 + idx][-1]
            # add isotropic and anisotropic Gaussian kernels for the blur kernels 
            # and downsample 
            clean_img_LR = imresize(img_HR, scale=1./scale, kernel=Kernel)
            # add noise
            img_LR = np.clip(clean_img_LR + np.random.randn(*clean_img_LR.shape)*noise_std, 0., 1.)
            # used cubic upsample 
            img_ILR = imresize(img_LR,scale=scale, output_shape=img_HR.shape, kernel='cubic')
            """

            input_task.append(LR_data_norm)
            label_task.append(HR_data_norm)
            
        
        input_meta.append(np.asarray(input_task))
        label_meta.append(np.asarray(label_task))
        
    
    input_meta = np.asarray(input_meta)
    label_meta = np.asarray(label_meta)
    inputa = input_meta[:,:TASK_BATCH_SIZE,:,:]
    labela = label_meta[:,:TASK_BATCH_SIZE,:,:]
   
    inputb = input_meta[:,TASK_BATCH_SIZE:,:,:]
    labelb = label_meta[:,TASK_BATCH_SIZE:,:,:]

    return inputa, labela, inputb, labelb

if __name__ == '__main__':
    make_data_tensor(4)