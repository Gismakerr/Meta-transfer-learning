import os
import time
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset
from myutils import *
from osgeo import gdal

import model
import mymodel
from config import *
import cv2

def GetNormalData(data_temp,max,min):
    normal_img = 2 * (data_temp - min) / (max - min + 10) - 1
    return normal_img

def read_tif(filepath):
    """
    功能：
    ——————
    读取tif文件
    
    参数：
    ——————
    filepath：tif文件路径
    
    输出：
    ——————
    [[宽，高，波段数，相似变换信息，投影信息，数据矩阵]]
    """
    
    dataset = gdal.Open(filepath)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    data = dataset.ReadAsArray(0, 0, width, height)
    info = []
    info.append([width, height, bands, geotrans, proj, data])
    return info

def Resize(img_temp, H, W, method):
    if isinstance(img_temp, torch.Tensor):
        device = img_temp.device
        datatype = img_temp.dtype
        
        img_temp = img_temp[0][0]
        img_temp = img_temp.detach().cpu().numpy()
        if method == "LINEAR":
            img_temp = cv2.resize(img_temp, (H, W),interpolation=cv2.INTER_LINEAR)
        elif method == "NEAREST":
            img_temp = cv2.resize(img_temp, (H, W), interpolation=cv2.INTER_NEAREST)
        result_tensor = torch.tensor(img_temp, dtype = datatype, device=device)
        result_tensor = result_tensor.reshape((1,1,H,W))
        return result_tensor
    else:
        if method == "LINEAR":
            return cv2.resize(img_temp, (H, W),interpolation=cv2.INTER_LINEAR)
        elif method == "NEAREST":
            return cv2.resize(img_temp, (H, W), interpolation=cv2.INTER_NEAREST)

def write_tif(im_data, info, save_path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    else:
        datatype = gdal.GDT_Float32
        
    driver = gdal.GetDriverByName('GTiff')
    if len(im_data.shape) == 2:
        dataset = driver.Create(save_path, im_data.shape[1], im_data.shape[0], 1, datatype)
        dataset.SetGeoTransform(info[0][3])
        dataset.SetProjection(info[0][4])
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        im_bands = int(im_data.shape[0])
        dataset = driver.Create(save_path, im_data[0].shape[1], im_data[0].shape[0], im_bands, datatype)
        dataset.SetGeoTransform(info[0][3])
        dataset.SetProjection(info[0][4])
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])    # i+1是因为GetRasterBand是从1开始
        
    dataset.FlushCache()
    del dataset
    driver = None
def ReverseNormalData(data_temp, max, min):
    reverse = (data_temp + 1) / 2 * (max - min + 10) + min
    #normal_img = 2 * (data_temp - min) / (max - min + 10) - 1
    return reverse

model = mymodel.TfaSR(16, 6).to("cuda:0")
checkpoint = torch.load(r'H:\MZSR-pytorch-master\MZSR-pytorch-master\checkpoint\tfasr_099.pth', map_location="cuda:0")
model.load_state_dict(checkpoint)
info = read_tif(r"H:\meta_train_data\gt\desert\desert1_0_0.tif")
data = Resize(info[0][5], 72, 72, "NEAREST")
data = GetNormalData(data, np.max(data), np.min(data))
data = torch.tensor(data, dtype=torch.float32) 
data = data[None, None, :, :].to("cuda:0")
result = np.squeeze(model(data).detach().cpu().numpy())
revese_data = ReverseNormalData(result, np.max(info[0][5]), np.min(info[0][5]))
label = read_tif(r"H:\meta_train_data\gt\desert\desert1_0_0.tif")[0][5]
print(np.mean((revese_data-label)**2))
write_tif(revese_data, info, r"H:\test_img\tfasr.tif")
