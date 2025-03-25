import model
import mymodel
import torch
import time
import imageio
import numpy as np
from  myutils import *
import cv2
from osgeo import gdal
import random

def ReNormalData(data_temp,max,min):
    
    return (0.5 * (data_temp + 1) * (max - min + 10) + min)
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

def read_dem(dem_path):
    """
    读取 DEM 文件并返回其高程数据以及投影信息。

    参数:
    dem_path (str): DEM 文件路径

    返回:
    - elevation_data (numpy.ndarray): 高程数据（二维数组）
    - geo_transform (tuple): 地理转换信息（如坐标系统、分辨率等）
    - projection (str): 投影信息
    """
    # 使用 GDAL 打开 DEM 文件
    dataset = gdal.Open(dem_path)

    if dataset is None:
        raise FileNotFoundError(f"文件 {dem_path} 无法打开或不存在")

    # 获取高程数据
    elevation_data = dataset.ReadAsArray()

    # 关闭数据集
    dataset = None

    return elevation_data

def add_noise(array, noise_type="gaussian", noise_level=0.1, device ="cpu"):
    """
    为输入的60×60数组添加噪声。

    参数:
    - array: numpy.ndarray, 输入的60×60数组
    - noise_type: str, 噪声类型，支持 "gaussian"（高斯噪声）或 "uniform"（均匀噪声）
    - noise_level: float, 噪声强度（标准差或幅度）

    返回:
    - numpy.ndarray, 添加噪声后的数组
    """
    
    
    if isinstance(array, torch.Tensor) and array.is_cuda:
        if array.is_cuda:
            temp_array = array.detach().cpu().numpy()  # 将 tensor 转为 numpy 数组
        else:
            temp_array = array.numpy()
    elif isinstance(array, np.ndarray):
        temp_array = array 

    if noise_type == "gaussian":
        noise = np.random.normal(loc=0, scale=noise_level, size=array.shape)
    elif noise_type == "uniform":
        noise = np.random.uniform(low=-noise_level, high=noise_level, size=array.shape)
    else:
        raise ValueError("不支持的噪声类型。请使用 'gaussian' 或 'uniform'")
    
    # 添加噪声并确保输出类型一致
    noisy_array = temp_array + noise
    
    if isinstance(array, np.ndarray):
        noisy_array = noisy_array.astype(array.dtype)

    # 如果输入是 tensor 类型，且最初是在 GPU 上，返回时转换回 GPU
    if isinstance(array, torch.Tensor):
        noisy_array = torch.tensor(noisy_array, dtype=array.dtype, device=device)
        #noisy_array = torch.from_numpy(noisy_array).clone().detach().to(device)

    return noisy_array


def GetMaxMin(img):
    # 如果 img 是张量，先转换为 numpy 数组
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()  # 将张量转为 numpy 数组，并确保是在 CPU 上

    # 如果 img 是四维数组（batch_size, channels, height, width），我们提取第一个通道
    if len(img.shape) == 4:
        img = img[0][0]  # 提取第一个 batch 和第一个 channel

    base_min = np.min(img)  # 计算最小值
    base_max = np.max(img)  # 计算最大值
    return base_max, base_min
    
def get_index(array, device):
    # 使用布尔索引修改非零值
    if isinstance(array, np.ndarray):
        result = np.where(array != 0, 1, 0)
    elif isinstance(array, torch.Tensor):
        result = torch.where(array != 0, torch.tensor(1, dtype=array.dtype, device=array.device), array)
    return result
def GetNormalData(data_temp,max,min):
    normal_img = 2 * (data_temp - min) / (max - min + 10) - 1
    return normal_img

def random_crop(hr,size):
    h, w = hr.shape[:-1]
    x = random.randint(0, w-size)
    y = random.randint(0, h-size)

    crop_hr = hr[y:y+size, x:x+size].copy()

    return crop_hr


def random_flip_and_rotate(im1):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        
    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
  

    # have to copy before be called by transform function
    return im1.copy()

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
        
def ReverseNormalData(data_temp, max, min):
    reverse = (data_temp + 1) / 2 * (max - min + 10) + min
    #normal_img = 2 * (data_temp - min) / (max - min + 10) - 1
    return reverse