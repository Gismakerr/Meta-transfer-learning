import imageio
import os
import numpy as np
import re
import math
from imresize import imresize
from osgeo import gdal

from time import strftime, localtime

def imread(dem_path):
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
    ele_max, ele_min = np.max(elevation_data), np.min(elevation_data)

    # 关闭数据集
    dataset = None

    return elevation_data, ele_max, ele_min

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
"""
def imread(path):
    img=imageio.imread(path).astype(np.float32)
    img=img/255.
    return img
"""

def save(saver, sess, checkpoint_dir, trial, step):
    model_name='model'
    checkpoint = os.path.join(checkpoint_dir, 'Model%d'% trial)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    saver.save(sess, os.path.join(checkpoint, model_name), global_step=step)

def count_param(scope=None):
    N=np.sum([np.prod(v.get_shape().as_list()) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)])
    print('Model Params: %d K' % (N/1000))

def psnr(img1, img2):
    img1=np.float32(img1)
    img2=np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    """
    if np.max(img1) <= 1.0:
        PIXEL_MAX= 1.0
    else:
        PIXEL_MAX = 255.0
    """
    PIXEL_MAX = max(np.max(img1), np.max(img2)) 
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def print_time():
    print('Time: ', strftime('%b-%d %H:%M:%S', localtime()))

''' color conversion '''
def rgb2y(x):
    if x.dtype==np.uint8:
        x=np.float64(x)
        y=65.481/255.*x[:,:,0]+128.553/255.*x[:,:,1]+24.966/255.*x[:,:,2]+16
        y=np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 /255

    return y


def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)

    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:int(sz[0]), 0:int(sz[1])]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:int(szt[0]), 0:int(szt[1]),:]

    return out

def back_projection(y_sr, y_lr, down_kernel, up_kernel, sf=None, ds_method='direct'):
    y_sr += imresize(y_lr - imresize(y_sr, scale=1.0/sf, output_shape=y_lr.shape, kernel=down_kernel, ds_method=ds_method),
                     scale=sf,
                     output_shape=y_sr.shape,
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)
