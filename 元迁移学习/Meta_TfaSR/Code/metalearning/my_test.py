import numpy as np
from osgeo import gdal
import os
import torch
import cv2

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
    geo_transform = dataset.GetGeoTransform()  # 返回一个包含 6 个元素的元组
    projection = dataset.GetProjection()  # 返回栅格的投影信息
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 获取栅格的波段数量
    bands = dataset.RasterCount

    # 获取数据类型
    data_type = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)

    # 关闭数据集
    dataset = None

    return elevation_data,geo_transform, projection, width, height, bands, data_type

def save_raster_to_tif(data, output_path, geo_transform=None, projection="EPSG:4326"):
    """
    使用 GDAL 将 NumPy 数组保存为 .tif 格式的栅格文件。

    :param data: 要保存的 NumPy 数组（二维或三维）。
    :param output_path: 输出文件的路径（.tif 文件）。
    :param geo_transform: 地理变换（6 个元素的元组）。
                           如果为 None，则默认使用左上角坐标 (0, rows)，像素大小为 1。
    :param projection: 坐标参考系统，默认为 "EPSG:4326"（WGS84坐标系）。
    """
    # 获取数据的维度
    rows, cols = data.shape
    if len(data.shape) == 3:  # 多波段数据
        bands = data.shape[2]
    else:
        bands = 1

    # 如果没有提供 geo_transform，则使用默认值（左上角在原点，单位为像素）
    if geo_transform is None:
        geo_transform = (0, 1, 0, rows, 0, -1)  # 左上角坐标(0, rows)，像素大小为1x1，Y轴方向向下

    # 创建一个新的 .tif 文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, bands, gdal.GDT_Float32)

    # 设置投影和地理变换信息
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)

    # 将数据写入文件
    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(data)  # 写入单波段数据
    else:
        for band in range(bands):
            dataset.GetRasterBand(band + 1).WriteArray(data[:, :, band])  # 写入多波段数据

    dataset.FlushCache()  # 刷新缓存，确保数据写入磁盘

    print(f"栅格文件已保存到: {output_path}")

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


for file in os.listdir(r'D:\Edge_download\MZSR-pytorch-master\MZSR-pytorch-master\gt_path'):
    elevation_data,geo_transform, projection, width, height, bands, data_type = read_dem(r'D:\Edge_download\MZSR-pytorch-master\MZSR-pytorch-master\gt_path\\'+file)
    resized = Resize(elevation_data, width // 2, height // 2, "LINEAR")
    save_raster_to_tif(resized, r'D:\Edge_download\MZSR-pytorch-master\MZSR-pytorch-master\img_path\\'+file, geo_transform=geo_transform, projection="EPSG:4326")
