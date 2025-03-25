from osgeo import gdal
import numpy as np
import sys, os

def read_tif(filepath):
    dataset = gdal.Open(filepath)
    rows = dataset.RasterXSize
    cols = dataset.RasterYSize
    bands = dataset.RasterCount
    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    print(proj)
    data = dataset.ReadAsArray(6, 6, rows-6, cols-6)
    info = []
    info.append((rows, cols, bands, geotrans, proj, data))
    return info

def write_tif(save_path, im_data, transform, proj):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    else:
        datatype = gdal.GDT_Float32
        
    im_bands = 1
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(save_path, im_data.shape[1], im_data.shape[0], im_bands, datatype)
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(proj)
    
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data)    # i+1是因为GetRasterBand是从1开始
        
    dataset.FlushCache()
    del dataset
    driver = None

def split(target_path, origin_info, output_size, start_i, end_i, start_j, end_j, name=''):

    origin_data = origin_info[0][5]
    origin_size = origin_info[0][5].shape
    origin_transform = origin_info[0][3]
    x = origin_transform[0]
    y = origin_transform[3]
    x_step = origin_transform[1]
    y_step = origin_transform[5]
    output_x_step = x_step
    output_y_step = y_step

    step_length = 0 #滑动步长(重叠部分)1072

    if end_i == '':
        end_i = origin_size[0] // (output_size[0]-step_length) - 1
    if end_j == '':
        end_j = origin_size[1] // (output_size[1]-step_length) - 1

    
    #j = (origin_size[0]-(output_size[0]-step_length)) // step_length
    #for i in range(start_i, origin_size[0] // (output_size[0]-step_length)):
        #for j in range(start_j, origin_size[1] // (output_size[1]-step_length)):
    for i in range(start_i, end_i+1):
        for j in range(start_j, end_j+1):
            output_data = origin_data[i*(output_size[0]-step_length): i*(output_size[0]-step_length)+output_size[0], j*(output_size[1]-step_length): j*(output_size[1]-step_length)+output_size[1]]
            output_transform = (x+j*output_x_step*(output_size[0]-step_length),
                                output_x_step,
                                origin_transform[2], 
                                y+i*output_y_step*(output_size[1]-step_length), 
                                origin_transform[4], 
                                output_y_step)
            write_tif(target_path+'\\'+name+str(i)+'_'+str(j)+'.tif', output_data, output_transform, origin_info[0][4])

"""
info1 = read_tif(r'F:\DL_Dataset\test1\test1.tif')
print(info1)
split(r'F:\DL_Dataset\1\2', info1, [256, 256])
info2 = read_tif(r'F:\DL_Dataset\test1\label.tif')
print(info2) 
split(r'F:\DL_Dataset\1\label', info2, [256, 256])
"""

for file in os.listdir(r'G:\pretrain_data\fab'):
    info1 = read_tif(r'G:\pretrain_data\fab\\'+file)
    work_dir_aster = r'G:\pretrain_data\meta_data\\'+file[:-5]
    if not os.path.exists(work_dir_aster):
        os.makedirs(work_dir_aster)
    split(work_dir_aster, info1, [72, 72], start_i=0, end_i='',  start_j=0, end_j='', name=file[:-4]+'_')

info1 = read_tif(r"E:\论文\arcgis_porj\4.预测结果图\MyProject\area3\area3_composite.tif")
split(r"F:\mask_test\test1024", info1, [72, 72], start_i=0, end_i='',  start_j=0, end_j='', name='area3_')

#info1 = read_tif(r"F:\mask_test\dv_test\image\composite.tif")
#split(r"F:\mask_test\test1024", info1, [1024, 1024], start_i=42, end_i=45,  start_j=46, end_j=50, name='predict_')

#split(r"F:\mask_test\test2", info1, [1024, 1024], start_i=15, end_i=32,  start_j=28, end_j=35, name='predict_')
#info2 = read_tif(r'E:\ArcGIS_Pro_file\alluvial_fan\MyProject\label_reclass\230_re.tif')
#split(r'F:\DL_Dataset\1\label', info2, [256, 256], '10_')
