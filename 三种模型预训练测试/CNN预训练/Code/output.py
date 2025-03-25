import os
import math
import cv2
import sys
import torch
import random
import argparse
from osgeo import gdal
import numpy as np
import torch.nn as nn
from skimage import io
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import torchvision.datasets as datasets
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
from SRCNN_model import SRCNN as srnet
from DeviceSetting import device
from DEM_features import Slope_net


model = srnet().to(device)
seed = 10
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str,
                    default=r'srcnn',
                    help='path to dataset')

parser.add_argument('--testdataroot', type=str,
                    default=r'D:\元迁移学习\To_黄涛\Data\裁剪DEM500\Test\Desert',
                    help='path to dataset')
parser.add_argument('--output',type=bool,
                    default = False,
                    help='path to outputh')
parser.add_argument('--outputpath',type=str,
                    default=r'E:\毕业论文\图\Resnet\Mountain',
                    help='path to output')
parser.add_argument('--outtext',type=str,
                    default=r'D:\元迁移学习\To_黄涛\三种模型预训练测试\Result.txt',
                    help='path to output')
parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
opt = parser.parse_args()
print(opt)

testdataset = datasets.ImageFolder(root = opt.testdataroot)
assert testdataset
testlen = len(testdataset.imgs)
content_criterion = nn.MSELoss().to(device)
print('output')
with torch.no_grad():
    mean_dem_loss1 = 0.0
    cubic_dem_loss1 = 0.0
    cubic_dem_slope_loss1 = 0.0
    mean_slope_loss1 = 0.0
    #weight = '%s/generator_final_%03d.pth' % (opt.out, epoch)
    weight = r"D:\元迁移学习\To_黄涛\三种模型预训练测试\CNN预训练\Code\CNN_099.pth"
    model.load_state_dict(torch.load(weight,map_location='cuda:0'))
    model.eval()
    for i in range(testlen):
        outpath = opt.outputpath
        try:
            os.makedirs(outpath)
        except OSError:
            pass
        img_temp1, _ = testdataset.imgs[i]
        datasets = gdal.Open(img_temp1)
        band = datasets.GetRasterBand(1)
        Datatype = band.DataType
        DProjection = datasets.GetProjection()
        DGeoTransform = datasets.GetGeoTransform()
        Name = img_temp1.split("\\")[-1]
        outname = outpath+"\\"+img_temp1.split("\\")[-1]
        
        img_temp1 = io.imread(img_temp1)
        H, W = img_temp1.shape
        # low-resolution image
        low_img_temp1 = cv2.resize(img_temp1, (H // opt.upSampling, W // opt.upSampling),
                                    interpolation=cv2.INTER_LINEAR)
        base_min1 = np.min(low_img_temp1)
        base_max1 = np.max(low_img_temp1)
        bicubic_high_img_temp1 = cv2.resize(low_img_temp1, (H, W), interpolation=cv2.INTER_NEAREST)
        B_temp1 = cv2.resize(low_img_temp1, (H, W), interpolation=cv2.INTER_CUBIC)
        img_temp1 = torch.tensor(img_temp1)  # 1*imagesize*imagesize                   #highdem
        OOO_dem1 = img_temp1
        OOO_dem1 = torch.tensor(OOO_dem1)
        OOO_dem1 = OOO_dem1.reshape(1,1, H, W)
        B_temp1 = 2 * (B_temp1 - base_min1) / (base_max1 - base_min1 + 10) - 1
        B_temp1 = torch.tensor(B_temp1) 
        B_temp1 = B_temp1.reshape(1,1, H, W)
        #bicubic highdem
        # 10 is a default value to keep safe
        img_temp1 = 2 * (img_temp1 - base_min1) / (base_max1 - base_min1 + 10) - 1
        bicubic_high_img_temp1 = 2 * (bicubic_high_img_temp1 - base_min1) / (base_max1 - base_min1 + 10) - 1    
        bicubic_high_img_temp1 = bicubic_high_img_temp1.reshape(1,1,H, W)
        bicubic_high_img_temp1 = torch.tensor(bicubic_high_img_temp1) #lowdem
       
       
        high_res_fake1 = model(bicubic_high_img_temp1.to(device))
        
        fake_dem1 = (0.5 * (high_res_fake1 + 1) * (base_max1 - base_min1 + 10) + base_min1).to(device)
        dem_loss_temp1 = math.sqrt(content_criterion(OOO_dem1.to(device), fake_dem1))
        mean_dem_loss1 += dem_loss_temp1
        
        fake_dem_chazhi1 = (0.5 * (B_temp1 + 1) * (base_max1 - base_min1 + 10) +base_min1).to(device)
        cublic_dem_loss_temp1 = math.sqrt(content_criterion(OOO_dem1.to(device), fake_dem_chazhi1))
        cubic_dem_loss1 += cublic_dem_loss_temp1

        slope_loss_temp1 = math.sqrt(content_criterion(Slope_net(OOO_dem1.to(device)), Slope_net(fake_dem1.reshape(1,1,H,W))))
        mean_slope_loss1 += slope_loss_temp1
        
        fake_dem_chazhi1_slope = (0.5 * (B_temp1 + 1) * (base_max1 - base_min1 + 10) +base_min1).to(device)
        cublic_dem_slope_loss_temp1 = math.sqrt(content_criterion(Slope_net(OOO_dem1.to(device)), Slope_net(fake_dem_chazhi1)))
        cubic_dem_slope_loss1 += cublic_dem_slope_loss_temp1
        
        if (opt.output == True):
            driver = gdal.GetDriverByName("GTiff")
            newdata = driver.Create(outname,H,W,1,Datatype)
            newdata.SetProjection(DProjection)
            newdata.SetGeoTransform(DGeoTransform)
            newband = newdata.GetRasterBand(1)
            newband.WriteArray(fake_dem1[0][0].cpu().numpy())
            del newdata
        
    cubic_dem_loss1 = cubic_dem_loss1  / testlen
    cubic_dem_slope_loss1 = cubic_dem_slope_loss1 / testlen
    mean_dem_loss1 = mean_dem_loss1 / testlen
    mean_slope_loss1 = mean_slope_loss1 / testlen
    
    improve_dem = (cubic_dem_loss1 - mean_dem_loss1) /  cubic_dem_loss1
    improve_slope = (cubic_dem_slope_loss1 - mean_slope_loss1  ) / cubic_dem_slope_loss1
    type = outpath.split("\\")[-1]
    with open(opt.outtext, 'a') as file:
    # 将信息格式化并写入文件，包含每一项的名称
        file.write(f"名称: {opt.modelname}, 地貌类型：{type},  插值DEM损失: {cubic_dem_loss1}，插值坡度损失: {cubic_dem_slope_loss1}, DEM损失: {mean_dem_loss1}, DEM坡度损失: {mean_slope_loss1}, DEM提升: {improve_dem}, 坡度提升: {improve_slope }\n")
