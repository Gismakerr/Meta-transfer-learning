import model
import mymodel
import torch
import time
import imageio
import numpy as np
from  myutils import *
from  myutils1 import *
import cv2
from osgeo import gdal
import random
import Models

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

class Test(object):
    def __init__(self, model_path, save_path,kernel, scale, method_num, num_of_adaptation):
        methods=['direct', 'direct', 'bicubic', 'direct']
        self.save_results=True
        self.max_iters=num_of_adaptation
        self.display_iter = 1

        self.upscale_method= 'cubic'
        self.noise_level = 0.0

        self.back_projection=False
        self.back_projection_iters=4

        self.model_path=model_path
        self.save_path=save_path
        self.method_num=method_num

        self.ds_method=methods[self.method_num]

        self.kernel = kernel
        self.scale=scale
        self.scale_factors = [self.scale, self.scale]

        self.mse, self.mse_tfasr, self.mse_final, self.mse_tfasr_final, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], [], [], [], []
        #self.psnr=[]
        self.iter = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mymodel.TfaSR(16,6)
        self.model_tfasr = mymodel.TfaSR(16,6)

        self.learning_rate = 0.00001
        self.loss_fn = torch.nn.L1Loss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)
        
        """
    def __call__(self, img, gt, img_name):
        
        self.img_max = np.max(img)
        self.img_min = np.min(img)
        self.gt_max = np.max(gt)
        self.gt_min = np.min(gt)
        self.img = img
        self.psnr=[]
        self.gt=gt
        #self.gt = modcrop(gt, self.scale)

        self.img_name = img_name
        print('** Start Adaptation for X', self.scale, os.path.basename(self.img_name), ' **')
        """
    def __call__(self, test_path, train_path):
        self.test_path = test_path
        self.train_path = train_path
        self.gt_max = None
        self.gt_min = None
        self.img_max = None
        self.img_min = None
        self.psnr = []
        
        self.model.load_state_dict(torch.load(r"D:\metalearning\checkpoints\Model_huangtu0\model_100.pth", map_location=torch.device("cuda:0")))
        self.model = self.model.to(self.device)
        
        self.model_tfasr.load_state_dict(torch.load(r"D:\元迁移学习\实验\不同模型进行元迁移学习\TfaSR预训练\Code\generator_final_099.pth", map_location=torch.device("cuda:0")))
        self.model_tfasr = self.model_tfasr.to(self.device)
        
        self.sf = np.array(self.scale_factors)
        #self.output_shape = np.uint(np.ceil(np.array([18, 18]) * self.scale))

        self.quick_test()

        print('[*] Baseline ')
        self.train()

        post_processed_output = self.final_test()

        return post_processed_output, self.psnr


    def train(self):
        #self.hr_father,_,_ = imread(r"H:\meta_train_data\gt\huangtu1_0_1.tif")
        #self.lr_son,_,_ = imread(r"H:\meta_train_data\img\huangtu1_0_1.tif")
        #self.hr_father = self.img
        #self.lr_son = self.Resize(self.img, int(self.img.shape[0] // self.scale), int(self.img.shape[1] // self.scale), "LINEAR")
        #self.add_noise(self.lr_son, "gaussian", 3, "cpu")
        #self.lr_son = imresize(self.img, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method)
        #self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.)
        t1=time.time()
        
        for i in range(self.iter):
            train_list = random.sample(self.train_path, 2)
            
            if self.iter==0:
                self.learning_rate = 0
                #self.learning_rate=2e-2
            elif self.iter < 4:
                self.learning_rate=1e-4
            else:
                self.learning_rate=5e-3
            
            for hr_path in (train_list):
                hr_father = read_tif(hr_path)[0][5]
                #icesat = r"H:\\meta_train_data\\gt\\huangtu\\"+hr_path.split("\\")[-1].split("_")[0]+"_icesat_"+hr_path.split("\\")[-1].split("_")[1]+"_"+hr_path.split("\\")[-1].split("_")[2]
                #icesat = read_tif(icesat)[0][5]
                lr_son = Resize(hr_father, int(self.gt.shape[0] / self.scale), int(self.gt.shape[1] / self.scale), "LINEAR")
                
                max, min = GetMaxMin(lr_son)
                lr_son  = Resize(lr_son, int(self.gt.shape[0]), int(self.gt.shape[1]), "NEAREST")
                lr_son = GetNormalData(lr_son,max,min)
                self.train_output = self.forward_backward_pass((lr_son), hr_father,max, min)
            # Display information
            if self.iter % self.display_iter == 0:
                    #print('Scale: ', self.scale, ', iteration: ', (self.iter+1), ', loss: ', self.loss[self.iter])
                    print('Scale: ', self.scale, ', iteration: ', (self.iter+1), ', loss: ', self.loss)

        t2 = time.time()
        print('%.2f seconds' % (t2 - t1))
    
    def quick_test(self):
        for i,gt_path in enumerate(self.test_path):
            print(i)
            gt_info = read_tif(gt_path)
            #self.data_list[index].split("_")[0]+"_icesat_"+self.data_list[index].split("_")[1]+"_"+self.data_list[index].split("_")[2]
            #self.icesat = read_tif(r"H:\\meta_train_data\\gt\\huangtu\\"+gt_path.split("\\")[-1].split("_")[0]+"_icesat_"+gt_path.split("\\")[-1].split("_")[1]+"_"+gt_path.split("\\")[-1].split("_")[2])[0][5]
            
            self.gt = gt_info[0][5]
            self.img = Resize(self.gt, int(self.gt.shape[0] / self.scale), int(self.gt.shape[1] / self.scale), "LINEAR")
            #write_tif(Resize(self.img, 60, 60, "NEAREST"), gt_info, r"H:\meta_train_data\img\\"+gt_path.split('\\')[-1].split(".")[0]+"_tfasr_reshape.tif")
            #self.gt_max, self.gt_min = GetMaxMin(self.gt)
            self.img_max, self.img_min = GetMaxMin(self.img)
            self.img = Resize(self.img, int(self.gt.shape[0]), int(self.gt.shape[1]), "NEAREST")
            self.img = GetNormalData(self.img,self.img_max,self.img_min)
            # 1. True MSE
            self.sr = self.ReverseNormalData(self.forward_pass((self.img), self.gt.shape), self.img_max, self.img_min)#.transpose(1,2,0)
            self.mse = self.mse + [math.sqrt(np.mean((self.gt - self.sr)**2))]
            #write_tif(self.sr, gt_info, r"H:\meta_train_data\img\\"+gt_path.split('\\')[-1])
            
            self.sr1 = self.ReverseNormalData(self.forward_pass_tfasr((self.img), self.gt.shape), self.img_max, self.img_min)#.transpose(1,2,0)
            self.mse_tfasr = self.mse_tfasr + [math.sqrt(np.mean((self.gt - self.sr1)**2))]
            #write_tif(self.sr1, gt_info, r"H:\meta_train_data\img\\"+gt_path.split('\\')[-1].split(".")[0]+"_tfasr.tif")
        """
        '''Shave'''
        scale = int(self.scale)
        PSNR = psnr(self.gt[scale:-scale, scale:-scale], self.sr[scale:-scale, scale:-scale])
        #PSNR = psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
                  #rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale])
        self.psnr.append(PSNR)
        """
        """
        # 2. Reconstruction MSE
        self.reconstruct_output = self.ReverseNormalData(self.forward_pass(self.hr2lr(self.img), self.img.shape), np.max(self.hr2lr(self.img)), np.min(self.hr2lr(self.img))) # .transpose(1,2,0)
        self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))
        #processed_output=np.round(np.clip(self.sr*255, 0., 255.)).astype(np.uint8)
        processed_output = self.sr
        print(np.mean((self.gt - self.sr)**2), np.mean((self.img - self.reconstruct_output)**2))
        #print('iteration: ', self.iter, 'recon mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)
        
        return processed_output"""
    
    def forward_pass_tfasr(self, input, output_shape=None):
        #ILR = torch.tensor(imresize(input, self.scale, output_shape, self.upscale_method),dtype=torch.float32).permute(2, 0, 1).to("cuda:0")
        ILR = torch.tensor(input, dtype=torch.float32)   #.permute(2, 0, 1).to("cuda:0")
        ILR = ILR.reshape((output_shape[0], output_shape[1])) #.permute(2, 0, 1).to("cuda:0")
        #icesat_index = get_index(input[1], "")
        #self.img_max = np.max(ILR.cpu().numpy())
        #self.img_min = np.min(ILR.cpu().numpy())
        #icesat = GetNormalData(input[1], np.max(ILR.cpu().numpy()), np.min(ILR.cpu().numpy())) * icesat_index
        ILR = ILR[None, None, :, :].to("cuda:0")
        #icesat = torch.tensor(icesat[None, None, :, :], dtype=torch.float32).to("cuda:0")
        #ILR = self.GetNormalData(ILR)
        #self.model.to("cpu")
        #self.model_tfasr.to(self.device)
        self.model_tfasr.eval()
        output = self.model_tfasr(ILR)
        output = output.detach().cpu().numpy()######
        return np.squeeze(output)
        #return np.clip(np.squeeze(output), 0., 1.)

    def forward_pass(self, input, output_shape=None):
        #ILR = torch.tensor(imresize(input, self.scale, output_shape, self.upscale_method),dtype=torch.float32).permute(2, 0, 1).to("cuda:0")
        ILR = torch.tensor(input, dtype=torch.float32)   #.permute(2, 0, 1).to("cuda:0")
        ILR = ILR.reshape((output_shape[0], output_shape[1]))
        #icesat_index = get_index(input[1], "")
        # self.img_max = np.max(ILR.cpu().numpy())
        # self.img_min = np.min(ILR.cpu().numpy())
        #icesat = GetNormalData(input[1], np.max(ILR.cpu().numpy()), np.min(ILR.cpu().numpy())) * icesat_index
        ILR = ILR[None, None, :, :].to("cuda:0")
        #icesat = torch.tensor(icesat[None, None, :, :], dtype=torch.float32).to("cuda:0")
        #ILR = self.GetNormalData(ILR)
        #self.model_tfasr.to("cpu")
        #self.model.to(self.device)
        self.model.eval()
        output = self.model(ILR)
        output = output.detach().cpu().numpy()######
        return np.squeeze(output)
        #return np.clip(np.squeeze(output), 0., 1.)
    
    def GetNormalData(self, data_temp,max, min):
        #max, min = torch.max(data_temp), torch.min(data_temp)
        normal_img = 2 * (data_temp - min) / (max - min + 10) - 1
        return normal_img
    
    def ReverseNormalData(self, data_temp, max, min):
        reverse = 0.5 * (data_temp + 1)  * (max - min + 10) + min
        #normal_img = 2 * (data_temp - min) / (max - min + 10) - 1
        return reverse

    def Resize(self, img_temp, H, W, method):
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
    
    def add_noise(self, array, noise_type="gaussian", noise_level=0.1, device ="cpu"):
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

    def forward_backward_pass(self, input, hr_father,max, min):
        #ILR = torch.tensor(self.Resize(input[0], hr_father.shape[0], hr_father.shape[1], "NEAREST"), dtype=torch.float32).to("cuda:0")   #.permute(2, 0, 1).to("cuda:0")
        ILR = torch.tensor(input, dtype=torch.float32).to("cuda:0")   #.permute(2, 0, 1).to("cuda:0")
        ILR = ILR.reshape((hr_father.shape[0], hr_father.shape[1]))
        #ILR = torch.tensor(imresize(input, self.scale, hr_father.shape, self.upscale_method),dtype=torch.float32).permute(2, 0, 1).to("cuda:0")
        #icesat_index = get_index(input[1], "")
        
        #icesat = GetNormalData(input[1], np.max(ILR.cpu().numpy()), np.min(ILR.cpu().numpy())) * icesat_index
        #icesat = torch.tensor(icesat[None, None, :, :], dtype=torch.float32).to("cuda:0")
        ILR = ILR[None, None, :, :]
        HR = torch.tensor(hr_father[None, None, :, :],dtype=torch.float32).to("cuda:0")
        
        #train_output = self.model(ILR_norm, icesat)
        train_output = self.model(ILR)
        self.loss = self.loss_fn(train_output, self.GetNormalData(HR,max, min))
        
        self.opt.zero_grad()
        for param in self.opt.param_groups:
            print(param["lr"])
        self.loss.backward()
        self.opt.step()
        train_output = train_output.detach().cpu().numpy()
        return np.squeeze(train_output)
        #return np.clip(np.squeeze(train_output), 0., 1.)
    
    def hr2lr(self, hr):
        lr = self.Resize(hr, int(hr.shape[0] // self.scale), int(hr.shape[1] // self.scale), method="LINEAR")
        lr = self.add_noise(lr, "gaussian", 3, "cpu")
        return lr
    """
    def hr2lr(self, hr):
        lr = imresize(hr, 1.0 / self.scale, kernel=self.kernel, ds_method=self.ds_method)
        return np.clip(lr + np.random.randn(*lr.shape) * self.noise_level, 0., 1.)
    """
    def final_test(self):
        for gt_path in self.test_path:
            gt_info = read_tif(gt_path)
            #self.icesat = read_tif(r"H:\\meta_train_data\\gt\\huangtu\\"+gt_path.split("\\")[-1].split("_")[0]+"_icesat_"+gt_path.split("\\")[-1].split("_")[1]+"_"+gt_path.split("\\")[-1].split("_")[2])[0][5]
            self.gt = gt_info[0][5]
            self.img = Resize(self.gt, int(self.gt.shape[0] / self.scale), int(self.gt.shape[1] / self.scale), "LINEAR")
            #self.gt_max, self.gt_min = GetMaxMin(self.gt)
            self.img_max, self.img_min = GetMaxMin(self.img)
            # 1. True MSE
            #self.sr = self.ReverseNormalData(self.forward_pass((self.img, self.icesat), self.gt.shape), self.img_max, self.img_min)#.transpose(1,2,0)
            self.mse_final = self.mse_final + [np.mean((self.gt - self.sr)**2)]
            
            #self.sr1 = self.ReverseNormalData(self.forward_pass_tfasr((self.img, self.icesat), self.gt.shape), self.img_max, self.img_min)#.transpose(1,2,0)
            self.mse_tfasr_final = self.mse_tfasr_final + [np.mean((self.gt - self.sr1)**2)]
    
        """
        output = self.forward_pass(self.img, self.gt.shape)
        if self.back_projection == True:
            for bp_iter in range(self.back_projection_iters):
                output = back_projection(output, self.img, down_kernel=self.kernel,
                                                  up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)
        processed_output = self.ReverseNormalData(output, self.img_max, self.img_min)
        #processed_output=np.round(np.clip(output*255, 0., 255.)).astype(np.uint8)
        
        '''Shave'''
        scale=int(self.scale)
        PSNR=psnr(self.gt[scale:-scale, scale:-scale],
                  processed_output[scale:-scale, scale:-scale])

        print(np.mean((processed_output - self.gt)**2))

        # PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)),
        #           rgb2y(processed_output))

        self.psnr.append(PSNR)
        """
        return processed_output