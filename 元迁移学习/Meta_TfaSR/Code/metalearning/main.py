import dataset
import train
from config import *
from myutils import imread
import glob
import scipy.io
import mytest1 as test
import os
import numpy as np
from dataset import resize
import random

random.seed(1)

# 在这运行
def main():
    if args.is_train==True:

        Trainer = train.Train(trial=args.trial, step=args.step, size=[HEIGHT,WIDTH,CHANNEL],
                              scale_list=SCALE_LIST, meta_batch_size=META_BATCH_SIZE, meta_lr=META_LR, meta_iter=META_ITER, task_batch_size=TASK_BATCH_SIZE,
                              task_lr=TASK_LR, task_iter=TASK_ITER,  checkpoint_dir=CHECKPOINT_DIR)

        Trainer()
    else:
        # 训练好的模型
        model_path = r'D:\元迁移学习\To_黄涛\三种模型预训练测试\TfaSR预训练\Code\TfaSR_final_099.pth'
        train_epoch = 100

        img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.tif')))
        gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.tif')))
        
        img_path = list(filter(file_filter, img_path))
        
        train_path = random.sample(img_path, 2)
        for i in train_path:
            img_path.remove(i)
        #test_path = random.sample(img_path, 20)
        test_path = img_path

        # 倍数
        scale=4.0

        try:
            kernel=scipy.io.loadmat(args.kernelpath)['kernel']
        except:
            kernel='cubic'

        Tester=test.Test(model_path, args.savepath, kernel, scale, args.model, args.num_of_adaptation)
        P=[]
        for i in range(train_epoch):
            #img, img_max, img_min=imread(img_path[i])
            #gt, gt_max, gt_min=imread(gt_path[i])

            _, pp =Tester(test_path, train_path)

            P.append(pp)

        avg_PSNR=np.mean(P, 0)

        print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))
        
def file_filter(f):
    if "icesat" in f:
        return False
    else:
        return True

if __name__ == '__main__':
    main()