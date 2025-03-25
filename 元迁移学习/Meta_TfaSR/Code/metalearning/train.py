import os
import time
from time import localtime, strftime
import collections 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset

import model
import mymodel
from config import *
from myutils1 import *
import Models

torch.autograd.set_detect_anomaly(True)

class Train(object):
    def __init__(self, trial, step, size, scale_list, meta_batch_size, meta_lr, meta_iter, task_batch_size, task_lr, task_iter, checkpoint_dir):
    #def __init__(self, trial, step, size, scale_list, meta_batch_size, meta_lr, meta_iter, task_batch_size, task_lr, task_iter, data_generator, checkpoint_dir):
        print('[*] Initialize Training')
        self.trial = trial
        self.step=step
        self.HEIGHT=size[0]
        self.WIDTH=size[1]
        self.CHANNEL=size[2]
        self.scale_list=scale_list

        self.META_BATCH_SIZE = meta_batch_size
        self.META_LR = meta_lr
        self.META_ITER = meta_iter

        self.TASK_BATCH_SIZE = task_batch_size
        self.TASK_LR = task_lr
        self.TASK_ITER = task_iter

        # self.data_generator=data_generator
        self.checkpoint_dir=checkpoint_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """
        "Models"
        self.model = Models.MSRResNet1(1, 1)
        checkpoint = torch.load(self.checkpoint_dir, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        """
        
        "mymodel"
        self.model = mymodel.TfaSR(16, 6)
        #self.metamodel = mymodel.TfaSR(16, 6).to(self.device)
        checkpoint = torch.load(self.checkpoint_dir, map_location=self.device)

        self.model.load_state_dict(checkpoint)
        #self.metamodel.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        
        self.weights = collections.OrderedDict(self.model.named_parameters())
        

        '''model'''
        #self.model = model.Net()
        #checkpoint = torch.load(self.checkpoint_dir)
        #self.model.load_state_dict(checkpoint)
        #self.model = self.model.to(self.device)
        '''loss'''
        self.loss_fn = nn.L1Loss()

        '''Optimizers'''
        
        self.opt = optim.Adam(self.model.parameters(), lr=self.META_LR)
        
    def functional_forward(self, image, weights=None):
        if weights is not None:
            for name,param in weights.items():
                setattr(self, name, param)
        return self.model.function_forward(image,weights)

    
    def construct_model(self, inf):
        def task_meta_learning(inputa, labela, inputb, labelb):
            self.model.train()
            #self.metamodel.train()
            #meta_weights = self.model.get_weights()
            """
            name_a = name_a[0]
            info = read_tif(r"H:\meta_data\\"+name_a.split('_')[0][:-1]+'\\'+name_a)
            write_tif(labela[0], info, r'H:\test\\'+name_a.split('.')[0]+"_labela.tif")
            write_tif(inputa[0], info, r'H:\test\\'+name_a.split('.')[0]+"_inputa.tif")
            write_tif(ReverseNormalData(inputa[0], np.max(info[0][5]), np.min(info[0][5])), info, r'H:\test\\'+name_a.split('.')[0]+"_inputa1.tif")
            """
            """少了一个波段数"""
            inputa = inputa[:, :, :, None]
            inputb = inputb[:, :, :, None]
            labela = labela[:, :, :, None]
            labelb = labelb[:, :, :, None]


            inputa = torch.as_tensor(inputa).type(torch.FloatTensor).to(self.device).permute(0,3,1,2)
            labela = torch.as_tensor(labela).type(torch.FloatTensor).to(self.device).permute(0,3,1,2)
            inputb = torch.as_tensor(inputb).type(torch.FloatTensor).to(self.device).permute(0,3,1,2)
            labelb = torch.as_tensor(labelb).type(torch.FloatTensor).to(self.device).permute(0,3,1,2)


            task_outputs, task_lossesb = [],[]
            
            #task_outputa = self.model(inputa)
            #self.model.zero_grad()
            #task_lossa.backward(retain_graph=True)   
            #for param in self.weights:
                #param.data.sub_(param.grad.data * self.TASK_LR)
                

            task_outputa = self.functional_forward(inputa, self.weights)
            task_lossa = self.loss_fn(labela, task_outputa)
            grads = torch.autograd.grad(task_lossa, self.weights.values(), create_graph=True)
            self.weights = collections.OrderedDict((name, param - self.TASK_LR * grad) for ((name, param), grad) in zip(self.weights.items(), grads))         
            """
            output = self.model(inputb)
            #output = self.model(inputb, icesatb)
            task_outputs.append(output)
            task_lossesb.append(self.loss_fn(labelb,output))
            """
            

            output = self.functional_forward(inputb, self.weights)
            task_outputs.append(output)
            task_lossesb.append(self.loss_fn(labelb, output))
            for i in range(self.TASK_ITER-1):
                """
                output_s = self.model(inputa)
                #output_s = self.model(inputa, icesata)
                loss = self.loss_fn(labela, output_s)
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                for param in self.weights:
                    param.data.sub_(param.grad.data * self.TASK_LR)
                """
                task_outputa = self.model.function_forward(inputa, self.weights)
                task_lossa = self.loss_fn(labela, task_outputa)
    

                grads = torch.autograd.grad(task_lossa, self.weights.values(), create_graph=True)
              
                self.weights = collections.OrderedDict((name, param - self.TASK_LR * grad) for ((name, param), grad) in zip(self.weights.items(), grads))         
                """
                for name_temp, param_temp in self.weights.items():
                    self.name_temp = name_temp
                    self.param_temp = param_temp
                    break;"""
                """
                output = self.model(inputb)
                task_outputs.append(output)
                task_lossesb.append(self.loss_fn(labelb,output))
                """
                
                output = self.model.function_forward(inputb, self.weights)
                task_outputs.append(output)
                task_lossesb.append(self.loss_fn(labelb, output))
            
            task_output = [task_outputa, task_outputs, task_lossa, task_lossesb]

            return task_output

        self.total_lossa = 0.
        self.total_lossesb = []
        inputa, labela, inputb, labelb = inf
        for i in range(self.META_BATCH_SIZE):
            """
            for name_eternal, param_eternal in self.model.named_parameters():
                self.name_eternal = name_eternal
                self.param_eternal = param_eternal
                break"""
            self.weights = collections.OrderedDict(self.model.named_parameters())
            res = task_meta_learning(inputa[i], labela[i], inputb[i], labelb[i])
            self.total_lossa += res[2]
            self.total_lossesb.append(sum(res[3])/self.TASK_ITER)
        self.total_lossa /= self.META_BATCH_SIZE
        self.weighted_total_lossesb = torch.mean(torch.as_tensor(self.total_lossesb))
        self.weighted_total_lossesb.requires_grad_()

        self.opt.zero_grad()
        self.weighted_total_lossesb.backward()
        self.opt.step()


    def __call__(self):
        PRINT_ITER=1
        #PRINT_ITER=1000
        SAVE_ITET=1

        print('[*] Training Starts')
        step = self.step

        t2 = time.time()
        while True:
            inf = dataset.make_data_tensor(4)
            
            self.construct_model(inf)

            step += 1

            if step % PRINT_ITER == 0:
                t1 = t2
                t2 = time.time()
                print('Iteration:',step, '(Pre, Post) Loss', self.total_lossa, self.weighted_total_lossesb, 'Time: %.2f'%(t2-t1))

            if step % SAVE_ITET == 0:
                print_time()
                save(self.model, r'D:\metalearning\checkpoints', self.trial, step)

            if step == self.META_ITER:
                print('Done Training')
                print_time()
                break

def print_time():
    print('Time: ',strftime('%b-%d %H:%M:%S', localtime()))

def save(model, checkpoint_dir, trial, step):
    model_name = 'model_{}.pth'.format(str(step))
    checkpoint = os.path.join(checkpoint_dir, 'Model_huangtu%d' % trial)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    torch.save(model.state_dict(), os.path.join(checkpoint, model_name))
            
