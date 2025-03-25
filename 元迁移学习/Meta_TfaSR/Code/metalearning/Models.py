import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from torch.nn import functional as F
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN

#correction

class Correction(nn.Module):
    def __init__(self):
        super(Correction, self).__init__()
        
        self.conv = nn.Conv2d(3, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, Slope, Distance, Level, Index):
        #########################original version########################
        if not (Slope.shape == Distance.shape == Level.shape):
            raise ValueError("输入的张量形状必须一致！")
        # 沿着 dimension（第 1 个维度）拼接
        data = torch.cat((Slope, Distance, Level), dim=1)
        x = self.conv(data)
        return x * Index

#TfaSR
def swish(x):
    return F.relu(x)
class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x
    
    

class TfaSR(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(TfaSR, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x):
        #########################original version########################
        x = self.conv1(x)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        x = swish(self.dconv2_2(x))
        x = swish(self.dconv2_3(x))
        return self.conv4(self.dconv2_4(x))  
    
    
class TfaSR_Point(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(TfaSR_Point, self).__init__()
        self.conv = nn.Conv2d(3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, LowDEM, Point_Ele, Slope, Distance, Level):
        #########################original version########################
        if not (Slope.shape == Distance.shape == Level.shape):
            raise ValueError("输入的张量形状必须一致！")
        data = torch.cat((Slope, Distance, Level), dim=1)
        error = self.conv(data)
        error = error * (Level != 0).float() * 1
        ReverseIndex = (1 - (Level != 0).float() * 1)
        x = LowDEM * ReverseIndex + Point_Ele + error
        x = self.conv1(x)
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        x = swish(self.dconv2_2(x))
        x = swish(self.dconv2_3(x))
        return self.conv4(self.dconv2_4(x))   
    
    
class TfaSR_Point1(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(TfaSR_Point1, self).__init__()
        self.conv = nn.Conv2d(2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride = 1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, LowDEM, Point_Ele):
        #########################original version########################
        
        data = torch.cat((LowDEM, Point_Ele), dim=1)
        x = self.conv(data)
        
        x = self.conv1(x)
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        x = swish(self.dconv2_2(x))
        x = swish(self.dconv2_3(x))
        return self.conv4(self.dconv2_4(x))    

    
     
class Blocks(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Blocks, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)  





#SRCNN
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


#SRGAN
class SRGAN(nn.Module):
  def __init__(self, in_channels, features):
    super(SRGAN, self).__init__()
    self.first_layer = nn.Sequential(
        nn.Conv2d(in_channels, features, 3, 1, 1),
        nn.PReLU(),
    )
    self.RB1 = CNNBlocks(features)
    self.RB2 = CNNBlocks(features)
    self.RB3 = CNNBlocks(features)
    self.RB4 = CNNBlocks(features)

    self.mid_layer = nn.Sequential(
        nn.Conv2d(features, features*4, 3, 1, 1),
        nn.PReLU(),
    )
    self.PS1 = PixelShuffle(features*4, features*8,2)
    self.PS2 = PixelShuffle(features*2, features*4,2)

    self.final_layer = nn.Sequential(
        nn.Conv2d(features, in_channels, 3, 1, 1),
        nn.Tanh(),
    )


  def forward(self, x):
    x1 = self.first_layer(x)
    x2 = self.RB1(x1)
    x3 = self.RB2(x2)
    x4 = self.RB3(x3)
    x5 = self.RB4(x4)
    x6 = self.mid_layer(x5+x1)
    x7 = self.PS1(x6)
    x8 = self.PS2(x7)
    return self.final_layer(x8)

class CNNBlocks(nn.Module):
  def __init__(self, in_channels ):
    super(CNNBlocks, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        nn.BatchNorm2d(in_channels),
        nn.PReLU(),
        nn.Conv2d(in_channels, in_channels, 3, 1, 1),
        nn.BatchNorm2d(in_channels),
    )

  def forward(self, x):
      return self.conv(x)+x


class PixelShuffle(nn.Module):
  def __init__(self, in_channels, out_channels, upscale_factor):
    super(PixelShuffle, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels,out_channels, 3, 1, 1),
        nn.PixelShuffle(upscale_factor),
        nn.PReLU(),
    )

  def forward(self,x):
    return self.conv(x)

#Discriminator

class Block(nn.Module):
  def __init__(self, in_channels, out_channels, stride):
    super(Block, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

  def forward(self, x):
    return self.conv(x)  

class Discriminator(nn.Module):
  def __init__(self, in_channels, features):
    super(Discriminator, self).__init__()
    self.first_layer= nn.Sequential(
        nn.Conv2d(in_channels, features, 3, 2 ,1),
        nn.LeakyReLU(0.2),
    )
    self.Block1 = Block(features, features*2, stride=2)
    self.Block2 = Block(features*2, features*2, stride=1)
    self.Block3 = Block(features*2, features*4, stride=2)
    self.Block4 = Block(features*4, features*4, stride=1)
    self.Block5 = Block(features*4, features*8, stride=2)
    self.Block6 = Block(features*8, features*8, stride=1)
    self.Block7 = Block(features*8, features*8, stride=2)
    self.Block8 = Block(features*8, features*8, stride=2)
    self.Block9 = nn.Sequential(
        nn.Conv2d(features*8, features*4, 3, 2, 1),
        
        nn.LeakyReLU(0.2),
    )
    self.final_layer = nn.Sequential(
        nn.Linear(features*4, 1),
        nn.Sigmoid(),
    )

  def forward(self, x):
    x =  self.first_layer(x)
    x =  self.Block1(x)
    x =  self.Block2(x)
    x =  self.Block3(x)
    x =  self.Block4(x)
    x =  self.Block5(x)
    x =  self.Block6(x)
    x =  self.Block7(x)
    x =  self.Block8(x)
    x = self.Block9(x)
    x = x.view(x.size(0), -1)
    return self.final_layer(x)

#SRResRet


class ResBlock1(nn.Module):
    """残差模块"""
    def __init__(self,inChannals,outChannals):
        """初始化残差模块"""
        super(ResBlock1,self).__init__()
        self.conv1 = nn.Conv2d(inChannals,outChannals,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(outChannals)
        self.conv2 = nn.Conv2d(outChannals,outChannals,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(outChannals)
        self.conv3 = nn.Conv2d(outChannals,outChannals,kernel_size=1,bias=False)
        self.relu = nn.PReLU()
       
    def forward(self,x):
        """前向传播过程"""
        resudial = x
       
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
       
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
       
        out = self.conv3(x)
       
        out += resudial
        out = self.relu(out)
        return out

class SRResNet(nn.Module):
    """SRResNet模型(4x)"""
   
    def __init__(self):
        """初始化模型配置"""
        super(SRResNet,self).__init__()
       
        #卷积模块1
        self.conv1 = nn.Conv2d(1,64,kernel_size=9,padding=4,padding_mode='reflect',stride=1)
        self.relu = nn.PReLU()
        #残差模块
        self.resBlock = self._makeLayer_(ResBlock1,64,64,16)
        #卷积模块2
        self.conv2 = nn.Conv2d(64,64,kernel_size=1,stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.PReLU()
       
        #子像素卷积
        self.convPos1 = nn.Conv2d(64,256,kernel_size=3,stride=1,padding=2,padding_mode='reflect')
        self.pixelShuffler1 = nn.PixelShuffle(2)
        self.reluPos1 = nn.PReLU()
       
        self.convPos2 = nn.Conv2d(64,256,kernel_size=3,stride=1,padding=1,padding_mode='reflect')
        self.pixelShuffler2 = nn.PixelShuffle(2)
        self.reluPos2 = nn.PReLU()
       
        self.finConv = nn.Conv2d(64,1,kernel_size=9,stride=1)
       
    def _makeLayer_(self,block,inChannals,outChannals,blocks):
        """构建残差层"""
        layers = []
        layers.append(block(inChannals,outChannals))
       
        for i in range(1,blocks):
            layers.append(block(outChannals,outChannals))
       
        return nn.Sequential(*layers)
   
    def forward(self,x):
        """前向传播过程"""
        x = self.conv1(x)
        x = self.relu(x)
        residual = x
       
        out = self.resBlock(x)
       
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.convPos1(out)  
        out = self.pixelShuffler1(out)
        out = self.reluPos1(out)
       
        out = self.convPos2(out)  
        out = self.pixelShuffler2(out)
        out = self.reluPos2(out)
        out = self.finConv(out)
       
        return out

#MSResNet
#用于顺序地将多个神经网络模块（如卷积层、激活层、批量归一化等）连接起来，形成一个新的 nn.Sequential 模型
def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


#外部定义的 conv 函数或类来封装卷积操作
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == "D":
            L.append(DCN(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    return sequential(*L)

#实现了“残差连接”，解决梯度消失和加速训练等问题
class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()

        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr
    

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res
    

    

class MSRResNet0(nn.Module):
    def __init__(self, in_nc, out_nc, nc=64, nb=16, act_mode='R'):
        super(MSRResNet0, self).__init__() #调用了父类（超类）的构造函数
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
      

        m_head = conv(in_nc, nc, mode='D')

        m_body = [ResBlock(nc, nc, mode='D'+act_mode+'D') for _ in range(nb)]
        m_body.append(conv(nc, nc, mode='D'))

        H_conv0 = conv(nc, nc, mode='D'+act_mode)
        H_conv1 = conv(nc, out_nc, bias=False, mode='D')
        m_tail = sequential(H_conv0, H_conv1)

        self.model = sequential(m_head, ShortcutBlock(sequential(*m_body)),m_tail)

    def forward(self, x, y):  
        x += y     
        x = self.model(x)     
        return x    
    
class MSRResNet00(nn.Module):
    def __init__(self, in_nc, out_nc, nc=64, nb=16, act_mode='R'):
        super(MSRResNet00, self).__init__() #调用了父类（超类）的构造函数
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
      

        m_head = conv(in_nc, nc, mode='C')

        m_body = [ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(conv(nc, nc, mode='C'))

        H_conv0 = conv(nc, nc, mode='C'+act_mode)
        H_conv1 = conv(nc, out_nc, bias=False, mode='C')
        m_tail = sequential(H_conv0, H_conv1)

        self.model = sequential(m_head, ShortcutBlock(sequential(*m_body)),m_tail)

    def forward(self, x):  
        x = self.model(x)     
        return x            


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc=1, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 64
        conv2 = conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 128
        conv4 = conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 256
        conv6 = conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv8 = conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x, point):
        assert x.size(0) == point.size(0), "Batch sizes of x and point must be the same"
        assert x.size(2) == point.size(2) and x.size(3) == point.size(3), "Height and width of x and point must be the same"

        ReverseIndex = (1 - (point != 0).float() * 1)
        x = x * ReverseIndex + point
        x = self.features(x)
        #print("Shape after convolution layers:", x.shape)  # 打印卷积层后的形状
        x = x.view(x.size(0), -1)
        #print("Shape after convolution layers:", x.shape)
        x = self.classifier(x)
        return x
  
class MSRResNet1(nn.Module):
    def __init__(self, in_nc, out_nc, nc=64, nb=16, act_mode='R'):
        super(MSRResNet1, self).__init__() #调用了父类（超类）的构造函数
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        
        m_head = conv(in_nc, nc, mode='D')
        m_body = [ResBlock(nc, nc, mode='D'+act_mode+'D') for _ in range(nb)]
        m_body.append(conv(nc, nc, mode='D'))
        H_conv0 = conv(nc, nc, mode='D'+act_mode)
        H_conv1 = conv(nc, out_nc, bias=False, mode='D')
        m_tail = sequential(H_conv0, H_conv1)

        self.model = sequential(m_head, ShortcutBlock(sequential(*m_body)),m_tail)

    def forward(self, LowDEM, Point_Ele): 
         
        ReverseIndex = (1 - (Point_Ele != 0).float() * 1)
        LowDEM = LowDEM * ReverseIndex + Point_Ele 
        SRDEM = self.model(LowDEM)     
        return SRDEM
   
# 训练时创建模型
def CreatModels(name, device):
    if name == "MSRResNet":
        model = MSRResNet1(in_nc=1, out_nc=1, nc=64, nb=16).to(device)
        testmodel = MSRResNet1(in_nc=1, out_nc=1, nc=64, nb=16).to(device)
        discriminator = Discriminator_VGG_96(in_nc=1, base_nc=64).to(device)
        testdiscriminator = Discriminator_VGG_96(in_nc=1, base_nc=64).to(device) 
        return model, testmodel, discriminator, testdiscriminator
    elif name == "TfaSR":
        model = TfaSR(16,6).to(device)
        testmodel = TfaSR(16,6).to(device)
        return model, testmodel
    elif name == "TfaSR_Point":
        model = TfaSR_Point(16,6).to(device)
        testmodel = TfaSR_Point(16,6).to(device)
        return model, testmodel
    elif name == "TfaSR1_Point":
        model = TfaSR_Point1(16,6).to(device)
        testmodel = TfaSR_Point1(16,6).to(device)
        return model, testmodel
    
def CreatOptimiser(G,D,opt,milestones):
    optimiser_G = optim.Adam(G.parameters(), lr=opt.lr)  # 生成器优化器
    optimizer_D = optim.Adam(D.parameters(), lr=opt.lr)  # 判别器优化器
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimiser_G, milestones, gamma=0.1)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones, gamma=0.1)
    schedulers = [scheduler_G,scheduler_D] 
    return optimiser_G, optimizer_D, schedulers