import torch
import torch.nn as nn
from collections import OrderedDict
import torch.optim as optim
from torch.nn import functional as F
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
"""
# Define the ModulatedDeformConv2dPack class (shortened here, using your provided implementation)
class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_offset = nn.Conv2d(
            in_channels,
            self.deform_groups * 3 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True)
        self.init_weights()

    def init_weights(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)


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
    
"""

"""import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d"""
# Define the ModulatedDeformConv2dPack class (shortened here, using your provided implementation)
# Assuming ModulatedDeformConv2dPack is already defined
def swish(x):
    return F.relu(x)
class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()
        # Use conv1 and conv2 instead of conv1 and conv2
        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))  # Update to use conv1 and bn1
        return self.bn2(self.conv2(y)) + x  # Update to use conv2 and bn2

    

    def functional_forward(self, x, params, i):
        # 使用正确的键名来获取卷积层和批归一化层的权重
        y = F.conv2d(x, params['residual_block'+ str(i) +'.conv1.weight'], params.get('residual_block'+ str(i) +'.conv1.bias'), 
                    stride=self.conv1.stride, padding=self.conv1.padding)
        y = F.batch_norm(y, None,None,
                        weight=params.get('residual_block'+ str(i) +'.bn1.weight'), bias=params.get('residual_block'+ str(i) +'.bn1.bias'), training=True)
        y = swish(y)

        y = F.conv2d(y, params['residual_block'+ str(i) +'.conv2.weight'], params.get('residual_block'+ str(i) +'.conv2.bias'),
                    stride=self.conv2.stride, padding=self.conv2.padding)
        y = F.batch_norm(y, None,None,
                        weight=params.get('residual_block'+ str(i) +'.bn2.weight'), bias=params.get('residual_block'+ str(i) +'.bn2.bias'), training=True)
        return y + x

class TfaSR(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(TfaSR, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        # Add residual blocks
        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Deformable convolution layers (Modulated Deformable Convolution)
        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 =  DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        
        y = x.clone()
        x1 = y
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)
        y = self.conv2(y)
        y = self.bn2(y)
        x = y + x
        x = self.dconv2_2(x)
        x = swish(x)
        x = swish(self.dconv2_3(x))
        x = self.dconv2_4(x)
        x = self.conv4(x)
        
        return  x

    def function_forward(self, x, params):
        x = F.conv2d(x, params['conv1.weight'], params.get('conv1.bias'), stride=self.conv1.stride, padding=self.conv1.padding)
        y = x.clone()
        x1 = y
        for i in range(self.n_residual_blocks):
            i += 1
            residual_block = self.__getattr__('residual_block' + str(i))
            y = residual_block.functional_forward(y, params, i)  # 在这里调用 residualBlock 的 function_forward
        y = F.conv2d(y, params['conv2.weight'], params.get('conv2.bias'), stride=self.conv2.stride, padding=self.conv2.padding)
        y = F.batch_norm(y, None,None, weight=params.get('bn2.weight'), bias=params.get('bn2.bias'), training=True)
        x = x + y
        x = self.dconv2_2.functional_forward(x, params, "dconv2_2")
        x = swish(x)
        x = self.dconv2_3.functional_forward(x, params, "dconv2_3")
        x = swish(x)
        x = self.dconv2_4.functional_forward(x, params, "dconv2_4")
        x = F.conv2d(x, params['conv4.weight'], params.get('conv4.bias'), stride=self.conv4.stride, padding=self.conv4.padding)
        return  x




