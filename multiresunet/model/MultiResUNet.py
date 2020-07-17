from cv2 import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3,stride=1,padding=1,act='relu'):
        super(conv_block,self).__init__()
        if act == None:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
                nn.BatchNorm2d(ch_out)
            )
        elif act == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )
        elif act == 'sigmoid':
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.Sigmoid()
            )

    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class res_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_block,self).__init__()
        self.res = conv_block(ch_in,ch_out,1,1,0,None)
        self.main = conv_block(ch_in,ch_out)

    def forward(self,x):
        res_x = self.res(x)
        main_x = self.main(x)
        out = res_x.add(main_x)
        out = nn.ReLU(inplace=True)(out)
        out = nn.BatchNorm2d(out.size[1])(out)
        return out


class ResPath(nn.Module):
    def __init__(self,ch,stage):
        super(ResPath,self).__init__()
        self.stage = stage
        self.block = res_block(ch, ch)

    def forward(self, x):
        out = self.block(x)
        for i in range(self.stage-1):
            out = self.block(out)


class MultiResBlock(nn.Module):
    def __init__(self,U,alpha=1.67):
        super(MultiResBlock,self).__init__()
        self.W = alpha * U
        self.residual_layer = conv_block(3, int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5), 1, 1, 0, act=None)
        self.conv3x3 = conv_block(3, int(self.W*0.167))
        self.conv5x5 = conv_block(int(self.W*0.167), int(self.W*0.333))
        self.conv7x7 = conv_block(int(self.W*0.333), int(self.W*0.5))
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm_1 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
        self.batchnorm_2 = nn.BatchNorm2d(int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5))
        
    def forward(self, x):
        res = self.residual_layer(x)
        sbs = self.conv3x3(x)
        obo = self.conv5x5(sbs)
        cbc = self.conv7x7(obo)
        all_t = torch.cat((sbs, obo, cbc), 1)
        all_t_b = self.batchnorm_1(all_t)
        print(all_t_b.size())
        print(res.size())
        out = all_t_b.add(res)
        out = self.relu(out)
        out = self.batchnorm_2(out)

        return out


class MultiResUNet(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MultiResUNet,self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.mresblock1 = MultiResBlock(32) 
        self.up_conv1 = up_conv(32*2, 32)
        self.res_path1 = ResPath(32, 4)
        self.mresblock2 = MultiResBlock(32*2)
        self.up_conv2 = up_conv(32*4, 32*2)
        self.res_path2 = ResPath(32*2, 3)
        self.mresblock3 = MultiResBlock(32*4)
        self.up_conv3 = up_conv(32*8, 32*4)
        self.res_path3 = ResPath(32*4, 2)
        self.mresblock4 = MultiResBlock(32*8)
        self.up_conv4 = up_conv(32*16, 32*8)
        self.res_path4 = ResPath(32*8, 1)
        self.mresblock5 = MultiResBlock(32*16)
        self.obo = conv_block(32, 1, act='sigmoid')

    def forward(self, x):
        x1 = self.mresblock1(x)
        res_x1 = self.res_path1(x1)
        x2 = self.Maxpool(x1)

        x2 = self.mresblock2(x2)
        res_x2 = self.res_path2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.mresblock2(x3)
        res_x3 = self.res_path2(x3)
        x4 = self.Maxpool(x3)

        x4 = self.mresblock2(x4)
        res_x4 = self.res_path2(x4)
        x5 = self.Maxpool(x4)

        x5 = self.mresblock5(x5)


