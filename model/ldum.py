#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-01-29 17:54:45
LastEditTime: 2022-06-21 15:56:09
@Description: 
'''

import os
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
# from torchvision.transforms import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, args):
    # channels, denselayer, growthrate):
        super(Net, self).__init__()

        self.args = args
        num_channels = self.args['data']['n_colors']
        scale_factor = self.args['data']['upsacle']

        self.D = 20
        self.C = 6
        self.G = 32
        self.G0 = 64

        self.sfe1 = ConvBlock(num_channels, self.G0, 3, 1, 1, activation='relu', norm=None)
        self.sfe2 = ConvBlock(self.G0, self.G0, 3, 1, 1, activation='relu', norm=None)

        self.RDB1 = RDB(self.G0, self.C, self.G)
        self.RDB2 = RDB(self.G0, self.C, self.G)
        self.RDB3 = RDB(self.G0, self.C, self.G)

        self.GFF_1x1 = nn.Conv2d(self.G0*3, self.G0, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(self.G0, self.G0, kernel_size=1, padding=0, bias=True)

        self.up = Upsampler(scale_factor, self.G0, activation=None)
        self.output_conv = ConvBlock(self.G0, num_channels, 3, 1, 1, activation=None, norm=None)

        ## res block
        res = [
            # ResnetBlock_triple(3, 3, 1, 1, False, 0.1, activation='prelu', norm=None, middle_size=32, output_size=32) for _ in range(3)
            ConvBlock(3, 32, 3, 1, 1, activation='relu', norm=None, bias = True)
        ]
        self.res = nn.Sequential(*res)
        res1 = [
            # ResnetBlock_triple(3, 3, 1, 1, False, 0.1, activation='prelu', norm=None, middle_size=32, output_size=3) for _ in range(3)
            ConvBlock(32, 3, 3, 1, 1, activation='relu', norm=None, bias = True) 
        ]
        self.res1 = nn.Sequential(*res1)

        ## prox
        prox1 = [
            RCAB(32, 3, 16, act=nn.ReLU(True), res_scale=1)
            # ResnetBlock_triple(32, 3, 1, 1, False, 1, activation='relu', norm=None, middle_size=32, output_size=32) for _ in range(3)
        ]
        self.prox1 = nn.Sequential(*prox1)

        prox2 = [
            # RCAB(3, 3, 16, act=nn.ReLU(True), res_scale=1)
            ResnetBlock_triple(3, 3, 1, 1, False, 1, activation='relu', norm=None, middle_size=32, output_size=3) for _ in range(1)
        ]
        self.prox2 = nn.Sequential(*prox2)

        self.lap2 =  torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lap2.data.fill_(0.1)

        self.up2 = Conv_up(32, 4)
        self.down = Conv_down(3, 4)

        self.up1 = Conv_up(3, 4)

        operations = []
        for i in range(2):
            u = unfolding_block()
            operations.append(u)
        self.operations = nn.ModuleList(operations)


    def forward(self, x):
        lr = x
        ## itera 0
        x = self.sfe1(x)
        x_0 = self.sfe2(x)
        x_1 = self.RDB1(x_0)
        x_2 = self.RDB2(x_1)
        x_3 = self.RDB3(x_2)
        xx = torch.cat((x_1, x_2, x_3), 1)
        x_LF = self.GFF_1x1(xx)
        x_GF = self.GFF_3x3(x_LF)
        x = x_GF + x
        x = self.up(x)

        h_0 = self.output_conv(x)

        r_0 =h_0 - F.interpolate(lr, scale_factor=4, mode='bicubic')

        ## itera 1

        r_1 = r_0
        r_1 = self.res(r_1)
        r = self.prox1(r_1)

        R = self.res1(r)
        h_h = lr + self.down(R) - self.down(h_0)
        h_h = self.lap2 * self.up1(h_h)
        h_h = h_0 + h_h
        x = self.prox2(h_0 )
        x_1 = x
        # h = self.lap2*h_h
        # x = self.prox2(h_0 - h)

        x_list = []
        for op in self.operations:
            x, r, R = op.forward(x, r, lr)
            x_list.append(R)
  
        return x

class unfolding_block(nn.Module):
    def __init__(self):
        super(unfolding_block, self).__init__()

        feature_num = 32

        ## prox
        prox1 = [
            ResnetBlock_triple(feature_num, 3, 1, 1, False, 0.1, activation='relu', norm=None, middle_size=feature_num, output_size=feature_num) for _ in range(1)
        ]
        self.prox1 = nn.Sequential(*prox1)

        prox2 = [
            ResnetBlock_triple(3, 3, 1, 1, False, 0.1, activation='relu', norm=None, middle_size=feature_num, output_size=3) for _ in range(1)
        ]
        self.prox2 = nn.Sequential(*prox2)
        
        ## hyper-parameter
        self.lap1 =  torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lap2 =  torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lap1.data.fill_(0.1)
        self.lap2.data.fill_(0.1)

        ## res block
        res = [
            ConvBlock(32, 3, 3, 1, 1, activation='relu', norm=None, bias = True)
        ]
        self.res = nn.Sequential(*res)

        ## res1 block
        res1 = [
            # ResnetBlock_triple(3, 3, 1, 1, False, 0.1, activation='prelu', norm=None, middle_size=feature_num, output_size=3) for _ in range(3)
            ConvBlock(3, 32, 3, 1, 1, activation='relu', norm=None, bias = True)
        ]
        self.res1 = nn.Sequential(*res1)

        ## res2 block
        res2 = [
            ConvBlock(32, 3, 3, 1, 1, activation='relu', norm=None, bias = True)
        ]
        self.res2 = nn.Sequential(*res2)

        ## res2 block
        res3 = [
            ConvBlock(3, 32, 3, 1, 1, activation='relu', norm=None, bias = True)
        ]
        self.res3 = nn.Sequential(*res3)

        self.up = Conv_up(3, 4)
        self.down = Conv_down(3, 4)

    def candy_f(self, im):
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') 
        sobel_kernel = sobel_kernel.reshape((1, 3, 3, 3))
        weight = Variable(torch.from_numpy(sobel_kernel))
        edge_detect = F.conv2d(Variable(im.cpu()), weight, padding=1)
        #edge_detect = edge_detect.squeeze().detach().numpy()
        return edge_detect.cuda()

    def lap_f(self, img):
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        a = torch.from_numpy(a).float().unsqueeze(0)
        a = a.repeat(3, 3, 1, 1)
        # a = torch.stack((a, a, a, a))
        conv1.weight = nn.Parameter(a, requires_grad=False)
        conv1 = conv1.cuda()
        G_x = conv1(img)

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        b = torch.from_numpy(b).float().unsqueeze(0)
        b = b.repeat(3, 3, 1, 1)
        conv2.weight = nn.Parameter(b, requires_grad=False)
        conv2 = conv2.cuda()
        G_y = conv2(img)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return G

    def forward(self, x, r, lr):   

        ## Res
        tem = F.interpolate(x, scale_factor=1/4, mode='bicubic') - lr
        tem = self.candy_f(F.interpolate(x, scale_factor=1/4, mode='bicubic'))
        r_h = self.res(r)
        r_h = self.down(r_h)
        r_h = tem - r_h
        r_h  = self.up(r_h)
        r_t =  self.lap1 * self.res1(r_h)
        r_t = r_t + r
        r_t = self.prox1(r_t)

        ## HR
        R = self.res2(r_t)
        tem = self.down(R)
        tem1 = self.down(x)
        h_h = lr + tem - tem1
        h_h = self.lap2* self.up(h_h)
        h_h = x + h_h
        h = self.prox2(h_h)

        # lr = F.interpolate(lr, scale_factor=4, mode='bicubic')
        # r_t = (1 + self.lap1) * (x-lr)
        # r_t = r_t- self.lap1* x + lr * self.lap1
        # r_t = self.prox1(r_t)
        # h_t = (1 - self.lap2) * x + self.lap2 * r_t + lr * self.lap2
        # h_t = self.prox2(h_t)
        return h, r_t, R


class Conv_up(nn.Module):
    def __init__(self, c_in,up_factor=4):
        super(Conv_up, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)

        if up_factor==2:
            modules_tail = [
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=up_factor,padding=1,output_padding=1),
                ConvBlock(64, c_in, 3, 1, 1, activation='relu', norm=None, bias = True)]
                # conv(64, c_in, 3)]
        elif up_factor==3:
            modules_tail = [
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=up_factor,padding=0,output_padding=0),
                # conv(64, c_in, 3)]
                ConvBlock(64, c_in, 3, 1, 1, activation='relu', norm=None, bias = True)]
        elif up_factor==4:
            modules_tail = [
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
                ConvBlock(64, c_in, 3, 1, 1, activation='relu', norm=None, bias = True)]
                # conv(64, c_in, 3)]
        self.tail = nn.Sequential(*modules_tail)    

    def forward(self, input):
        
        out = self.body(input)
        out = self.tail(out)
        return out

class Conv_down(nn.Module):
    def __init__(self, c_in,up_factor):
        super(Conv_down, self).__init__()

        body = [nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2), nn.ReLU(),
                ]
        self.body = nn.Sequential(*body)
        if up_factor==4:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=2),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=2),
                ConvBlock(64, c_in, 3, 1, 1, activation='relu', norm=None, bias = True)]
                # conv(64, c_in, 3)]
        elif up_factor==3:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=up_factor),
                ConvBlock(64, c_in, 3, 1, 1, activation='relu', norm=None, bias = True)]
                # conv(64, c_in, 3)]
        elif up_factor==2:
            modules_tail = [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,stride=up_factor),
                ConvBlock(64, c_in, 3, 1, 1, activation='relu', norm=None, bias = True)]
                # conv(64, c_in, 3)]                
        self.tail = nn.Sequential(*modules_tail)    

    def forward(self, input):
        
        out = self.body(input)
        out = self.tail(out)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ConvBlock(n_feat, n_feat, 3, 1, 1, activation=None, norm=None))
            # if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = res + x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(ConvBlock(n_feat, n_feat, 3, 1, 1, activation=None, norm=None))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

class DenseBlock_rdn(torch.nn.Module):
    def __init__(self, in_filter, out_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(DenseBlock_rdn, self).__init__()

        self.conv = torch.nn.Conv2d(in_filter, out_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm

        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            out = self.act(out)
        else:
            out = out
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        G0_ = G0
        convs = []
        for i in range(C):
            convs.append(DenseBlock_rdn(G0_, G, 3, 1, 1, activation='relu', norm=None))
            G0_ = G0_ + G
        self.dense_layer = nn.Sequential(*convs)
        self.conv_1x1 = nn.Conv2d(G0_, G0, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x_out = self.dense_layer(x)
        x_out = self.conv_1x1(x_out)
        x = x_out + x
        return x