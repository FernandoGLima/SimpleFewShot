import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import transpose as t
from torch import inverse as inv
from torch import mm
import numpy as np

# Meta-learning with differentiable closed-form solvers
# https://arxiv.org/abs/1805.08136
# Based on the open-source implementation of https://github.com/bertinetto/r2d2

class RRFeatures(nn.Module):
    def __init__(self):
        super(RRFeatures, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96, momentum=0.1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.1))
        self.features2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192, momentum=0.1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.1))
        self.features3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, padding=1),
            nn.BatchNorm2d(384, momentum=0.1),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1))
        self.features4 = nn.Sequential(
            nn.Conv2d(384, 512, 3, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.MaxPool2d(2, stride=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1))

        self.pool3 = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x3 = self.pool3(x)
        x3 = x3.view(x3.size(0), -1)
        x = self.features4(x)
        x4 = x.view(x.size(0), -1)
        x = torch.cat((x3, x4), 1)
        return x



def t_(x):
    return t(x, 0, 1)

#foi retirado o linsys (linear system solver) que é usado no lugar de fazer a inversão da matriz
class RRNet(nn.Module):
    def __init__(self, backbone, learn_lambda=True, init_lambda=1, init_adj_scale=1e-4, lambda_base=1, adj_base=1):
        super(RRNet, self).__init__()
        self.backbone = backbone
        self.lambda_rr = LambdaLayer(learn_lambda, init_lambda, lambda_base)
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)

    def forward(self, train_imgs, train_labels, query_imgs, query_labels):
        self.n_way = train_labels.size(1)
        self.n_shot = train_labels.size(0) // self.n_way

        zs = self.backbone.forward(train_imgs)
        zq = self.backbone.forward(query_imgs)

        self.output_dim = zs.size(1)
        
        if self.n_way * self.n_shot > self.output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(self.output_dim + 1).cuda())
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(self.n_way * self.n_shot).cuda())

        y_inner = train_labels / np.sqrt(self.n_way * self.n_shot)
        
        # add a column of ones for the bias
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((zs, ones), 1), self.n_way, self.n_shot, I, y_inner)

        else:
            wb = self.rr_standard(torch.cat((zs, ones), 1), self.n_way, self.n_shot, I, y_inner)

        w = wb.narrow(dim=0, start=0, length=self.output_dim)
        b = wb.narrow(dim=0, start=self.output_dim, length=1)
        out = mm(zq, w) + b
        y_hat = self.adjust(out)
        # print("%.3f  %.3f  %.3f" % (w.mean()*1e5, b.mean()*1e5, y_hat.max()))
        # print('Loss: %.3f Acc: %.3f' % (loss_val.data[0], acc_val.data[0]))
        return y_hat

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary):
        x /= np.sqrt(n_way * n_shot)

        
        w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)

        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary):
        x /= np.sqrt(n_way * n_shot)

        w = mm(mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary)

        return w

class AdjustLayer(nn.Module):
    def __init__(self, init_scale=1e-4, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale]).cuda())
        self.bias = nn.Parameter(torch.FloatTensor([init_bias]).cuda())
        self.base = base

    def forward(self, x):
        if self.base == 1:
            return x * self.scale + self.bias
        else:
            return x * (self.base ** self.scale) + self.base ** self.bias - 1


class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=True, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda]).cuda()
        self.base = base
        if learn_lambda:
            self.l = nn.Parameter(self.l)
        else:
            self.l = Variable(self.l)

    def forward(self, x):
        if self.base == 1:
            return x * self.l
        else:
            return x * (self.base ** self.l)