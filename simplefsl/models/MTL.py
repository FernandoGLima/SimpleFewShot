import  torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

# Meta-Transfer Learning for Few-Shot Learning
# https://arxiv.org/abs/1812.02391
# Based on the open-source implementation of https://github.com/yaoyao-liu/meta-transfer-learning/tree/main

class _ConvNdMtl(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        self.weight.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.requires_grad=False
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dMtl(_ConvNdMtl):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMtl, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inp):
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        new_weight = self.weight.mul(new_mtl_weight)
        if self.bias is not None:
            new_bias = self.bias + self.mtl_bias
        else:
            new_bias = None
        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
def convert_mtl_module(module):
    for name, child in module.named_children():
        convert_mtl_module(child)

        if isinstance(child, torch.nn.Conv2d):
            conv2d_mtl = Conv2dMtl(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                child.dilation,
                child.groups,
                child.bias is not None
            )
            with torch.no_grad():
                conv2d_mtl.weight.copy_(child.weight)
                if child.bias is not None:
                    conv2d_mtl.bias.copy_(child.bias)

            setattr(module, name, conv2d_mtl)

    return module

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, ways, z_dim, device=None):
        super().__init__()
        self.way = ways
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.way, self.z_dim], device=device))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.way, device=device))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, backbone, ways=2, base_lr=0.01, update_step=100):
        super().__init__()
        self.update_lr = base_lr
        self.update_step = update_step
        self.z_dim = backbone.num_features
        self.backbone = backbone
        # self.base_learner = BaseLearner(ways, self.z_dim)
        self.flag = True
        convert_mtl_module(self.backbone) #- Foi optado por não maner a conversão das 

    def forward(self,  train_imgs, train_labels, query_imgs):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        if self.flag == True:
            self.base_learner = BaseLearner(query_imgs.size(0), self.z_dim, device=query_imgs.device)
            self.flag = False
        
        embedding_query = self.backbone(query_imgs)
        embedding_shot = self.backbone(train_imgs)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, train_labels)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, train_labels)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q