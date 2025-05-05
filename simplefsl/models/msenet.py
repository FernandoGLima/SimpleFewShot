import torch
import torch.nn as nn
from timm.models import FeatureListNet

# MSENet: Multi-Scale Enhanced Network
# https://arxiv.org/abs/2409.07989v2
# Based on the open-source implementation of https://github.com/FatemehAskari/MSENet/


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim: int, gamma: float = 0.2, gamma1: float = 0.2):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim , out_channels=in_dim//8 , kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim , out_channels=in_dim//8 , kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim , out_channels=in_dim , kernel_size=1)
        self.gamma = gamma
        self.gamma1 = gamma1
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out


class MSENet(nn.Module):
    def __init__(self, backbone: nn.Module, skip_layers: int = 1):
        super().__init__()
        self.backbone = FeatureListNet(backbone) # to extract all features maps
        self.skip_layers = skip_layers # to reduce memory usage
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        feature_dims = self.backbone.feature_info.channels()[skip_layers:]
        self.attn_blocks = nn.ModuleList([Self_Attn(dim) for dim in feature_dims])
        weights = torch.tensor([1.0 + 0.1 * i for i in range(len(feature_dims))])
        self.weights = nn.Parameter(weights, requires_grad=True)

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor = None,
    ) -> torch.Tensor:

        z_support = self.backbone.forward(support_images)[self.skip_layers:]
        z_query = self.backbone.forward(query_images)[self.skip_layers:]

        ways = support_labels.shape[1]
        support_labels = support_labels.argmax(1)

        logits = 0
        for support, query, w, attn in zip(z_support, z_query, self.weights, self.attn_blocks): 
            support = attn(support) # [ways*shot, c, h, w]
            query = attn(query) # [ways, c, h, w]

            support = self.avg_pool(support).flatten(1) # [ways*shot, hdim]
            query = self.avg_pool(query).flatten(1)     # [ways, hdim]

            proto = torch.cat([
                support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(ways)
            ]) # shape: [ways, hdim]

            dist = -torch.cdist(query, proto) # [query, ways]
            logits += w * dist
        
        return logits
