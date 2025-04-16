import torch
import torch.nn as nn
from torch.autograd import Variable

# Matching Networks 
# https://arxiv.org/abs/1606.04080
# Original implementation: 
# https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/main/matching_network.py


class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim*2, feat_dim)
        self.softmax = nn.Softmax(dim=0) #####
        self.c_0 = Variable(torch.zeros(1,feat_dim))
        self.feat_dim = feat_dim

    def forward(self, f, G, n_ways):
        h = f
        c = self.c_0.expand_as(f).to(f.device)
        G_T = G.transpose(0,1)
        for _ in range(n_ways):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r),1)
            h, c = self.lstmcell(x, (h, c))
            h = h + f

        return h


class MatchingNetworks(nn.Module):
    def __init__(self, backbone):
        super(MatchingNetworks, self).__init__()
        feat_dim = backbone.num_features
        self.backbone = backbone
        self.FCE = FullyContextualEmbedding(feat_dim)
        self.G_encoder = nn.LSTM(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True)
        self.softmax = nn.Softmax(dim=0) #####
        self.feat_dim = feat_dim


    def encode_training_set(self, S):
        out_G = self.G_encoder(S.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
        G_norm = G.pow(2).sum(1, keepdim=True).pow(0.5)  # keepdim=True ensures correct broadcasting #####
        G_normalized = G / (G_norm + 1e-5)  # Use broadcasting instead of expand_as() #####
        return G, G_normalized

    def get_logprobs(self, f, G, G_normalized, Y_S):
        n_ways = f.size(0) ##### estamos usando n_ways = query_shots 
        F = self.FCE(f, G, n_ways)
        scores = F.mm(G_normalized.transpose(0,1))
        softmax = self.softmax(scores)
        logprobs = softmax.mm(Y_S).log()
        return logprobs



    def forward(self, S, Y_S, f, *args):
        S = self.backbone.forward(S) #####
        f = self.backbone.forward(f) #####
        G, G_normalized = self.encode_training_set(S)
        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)
        return logprobs

