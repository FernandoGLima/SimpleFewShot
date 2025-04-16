import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# FEAT: Few-Shot Image Classification with Feature-Aware Transformer
# https://arxiv.org/abs/1812.03664
# Based on the open-source implementation of https://github.com/Sha-Lab/FEAT/


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output

class FEAT(nn.Module):
    def __init__(self, 
        backbone: nn.Module,
        use_euclidean: bool = False,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.temperature = temperature
        self.distance = self._euclidean_distance if use_euclidean else self._cosine_distance

        self.hdim = backbone.num_features
        self.slf_attn = MultiHeadAttention(1, self.hdim, self.hdim, self.hdim, dropout=0.5)

    def _euclidean_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x1, x2, p=2)

    def _cosine_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        return -torch.mm(x1, x2.T) 

    def forward(self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:       

        # extract features
        z_support = self.backbone.forward(support_images) # [ways*shots, hdim]
        z_query = self.backbone.forward(query_images)     # [query, hdim]

        ways = support_labels.shape[1] 

        support_labels = support_labels.argmax(1) 
        proto = torch.cat([
            z_support[torch.nonzero(support_labels == label)].mean(0)
            for label in range(ways)
        ]) # shape: [ways, hdim]

        # this hack is necessary because we are using "batch_size" == 1 during training
        proto = proto.unsqueeze(0) # [1, ways, hdim]
        proto = self.slf_attn(proto, proto, proto).squeeze(0) # [ways, hdim]

        logits = -self.distance(z_query, proto) # [query, ways]

        # contrastive regularization 
        query_labels = query_labels.argmax(1)
        all_labels = torch.cat([support_labels, query_labels], dim=0) # [ways*shot+query]
        all_tasks = torch.cat([z_support, z_query], dim=0) # [ways*shot+query, D]
        
        all_tasks = all_tasks.unsqueeze(0)
        all_emb = self.slf_attn(all_tasks, all_tasks, all_tasks).squeeze(0) # [ways*shot+query, D]

        ways_center = torch.zeros(ways, self.hdim).to(all_tasks.device) 
        ways_center = torch.stack([
            all_emb[all_labels == i].mean(0)
            for i in range(ways)
        ])  # [ways, D]

        logits_reg = -self.distance(all_emb, ways_center) # [ways*shot+query, ways]
        
        return logits, logits_reg

