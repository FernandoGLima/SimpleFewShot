import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO
class BidirectionalLSTM(nn.Module):
    def __init__(self):
        super(BidirectionalLSTM, self).__init__()

    def forward(self):
        pass


class MatchingNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(MatchingNetworks, self).__init__()
        self.backbone = backbone
        self.LSTM = BidirectionalLSTM()

    def g(self, z_support: torch.Tensor):
        # TODO
        return z_support


    def f(self, z_query: torch.Tensor, z_support: torch.Tensor):
        # TODO
        return z_query


    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        fce: bool = True,
    ) -> torch.Tensor:
        
        # computa features 
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        
        # media de todas as features de cada label
        n_way = len(torch.unique(support_labels))
        z_support = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        )
    
        # full context embedding
        if fce:
            z_support = self.g(z_support)
            z_query = self.f(z_query, z_support)

        # similaridade cosseno 
        attention = F.cosine_similarity(z_query.unsqueeze(1), z_support.unsqueeze(0), dim=-1)  
        attention = F.softmax(attention, dim=1) # shape: [n_query, n_way]

        return attention

        