import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalNetworks(nn.Module):
    '''
    Implementacao de um notebook de exemplo do easyfsl: https://github.com/sicara/easy-few-shot-learning
    a classe que esta na lib e pode ser importada está desatualizada ;-;
    '''
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        
        # computa features 
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        # um prototype eh a media de todas as features de cada label
        n_way = len(torch.unique(support_labels))
        z_proto = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(0)
                for label in range(n_way)
            ]
        ) # shape: [n_way, feature_size]

        # distancia euclidiana
        dists = torch.cdist(z_query, z_proto) 

        # scores sao as distancias negativas (quanto menor a distancia, maior o score)
        scores = -dists
        
        return F.softmax(scores, dim=1)