import torch
import torch.nn as nn
import torch.nn.functional as F

# Prototypical Networks for Few-shot Learning
# https://arxiv.org/abs/1703.05175
# based on the open-source implementation of https://github.com/sicara/easy-few-shot-learning

class PrototypicalNetworks(nn.Module):
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

        # compute prototype
        ways = support_labels.shape[1] 
        support_labels = support_labels.argmax(1) 
        z_proto = torch.cat([
            z_support[torch.nonzero(support_labels == label)].mean(0)
            for label in range(ways)
        ]) # [ways, hdim]

        # compute distances
        logits = -torch.cdist(z_query, z_proto) 

        return logits