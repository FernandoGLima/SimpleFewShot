import torch
import torch.nn as nn
import torch.nn.functional as F

# Learning to Compare: Relation Network for Few-Shot Learning
# https://arxiv.org/abs/1711.06025
 

class ConvRelationModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.relation_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relation_layer(x)
        return self.fc(x)


class RelationNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.relation_module = ConvRelationModule(2 * backbone.num_features)

    def forward(
        self,
        support_images: torch.Tensor,    
        support_labels: torch.Tensor,    
        query_images: torch.Tensor,
        *args
    ) -> torch.Tensor:

        # extract last feature maps
        z_support = self.backbone.forward_features(support_images) # [ways*shots, c, h, w]
        z_query = self.backbone.forward_features(query_images)     # [query, c, h, w]

        ways = support_labels.shape[1]
        query = z_query.size(0) 

        # compute prototype
        support_labels = support_labels.argmax(1)     
        z_proto = torch.stack([
            z_support[support_labels == label].mean(0)
            for label in range(ways)
        ])  # [ways, c, h, w]

        # pairwise relations
        z_proto_ext = z_proto.unsqueeze(0).repeat(query, 1, 1, 1, 1) # [query, ways, C, H, W]
        z_query_ext = z_query.unsqueeze(1).repeat(1, ways, 1, 1, 1)  # [query, ways, C, H, W]

        relation_pairs = torch.cat((z_proto_ext, z_query_ext), dim=2) # [query, ways, 2*C, H, W]
        relation_pairs = relation_pairs.view(-1, *relation_pairs.shape[2:]) # [query * ways, 2*C, H, W]

        relations = self.relation_module(relation_pairs) # [query * ways, 1]
        relations = relations.view(query, ways) # [query, ways]

        return relations
