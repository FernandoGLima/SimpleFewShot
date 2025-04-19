import torch
import torch.nn as nn
import torch.nn.functional as F

# DN4: Deep Nearest Neighbor Neural Network 
# https://arxiv.org/abs/1903.12290 


class DN4(nn.Module):
    def __init__(self, backbone: nn.Module, k: int = 3):
        super().__init__()
        self.backbone = backbone
        self.k = k

    def forward(self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor = None,
    ) -> torch.Tensor:       

        # extract last feature maps
        z_support = self.backbone.forward_features(support_images) # [ways*shots, c, h, w]
        z_query = self.backbone.forward_features(query_images)     # [query, c, h, w]

        z_support = z_support.flatten(2, 3).permute(0, 2, 1) # [ways*shots, h*w, c]
        z_query = z_query.flatten(2, 3).permute(0, 2, 1)     # [query, h*w, c]

        z_support = F.normalize(z_support, dim=2) 
        z_query = F.normalize(z_query, dim=2)     

        # group descriptors by label
        descriptors = []
        support_labels = support_labels.argmax(1) # [ways*shots]
        for label in torch.unique(support_labels):
            label_features = z_support[support_labels == label] # [shots, h*w, c]
            label_features = label_features.flatten(0, 1)       # [shots*h*w, c]
            descriptors.append(label_features)                  # [ways, shots*h*w, c]

        # for each query, compute the distance to all support descriptors
        query = z_query.shape[0]
        ways = len(descriptors)
        scores = torch.empty(query, ways, device=z_query.device)

        for i in range(query):
            query_feat = z_query[i]  # [h*w, c]
            for j, class_feat in enumerate(descriptors):
                sim = torch.matmul(query_feat, class_feat.T)  # [h*w, shots*h*w]
                top_k = torch.topk(sim, self.k, dim=1).values # [h*w, k]
                scores[i, j] = top_k.mean() 

        return scores # [query, ways]





