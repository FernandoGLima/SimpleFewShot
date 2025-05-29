import torch
import torch.nn as nn
import torch.nn.functional as F

# TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning
# https://arxiv.org/abs/1905.06549


class ProjectionNetwork(nn.Module):
    def __init__(self, hdim: int, proj_dim: int):
        super().__init__()
        self.hdim = hdim
        self.proj_dim = proj_dim
        
        self.generator = nn.Sequential(
            nn.Linear(hdim, hdim), 
            nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim * proj_dim)
        )

    def forward(self, task_embedding: torch.Tensor) -> torch.Tensor:
        proj_matrix = self.generator(task_embedding)
        proj_matrix = proj_matrix.view(task_embedding.size(0), self.proj_dim, self.hdim)
        q, _ = torch.linalg.qr(proj_matrix.squeeze(0).T) # orthogonalization via QR decomposition
        proj_matrix = q.T.unsqueeze(0)  
        
        return proj_matrix


class TapNet(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int = 64):
        super().__init__()
        self.backbone = backbone
        self.proj_dim = proj_dim
        self.projection_net = ProjectionNetwork(backbone.num_features, proj_dim)

    def forward(self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:   
        
        # extract features
        z_support = self.backbone.forward(support_images) # [ways*shots, hdim]
        z_query = self.backbone.forward(query_images)     # [query, hdim]

        # compute task embedding
        task_embedding = z_support.mean(dim=0, keepdim=True)
        
        # compute projection matrix (M)
        proj_matrix = self.projection_net(task_embedding) # [1, proj_dim, hdim]

        # project features
        z_support_proj = self._project_features(z_support, proj_matrix) # [ways*shots, proj_dim]
        z_query_proj = self._project_features(z_query, proj_matrix)     # [query, proj_dim]

        # compute class prototypes
        ways = support_labels.shape[1] 
        support_labels = support_labels.argmax(dim=1)
        z_proto = torch.stack([
            z_support_proj[support_labels == i].mean(0) 
            for i in range(ways)
        ]) # [ways, proj_dim]

        # compute similarity scores
        logits = -torch.cdist(z_query_proj, z_proto)  # [n_query, n_way]

        return logits

    def _project_features(self, features: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
        proj_features = torch.matmul(features, proj_matrix.squeeze(0).T)  
        return F.normalize(proj_features, dim=-1)
