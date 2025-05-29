import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

class MAML(nn.Module):
    def __init__(self, 
        backbone: nn.Module,
        num_classes: int = 2, 
        inner_lr: float = 0.01,
        inner_steps: int = 1
    ):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = backbone.num_features 
        self.classifier = nn.Linear(self.feat_dim, num_classes)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps

    def forward(self, 
        support_images: torch.Tensor,
        support_labels: torch.Tensor, 
        query_images: torch.Tensor,
        query_labels: torch.Tensor = None
    ):
        support_feats = self.backbone(support_images)
        query_feats = self.backbone(query_images)

        fast_weights = {
            name: param.clone().detach().requires_grad_(True) 
            for name, param in self.classifier.named_parameters()
        }
        
        support_labels = support_labels.argmax(dim=1)
        for _ in range(self.inner_steps):
            logits = functional_call(self.classifier, fast_weights, (support_feats,))
            loss = F.cross_entropy(logits, support_labels)

            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }

        query_logits = functional_call(self.classifier, fast_weights, (query_feats,))

        return query_logits

