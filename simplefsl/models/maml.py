import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from collections import OrderedDict

class TimmWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int = 2):
        super().__init__()
        self.backbone = backbone
        self.output_head = nn.Linear(backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.output_head(features)

    def functional_forward(self, x: torch.Tensor, weights: dict) -> torch.Tensor:
        return functional_call(self, weights, (x,))


class MAML(nn.Module):
    def __init__(self, 
        backbone: nn.Module, 
        inner_lr: float = 0.01, 
        meta_lr: float = 0.001, 
        inner_steps: int = 5,
        num_classes: int = 2,
    ):
        super().__init__()
        self.model = TimmWrapper(backbone, num_classes)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def _inner_loop(self, 
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
    ):
        fast_weights = {
            name: param.clone().detach().requires_grad_(True)
            for name, param in self.model.named_parameters()
        }
        
        for _ in range(self.inner_steps):
            logits = self.model.functional_forward(support_images, fast_weights)
            loss = F.cross_entropy(logits, support_labels)

            grads = torch.autograd.grad(
                loss, 
                fast_weights.values(), 
                create_graph=True,
                allow_unused=True
            )

            grads = [ 
                grad if grad is not None else torch.zeros_like(param)
                for grad, param in zip(grads, fast_weights.values())
            ]

            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }

        return fast_weights

    def forward(self, 
        support_images: torch.Tensor,
        support_labels: torch.Tensor, 
        query_images: torch.Tensor,
        query_labels: torch.Tensor = None,
    ):
        support_labels = support_labels.argmax(dim=1)

        adapted_weights = self._inner_loop(support_images, support_labels)
        query_logits = self.model.functional_forward(query_images, adapted_weights)

        return query_logits


