import torch
import torch.nn as nn
import torch.nn.functional as F

# TADAM: Task dependent adaptive metric for improved few-shot learning
# https://arxiv.org/abs/1805.10123


class TaskEncoder(nn.Module):
    def __init__(self, hdim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(hdim), 
            nn.ReLU(inplace=True),
            nn.Linear(hdim, hdim),
            nn.ReLU(inplace=True)
        )

    def forward(self, z_proto: torch.Tensor) -> torch.Tensor:
        return self.encoder(z_proto)


class FiLMGenerator(nn.Module):
    def __init__(self, task_embedding_dim: int, hdim: list[int]):
        super().__init__()
        self.gamma_layers = nn.ModuleList([nn.Linear(task_embedding_dim, d) for d in hdim])
        self.beta_layers = nn.ModuleList([nn.Linear(task_embedding_dim, d) for d in hdim])

    def forward(self, task_embedding: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        gammas = [layer(task_embedding) for layer in self.gamma_layers]
        betas = [layer(task_embedding) for layer in self.beta_layers]
        return gammas, betas


class FiLMResNetBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x * gamma[:, :, None, None] + beta[:, :, None, None]


class TADAM(nn.Module):
    def __init__(self, 
            backbone: nn.Module, 
            film_layers: set[str] = {'layer4'}, # to see the available layers use `backbone.feature_info`
            task_embedding_dim: int = 128,
            scale: float = 5.0
        ):
        super().__init__()
        self.backbone = backbone
        self.task_embedding_dim = task_embedding_dim

        self.hdim = [feat['num_chs'] for feat in backbone.feature_info if feat['module'] in film_layers]

        self.task_encoder = TaskEncoder(task_embedding_dim)
        self.film_generator = FiLMGenerator(task_embedding_dim, self.hdim)
        self.scale = nn.Parameter(torch.tensor(scale))

        self.film_layers_map = nn.ModuleDict({
            name: FiLMResNetBlock(getattr(backbone, name)) for name in film_layers
        })

    def forward(self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:       

        z_support = self._extract_film_features(support_images)

        # compute class prototypes
        ways = support_labels.shape[1] 
        support_labels = support_labels.argmax(dim=1)
        z_proto = torch.stack([
            z_support[support_labels == i].mean(0) 
            for i in range(ways)
        ])

        # generate FiLM parameters
        task_embedding = self.task_encoder(z_proto.flatten().unsqueeze(0))
        gammas, betas = self.film_generator(task_embedding)

        # extract features for query images
        z_query = self._extract_film_features(query_images, gammas, betas)

        # compute similarity scores
        scores = -self.scale * torch.cdist(z_query, z_proto)
        return F.softmax(scores, dim=1)

    def _extract_film_features(self, 
        x: torch.Tensor, 
        gammas: list[torch.Tensor] = None, 
        betas: list[torch.Tensor] = None
    ) -> torch.Tensor:

        device = x.device
        idx = 0

        for name, module in self.backbone.named_children():
            if name in self.film_layers_map:
                gamma = gammas[idx] if gammas else torch.ones(1, self.hdim[idx], device=device)
                beta = betas[idx] if betas else torch.zeros(1, self.hdim[idx], device=device)

                x = self.film_layers_map[name](x, gamma, beta)
                idx += 1
            else:
                x = module(x)
        return x
