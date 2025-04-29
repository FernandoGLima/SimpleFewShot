import random
import numpy as np
import torch
import torch.nn as nn


def get_model_loss(model: nn.Module) -> nn.Module:
    loss_map = {
            "DN4": nn.CrossEntropyLoss(),
            "FEAT": nn.CrossEntropyLoss(),
            "MatchingNetworks": nn.CrossEntropyLoss(),
            "MetaOptNet": nn.CrossEntropyLoss(),
            "MSENet": nn.MSELoss(),
            "PrototypicalNetworks": nn.CrossEntropyLoss(),
            "RelationNetworks": nn.MSELoss(),
            "TADAM": nn.CrossEntropyLoss(),
            "TapNet": nn.CrossEntropyLoss(),
        }
    
    model_name = model.__class__.__name__
    try:
        return loss_map[model_name]
    except KeyError:
        raise ValueError(
            f"Unknown model_name {model_name!r}\nAvailable options: {list(loss_map.keys())}"
        )
    
def seed_everything(seed : int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    