import random
import numpy as np
import torch
import torch.nn as nn
import timm

def load_model(model: str, backbone: str) -> nn.Module:
    backbone_instance = timm.create_model(backbone, pretrained=True)
    backbone_instance.reset_classifier(num_classes=0)

    Method = None

    if model == 'dn4':
        from simplefsl.models.dn4 import DN4 as Method
    elif model == 'feat':
        from simplefsl.models.feat import FEAT as Method
    elif model == 'matching_net':
        from simplefsl.models.matching_net import MatchingNetworks as Method
    elif model == 'metaopt_net':
        from simplefsl.models.metaopt_net import MetaOptNet as Method
    elif model == 'msenet':
        from simplefsl.models.msenet import MSENet as Method
    elif model == 'proto_net':
        from simplefsl.models.proto_net import PrototypicalNetworks as Method
    elif model == 'relation_net':
        from simplefsl.models.relation_net import RelationNetworks as Method
    elif model == 'tadam':
        from simplefsl.models.tadam import TADAM as Method
    elif model == 'tapnet':
        from simplefsl.models.tapnet import TapNet as Method
    elif model == 'R2D2':
        from simplefsl.models.R2D2 import RRNet as Method
    elif model =='DSN':
        from simplefsl.models.DSN import DSN as Method
    elif model =='MetaQDA':
        from simplefsl.models.MetaQDA import MetaQDA as Method
    elif model =='NegativeMargin':
        from simplefsl.models.NegativeMargin import NegativeMargin as Method
    else:
        raise ValueError(f"Unsupported model type: {model}")

    model_instance = Method(backbone_instance)

    return model_instance


def get_model_loss(model: nn.Module) -> nn.Module:
    loss_map = {
        "DN4": nn.CrossEntropyLoss(),
        "FEAT": nn.CrossEntropyLoss(),
        "MatchingNetworks": nn.NLLLoss(),
        "MetaOptNet": nn.CrossEntropyLoss(),
        "MSENet": nn.CrossEntropyLoss(),
        "PrototypicalNetworks": nn.CrossEntropyLoss(),
        "RelationNetworks": nn.MSELoss(),
        "TADAM": nn.CrossEntropyLoss(),
        "TapNet": nn.CrossEntropyLoss(),
        "R2D2": nn.CrossEntropyLoss(),
        "DSN": nn.CrossEntropyLoss(),
        "MetaQDA": nn.CrossEntropyLoss(),
        "NegativeMargin": nn.CrossEntropyLoss(),
    }
    
    model_name = model.__class__.__name__
    try:
        return loss_map[model_name]
    except:
        print(f"Model {model_name} not found in loss map. Using default CrossEntropyLoss")
        return nn.CrossEntropyLoss()
       
    
def seed_everything(seed : int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    