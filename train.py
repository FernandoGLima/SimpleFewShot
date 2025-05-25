import argparse
import timm
import torch

from simplefsl.data.manager import FewShotManager
from simplefsl.trainer import Trainer
from simplefsl.utils import seed_everything, get_model_loss

def import_model(model: str):
    global Method  
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
    elif model=='DSN':
        from simplefsl.models.DSN import DSN as Method
    elif model=='MetaQDA':
        from simplefsl.models.MetaQDA import MetaQDA as Method
    elif model=='NegativeMargin':
        from simplefsl.models.NegativeMargin import NegativeMargin as Method
    else:
        raise ValueError(f"Unsupported model type: {model}")


def main(model: str, ways: int, shots: int, gpu: int, lr: float, l2_weight: float):
    import_model(model)

    seed = 42
    backbone_name = 'resnet50.a3_in1k'
    episodes = 500
    epochs = 40
    validate_every = 2

    # setup
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # model
    backbone = timm.create_model(backbone_name, pretrained=True)
    backbone.reset_classifier(num_classes=0)
    model = Method(backbone).to(device)

    label_path = '/home/victornasc/Metodos-FSL/BRSET/data/clean1.csv'
    train_classes = [
        'diabetic_retinopathy', 'scar', 'amd', 'hypertensive_retinopathy',
        'drusens', 'myopic_fundus', 'increased_cup_disc', 'other'
    ]
    test_classes = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']

    manager = FewShotManager(label_path, 
                             train_classes, 
                             test_classes, 
                             ways, 
                             shots, 
                             backbone_name, 
                             augment=None,
                             seed=seed)

    # training
    criterion = get_model_loss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, criterion, optimizer, l2_weight=l2_weight)
    # trainer.load_checkpoint('model.pth')

    print(f'training {model.__class__.__name__} with {ways}-way-{shots}-shot on {backbone_name}')
    trainer.train(manager, epochs, episodes, validate_every)

    # trainer.save_checkpoint('model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Few-Shot Learning model.")
    parser.add_argument('--ways', type=int, default=2, help='Number of classes per episode (N-way)')
    parser.add_argument('--shots', type=int, default=5, help='Number of examples per class (K-shot)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.001)')
    parser.add_argument('--l2_weight', type=float, default=0.0, help='L2 regularization term (default: 0.0001)')
    
    args = parser.parse_args()
    main(args.model, args.ways, args.shots, args.gpu, args.lr, args.l2_weight)
