import argparse
import timm
import torch

from simplefsl.datasets.manager import BRSETManager
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


def main(model: str, ways: int, shots: int, gpu: int):
    import_model(model)

    seed = 42
    backbone_name = 'resnet50.a3_in1k'
    episodes = 500
    epochs = 40
    validate_every = 2

    TRAINING_CLASSES = [
        'diabetic_retinopathy', 'scar', 'amd', 'hypertensive_retinopathy',
        'drusens', 'myopic_fundus', 'increased_cup_disc', 'other'
    ]
    TEST_CLASSES = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']

    # setup
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # model
    backbone = timm.create_model(backbone_name, pretrained=True)
    backbone.reset_classifier(num_classes=0)
    model = Method(backbone).to(device)

    # data manager
    if backbone_name in ["resnet50.a3_in1k", "swin_s3_tiny_224.ms_in1k"]:
        mean_val, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif backbone_name == "vit_small_patch32_224.augreg_in21k_ft_in1k":
        mean_val, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        raise ValueError(f"Unsupported model type: {backbone_name}")

    manager = BRSETManager(
        TRAINING_CLASSES, TEST_CLASSES, shots, ways, mean_val, std,
        augment=None, batch_size=ways * shots, seed=seed
    )

    # training
    criterion = get_model_loss(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, criterion, optimizer)
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
    
    args = parser.parse_args()
    main(args.model, args.ways, args.shots, args.gpu)
