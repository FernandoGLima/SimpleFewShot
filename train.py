import argparse
import timm
import torch

from simplefsl.data.manager import FewShotManager
from simplefsl.trainer import Trainer
from simplefsl.utils import seed_everything, get_model_loss, load_model

def main(model_name: str, backbone: str, ways: int, shots: int, gpu: int, lr: float, l2_weight: float, aug: str):
    seed = 42
    episodes = 500
    epochs = 40
    validate_every = 2
    
    if backbone == "resnet":
        backbone_name = 'resnet50.a3_in1k'
    elif backbone == "swin":
        backbone_name = 'swin_s3_tiny_224.ms_in1k'


    # setup
    seed_everything(seed)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # model
    model = load_model(model_name, backbone_name).to(device)

    label_path = '/home/rodrigocm/datasets/brset/data/clean1.csv'
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
                             augment=aug,
                             seed=seed)

    # training
    criterion = get_model_loss(model)
    if backbone == "resnet":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    elif backbone == "swin":
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 0.005)
    trainer = Trainer(model, criterion, optimizer, l2_weight=l2_weight, augment_type = aug)
    # trainer.load_checkpoint('model.pth')

    train_description = f'training {model.__class__.__name__} with {ways}-way-{shots}-shot on {backbone_name}'
    if aug is not None:
        train_description += f' using {aug}'
    print(train_description)
    trainer.train(manager, epochs, episodes, validate_every)

    # trainer.save_checkpoint(f'./checkpoints/{model.__class__.__name__}{ways}w{shots}s.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Few-Shot Learning model.")
    parser.add_argument('--ways', type=int, default=2, help='Number of classes per episode (N-way)')
    parser.add_argument('--shots', type=int, default=5, help='Number of examples per class (K-shot)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--backbone', type=str, default="resnet", help='Backbone name')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate (default: 0.001)')
    parser.add_argument('--l2_weight', type=float, default=0.0, help='L2 regularization term (default: 0.0001)')
    parser.add_argument('--augment', type=str, default=None, help="Data augmentation technique")

    args = parser.parse_args()
    main(args.model, args.backbone, args.ways, args.shots, args.gpu, args.lr, args.l2_weight, args.augment)
