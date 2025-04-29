import timm
import torch

from simplefsl.datasets.manager import BRSETManager
from simplefsl.models.tadam import TADAM as Method
from simplefsl.trainer import Trainer
from simplefsl.utils import seed_everything, get_model_loss

TRAINING_CLASSES = ['diabetic_retinopathy',
                        'scar', 'amd', 'hypertensive_retinopathy', 'drusens', 
                        'myopic_fundus', 'increased_cup_disc', 'other']
TEST_CLASSES = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

seed = 42
seed_everything(seed)

ways = 2
shots = 5
model_type = 'resnet50.a3_in1k'

batch_size = ways*shots

backbone = timm.create_model(model_type, pretrained=True)
backbone.reset_classifier(num_classes=0) 
model = Method(backbone).to(device)


if model_type == "resnet50.a3_in1k" or model_type == "swin_s3_tiny_224.ms_in1k": 
    mean_val, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
elif model_type == "vit_small_patch32_224.augreg_in21k_ft_in1k":
    mean_val, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


manager = BRSETManager(TRAINING_CLASSES, TEST_CLASSES, shots, ways, mean_val, std, 
                       augment=None, batch_size=batch_size, seed=seed)


criterion = get_model_loss(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

episodes = 500
epochs = 20
validate_every = 2

print(f'training {model_type} on {ways} ways, {shots} shots')
trainer = Trainer(model, criterion, optimizer)

trainer.train(manager, epochs, episodes, validate_every)

