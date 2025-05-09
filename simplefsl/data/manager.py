import os
import torch
import random 
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FewShotBRSET(Dataset):
    """
    Few-Shot Dataset for BRSET dataset
    BRSET: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos

    !! Use this class to create your own dataset for few-shot learning tasks
    """
    train_classes = [
        'diabetic_retinopathy', 'scar', 'amd', 'hypertensive_retinopathy',
        'drusens', 'myopic_fundus', 'increased_cup_disc', 'other'
    ]
    test_classes = ['hemorrhage', 'vascular_occlusion', 'nevus', 'healthy']

    def __init__(self, data_path, img_ids, labels, transforms=None):
        self.img_ids = img_ids
        self.transforms = transforms
        self.labels = labels
        self.img_dir = data_path 
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, f"{self.img_ids[idx]}.jpg")
        image = read_image(img_name) / 255
        label = self.labels[idx].float()

        if self.transforms:
            image = self.transforms(image)

        return image, label


class FewShotManager():
    def __init__(
        self, 
        dataset: Dataset,
        data_path: str,
        labels_path: str,
        ways: int, 
        shots: int, 
        backbone_name: str = None,
        augment: str = None,
        seed: int = None
    ):
        self.data_path = data_path
        self.dataset = dataset
        self.shots = shots
        self.ways = ways
        self.augment = augment
        self.seed = seed
        
        self.label_map = pd.read_csv(labels_path)

        self.mean, self.std = self._get_mean_std(backbone_name)
        self.eval_task_count = 0

        if self.seed is not None:
            torch.manual_seed(seed)
    
    @staticmethod
    def _get_mean_std(backbone_name: str):
        if 'resnet' in backbone_name or 'swin' in backbone_name:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif 'vit' in backbone_name:
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            print(f"Backbone {backbone_name} not supported. Using mean = 0 and std = 1")
            mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

        return mean, std

    def get_fewshot_task(self, train: bool):
        if self.seed is not None:
            seed = self.seed + self.eval_task_count
            random.seed(seed)
        else:
            seed = None

        if train:
            classes = random.sample(self.dataset.train_classes, self.ways)
        else:
            classes = random.sample(self.dataset.test_classes, self.ways)

        train_ids = []
        train_labels = torch.empty((0, self.ways), dtype=torch.float32)

        query_ids = []
        query_labels = []

        for clss in classes:
            objs = self.label_map[self.label_map[clss] == 1].sample(n=self.shots+1, random_state=seed)

            train_ids += list(objs["image_id"])[:-1]
            train_labels = torch.cat((train_labels, torch.tensor(objs[classes].to_numpy()[:-1])), axis = 0)
            
            query_ids.append(list(objs["image_id"])[-1])
            query_labels.append(torch.tensor(objs[classes].to_numpy()[-1]))
        
        query_labels = torch.stack(query_labels)
            
        if self.augment == None or self.augment == "mixup" or self.augment == "cutmix":
            dataset = self.dataset(self.data_path, train_ids, train_labels, transforms = transforms.Normalize(self.mean, self.std))
        elif self.augment == "basic":
           t = transforms.Compose([
               transforms.RandomHorizontalFlip(p=0.5),
               transforms.ColorJitter(contrast=(0.5, 1.5), saturation=(0.5,1.5), hue=(-0.1,0.1)),
               transforms.Normalize(self.mean, self.std)
           ])
           dataset = self.dataset(self.data_path, train_ids, train_labels, transforms = t)
        else: 
            raise NotImplementedError(f"Augmentation {self.augment} not implemented")
        
        train_loader = DataLoader(dataset, batch_size=self.ways * self.shots, shuffle=True)
        
        dataset = self.dataset(self.data_path, query_ids, query_labels, transforms = transforms.Normalize(self.mean, self.std))
        query_loader = DataLoader(dataset, batch_size=self.ways, shuffle=True)

        self.eval_task_count += 1

        return train_loader, query_loader, classes
