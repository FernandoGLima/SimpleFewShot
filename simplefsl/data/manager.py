import torch
import random 
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FewShotDataset(Dataset):
    '''Generic dataset for few-shot learning tasks'''
    def __init__(self, img_ids, labels, transforms=None):
        self.img_ids = img_ids
        self.transforms = transforms
        self.labels = labels
    
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = read_image(self.img_ids[idx]) / 255
        label = self.labels[idx].float()

        if self.transforms:
            image = self.transforms(image)

        return image, label


class FewShotManager():
    '''Data manager for few-shot learning tasks'''
    def __init__(
        self, 
        labels_path: str,
        train_classes: list,
        test_classes: list,
        ways: int, 
        shots: int, 
        backbone_name: str = None,
        augment: str = None,
        seed: int = None,
        remove_img: str = None
    ):
        self.train_classes = train_classes
        self.test_classes = test_classes
        self.shots = shots
        self.ways = ways
        self.augment = augment
        self.seed = seed
        
        self.label_map = pd.read_csv(labels_path)

        self.mean, self.std = self._get_mean_std(backbone_name)
        self.eval_task_count = 0

        if self.seed is not None:
            torch.manual_seed(seed)
        
        if remove_img is not None:
            self.label_map = self.label_map[self.label_map.image_id != remove_img]
            print("removed img", remove_img)
    
    @staticmethod
    def _get_mean_std(backbone_name: str):
        name = backbone_name.lower()
        if 'resnet' in name or 'swin' in name:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif 'vit' in name:
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
            classes = random.sample(self.train_classes, self.ways)
        else:
            classes = random.sample(self.test_classes, self.ways)

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
            dataset = FewShotDataset(train_ids, train_labels, transforms = transforms.Normalize(self.mean, self.std))
        elif self.augment == "basic":
           t = transforms.Compose([
               transforms.RandomHorizontalFlip(p=0.5),
               transforms.ColorJitter(contrast=(0.5, 1.5), saturation=(0.5,1.5), hue=(-0.1,0.1)),
               transforms.Normalize(self.mean, self.std)
           ])
           dataset = FewShotDataset(train_ids, train_labels, transforms = t)
        else: 
            raise NotImplementedError(f"Augmentation {self.augment} not implemented")
        
        train_loader = DataLoader(dataset, batch_size=self.ways * self.shots, shuffle=True)
        
        dataset = FewShotDataset(query_ids, query_labels, transforms = transforms.Normalize(self.mean, self.std))
        query_loader = DataLoader(dataset, batch_size=self.ways, shuffle=True)

        self.eval_task_count += 1

        return train_loader, query_loader, classes
