import torch
import numpy as np
import timm
import cv2
import argparse
import torch.nn as nn

from simplefsl.utils import load_model
from simplefsl.data.manager import FewShotManager

from torchvision import transforms
from torchvision.transforms import v2
from torchvision.io import read_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Model wrapper, used to input
only a single image to the 
model forward pass
'''
class CustomModelWrapper(nn.Module):
    def __init__(self, few_shot_model, support_imgs, support_labels):
        super().__init__()
        self.few_shot_model = few_shot_model
        self.support_imgs = support_imgs
        self.support_labels = support_labels

    def forward(self, x):
        return self.few_shot_model.forward(self.support_imgs, self.support_labels, x)

'''
Reshape images
'''
def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

'''
Apply data augmentation to images during
training loop
'''
def get_augmented_images_and_labels(aug, support_images, support_labels, query_images, query_labels_indices):
    all_images = torch.cat([support_images, query_images], dim=0)
    all_labels_indices = torch.cat([torch.argmax(support_labels, dim=1), query_labels_indices], dim=0)

    augmented_images, augmented_labels = aug(all_images, all_labels_indices)

    current_support_images = augmented_images[:len(support_images)]
    current_query_images = augmented_images[len(support_images):]

    current_query_labels_indices = augmented_labels[len(support_images):]

    return current_support_images, current_query_images, query_images, current_query_labels_indices

'''
Fine tune model and apply GradCAM 
for image on unseen class during
training.
'''
def apply_cam(
        model_name: str,
        model_path: str,
        ways: int,
        shots: int,
        img: str,
        target_class: str,
        classes: str,
        augment: str,
        save_path: str
    ):
    print(f"Applying GradCAM to image: {img}")

    base_path = "/home/rodrigocm/datasets/brset/data"

    orig_image = read_image(f"{base_path}/imgs/{img}.jpg") / 255 #img06628
    t = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = t(orig_image)
    image = image.reshape(1,3,224,224).to(device)

    seed = 42
    n_updates = 52
    epochs = n_updates * 5 // shots

    backbone_name = 'resnet50.a3_in1k'
    label_path = f'{base_path}/clean1.csv'

    classes = classes.split(",")

    removed_image_name = f'{base_path}/imgs/{img}.jpg'    
    manager = FewShotManager(label_path,
                             [],
                             classes,
                             ways,
                             shots,
                             backbone_name,
                             augment=None,
                             seed=seed,
                             remove_img=removed_image_name)

    if augment == "cutmix":
        aug = v2.CutMix(num_classes=2)
    elif augment == "mixup":
        aug = v2.MixUp(num_classes=2)

    hit_count = 0
    masks = 0

    with tqdm(total=100) as pbar:
        while hit_count < 100:
            #Load model
            model = load_model(model_name, backbone_name).to(device)

            if model_path is not None:
                checkpoint = torch.load(model_path, weights_only=False)['model_state_dict']
                model.load_state_dict(checkpoint, strict=False)

            model.train()

            #Train model
            train_loader, query_loader, class_names = manager.get_fewshot_task(train=False)

            support_images, support_labels = next(iter(train_loader))
            query_images, query_labels = next(iter(query_loader))

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            query_labels_indices = query_labels.argmax(1)

            if hit_count == 0:
                print(f"Current task classes: {class_names}")

            criterion = torch.nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=0.0001)

            for epoch in range(epochs):
                if augment == "cutmix" or augment == "mixup":
                    support_images, support_labels, query_images, query_labels_indices = \
                    get_augmented_images_and_labels(aug,
                                                    support_images,
                                                    support_labels,
                                                    query_images,
                                                    query_labels_indices)

                opt.zero_grad()

                scores = model(support_images, support_labels, query_images)

                loss = criterion(scores, query_labels_indices)
                loss.backward()

                opt.step()

            model.eval()
            wrapped_model = CustomModelWrapper(model, support_images, support_labels)

            pred = int(torch.argmax(wrapped_model(image)))

            if class_names[pred] == target_class:
                if backbone_name == 'resnet50.a3_in1k':
                    target_layers = [model.backbone.layer4[-1]]
                    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
                elif backbone_name == 'swin_s3_tiny_224.ms_in1k':
                    target_layers = [model.layers[-1].blocks[-1].norm2]
                    cam = GradCAM(model=wrapped_model, target_layers=target_layers, reshape_transform=reshape_transform)

                mask = cam(input_tensor=image, targets=None)

                masks = masks + mask
                hit_count += 1
                pbar.update(1)

    masks = masks / 100
    final_mask = masks[0, :]

    rgb_img = cv2.imread(f"{base_path}/imgs/{img}.jpg", 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    cam_image = show_cam_on_image(rgb_img, final_mask)
    cv2.imwrite(save_path, cam_image)

    print(f"Applied GradCAM to {img} and saved it to {save_path}.")
    return

# python explain.py --model "metaopt_net" --model-path "model.pth" --ways 2 --shots 5 --image "img02000" --target-class "hemorrhage" --classes "hemorrhage,healthy" --save-path "heatmap.jpg"
def main(args):
    apply_cam(args.model,
                args.model_path, 
                args.ways, 
                args.shots, 
                args.image, 
                args.target_class, 
                args.classes, 
                args.augment, 
                args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GradCAM for FSL model.")

    ###### EXPERIMENT SETTINGS ######
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--ways", type=int, default=2)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--augment", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="gradcam.jpg")

    args = parser.parse_args()

    main(args)