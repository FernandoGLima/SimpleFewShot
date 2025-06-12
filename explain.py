import os
import torch
import numpy as np
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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

'''
Model wrapper, used to input
only a single image to the 
model forward pass
'''
class CustomModelWrapper(nn.Module):
    def __init__(self, model, support_imgs, support_labels):
        super().__init__()
        self.model = model
        self.support_imgs = support_imgs
        self.support_labels = support_labels

    def forward(self, x):
        return self.model.forward(self.support_imgs, self.support_labels, x)

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
def main(args):
    print(f"Applying GradCAM to image: {args.image}")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    data_path = "/home/rodrigocm/datasets/brset/data"
    # args.weights = os.path.join("/home/rodrigocm/research/SimpleFewShot/checkpoints", args.weights)
    args.weights = os.path.join("/home/carlosmarquesr/projects/SimpleFewShot/checkpoints/best", args.weights)
    save_path = "/home/rodrigocm/research/SimpleFewShot/gradcam"

    orig_image = read_image(f"{data_path}/imgs/{args.image}.jpg").float() / 255.0 #img06628
    t = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = t(orig_image)
    image = image.reshape(1,3,224,224).to(device)

    seed = 42
    n_updates = 52
    epochs = n_updates * 5 // args.shots

    # backbone_name = 'resnet50.a3_in1k'
    backbone_name = 'swin_s3_tiny_224.ms_in1k'
    label_path = f'{data_path}/clean1.csv'

    args.classes = args.classes.split(",")

    removed_image_name = f'{data_path}/imgs/{args.image}.jpg'    
    manager = FewShotManager(label_path,
                             [],
                             args.classes,
                             args.ways,
                             args.shots,
                             backbone_name,
                             args.augment,
                             seed=seed,
                             remove_img=removed_image_name)

    if args.augment == "cutmix":
        aug = v2.CutMix(num_classes=2)
    elif args.augment == "mixup":
        aug = v2.MixUp(num_classes=2)

    hit_count = 0
    masks = 0

    with tqdm(total=100) as pbar:
        while hit_count < 100:
            #Load model
            model = load_model(args.model, backbone_name).to(device)

            if args.weights is not None:
                checkpoint = torch.load(args.weights, weights_only=False)['model_state_dict']
                model.load_state_dict(checkpoint, strict=True)

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

            criterion = torch.nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=0.0001)

            for _ in range(epochs):
                if args.augment == "cutmix" or args.augment == "mixup":
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
            target = [ClassifierOutputTarget(pred)]

            if class_names[pred] == args.target_class:
                if backbone_name == 'resnet50.a3_in1k':
                    target_layer = wrapped_model.model.backbone.layer4[-1]
                    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
                elif backbone_name == 'swin_s3_tiny_224.ms_in1k':
                    target_layer = wrapped_model.model.backbone.layers[-1].blocks[-1].norm2
                    cam = GradCAM(model=wrapped_model, target_layers=[target_layer], reshape_transform=reshape_transform)
                
                mask = cam(input_tensor=image, targets=target)

                masks = masks + mask
                hit_count += 1
                pbar.update(1)

    masks = masks / hit_count
    final_mask = masks[0, :]

    rgb_img = cv2.imread(f"{data_path}/imgs/{args.image}.jpg", 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    cam_image = show_cam_on_image(rgb_img, final_mask)

    cv2.imwrite(os.path.join(save_path, f"{args.image}_{backbone_name}_{args.model}_{args.ways}w{args.shots}s.jpg"), cam_image)

    print(f"Applied GradCAM to {args.image} and saved it to {save_path}.")
    return

# python explain.py --model "metaopt_net" --weights "model.pth" --ways 2 --shots 5 --gpu 0 --image "img06628" --target-class "hemorrhage" --classes "hemorrhage,healthy"
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GradCAM for FSL model.")

    ###### EXPERIMENT SETTINGS ######
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--ways", type=int, default=2)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target-class", type=str, default=None)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--augment", type=str, default=None)

    args = parser.parse_args()

    main(args)