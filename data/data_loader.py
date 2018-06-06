'''
data_loader.py

Modified from https://github.com/carpedm20/BEGAN-pytorch/blob/master/data_loader.py
https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/model/data_loader.py
'''


import os
import torch
from torchvision import transforms

from data.folder import ImageFolder

# Train set transformation
train_transformer = transforms.Compose([
    transforms.CenterCrop(160),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor()
])

# Val and test set transformations
eval_transformer = transforms.Compose([
    transforms.CenterCrop(160),
    transforms.Resize(128),
    transforms.ToTensor()
])

def fetch_dataloader(root, split, params, shuffle=True):
    dataset_name = os.path.basename(root)
    image_root = os.path.join(root, 'splits', split)

    #if dataset_name == 'CelebA':
    dataset = ImageFolder(root=image_root, transform=transforms.Compose([
                                transforms.CenterCrop((params.crop_size, params.crop_size)), # Center crop to 160x160
                                # transforms.RandomHorizontalFlip(),
                                transforms.Resize((params.img_size, params.img_size)), # Resize to 128x128
                                transforms.ToTensor(),
                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    #else:
    #    dataset = ImageFolder(root=image_root, transform=transforms.Compose([
    #        transforms.Resize(128),
    #        transforms.ToTensor(),
    #        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #    ]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                                                shuffle=shuffle, num_workers=params.num_workers)
    # data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader
