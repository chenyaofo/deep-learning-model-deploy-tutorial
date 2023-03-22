import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

valdir = "test_images"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                # std=[1, 1, 1],
                                )

val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False,
    num_workers=4, pin_memory=True, sampler=None)        
