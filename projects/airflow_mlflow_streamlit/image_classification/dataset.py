from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import transforms as T

def create_dataset(train_path,val_path,train_batch=32,val_batch=32):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    data_transforms = {
        'train':
        T.Compose([
            T.Resize((224,224)),
            T.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ]),
        'validation':
        T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            normalize
        ]),
    }

    image_datasets = {
        'train': 
        datasets.ImageFolder(train_path, data_transforms['train']),
        'validation': 
        datasets.ImageFolder(val_path, data_transforms['validation'])
    }

    dataloaders = {
        'train':
        DataLoader(image_datasets['train'],
                                    batch_size=train_batch,
                                    shuffle=True, num_workers=4),
        'validation':
        DataLoader(image_datasets['validation'],
                                    batch_size=val_batch,
                                    shuffle=False, num_workers=4)
    }

    return image_datasets,dataloaders