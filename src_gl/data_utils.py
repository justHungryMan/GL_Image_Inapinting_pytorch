from torchvision import transforms

def preprocess(config):
    if config['type'] == 'pad':
        return transforms.Pad(**config['params'])
    elif config['type'] == 'resize':
        return transforms.Resize(**config['params'])
    elif config['type'] == 'randomcrop':
        return transforms.RandomCrop(**config['params'])
    elif config['type'] == 'randomresizecrop':
        return transforms.RandomResizedCrop(**config['params'])
    elif config['type'] == 'horizontal':
        return transforms.RandomHorizontalFlip()
    elif config['type'] == 'tensor':
        return transforms.ToTensor()
    elif config['type'] == 'normalize':
        return transforms.Normalize(**config['params'])
