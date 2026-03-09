import torch
import torchvision.transforms as transforms


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def get_normalize_transform(name: str):
    if name == "imagenet":
        mean = [0.193, 0.193, 0.193]
        std = [0.224, 0.224, 0.224]
    elif name == "stat":
        mean = [0.18136882, 0.18137674, 0.18136712]
        std = [0.1563932, 0.1563886, 0.15638869]
    elif name == "original_imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported normalizer: {name}")
    return transforms.Normalize(mean=mean, std=std)


def get_interpolation(name: str):
    if name == "nearest":
        return transforms.InterpolationMode.NEAREST
    if name == "bilinear":
        return transforms.InterpolationMode.BILINEAR
    raise ValueError(f"Unsupported interpolation: {name}")


def build_transform(args, is_train: bool):
    transform_list = [
        transforms.Resize(tuple(args.input_size), interpolation=get_interpolation(args.interpolation))
    ]

    if is_train and getattr(args, "colorjitter", 0) > 0:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
            )
        )

    if is_train and getattr(args, "randomaffine", 0) > 0:
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))

    transform_list.append(transforms.ToTensor())

    if is_train and getattr(args, "gaussian", 0.0) > 0:
        transform_list.append(AddGaussianNoise(0.0, args.gaussian))

    transform_list.append(get_normalize_transform(args.normalizer))
    return transforms.Compose(transform_list)