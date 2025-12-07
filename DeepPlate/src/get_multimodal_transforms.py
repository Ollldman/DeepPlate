import albumentations as A
import timm

from DeepPlate import Config

def get_multimodal_transforms(config: Config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)

    if ds_type == "train":
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.RandomCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(scale=(0.95, 1.05),
                        rotate=(-10, 10),
                        translate_percent=(-0.05, 0.05),
                        shear=(-5, 5),
                        fill=0,
                        p=0.5),
                A.CoarseDropout(num_holes_range=(2, 3),
                                hole_height_range=(int(0.07 * cfg.input_size[1]),
                                                int(0.12 * cfg.input_size[1])),
                                hole_width_range=(int(0.07 * cfg.input_size[2]),
                                                int(0.11 * cfg.input_size[2])),
                                fill=0,
                                p=0.5),
                A.ColorJitter(
                    brightness=0.1, contrast=0.15, saturation=0.5, hue=0.07, p=0.7),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ],
            seed=config.SEED,
        )
    else:
        transforms = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(
                    height=cfg.input_size[1], width=cfg.input_size[2], p=1.0),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.ToTensorV2(p=1.0)
            ]
        )

    return transforms 