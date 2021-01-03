from torchvision import transforms


def transform_for_training(image_shape):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )


def transform_for_infer(image_shape):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(image_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )


# def transform_for_training(image_shape):
#     return transforms.Compose([
#         transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
#         # transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#         ], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465],
#                              [0.2023, 0.1994, 0.2010])
#     ])


# def transform_for_infer(image_shape):
#     return transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
#                              std=(0.2023, 0.1994, 0.2010))
#     ])