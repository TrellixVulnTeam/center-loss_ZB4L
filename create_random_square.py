# Create random square
import cv2, os

img = cv2.imread("test-dataset/1.png")
print(f"image shape : {img.shape}")

img = cv2.rectangle(img, (40, 30), (140, 78), (0, 0, 0), -1)
cv2.imwrite("test-anomaly/anomaly_3.png", img)


"""
rename images
"""

# i = 1
# for img in os.listdir("test-dataset/"):
#     print(f"test-dataset/{img}")

#     old_name = f"test-dataset/{img}"
#     new_name = f"test-dataset/{i}.png"

#     os.rename(old_name, new_name)
#     i += 1

"""
crop images
"""

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

tfms = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize((192, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

# tfms = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
#         ),
#     ]
# )


img = Image.open("images/an1.png").convert("RGB")
# tfms = transform_for_infer((192, 256))
img = tfms(img)
print(f"Image shape : {img.shape}")

img = img.permute(1, 2, 0)
plt.imshow(np.uint8(img))
plt.show()

# img = tfms("images/an1.png")