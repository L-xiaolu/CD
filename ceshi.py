import numpy as np
from PIL import Image
from paddle.vision.transforms import Normalize,ToTensor

normalize = Normalize(mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        data_format='HWC')

fake_img = Image.fromarray((np.random.rand(300, 320, 3) * 255.).astype(np.uint8))
transform = ToTensor()
fake_img = normalize(fake_img)
fake_img=transform(fake_img)
print(fake_img)
# print(fake_img.shape)
# print(fake_img.max, fake_img.max)
