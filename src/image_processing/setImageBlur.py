import cv2
import numpy as np


# 高斯模糊图像
# 会在范围[min_size, max_size]取一个随机的奇数整数size, 并生成高斯核大小=(size, size)
def setImageBlur(filename: str, min_size: int = 12, max_size: int = 24):
    img = cv2.imread(filename)

    if img is None:
        raise Exception("Failed to load image {}".format(filename))

    size = np.random.randint(min_size, max_size + 1)
    # 高斯核的大小只能为奇数，不能为偶数
    if not (size & 1):
        size = size + 1 if size + 1 <= max_size else size - 1
    kernel_size = (size, size)

    img_blur = cv2.GaussianBlur(img, kernel_size, 0)
    return img_blur
