import cv2


# 用拉普拉斯算子的方差来计算指定路径图像的模糊程度
def getImageBlurriness(img_path: str) -> float:
    # imread函数默认读取图像的格式是BGR
    img = cv2.imread(img_path)
    if img is None:
        raise Exception("Failed to load image {}".format(filename))
    
    # BGR转灰度图
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯算子的方差（模糊程度）
    blurriness = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return blurriness
