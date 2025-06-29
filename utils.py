import cv2
def binarize_image(image, threshold=128):
    """
    将图像进行01二值化处理。

    参数:
    - image_path: 图像文件的路径。
    - threshold: 二值化的阈值，默认为128。

    返回:
    - 二值化后的图像（PIL Image对象）。
    """
    _, binarized_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return binarized_image