# 对灰度图进行直方图均衡化
import cv2
import numpy as np

def histogram_equalization(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Compute the histogram
    hist, bins = np.histogram(gray.flatten(), 256, [0,256])

    # Compute the cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Perform histogram equalization
    equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)

    # Reshape the equalized image to original shape
    equalized_image = equalized.reshape(gray.shape)

    return equalized_image.astype(np.uint8)

# 读取图像
image = cv2.imread('./data/cat.png')

# 调用直方图均衡化函数
equalized_image = histogram_equalization(image)

cv2.imwrite('./data/1gray_he.jpg', equalized_image)


