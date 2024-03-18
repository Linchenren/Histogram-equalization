# 彩色图片的HE
# 先将彩色图像转换为YUV颜色空间，然后对Y通道（亮度）进行直方图均衡化，最后将图像转换回BGR颜色空间。这样就可以保留彩色图像的色彩信息，并对亮度进行均衡化
import cv2
import numpy as np

def histogram_equalization_color(image):
    # Convert the image to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Equalize the histogram of the Y channel (luminance)
    yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])

    # Convert the image back to BGR color space
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return equalized_image

# Read the image
image = cv2.imread('./data/cat.png')

# Call the histogram equalization function for color images
equalized_image = histogram_equalization_color(image)

# Save the equalized image
cv2.imwrite('./data/2rgb_he.jpg', equalized_image)