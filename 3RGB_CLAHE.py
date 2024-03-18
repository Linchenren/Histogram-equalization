# 首先将彩色图像转换为LAB颜色空间。
# 然后，我们对L通道（亮度）应用CLAHE算法，保持A和B通道不变。最后，我们将CLAHE增强的L通道与原始的A和B通道合并，并将图像转换回BGR颜色空间。这样就可以保留彩色图像的色彩信息，并在受限制的对比度条件下进行直方图均衡化
# 参考资料： https://cloud.tencent.com/developer/article/1667213?areaId=106001
import cv2
import time

def clahe_color01(image, clip_limit=4, grid_size=(8, 8)):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE to the L channel
    clahe_l_channel = clahe.apply(l_channel)

    # Merge the CLAHE enhanced L channel with the original A and B channels
    clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

    # Convert the LAB image back to BGR color space
    clahe_color_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

    return clahe_color_image

def clahe_color02(image, clip_limit=4, grid_size=(8, 8)):
    # 避免使用cv2.split(lab_image)，速度会加快
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

    # Apply CLAHE to the L channel (first channel)
    clahe_l_channel = clahe.apply(lab_image[:,:,0])

    # Merge the CLAHE enhanced L channel with the original A and B channels
    clahe_lab_image = cv2.merge((clahe_l_channel, lab_image[:,:,1], lab_image[:,:,2]))

    # Convert the LAB image back to BGR color space
    clahe_color_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

    return clahe_color_image

# Read the image
image = cv2.imread('./data/cat.png')


# Call the CLAHE function for color images
start_time = time.time()
clahe_image = clahe_color02(image)
end_time = time.time()

print("Execution time:", end_time - start_time)

# Save the CLAHE enhanced image
cv2.imwrite('./data/3rgb_clahe4.jpg', clahe_image)