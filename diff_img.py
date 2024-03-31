import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import *
import scipy

print(cv2.__path__)
def time_domain_convolution(image, kernel):
    start_time = time.time()
    # Perform time domain convolution
    result = TimeDomainConvolution(image, kernel)
    # Your implementation here
    elapsed_time = time.time() - start_time
    return elapsed_time, result


def filter2d_convolution(image, kernel):
    start_time = time.time()
    # Perform time domain convolution
    result = filter2d(image, kernel)
    # Your implementation here
    elapsed_time = time.time() - start_time
    return elapsed_time, result


def frequency_domain_convolution(image, kernel):
    start_time = time.time()
    # result = image
    result = fft_convolve2d(image, kernel, is_show=False)
    # Perform frequency domain convolution
    # Your implementation here
    elapsed_time = time.time() - start_time
    return elapsed_time, result


def scipy_fft_convolution(image, kernel):
    dim = len(image.shape)
    if dim == 3:
        kernel = np.expand_dims(kernel, 2).repeat(1, axis=2)
    start_time = time.time()
    # Perform time domain convolution
    result = scipy.signal.convolve(image, kernel, mode="same", method="fft").astype(
        np.uint8
    )
    # Your implementation here
    elapsed_time = time.time() - start_time
    return elapsed_time, result


def scipy_tb_convolution(image, kernel):
    dim = len(image.shape)
    if dim == 3:
        kernel = np.expand_dims(kernel, 2).repeat(3, axis=2)
    start_time = time.time()
    # Perform time domain convolution
    result = scipy.signal.convolve(image, kernel, mode="same", method="direct").astype(
        np.uint8
    )
    # Your implementation here
    elapsed_time = time.time() - start_time
    return elapsed_time, result
# Load the image
folder_path = "test_img"
# 遍历文件夹 读取n张图片 添加到list中
N =11
image_list = []
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_list.append(cv2.imread(os.path.join(folder_path, file_name)))
        if len(image_list) == N:
            break
# Define the kernel sizes to test
kernel_size = 359
image_times = []
results = []
# Perform convolution for each image
kernel = cv2.getGaussianKernel(kernel_size, 0)
kernel = np.dot(kernel, kernel.T)
for image in image_list:
    image_time, result = scipy_fft_convolution(image, kernel)
    print("Kernel size: {}, Time domain: {:.4f} s".format(kernel_size, image_time))
    image_times.append(image_time)
    results.append(result)

# for kernel_size in kernel_sizes:
#     # 根据kernel_size生成高斯核
#     # kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
#     kernel = cv2.getGaussianKernel(kernel_size, 0)
#     kernel = np.dot(kernel, kernel.T)
#     frequency_domain_time, frequency_domain_result = scipy_fft_convolution(re_image, kernel)
#     print("Kernel size: {}, Frequency domain: {:.4f} s".format(kernel_size, frequency_domain_time))
#     frequency_domain_times.append(frequency_domain_time)
# Plot the results
#根据N的大小生成对应的子图
#然后 画出 results中的图片 并显示对应的时间
# 一排最多显示4张图片
# fig, axs = plt.subplots(1, N, figsize=(20, 20))
# for i in range(N):
#     axs[i].imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
#     axs[i].set_title(str(results[i].shape)+"Time: {:.4f} s".format(image_times[i]))
# plt.show()
# plt.close()
# 重新生成图片
plt.figure()
# fig, axs = plt.subplots(3, 4, figsize=(20, 20))
for i in range(N):
    axs=plt.subplot(3,4,i+1) 
    plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
    axs.set_title(str(results[i].shape)+"Time: {:.4f} s".format(image_times[i]))
plt.show()