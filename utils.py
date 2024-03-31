import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import scipy
def conv2d_numpy(input_data, kernel, stride=1, padding=0):
    # 获取输入数据的尺寸和通道数
    # 如果输入数据是三通道的图像，那么获取通道数
    dim= len(input_data.shape)
    kernel = np.expand_dims(kernel, 2).repeat(dim, axis=2)
    if dim == 2:
        input_data = np.expand_dims(input_data, 2).repeat(1, axis=2)
    input_height, input_width, input_channels = input_data.shape
    # 获取卷积核的尺寸和通道数
    kernel_height, kernel_width,kernel_channels = kernel.shape
    # 计算输出图像的尺寸
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1
    
    # 初始化输出图像
    output_data = np.zeros((output_height, output_width,input_channels))
    kernel_side = kernel_height//2
    # 填充输入数据（根据填充数量添加额外的行和列）
    if padding > 0:
        input_data = np.pad(input_data, ((padding, padding), (padding, padding),(0,0)), mode='constant',constant_values = (0,0))
    
    # 执行多通道卷积操作
    for i in range(0, output_height, stride):
        for j in range(0, output_width, stride):
            for k in range(input_channels):
                output_data[i // stride, j // stride,k] = np.sum(input_data[i:i+kernel_width, j:j+kernel_width, k] * kernel[:, :, k])
    
    return output_data
def fft_convolve2d(a, b, is_show=False,is_padding=True):
    # 将a和b的大小扩展到相同的大小
    # a 通常是3通道的图像，b是卷积核
    # b需要升

    # # 进行FFT
    original_shape = a.shape
    kernel_size = b.shape[0]
    #避免循环卷积需要对A进行填充
    padding = kernel_size // 2
    if is_padding:
        a = np.pad(a,((padding,padding),(padding,padding),(0,0)),mode='constant',constant_values = (0,0))
    shape = a.shape
    A = np.fft.fft2(a, axes=(0, 1))
    # A = np.fft.fftshift(A)
    B = np.fft.fft2(b, s=(shape[0], shape[1]))
    # 如果A是三通道的图像，那么B也要是三通道的
    # 进行点积
    # AB=A.transpose(2,0,1)*B
    # AB = AB.transpose(0,1,2)
    # AB = np.fft.ifft2(AB.transpose(1,2,0)).real
    if len(A.shape) == 3:
        B = np.expand_dims(B, 2).repeat(3, axis=2)
    C = A* B
    # C = np.fft.fftshift(B) * np.fft.fftshift(A)
    # C = np.fft.ifftshift(C)
    # AB = np.fft.ifft2(C, axes=(0, 1)).real
    AB = np.fft.ifft2(C, axes=(0, 1)).real
    AB = AB[kernel_size-1:AB.shape[0], kernel_size-1:AB.shape[1]]
    # AB_shift = np.fft.ifft2(C_shift, axes=(0, 1))
    # 三通道位置变了 012 要切换成 1206
    # 是沿第三个维度
    # AB = np.clip(AB,0,255) #会产生一些过大值需要截断
    # # 剪裁卷积后的结果以使其大小与输入大小相匹配
    # AB = AB[:a.shape[0], :a.shape[1]]
    return AB.astype(np.uint8)
def scipy_convolve2d(a, b):
    dim = len(a.shape)
    if dim == 3:
        b = np.expand_dims(b, 2).repeat(1, axis=2)
    return scipy.signal.convolve(a, b, mode='same',method='fft').astype(np.uint8)

def TimeDomainConvolution(a, b):
    # 时域卷积
    # a 通常是3通道的图像，b是卷积核
    return cv2.filter2D(a, -1, b, borderType=cv2.BORDER_CONSTANT)
    # return conv2d_numpy(a, b,padding=b.shape[0]//2).astype(np.uint8)

def filter2d(a, b):
    return cv2.filter2D(a, -1, b, borderType=cv2.BORDER_CONSTANT)

def cv2imgshow(name, img):
    cv2.namedWindow(name, cv2.WINDOW_FREERATIO)
    # cv2.resizeWindow(name, 400, 300)
    cv2.imshow(name, img)


if __name__ == "__main__":
    img_path = "test_img/IMG_20220704_075206.jpg"
    # test_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(img_path)
    test_img = cv2.resize(test_img, (5,5))
    test_img_spectrum = np.fft.fftshift(np.fft.fft2(test_img))
    kernel_size = 3
    kernel = cv2.getGaussianKernel(kernel_size, 0)
    kernel = np.dot(kernel, kernel.T)
    # kernel = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    kernel_spectrum = np.fft.fftshift(np.fft.fft2(kernel, s=(300, 300)))
    # K = np.fft.fft2(kernel, s=(20,20))
    # 执行时域卷积
    start = time.time()
    td_result =TimeDomainConvolution(test_img, kernel)
    end = time.time()
    TimeDomainTimer = end - start
    #观察频谱
    td_result_spectrum = np.fft.fftshift(np.fft.fft2(td_result))
    start = time.time()
    fft_result = scipy_convolve2d(test_img, kernel)
    end = time.time() 
    FFTDomainTimer = end - start  
    fft_result_spectrum = np.fft.fftshift(np.fft.fft2(fft_result))
    #   还是用plt库吧，cv2不好多张图一起显示
    plt.figure(1)
    plt.subplot(121)
    # 如果是三通道
    test_img = cv2.cvtColor(td_result, cv2.COLOR_BGR2RGB)
    plt.imshow(test_img)
    plt.title("time_domain_convolution"+ str(TimeDomainTimer)[0:5] + "s")
    plt.subplot(122)
    fft_result = cv2.cvtColor(fft_result, cv2.COLOR_BGR2RGB)
    plt.imshow(fft_result)
    plt.title("fft_convolution" + str(FFTDomainTimer)[0:5] + "s")
    plt.show()

    plt.figure(2)
    plt.subplot(221)
    plt.imshow(np.log(np.abs(td_result_spectrum)))
    plt.title("td_result_spectrum")
    plt.subplot(222)
    plt.imshow(np.log(np.abs(fft_result_spectrum)))
    plt.title("fft_result_spectrum")
    plt.subplot(223)
    plt.imshow(np.log(np.abs(test_img_spectrum)))
    plt.title("test_img_spectrum")
    plt.subplot(224)
    plt.imshow(np.log(np.abs(kernel_spectrum)))
    plt.title("kernel_spectrum")
    plt.show()
    # cv2imgshow("test_img", test_img)