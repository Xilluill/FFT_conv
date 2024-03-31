
<h1 align = "center">基于NumPy的FFT快速卷积算法</h1>


## 摘要

​	卷积在当前的图像处理领域中被广泛应用，是许多计算机视觉和图像处理任务的基础操作之一。然而，传统的时域卷积在处理大尺寸核（$kernel$​）时可能效率较低。为了优化计算速度，傅里叶变换被引入到卷积中，通过将卷积操作转换到频域进行计算。本文旨在讨论卷积在图像处理中的重要性，并比较不同实现方式的性能，包括*OpenCV*和*SciPy*自带的卷积函数。最重要的是，本文编写了基于*NumPy*的频域加速卷积算法，比较了其与时域卷积的时间复杂度。



## 1. 介绍

​	在图像处理领域，卷积是一种重要的操作，通过将图像与特定的核（$kernel$）进行卷积操作，可以实现一系列图像处理任务，如模糊、边缘检测、锐化等。卷积操作可以使图像在不同尺度上进行平滑或突出边缘等特征，从而帮助人们更好地理解图像中的信息。例如，通过应用不同的卷积核，可以检测图像中的边缘，提取纹理信息，或者进行图像去噪等处理，从而为后续的分析和识别提供更好的基础。

​	随着深度学习的发展，卷积神经网络（$Convolutional Neural Networks，CNNs$）成为图像处理和计算机视觉领域的重要工具。$CNNs$通过卷积层对图像进行特征提取，然后通过池化层和全连接层进行分类或回归等任务。卷积层的引入使得网络能够自动学习图像中的特征，从而在图像识别、分割、检测等任务中取得了巨大成功。

​	通过传统图像处理和深度学习两个角度介绍，可以更全面地展示卷积在图像处理中的重要性和意义。

<div align=center><img src="https://pic2.zhimg.com/80/v2-a18f53d4f4d60a0eb6d1940d06bd5af5_720w.webp" alt="image-20240308000656316" width="400px" /></div>
<div align = "center"><i>图 1 卷积示意图</i></div>



## 2. 相关工作

​	在实现基于傅里叶变换的频域加速卷积算法时，考虑了使用其他语言的实现方式，包括*C++*和*MATLAB*。这两种语言在性能方面都有一定优势，*C++*具有较高的性能，而*MATLAB*能够充分利用硬件进行矩阵加速，从而提高计算效率。

​	然而，最终选择了使用*Python*和*NumPy*进行实现。*NumPy*是*Python*中用于科学计算的重要库，其底层实现部分采用C语言编写，具有较高的性能。*NumPy*提供了丰富的数学函数和数据结构，能够高效地进行数组操作和线性代数运算，使得基于*NumPy*的算法在性能上能够接近*C++*的实现。

​	选择*Python*作为实现平台还有其他几个重要原因。首先，*Python*作为一种高级语言，具有简洁易读的语法，使得算法的编写和调试更为方便。其次，*Python*平台提供了丰富的第三方库支持，例如*OpenCV*，可以方便地进行图像处理相关的比较和验证。最重要的是，*Python*作为一种流行的科学计算语言，拥有庞大的社区和生态系统，例如*SciPy*提供了一个卷积接口，以选择使用*FFT*加速或者直接进行卷积。

​	因此，尽管Python受限于一些性质导致了编写的底层算法性能不佳，但考虑到*NumPy*和*Python*平台的诸多优势，选择了*Python*和*NumPy*作为实现基于傅里叶变换的频域加速卷积算法的平台。



## 3. 算法详细

### 3.1 图像卷积运算

​	在*OpenCV*的文档中可以找到 **cv2.filter2D()**的原理。这个函数其实是计算的相关运算。这个情况很常见，无论是深度学习里面的可学习参量，还是说这次实验用的是对称卷积核。所以这个相关运算不会干扰本文的实验。

$$
dst(x,y) =\sum_{\substack{x'=0\\ y'=0}}^{(height,width)}\text{kernel} (x',y')* \text{src} (x+x' ,y+y' )
$$


### 3.2 二维快速傅里叶变换

​	二维傅里叶变换其实是相当复杂的，关于他的快速实现有多种方法。但是不变的是其时间复杂度降至$O(nlogn)$，这为加速卷积运算提供了可能。

$$
F(u,v)=\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)e^{-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}
$$



### 3.3 OpenCV​​的filter2D()函数

​	根据官方文档API可以看得到该函数的接收变量。

| *参数*       | *类型*          | *默认值*             | *具体含义*     |
| ------------ | --------------- | -------------------- | -------------- |
| *src*        | *NumPy.ndarray* | */*                  | *原图像*       |
| *ddepth*     | *int*           | */*                  | *目标图像深度* |
| *kernel*     | *NumPy.ndarray* | */*                  | *卷积核*       |
| *anchor*     | *tuple*         | *(-1,-1)*            | *卷积锚点*     |
| *delta*      | */*             | *0*                  | *偏移量*       |
| *borderType* | *enum*          | *cv2.BORDER_DEFALUT* | *边缘类型*     |

<div align = "center"><i>表 1 filter2D()的接口</i></div>

​	并且*OpenCV*本身用*C++*实现该函数时，当卷积核大小比较小时用时域卷积，而很大时用*DFT*加速运算。而本文则使用*FFT*加速运算。

<div align=center><img src="https://bu.dusays.com/2024/03/31/660842019d1d8.png" width="400px" alt="img" /></div><div align = "center"><i>图 2 filter2D()性能测试</i></div>							    

​	在图像尺寸为$2048×2048$下进行多次卷积调用该函数，可以测出该函数的性能曲线。蓝色曲线可以看出在相对尺寸较小的卷积核是近似$O(n^2)$快速增长的，而达到一定大小之后使用*DFT*加速了运算。可以看出相近尺寸的卷积核时间用时基本一致，这是因为进行DFT时的长宽选择固定，最后呈现出阶段性上升和整体的$O(n^2)$增长。



### 3.4 NumPy实现的时域卷积

​	实际上无论是*NumPy*还是*OpenCV*都没有真正的时域卷积。因此我通过编写代码实现了一个多通道的时域卷积。

```python
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
```

​	虽然*NumPy*本身是*C++*实现具有高性能，但是*python*的循环速度非常慢，导致该算法性能极低。



### 3.5 NumPy实现的频域加速卷积

​	根据公式可以推导出频域相乘可以代替时域卷积。但是值得注意的是，在离散信号中，频域相乘对应的是时域**循环卷积**，这同样适用于二维图像。

$$
\begin{aligned}
\text{y[n]}& =x[n]®h[n]=\sum_{m=0}^{N-1}x[m]h[((n-m))_N],\quad n=0,1,\cdots,N-1\text{。}  \\
Y(k)& =\operatorname{DFT}\{y[n]\}  \\
&=\sum_{k=0}^{N-1}\sum_{m=0}^{N-1}x[m]h[((n-m))_N]\mathrm{e}^{-\mathrm{j}\frac{2\pi}Nkn} \\
\text{}& =\sum_{m=0}^{N-1}x[m]\sum_{k=0}^{N-1}h[((n-m))_N]\mathrm{e}^{-\text{j}\frac{2\pi}Nkn}  \\
{}\text{}& =\sum_{m=0}^{N-1}x[m]\mathrm{e}^{-\mathrm{j}\frac{2\pi}Nkm}\sum_{k=0}^{N-1}h[n]\mathrm{e}^{-\mathrm{j}\frac{2\pi}Nkn}  \\
&=X[k]H[k]\
\end{aligned}
$$



​	测试算法时做的实验均为灰度图计算，后续实验将在三通道*RGB*彩色图上进行实验。

<div align=center><img src="https://bu.dusays.com/2024/03/31/660843d53ebbd.png" alt="image-20240308011920511" width="400px" /><div align = "center"><i>图 3 频域相乘结果图</i>


​	可以观察右侧结果图的边缘非常奇怪，这其实就是时域上的循环卷积，使图像整体进行了循环。所以需要格外的填充操作修复这一问题。

<div align=center><img src="https://bu.dusays.com/2024/03/31/6608446549637.png" width="400px" /><div align = "center"><i>图 4 频域相乘结果图</i>


​	从频域也可以看出，直接相乘导致的方形分界线非常明显，而真正的时域卷积会略有不同。

<div align=center><img src="https://bu.dusays.com/2024/03/31/660844ff57744.png" width="400px" /><div align = "center"><i>图 5 加入边缘填充效果图</i>

​	加入填充和裁剪操作才能让频域相乘逆变换这一操作正确执行，右图为镜像填充对比左图的常量填充。   

 

### 3.6 $SciPy$的卷积

​	*SciPy*其实提供了一个卷积接口，并且均为*NumPy*实现，性能比起自己手写的更加高效和稳定，因此本文也将这一接口加入了实验。

| 参数*    | *类型*                            | *默认值* | *具体含义*        |
| -------- | --------------------------------- | -------- | ----------------- |
| *in1*    | *NumPy.ndarray*                   | */*      | *原图像*          |
| *in2*    | *NumPy.ndarray*                   | */*      | *卷积核*          |
| *mode*   | *str {'full'， 'valid'， 'same'}* | *full*   | *输出尺寸*        |
| *method* | *str {'auto'， 'direct'， 'FFT'}* | *auto*   | *是否使用FFT加速* |

<div align = "center"><i>表 2 Scipy的接口</i></div>



## 4. 实验

​	本文设计了*filter2D*方法和*FFT*方法对比，以及时域方法于*FFT*方法对比。前者在正常$2048×2048$尺寸下运行，后者在$360×360$分辨率下进行计算，确保在短时间内获得更多实验数据。同时在*SciPy*统一平台上做了相同的测试。



### 4.1 OpenCV与NumPy实现的FFT方法比较

<div align=center><img src="https://bu.dusays.com/2024/03/31/66084c825beff.png" width="400px" /><div align = "center"><i>图 6 filter2D和FFT对比</i>

​	可以看出频域相乘这套算法会存在周期性或者随机性负载增加，推测应该是不同大小的卷积核在固定尺寸下进行*FFT*计算会花费不同的时间。整体也是缓慢上升趋势。而前半段使用时域卷积，后半段使用*DFT*加速的*filter2D()*方法则呈现出稳定上升趋势。




### 4.2 在NumPy和SciPy两个平台上时域卷积和FFT方法比较

<div align=center><img src="https://bu.dusays.com/2024/03/31/660846e2c1201.png" width="200px" /><img src="https://bu.dusays.com/2024/03/31/66085363a8a60.png" width="200px" /><div align = "center"><i>图 7 时域卷积和FFT对比</i>




​	而时域卷积呈现出明显的$O(n^2)$​复杂度，符合理论推测。当然，这一方法显著慢于前两种方法的原因在于python作为解释性语言在密集计算场景下性能不佳。而*OpenCV*是*C++*编写的库，因此当有更多的计算交给*C++*完成，就会获得更高的性能。这两次实验只能比较出几种算法的复杂度区别，而想看到两种方法在某一卷积核大小附近完成了性能反超，则需要更为统一的计算平台。

​	右侧为*SciPy*则提供了统一的性能比较平台，在上述实验条件下分别调用计算性能。用不同实验平台得出了相似结果，即时域卷积在很小的时候和*FFT*速度相当，在大尺寸下远慢于*FFT*方法。有意思的是，*SciPy*的时域实现还没有自己编写的快。



### 4.3 OpenCV和SciPy时间复杂度比较

​	由于几种方法性能差异都太大，无法观察到FFT方法的时间复杂度，因此本文设计了*SciPy*和*OpenCV*的比较实验，其中图8右侧中*OpenCV*运行在2048×2048的方法下，由于已知*OpenCV*为$O(n^2)$复杂度，故通过这种方法比较得出*FFT*方法的复杂度。

<div align=center><img src="https://bu.dusays.com/2024/03/31/66085413991b4.png" width="200px" /><img src="https://bu.dusays.com/2024/03/31/6608b7b0ce913.png" width="200px" /><div align = "center"><i>图 8 时域卷积和FFT对比</i>





### 4.4 不同图像的性能表现

<div align=center><img src="https://bu.dusays.com/2024/03/31/6608c79560d4d.png" width="600px" /><div align = "center"><i>图 9 7×7卷积</i>

<div align=center><img src="https://bu.dusays.com/2024/03/31/6608c7cf00db2.png" width="600px" /><div align = "center"><i>图 10 359×359卷积</i>

| 图片序号        | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 7×7卷积用时     | 2.91 | 1.49 | 2.11 | 2.12 | 2.09 | 2.13 | 2.07 | 2.22 | 1.36 | 2.14 | 2.09 |
| 359×359卷积用时 | 3.29 | 1.78 | 2.61 | 2.59 | 2.57 | 2.60 | 2.62 | 2.64 | 1.67 | 2.46 | 2.47 |

<div align = "center"><i>表 3 不同图像卷积用时</i></div>

​	由实验数据可知，相同分辨率的在同一尺度下卷积没有太大的区别。这和*FFT*的具体实现有关系。例如*Cooley–Tukey*算法在素数矩阵上性能就会变差。



## 5.总结 

​	这份报告介绍了基于*NumPy*的*FFT*快速卷积算法。首先讨论了卷积在图像处理中的重要性，然后比较了不同实现方式的性能，包括*OpenCV*和*SciPy*自带的卷积函数。接着详细介绍了算法的实现过程，并通过实验对比了不同方法的性能表现。实验结果显示，在大尺寸卷积核情况下，*FFT*方法明显优于时域卷积。最后总结了实验数据，指出了不同尺寸矩阵对*FFT*算法性能的影响。整体而言，本文提出的基于*NumPy*的*FFT*快速卷积算法能够有效提高图像处理中卷积操作的计算速度。



## 6.参考资料

[1]  [SciPy.signal.convolve — SciPy v1.12.0 手册](https://docs.SciPy.org/doc/SciPy/reference/generated/SciPy.signal.convolve.html)

[2]  [NumPy FFT 性能提升|极客笔记 (deepinout.com)](https://deepinout.com/NumPy/NumPy-questions/64_numpy_improving_fft_performance_in_python.html)

[3]  [NumPy.FFT.fft2 — NumPy v1.26 Manual](https://NumPy.org/doc/stable/reference/generated/NumPy.FFT.fft2.html)

[4]  [OpenCV: Image Filtering](https://docs.OpenCV.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04)

[5] [NumPy手搓卷积 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/659650386)
