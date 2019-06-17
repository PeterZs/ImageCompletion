# Image Completion

## 项目依赖

* Linux系统下测试运行。其中，训练过程需要Pytorch GPU版本。

* Python3

* Python库依赖

  ```
  opencv_python
  torch(版本>1.1.0 如果需要运行train.py 则需要安装GPU版本)
  torchvision
  scikit_image
  skimage
  tqdm
  numpy
  pandas
  Pillow
  ```



## Train

1. 训练首先需要准备数据集，在下载好数据集之后，运行如下命令，则自动划分数据集为训练集与测试集。(默认比例为80%训练集，可以通过参数--proportion修改)。

```
python3 prepare_data.py /path/to/your/src_data_dir
```

2. 开始训练。训练只需要运行如下指令。其中all kinds of training options指训练的各种参数，详细参数在opt.py中查看。

```
python3 train.py [all kinds of training options]
```



## Test

我已经在项目中添加了训练好的人脸补全模型文件: weights/completion/completion_weights.pth 。进行测试只需要运行如下代码：

```
python3 inference.py --input_img test_imgs/test_7.jpg --output_img test_imgs/test_7_output.jpg --model weights/completion/completion_weights.pth --mode manual
```

上述代码中的四个参数依次为：输入图像路径，输出图像路径，模型存放路径，运行模式（其中manual 为手动涂抹代补全区域，random为随机产生代补全区域）。

或者全部直接采用默认参数：

```
python3 inference.py
```

