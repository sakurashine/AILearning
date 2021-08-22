''' 借助飞桨框架使用LeNet在MINIST数据集实现手写数字识别
    构建神经网络/深度学习模型主要有以下几大步骤
    数据处理：读取数据并完成数据预处理，如数据校验、格式转化等，保证模型可读取
    模型设计：网络结构设计
    模型训练：设定模型采用的寻解算法（优化器），并指定计算资源，循环调用训练过程，每轮都包括前向计算、损失函数和后向传播等
    模型预测：训练完成后，使用训练好的模型对测试集进行预测，计算损失与精度，验证模型的效果
'''


''' 环境配置
    首先需导入飞桨框架及其相关API
    paddle.nn 目录下包含了飞桨框架支持的神经网络层和相关函数的相关API
    paddle.vision 目录是飞桨在视觉领域的高层API，包括内置数据集、内置模型、视觉操作、数据处理等相关API
'''
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
from paddle.vision.transforms import Compose, Normalize
''' 注：一开始在此处添加了from matplotlib import pyplot as plt，结果在训练模型的过程中不知为何出现
    错误称async handler deleted by the wrong thread，设置渲染器为Agg后解决此错误。但是在之后模型预
    测阶段使用plt绘图时报错agg is a non-GUI backend，故现在于训练、预测两个步骤里分别设定了渲染器。
'''


#第一步，数据加载并做预处理
def loadData():
    #paddle提供了数据集读取方法paddle.vision.datasets.MNIST（）
    #导入数据集Compose的作用是将用于数据集预处理的接口以列表的方式进行组合。
    #导入数据集Normalize的作用是图像归一化处理。
    #问：为什么要做归一化处理，如何实现的？
    #在Normalize接口中，进行了如下运算，output[channel]=(input[channel]−mean[channel])/std[channel]
    #实现了将像素矩阵进行归一化处理，数值的范围从0 ~ 255压缩到-1 ~ 1，这有利于之后进行运算
    #data_format规定了数据的排列方式，默认为CHW，设置为 “CHW” 时，数据排列顺序为 [batch, channels, height, width]
    #问：设置为NHWC与NCHW时，有什么区别呢？
    #两种格式的区别在于对数据格式进行RGB->灰度的计算过程时的访存特点不一样。
    #基于 CPU开发时，NHWC 局部性更好，cache 利用率高，使用 NHWC 比 NCHW 稍快一些。
    #NCHW则是 Nvidia cuDNN 默认格式，使用 GPU 加速时用 NCHW 格式速度会更快
    transform = Compose([Normalize(mean=[127.5],
                                   std=[127.5],
                                   data_format='CHW')])
    print('加载训练集中......')
    #加载MINIST训练集
    train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
    #加载MINIST测试集
    test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
    print('加载完成')
    #MINIST数据集的格式是怎样的？
    #MNIST数据集包含60000个28像素*28像素的手写数字数据，其中50000个作为训练集，10000个作为测试集
    #像素图的shape是784，标签的shape是10
    #label的shape为什么是10？
    #MNIST数据的label表示该像素图所表示的是0~9的几，共有10分类。label采用了one hot独热编码，其中一个元素设为1，其余均设为0
    #第一个下标是第几个数据（28*28），第二个下标是该数据的标签（0~9的几）
    train_data0, train_label_0 = train_dataset[0][0],train_dataset[0][1]
    #reshape方法按行重新再塑形数组的排列格式
    train_data0 = train_data0.reshape([28,28])
    
    '''打印数据集图像
    plt.figure(figsize=(2,2))
    plt.imshow(train_data0, cmap=plt.cm.binary)
    plt.show()
    print("图像数据形状和对应数据为:", train_data0.shape)
    print("图像标签形状和对应数据为:", train_label_0.shape, train_label_0)
    print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(train_label_0))
    '''
    return train_dataset, test_dataset


''' 第二步，采用单隐层全连接网络，搭建模型网络结构
    其中输入层784个神经元（28像素*28像素），即待处理数据中输入变量的数量
    隐层512个神经元（自己取定），输出层10个神经元（输出结果是0~9个数字）
    问：输入输出层的神经元个数好确定，但是隐藏层的神经元个数该如何确定呢？
    关于这个问题，查阅相关资料，有表述说，隐层神经元个数可随意取定，但需大致遵循以下几个原则
    1、隐层神经元的数量应在输入层的大小和输出层的大小之间
    2、隐层神经元的数量应为输入层大小的2/3加上输出层大小的2/3
    3、隐层神经元的数量应小于输入层大小的两倍
    具体原理，暂时未能深入了解。
'''
def modelStruct():
    ''' 使用Sequential定义神经网络，Sequential接口是飞桨提供的顺序容器。
        1、Flatten接口将一个连续维度的张量展平成一维张量。简言之，就是将28*28像素拉平。
        问：什么是张量？
        可以简单理解为多维数组。0维张量是标量，1维张量是矢量，2维张量是矩阵，3维张量是矩阵数组等等。
        2、Linear接口将隐层和输出层设置为线性变换层。Out=XW+b
        3、ReLU接口，用relu激活函数处理神经元经过线性变换的结果，然后作为输出值输出到下一层。ReLU(x)=max(0,x)
    '''
    network = paddle.nn.Sequential(
        paddle.nn.Flatten(),           # 拉平，将 (28, 28) => (784)
        paddle.nn.Linear(784, 512),    # 隐层：线性变换层
        paddle.nn.ReLU(),              # 激活函数，ReLU(x)=max(0,x)
        paddle.nn.Linear(512, 10)      # 输出层
    )
    # 模型封装
    model = paddle.Model(network)
    # 模型可视化，summary方法打印网络的基础结构和参数信息
    model.summary((1, 28, 28))
    return model,network


# 第三步，模型训练，主要是配置损失函数、优化器以及评估指标等    
def modelTrain(train_dataset, test_dataset, model, network):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    # prepare接口配置模型所需的部件，包括优化器、损失函数和评价指标等
    # Adam优化器利用梯度下降方法动态调整每个参数的学习率，parameters接口返回一个包含模型所有参数的列表
    #CrossEntropyLoss计算输入input和标签label间的交叉熵损失
    #交叉熵刻画了标签值和预测值两个概率分布之间的距离，交叉熵越小，两个概率的分布越接近
    model.prepare(paddle.optimizer.Adam(learning_rate=0.001, parameters=network.parameters()),
                  paddle.nn.CrossEntropyLoss(),
                  paddle.metric.Accuracy())
                  
    # fit接口完成对模型的训练
    model.fit(train_dataset,  # 训练数据集
              test_dataset,   # 评估数据集
              epochs=5,       # 训练的总轮次
              batch_size=64,  # 训练使用的批大小
              verbose=1)      # 日志展示形式
    # 模型评估，根据prepare接口配置的loss和metric进行返回
    result = model.evaluate(test_dataset, verbose=1)
    print(result)


# 第四步，模型预测
def modelPredict(test_dataset, model):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    # 进行预测操作,飞桨提供了predict接口对训练好的模型进行预测验证
    #predict的返回格式是一个list，元素数目对应模型的输出数目
    result = model.predict(test_dataset)
    # 定义画图方法，三个参数：像素图、标签值、预测值
    def showImg(img, label, predict):
        plt.figure()
        #title展示了该像素图的标签值和预测值
        plt.title('label= {}'.format(label)+'predict= {}'.format(predict))
        plt.imshow(img.reshape([28, 28]), cmap=plt.cm.binary)
        plt.show()
    # 抽样验证
    samples = [2, 77, 520, 3456]
    for n in samples:
        showImg(test_dataset[n][0], test_dataset[n][1], np.argmax(result[0][n]))


# 主函数
if __name__ == '__main__':
    train, test = loadData()
    model, network = modelStruct()
    modelTrain(train, test, model, network)
    modelPredict(test, model)




