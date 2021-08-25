# 借助飞桨框架使用卷积神经网络在CIFAR10数据集实现图像分类


#环境配置，导入要用到的库
import paddle as paddle  #飞桨库
import paddle.fluid as fluid  #飞桨的深度学习框架
import numpy as np
from PIL import Image  #Python Image Library图像库
import matplotlib.pyplot as plt
import os, sys


''' 数据准备，使用CIFAR10数据集，包含60,000张32x32的三通道彩色图片，10个类别，
	每个类包含6,000张图片。其中50,000张图片作为训练集，10000张作为验证集。
'''
def loadData():

	print('加载数据集中......')

	#每批大小
	BATCH_SIZE = 128
	
	''' paddle内置了CIFAR10数据集读取方法paddle.vision.datasets.Cifar10
		paddle.dataset.cifar.train10()获取cifar训练集
		paddle.dataset.cifar.test10()获取cifar测试集
		paddle.reader.shuffle()表示每次缓存BUF_SIZE个数据项，并进行打乱
		paddle.batch()表示每BATCH_SIZE组成一个batch
	'''
	
	#trainReader：用于训练的数据提供器
	trainReader = paddle.batch(
	    paddle.reader.shuffle(paddle.dataset.cifar.train10(), 
	                          buf_size=128*100),           
	    batch_size=BATCH_SIZE)                                
	
	#testReader：用于测试的数据提供器			
	testReader = paddle.batch(
		paddle.dataset.cifar.test10(),                            
		batch_size=BATCH_SIZE)   

	print('加载完成')
	return trainReader, testReader
	#print(testReader)


''' CNN网络模型搭建
	使用卷积神经网络结构：输入的二维图像先经过三次卷积层、池化层和Batchnorm，再经过全连接层，最后使用softmax分类作为输出层
	什么是Batchnorm？Batchnorm是对每batch个数据同时做一个norm，作用就是在模型训练过程中使得每一层神经网络的输入保持相同分布
	什么是softmax？softmax是多分类问题的分类函数。对应二分类问题中的sigmoid函数。因为CIFAR10数据集有10类数据
'''
def cnnModel(img):

	# 第一个卷积——池化层
	#simple_img_conv_pool 由一个conv2d和一个pool2d神经元组成
    #返回经过conv2d和pool2d之后的结果，数据类型与input相同
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,         # 输入图像
        filter_size=5,     # 滤波器的大小
        num_filters=20,    # filter的数量。它与输出的通道相同
        pool_size=2,       # 池化核大小2*2
        pool_stride=2,     # 池化步长
        act="relu")        # 使用relu激活函数
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)

    # 第二个卷积——池化层
    #上一层神经元的输出即是这一层神经元的输入，通过relu函数激活
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.layers.batch_norm(conv_pool_2)

    # 第三个卷积——池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_2,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")

    # 以softmax为激活函数的全连接输出层，10类数据输出10个数字
    prediction = fluid.layers.fc(input=conv_pool_3, size=10, act='softmax')
    return prediction


#模型训练
def modelTrain(trainReader, testReader):

	BATCH_SIZE = 128

	#网络配置
	print('开始网络配置')
	#定义输入数据
	paddle.enable_static()
	data_shape = [3, 32, 32]  #cifar数据集的shape  
	images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
	label = fluid.layers.data(name='label', shape=[1], dtype='int64')

	# 获取分类器，用cnn进行分类
	predict =  cnnModel(images)

	# 获取损失函数和准确率
	#cross_entropy计算输入input和标签label间的交叉熵损失
    #交叉熵刻画了标签值和预测值两个概率分布之间的距离，交叉熵越小，两个概率的分布越接近
	cost = fluid.layers.cross_entropy(input=predict, label=label) 
	avg_cost = fluid.layers.mean(cost)                            # 因为是一个batch的损失，要求所有损失的平均值
	acc = fluid.layers.accuracy(input=predict, label=label)       #使用输入和标签计算准确率

	# 获取测试程序
	test_program = fluid.default_main_program().clone(for_test=True)
    
    # Adam优化器利用梯度下降方法动态调整每个参数的学习率
	# 定义优化方法
	optimizer =fluid.optimizer.Adam(learning_rate=0.001)
	optimizer.minimize(avg_cost)
	
	''' 至此，配置好了两个fluid.Program：fluid.default_startup_program() 与fluid.default_main_program() 
		参数初始化操作会被写入fluid.default_startup_program()。fluid.default_main_program()用于获取默认或
		全局main program(主程序)，用于训练和测试模型。
	'''
	print("网络配置完成")

	#训练模型
	print('开始训练')
	# 定义使用CPU
	place = fluid.CPUPlace()

	# 创建执行器，初始化参数
	#Executor:接收传入的program，通过run()方法运行program
	exe = fluid.Executor(place)
	exe.run(fluid.default_startup_program())

	#定义数据映射器
	#DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor
	feeder = fluid.DataFeeder( feed_list=[images, label],place=place)

	#绘制训练过程中损失值和准确率变化趋势
	all_train_iter=0
	all_train_iters=[]
	all_train_costs=[]
	all_train_accs=[]
	def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
	    plt.title(title, fontsize=24)
	    plt.xlabel("iter", fontsize=20)
	    plt.ylabel("cost/acc", fontsize=20)
	    #用红色表示损失值，绿色表示准确率
	    plt.plot(iters, costs,color='red',label=label_cost) 
	    plt.plot(iters, accs,color='green',label=lable_acc) 
	    plt.legend()
	    plt.grid()
	    plt.show()

	#训练过程
	EPOCH_NUM = 20
	#model_save_dir = "/home/aistudio/work/catdog.inference.model"
	model_save_dir = sys.path[0]+'/cifar10.inference.model'

	for pass_id in range(EPOCH_NUM):
	    # 开始训练
	    for batch_id, data in enumerate(trainReader()):                        #遍历train_reader的迭代器，并为数据加上索引batch_id
	        train_cost,train_acc = exe.run(program=fluid.default_main_program(),#运行主程序
	                             feed=feeder.feed(data),                        #喂入一个batch的数据
	                             fetch_list=[avg_cost, acc])                    #fetch均方误差和准确率

	        
	        all_train_iter=all_train_iter+BATCH_SIZE
	        all_train_iters.append(all_train_iter)
	        all_train_costs.append(train_cost[0])
	        all_train_accs.append(train_acc[0])
	        
	        #每100次batch打印一次训练、进行一次测试
	        if batch_id % 100 == 0:                                             
	            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' % 
	            (pass_id, batch_id, train_cost[0], train_acc[0]))
	            
	    # 开始测试
	    test_costs = []                                                         #测试的损失值
	    test_accs = []                                                          #测试的准确率
	    for batch_id, data in enumerate(testReader()):
	        test_cost, test_acc = exe.run(program=test_program,                 #执行测试程序
	                                      feed=feeder.feed(data),               #喂入数据
	                                      fetch_list=[avg_cost, acc])           #fetch 误差、准确率
	        test_costs.append(test_cost[0])                                     #记录每个batch的误差
	        test_accs.append(test_acc[0])                                       #记录每个batch的准确率
	    
	    # 求测试结果的平均值
	    test_cost = (sum(test_costs) / len(test_costs))                         #计算误差平均值（误差和/误差的个数）
	    test_acc = (sum(test_accs) / len(test_accs))                            #计算准确率平均值（ 准确率的和/准确率的个数）
	    print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
	    
	#保存模型
	# 如果保存路径不存在就创建
	if not os.path.exists(model_save_dir):
	    os.makedirs(model_save_dir)
	print ('save models to %s' % (model_save_dir))
	fluid.io.save_inference_model(model_save_dir,
	                              ['images'],
	                              [predict],
	                              exe)
	print('训练模型保存完成！')
	draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc")


#模型预测
def modelPredict():
	#创建预测用的创建预测用的Executor
	#定义使用CPU
	place = fluid.CPUPlace()
	infer_exe = fluid.Executor(place)
	inference_scope = fluid.core.Scope() 
	model_save_dir = sys.path[0]+'/cifar10.inference.model'
	paddle.enable_static()

	#预测之前预处理图像
	def load_image(file):
		im = Image.open(file)
		im = im.resize((32, 32), Image.ANTIALIAS)
		im = np.array(im).astype(np.float32)
		im = im.transpose((2, 0, 1))
		im = im / 255.0
		im = np.expand_dims(im, axis=0)
		#print('图片维度：',im.shape)
		return im

    #开始预测
    #通过fluid.io.load_inference_model，预测器从params_dirname中读取已经训练好的模型，来对从未遇见过的数据进行预测
	with fluid.scope_guard(inference_scope):
		#从指定目录中加载 推理model(inference model)
		[inference_program, # 预测用的program
		feed_target_names, # 是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
		fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。
		                                                    infer_exe)     #infer_exe: 运行 inference model的 executor
		    
		#要预测的图片完整路径
		infer_path='C:/Users/DJ/Desktop/研究生/神经网络/cat1.jpg'
		img = Image.open(infer_path)
		plt.imshow(img)   
		plt.show()    
		    
		img = load_image(infer_path)

		results = infer_exe.run(inference_program,                     #运行预测程序
		                            feed={feed_target_names[0]: img},  #喂入要预测的img
		                            fetch_list=fetch_targets)          #得到推测结果
		#print('results',results)
		label_list = [
		    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
		    "ship", "truck"
		    ]
		print("预测结果: %s" % label_list[np.argmax(results[0])])


# 主函数
if __name__ == '__main__':
    trainReader, testReader = loadData()
    #modelTrain(trainReader, testReader)
    modelPredict()
    
