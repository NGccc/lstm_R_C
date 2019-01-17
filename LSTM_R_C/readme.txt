1.数据:
data目录下，需要处理出分类和回归的标签
1).分类标签
文件名: labels_c.txt
每行是一句话对应的动作序列，空格隔开是这句话对应的每帧的动作。
例如:
1_2_1_0 2_2_1_0 1_1_1_2 1_2_1_0
表示4帧的动作。
然后需要把动作标签转换为一个整数（类别），得到
0 1 2 0

2).回归标签
文件名: labels_r.txt
4.52_2.211_1.2222_0.2 20.1_15.0_3.4_0 4.7_1_4.4_20.7 4.72_3_2.34_2.0
对应上面的4帧动作

3)
使用os.listdir处理出上述标签，数据中有一些是坏的，在errorlist.txt中，为了与已经处理好的句子对应，必须不处理这些文件。

4)
config.py中需要修改你对应的回归关键点的数量和分类总类别数目

#回归数
opts.key_num    = 8

#分类数
opts.class_num  = 4398

2.网络
单向lstm+分类&回归

3.train
初次训练
python train.py
继续前一个模型的训练
python train.py -model='./model/yourmodelname'

4.inference
python inference.py