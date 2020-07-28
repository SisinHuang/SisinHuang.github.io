### pytorch向tensorflow的转移

#### 1. pytorch的pth权重文件转换为tensorflow的ckpt文件[1]

```python
import tensorflow as tf
import torch

def convert(pth_path, ckpt_path):
	with tf.Session() as sess:
		# torch.load()得到的权重字典
		for var_name, value in torch.load(pth_path, map_location='cpu').items():
			# pytorch和tensorflow的变量名格式不同，需改成符合tensorflow的格式
			print(var_name)
			tf.Variable(initial_value=value, name=var_name)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		saver.save(sess, ckpt_path)
```

#### 2. 查看ckpt文件中的权重名称和权重值[2]

```python
import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = “ckpt文件位置”
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))
```

#### 3. pytorch中的num_batches_tracked[3]

从PyTorch 0.4.1开始, BN层中新增加了一个参数 track_running_stats,

```
BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
```

这个参数的作用如下:

​	训练时用来统计训练时的forward过的min-batch数目,每经过一个min-batch, track_running_stats+=1
如果没有指定momentum, 则使用1/num_batches_tracked 作为因数来计算均值和方差(running mean and variance).

#### 4. 加载pretrained model出现的大量adam变量丢失[4]

这是由于 要恢复的变量设置 和 optimizer的摆放位置出错造成的。原因很简单，在你指定

```
variables_to_restore = slim.get_variables_to_restore()
```


​	之前，声明了optimizer优化器时，则优化器里面的adam的一些参数也被加载到图中，但是预训练模型中并不含这些参数，则出现了大量的adam缺失。

解决办法：
**更换 指定恢复变量 和 optimizer 的摆放位置**：

之前是：

```
opt = tf.train.AdamOptimizer(learning_rate=lr_v)
variables_to_restore = slim.get_variables_to_restore()
```


更改为：

```
variables_to_restore = slim.get_variables_to_restore()
opt = tf.train.AdamOptimizer(learning_rate=lr_v)
```


问题就可以解决。


#### 函数方面

*“x”表示一个向量或矩阵*

##### 1. 向量的形状

```
x.size() ==> x.get_shape() 或 x.shape 或 tf.shape(x)
```

这三个获取向量形状的函数在使用时需注意：

​	x.get_shpae()和x.shape是动态的，在tensorflow中使用时会返回值为**None**的值，这可能会让后续的操作出现问题。比如：tf.reshape(x, [x.shape[1], x.shape[0]])会发生错误（x.shape[1]值为None，reshape函数无法操作）

​	对于h=tf.shape(x)[0] w=tf.shape(x)[1]，这里的h和w是(symbolic或graph-mode)向量形式，所以在运行期间会获得确定的值。在这种情况下tf.reshape(x, [w, h])也会产生一个（symbolic）向量，运行时reshape会得到确定的形状值。

​	总结：tf.shape(x)会得到一个整型类型的向量，这个向量的值在运行时确定且不是None；但x.shape()和x.get_shape()会直接返回一个代表x形状的静态列表或元组，如果x的形状没有被声明，函数会直接返回None。[5]

##### 2. 乘法

<https://zhuanlan.zhihu.com/p/77113823>

##### 3. 逆矩阵

```
torch.inverse(x) ==> tf.matrix_inverse(x)
```

##### 4. 正则化

```
torch.nn.functional.normalize(input, p=2, dim=1, eps=1e-12, out=None)
```

​	input - 输入张量的形状
​	p（float） - 规范公式中的指数值。默认值：2
​	dim（int） - 要缩小的维度。默认值：1
​	eps（float） - 小值以避免除以零。默认值：1e-12

​	按照某个维度计算范数，p表示计算p范数（等于2就是2范数），dim计算范数的维度（这里为1，一般就是通道数那个维度）

```
tf.norm(tensor, ord='elucidean', axis=None, keep_dims=False, name=None)
```

​	这个函数可以计算几个不同的向量范数(1-norm,Euclidean 或 2-norm,inf-norm,p> 0 的 p-norm)和矩阵范数(Frobenius,1-norm 和 inf -norm).

参数：

- tensor：float32,float64,complex64,complex128 类型的张量.
- ord：范数的顺序.支持的值是“fro”、“euclidean”、0、1 、2、np.inf 和任意正实数,得到相应的 p-norm.缺省值是 'euclidean',如果张量是一个矩阵,则相当于 Frobenius 范数；如果是向量,则相当于 2-norm.一些限制适用：1、所述的 Frobenius 范数不是为向量所定义；2、若轴为 2 元组(矩阵范数),仅支持 “euclidean”、“fro”、1 、np.inf .有关如何计算在张量中存储的一批向量或矩阵的准则,请参见轴的说明.
- axis：如果 axis 是 None(默认值),那么输入被认为是一个向量,并且在张量的整个值集合上计算单个向量范数,即 norm(tensor,ord=ord)是等价于norm(reshape(tensor, [-1]), ord=ord).如果 axis 是 Python 整数,则输入被认为是一组向量,轴在张量中确定轴,以计算向量的范数.如果 axis 是一个2元组的 Python 整数,则它被认为是一组矩阵和轴,它确定了张量中的坐标轴,以计算矩阵范数.支持负数索引.示例：如果您在运行时传递可以是矩阵或一组矩阵的张量,则通过 axis=[-2,-1],而不是 axis=None 确保计算矩阵范数.
- keep_dims：如果为 True,则 axis 中指定的轴将保持为大小 1.否则,坐标轴中的尺寸将从 "输出" 形状中移除.
- name：操作的名字.

返回值：

- output：与张量具有相同类型的 Tensor,包含向量或矩阵的范数.如果 keep_dims 是 True,那么输出的排名等于张量的排名.否则, 如果轴为 none,则输出为标量；如果轴为整数,则输出的秩小于张量的秩；如果轴为2元组,则输出的秩比张量的秩低两倍.

可能引发的异常：

- ValueError：如果 ord 或者 axis 是无效的.

##### 5. torch.one_hot(x, depth) ==> tf.one_hot(x, depth)

##### 6. torch.std(x, axis=None, keepdims=False) ==> tf.keras.backend.std(x, axis=None, keepdims=False)

##### 7. torch.exp(x) ==> tf.exp(x)

##### 8. 元素取反或根据条件改变值

​	condition表示特定的条件值

​	torch下的操作有：~x； x != condition

​	tensorflow可以用：tf.where(x!=condition, x=条件为真的设定值, y=条件为假的设定值)

​		

参考资料：

[pytorch的pth权重文件如何转换为]: https://www.zhihu.com/question/317353538/answer/924705912
[tensorflow怎么查看.ckpt内保存的权重名和权重值]: <https://blog.csdn.net/Geoffrey_MT/article/details/82690340>
[PyTorch中BN层中新加的 num_batches_tracked 有什么用?]: <https://blog.csdn.net/shanglianlm/article/details/101394508>
[【tensorflow】加载pretrained model出现的大量adam变量丢失]: https://blog.csdn.net/shwan_ma/java/article/details/82868751
[TypeError: Failed to convert object of type &amp;lt;class &amp;#39;list&amp;#39;&amp;gt; to Tensor. Contents: [1, 1, Dimension(None)\]]:<https://github.com/tensorflow/models/issues/6245>

