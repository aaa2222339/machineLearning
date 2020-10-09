# Machine Learning学习笔记

## 常用库函数

### pandas

[pandas API文档](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

```python
# 生成DataFrame
pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)

# 显示开头的部分数据
DataFrame.head(n=5)

# 输出DataFrame的总结性信息
DataFrame.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None)

# 输出数据的描述性统计信息
DataFrame.describe(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)
```



### matplotlib

`matplotlib`的默认字体无法显示中文，所以需要切换字体。在`Windows`下，可以切换到`SimHei`字体；在该字体中，负号无法正常显示，所以需要加入第二行的参数设置。

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
```

在`Mac`下，默认并没有`SimHei`字体，我们可以切换到`Arial Unicode MS`字体，同样可以显示中文：

```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
```



```python
# 散点图
# s:size c:color
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=<deprecated parameter>, edgecolors=None, *, plotnonfinite=False, data=None, **kwargs)[source]

# 标题
matplotlib.pyplot.title(label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs)

# x轴标签
matplotlib.pyplot.xlabel(xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)[source]

# y轴标签
matplotlib.pyplot.ylabel(ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs)[source]

# 显示所有的figure
matplotlib.pyplot.show(*, block=None)
```



## LinearRegression

线性回归在`sklearn`的`linear_model`模块中，可以在头部导入：

```python
from sklearn.linear_model import LinearRegression
```

此外，还要导入一些工具库：

```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
```

在`sklearn`中有一个`datasets`模块，该模块中可以直接导入很多常用的数据集，这里我们使用波士顿房价的数据集：

```python
from sklearn.datasets import load_boston
```

下面就开始导入数据：

```python
data = load_boston()
```

得到的`data`是一个字典，其中包括：

- `data`：自变量数据
- `target`：因变量数据
- `feature_names`：每一个自变量的名字
- `DESCR`：该数据集的描述信息
- `filename`：数据集在本地的存储位置

为了方便操作数据，可以将数据转成`pandas.DataFrame`结构：

```python
# DataFrame是一个表格的样式，第一个参数传入数据，columns是每一列数据的名字
df = pd.DataFrame(data.data,columns=data.feature_names)
# 可以直接将变量名当做索引来得到数据，若没有该索引则可以新建一个
# 下列操作可以直接将因变量-房价，放到名为price的列中
df['price'] = data.target
```

得到的`DataFrame`如下：

![image-20200927165836836](tutorial.assets/image-20200927165836836.png)

为了方便看自变量和因变量的关系，可以通过散点图来直观观察：

```python
# 更换能显示中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 以CRIM为x轴，price为y轴画散点图
plt.scatter(df['CRIM'], data['price'])
# 设置图的标题
plt.title('城镇人均犯罪率与-房价散点图')
# 设置x轴的标题
plt.xlabel('城镇人均犯罪率')
# 设置y轴的标题
plt.ylabel('房价')
# 显示网格
plt.grid()
# 显示图像
plt.show()
```

得到结果如下：

<img src="tutorial.assets/image-20200927170152257.png" alt="image-20200927170152257" style="zoom:50%;" />

同样的方法，可以观察其他变量之间的关系。

模型拟合很简单：

```python
lm = LinearRegression()
# 拟合数据
lm.fit(train_X,train_Y)
```

拟合完成之后，可以查看模型的参数：

```python
# 不同特征变量的系数
lm.coef_
# 拟合模型的截距
lm.intercept_
# R squre
lm.score(X, y)
```

此外，还可以直接用该模型预测新的值：

```python
test_y = lm.predict(test_X)
```

为了更客观、更有说服力，可以将数据分成训练集和测试集：

```python
train_X,test_X,train_Y,test_Y = train_test_split(X,y,test_size=0.2)
```

此外，`sklearn`中也自带交叉验证的方法：

```python
kf = KFold(n_splits=5, shuffle=True)
scores = cross_val_score(linear_model, X, y, cv=kf)
```

`scores`中有`n_splits`个元素，即`n_splits`次`KFold`得到的$R^2$。



## Logistic regression

`sklearn`中`logistic regression`的函数：

```python
sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, 	fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, 		
  solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, 	
  n_jobs=None, l1_ratio=None)
```

两个较为重要的参数：

- `solver`:**{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’**

  | 正则项 | Solver              |                                                              |
  | ------ | ------------------- | ------------------------------------------------------------ |
  | L1     | liblinear           | liblinear适用于小数据集；如果选择L2正则化发现还是过拟合，即预测效果差的时候，就可以考虑L1正则化；如果模型的特征非常多，希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，也可以使用L1正则化。 |
  | L2     | liblinear           | libniear只支持多元逻辑回归的OvR，不支持MvM，但MVM相对精确。  |
  | L2     | lbfgs/newton-cg/sag | 较大数据集，支持one-vs-rest(OvR)和many-vs-many(MvM)两种多元逻辑回归。 |
  | L2     | sag                 | 如果样本量非常大，比如大于10万，sag是第一选择；但不能用于L1正则化。 |

- `multi_class`:**{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’**

`ovr`和`multinomial`的[详细介绍](https://blog.csdn.net/keeppractice/article/details/107088538)来自一篇博客，防止网站丢失，部分内容摘抄如下

### 多分类Logistic

#### 前言

逻辑回归分类器（Logistic Regression Classifier）是机器学习领域著名的分类模型。其常用于解决二分类（Binary Classification）问题。
利用二分类学习器进行的多分类学习可以分为三种策略：

一对一 （One vs. One, 简称OvO）
一对其余 （One vs. Rest，简称 OvR）也可是OvA（One vs. All）但是不严格
多对多（Many vs. Many，简称 MvM）

#### One-VS-Rest

假设我们要解决一个分类问题，该分类问题有三个类别，分别用△，□和×表示，每个实例（Entity）有两个属性（Attribute），如果把属性 1 作为 X 轴，属性 2 作为 Y 轴，训练集（Training Dataset）的分布可以表示为下图：

![img](https://img-blog.csdnimg.cn/20200702185736524.png)

One-Vs-Rest 的思想是把一个多分类的问题变成多个二分类的问题。转变的思路就如同方法名称描述的那样，选择其中一个类别为正类（Positive），使其他所有类别为负类（Negative）。比如第一步，我们可以将三角形所代表的实例全部视为正类，其他实例全部视为负类，得到的分类器如图：

![img](https://img-blog.csdnimg.cn/20200702185814831.png)

同理我们把 X 视为正类，其他视为负类，可以得到第二个分类器：

![img](https://img-blog.csdnimg.cn/20200702185834906.png)

最后，第三个分类器是把正方形视为正类，其余视为负类：

![img](https://img-blog.csdnimg.cn/20200702185850910.png)

对于一个三分类问题，我们最终得到 3 个二元分类器。在预测阶段，每个分类器可以根据测试样本，得到当前正类的概率。即 P(y = i | x; θ)，i = 1, 2, 3。选择计算结果最高的分类器，其正类就可以作为预测结果。

**One-Vs-Rest** 最为一种常用的二分类拓展方法，其优缺点也十分明显。

优点：普适性还比较广，可以应用于能输出值或者概率的分类器，同时效率相对较好，有多少个类别就训练多少个分类器。

缺点：很容易造成训练集样本数量的不平衡（Unbalance），尤其在类别较多的情况下，经常容易出现正类样本的数量远远不及负类样本的数量，这样就会造成分类器的偏向性。

#### One-Vs-One

相比于 One-Vs-Rest 由于样本数量可能的偏向性带来的不稳定性，One-Vs-One 是一种相对稳健的扩展方法。对于同样的三分类问题，我们像举行车轮作战一样让不同类别的数据两两组合训练分类器，可以得到 3 个二元分类器。

它们分别是三角形与 x 训练得出的分类器：

![img](https://img-blog.csdnimg.cn/20200702190039762.png)

三角形与正方形训练的出的分类器：

![img](https://img-blog.csdnimg.cn/20200702190052314.png)

以及正方形与 x 训练得出的分类器：

![img](https://img-blog.csdnimg.cn/20200702190105300.png)

假如我们要预测的一个数据在图中红色圆圈的位置，那么第一个分类器会认为它是 x，第二个分类器会认为它偏向三角形，第三个分类器会认为它是 x，经过三个分类器的投票之后，可以预测红色圆圈所代表的数据的类别为 x。

任何一个测试样本都可以通过分类器的投票选举出预测结果，这就是 One-Vs-One 的运行方式。

当然这一方法也有显著的优缺点，其缺点是训练出更多的 Classifier，会影响预测时间。

虽然在本文的例子中，One-Vs-Rest 和 One-Vs-One 都得到三个分类器，但实际上仔细思考就会发现，如果有 k 个不同的类别，对于 One-Vs-Rest 来说，一共只需要训练 k 个分类器，而 One-Vs-One 则需训练 C(k, 2) 个分类器，只是因为在本例种，k = 3 时恰好两个值相等，一旦 k 值增多，One-Vs-One 需要训练的分类器数量会大大增多。

当然 One-Vs-One 的优点也很明显，它在一定程度上规避了数据集 unbalance 的情况，性能相对稳定，并且需要训练的模型数虽然增多，但是每次训练时训练集的数量都降低很多，其训练效率会提高。

#### 比较 OvO 和 OvR

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070323082443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2tlZXBwcmFjdGljZQ==,size_16,color_FFFFFF,t_70)

容易看出，OvR只需训练N个分类器，而OvO则需要训练N(N-1)/2个分类器，因此，**OvO的存储开销和测试时间开销通常比OvR更大**。但在训练时，OvR的每个分类器均使用全部的训练样例，而OvO的每个分类器仅用到两个类的样例，因此，**在类别很多的时候，OvO的训练时间开销通常比OvR更小**。至于预测性能，则取决于具体的数据分布，在多数情况下两者差不多。

#### 多对多 （Many vs Many）

多对多是每次将若干类作为正例，若干其他类作为负例。MvM的正反例构造有特殊的设计，不能随意选取。我们这里介绍一种常用的MvM技术：纠错输出码（EOOC）。

- 编码：对N个类做M次划分，每次划分将一部分类别划分为正例，一部分划分为反例，从而形成一个二分类的训练集：这样共有M个训练集，则可训练出M个分类器。
- 解码：M个分类器分别对测试样本进行预测，这些预测样本组成一个编码。将这个编码与每个类各自的编码进行比较，返回其中距离最小的类别作为最终预测结果。

类别划分通过"编码矩阵" (coding matrix) 指定.编码矩阵有多种形式，**常见的主要有二元码 [Dietterich and iri 1995] 和三元码 [Allwein et al.,2000]**. 前者将每个类别分别指定为正类和反类，**后者在正、反类之外，还可指定"停用类"**因 3.5 给出了一个示意图，在图 3.5(a) 中，分类器 Cl 类和C3 类的样例作为正例 C2 类和 C4 类的样例作为反例;在图 3.5(b) 中，分类器14 类和 C4 类的样例作为正例 C3 类的样例作为反例.在解码阶段，各分类器的预测结果联合起来形成了测试示例的编码，该编码与各类所对应的编码进行比较?将距离最小的编码所对应的类别作为预测结果.例如在图 3.5(a) 中，若基于欧民距离，预测结果将是 C3.

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200704001238483.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2tlZXBwcmFjdGljZQ==,size_16,color_FFFFFF,t_70)
为什么要用纠错输出码呢？因为在测试阶段，ECOC编码对分类器的错误有一定的容忍和修正能力。例如上图中对测试示例正确的预测编码是（-1，1，1，-1，1），但在预测时f2出错从而导致了错误的编码（-1， -1， 1， -1，1）。但是基于这个编码仍然能产生正确的最终分类结果C3。



### 标准化和归一化

`sklearn.processing`中有标准化和归一化的函数：

```python
# 标准化
sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)

# 归一化
sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), *, copy=True)

# Methods
fit(X[, y]) # Compute the mean and std to be used for later scaling.
fit_transform(X[, y]) # Fit to data, then transform it.
get_params([deep]) # Get parameters for this estimator.
inverse_transform(X[, copy]) # Scale back the data to the original representation
partial_fit(X[, y]) # Online computation of mean and std on X for later scaling.
set_params(**params) # Set the parameters of this estimator.
transform(X[, copy]) # Perform standardization by centering and scaling
```



### 鸢尾花分类

同样的，`sklearn.datasets`中有`iris`的数据集：

```python
from sklearn.datasets import load_iris
data = load_iris()
```

数据的使用方法与`linearRegression`部分相同。

主要代码部分：

```python
lm = LogisticRegression(max_iter=2000, solver="sag")
X = data.data
y = data.target
lm.fit(X, y)
```

默认的`max_iter=100`在这里不能达到收敛状态，这里提高到了`2000`。

拟合完之后，查看正确率：

```python
lm.score(X, y)
```

> 0.98



待补充：

- 用`np.meshgrid`画分类结果图



## sklearn中对数据的处理

### 划分训练集/测试集

[API链接](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test#sklearn.model_selection.train_test_split)

```python
X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(train_data,train_target,test_size=0.25)
```



### KFold

```python
sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)

# 用法示例
import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
```



### Cross-Validation

[API链接](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html?highlight=cross_validation)

```python
sklearn.model_selection.cross_validate(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False, return_estimator=False, error_score=nan)

# 用法示例
>>> from sklearn import datasets, linear_model
>>> from sklearn.model_selection import cross_validate
>>> from sklearn.metrics import make_scorer
>>> from sklearn.metrics import confusion_matrix
>>> from sklearn.svm import LinearSVC
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> lasso = linear_model.Lasso()

>>> cv_results = cross_validate(lasso, X, y, cv=3)
>>> sorted(cv_results.keys())
['fit_time', 'score_time', 'test_score']
>>> cv_results['test_score']
array([0.33150734, 0.08022311, 0.03531764])

# 也可以直接得到scores
scores = sklearn.model_selection.cross_val_score(estimator, X, y=None, *, groups=None, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=nan)[source]
```

