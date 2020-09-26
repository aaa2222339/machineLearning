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

