# import theano
# import numpy as np
# import keras
# from keras.models import Sequential
# id(numpy.dot == id.numpy.core.multiarray.dot);
# theano.test();
# a = np.array([[1,2],[3,4]])
# sum0 = np.sum(a, axis=0)
# sum1 = np.sum(a, axis=1)
# print(sum0)
# print(sum1)
# print(Sequential.constraints)

# learning pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1.创建对象
# 可以通过传递一个list对象来创建一个Series，pandas会默认创建整型索引
# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

# 通过传递一个numpy array，时间索引以及列标签来创建一个DataFrame
# dates = pd.date_range('20130101', periods = 6)
# print(dates)
# df = pd.DataFrame(np.random.randn(6, 4), index = dates, columns = list('ABCD'))
# print(df)

# 通过传递一个能够被转换成类似序列结构的字典对象来创建一个DataFrame
# df2 = pd.DataFrame({
#     'A': 1,
#     'B': pd.Timestamp('20130102'),
#     'C': pd.Series(1, index = list(range(4)), dtype = 'float32'),
#     'D': np.array([3] * 4, dtype = 'int32'),
#     'E': pd.Categorical(['test', 'train', 'test', 'train']),
#     'F': 'foo'
# })
# print(df2)

# 查看不同列的数据类型
# print(df2.dtypes)

# 如果你使用的是IPython，使用Tab自动补全功能会自动识别所有的属性以及自定义的列，下图中是所有能够被自动识别的属性的一个子集
# df2.<TAB>

# 2.查看数据
# 查看frame中头部和尾部的行
# print(df.head())# 从头部开始查
# print(df.tail(2))# 从尾部开始查

# 显示索引,列,和底层numpy数据
# print(df.index)
# print(df.columns)
# print(df.values)

# 描述显示数据快速统计摘要
# print(df.describe()) #count, mean, std, min, 25%, 50%, 75%, max

# 转置数据
# print(df.T)

# 按轴排序
# print(df.sort_index(axis = 0, ascending = False))
# print(df.sort(columns = 'B'))

# 3.选择器
# 注释: 标准Python / Numpy表达式可以完成这些互动工作, 但在生产代码中, 我们推荐使用优化的pandas数据访问方法, .at, .iat, .loc, .iloc 和 .ix.

# 读取
# 选择单列, 这会产生一个序列, 等价df.A
# print(df['A'])

# 使用[]选择行片断
# print(df[0:3])
# print(df['20130102': '20130104'])

# 使用标签选择
# 使用标签获取横截面
# print(df.loc[dates[0]])

# 使用标签选择多轴
# print(df.loc[:, ['A', 'B']])

# 显示标签切片, 包含两个端点
# print(df.loc['20130102': '20130104', ['A', 'B']])

# 降低返回对象维度
# print(df.loc['20130102', ['A', 'B']])

# 获取标量值
# print(df.loc[dates[0], 'A'])

# 快速访问并获取标量数据 (等价上面的方法)
# print(df.at[dates[0], 'A'])

# 按位置选择
# 传递整数选择位置
# print(df.iloc[3])

# 使用整数片断,效果类似numpy/python
# print(df.iloc[3: 5, 0: 2])

# 使用整数偏移定位列表,效果类似 numpy/python 样式
# print(df.iloc[[1, 2, 4], [0, 2]])

# 显式行切片
# print(df.iloc[1:3, :])

# 显式列切片
# print(df.iloc[:, 1:3])

# 显式获取一个值
# print(df.iloc[1, 1])

# 快速访问一个标量（等同上个方法）
# print(df.iat[1, 1])

# 布尔索引
# 使用单个列的值选择数据
# print(df[df.A > 0])

# where 操作，类似于点乘的逐个点于零比较，小于0的删除，大于零的留下
# print(df[df > 0])

# 使用 isin() 筛选
# df2 = df.copy()
# df2['E'] = ['one', 'two', 'three', 'four', 'five', 'six']
# print(df2[df2['E'].isin(['two', 'four'])])

# 赋值
# 赋值一个新列，通过索引自动对齐数据
# s1 = pd.Series([1, 2, 3, 4, 5, 6], index = pd.date_range('20130102', periods = 6))
# print(s1)
# df['F'] = s1
# print(df)

# 按标签赋值
# df.at[dates[0], 'A'] = 0
# print(df)

# 按位置赋值
# df.iat[0, 1] = 0
# print(df)

# 通过numpy数组分配赋值
# df.loc[:, 'D'] = np.array([5] * len(df)) # np.array([5] * len(df)) = [5 5 5 5 5 5]
# print(df)

# where 操作赋值
# df2 = df.copy()
# df2[df2 > 0] = -df2
# print(df2)

# 丢失的数据
# pandas主要使用np.nan替换丢失的数据. 默认情况下它并不包含在计算中

# 重建索引允许更改/添加/删除指定轴索引,并返回数据副本
# df1 = df.reindex(index = dates[0: 4], columns = list(df.columns) + ['E'])
# df1.loc[dates[0]: dates[1], 'E'] = 1
# print(df1)

# 删除任何有丢失数据的行
# print(df1.dropna(how = 'any'))

# 填充丢失数据
# print(df1.fillna(value = 5))

# 获取值是否nan的布尔标记
# print(pd.isnull(df1))

# 运算
# 统计
# 计算时一般不包括丢失的数据

# 执行描述性统计
# print(df.mean())

# 在其他轴做相同的运算
# print(df.mean(1))

# 用于运算的对象有不同的维度并需要对齐.除此之外，pandas会自动沿着指定维度计算
# s = pd.Series([1, 3, 5, np.nan, 6, 8], index = dates).shift(2)
# print(s)
# print(df)
# print(df.sub(s, axis = 'index')) # sub（Subtraction）减去

# 在数据上使用函数
# print(df)
# print(df.apply(np.cumsum)) #按列逐个叠加
# print(df.apply(lambda x: x.max() - x.min())) # lambda匿名函数，而def是有名称的

# 直方图
# s = pd.Series(np.random.randint(0, 7, size = 10))
# print(s)
# print(s.value_counts()) #频率分析

# 字符串方法
# 序列可以使用一些字符串处理方法很轻易操作数据组中的每个元素,比如以下代码片断。 注意字符匹配方法默认情况下通常使用正则表达式（并且大多数时候都如此）
# s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
# print(s.str.lower()) #小写
# print(s.str.upper())

# 合并
# 连接
# pandas提供各种工具以简便合并序列,数据桢,和组合对象, 在连接/合并类型操作中使用多种类型索引和相关数学函数
# 把pandas对象连接到一起
# df = pd.DataFrame(np.random.randn(10, 4))
# print(df)
# pieces = [df[: 3], df[3: 7], df[7:]]
# print(pd.concat(pieces))

# 连接
# SQL样式合并
# left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
# right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
# print(left)
# print(right)
# print(pd.merge(left, right, on = 'key'))

# 添加
# 添加行到数据增
# df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
# s = df.iloc[3]
# print(df.append(s, ignore_index = True))

# 分组
# 对于“group by”指的是以下一个或多个处理
# 1.将数据按某些标准分割为不同的组
# 2.在每个独立组上应用函数
# 3.组合结果为一个数据结构
# df = pd.DataFrame({
#     'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
#     'B': ['one', 'two', 'three', 'three', 'one', 'two', 'three', 'three'],
#     'C': np.random.randn(8),
#     'D': np.random.randn(8)})
# print(df)

# 分组然后应用函数统计总和存放到结果组
# print(df.groupby('A').sum())

# 按多列分组为层次索引,然后应用函数
# print(df.groupby(['A', 'B']).sum())

# 重塑
# 堆叠
# tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
#                     ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'],
#                     ['1', '2', '3', '4', '5', '6', '7', '8']]))
# print(tuples)
# index = pd.MultiIndex.from_tuples(tuples, names = ['first', 'second', 'third'])
# print(index)
# df = pd.DataFrame(np.random.randn(8, 2), index = index, columns = ['A', 'B'])
# df2 = df[:4]
# print(df)

# 堆叠 函数 “压缩” 数据桢的列一个级别
# print(df2.stack())

# 被“堆叠”数据桢或序列(有多个索引作为索引), 其堆叠的反向操作是未堆栈, 上面的数据默认反堆叠到上一级别
# print(df2.stack().unstack())

# 数据透视表
# df = pd.DataFrame({'A': ['one', 'two', 'three', 'three'] * 3,
#                    'B': ['A', 'B','C'] * 4,
#                    'C': ['foo', 'bar', 'baz', 'qux', 'foo', 'bar', 'baz', 'qux', 'foo', 'bar', 'baz', 'qux'],
#                    'D': np.random.randn(12),
#                    'E': np.random.randn(12)})
# print(df)

# 我们可以从此数据非常容易的产生数据透视表
# df1 = pd.pivot_table(df, values = 'D', index = ['A', 'B'], columns = [])
# print(df1)

# 时间序列
# pandas有易用,强大且高效的函数用于高频数据重采样转换操作(例如,转换秒数据到5分钟数据), 这是很普遍的情况，但并不局限于金融应用
# rng = pd.date_range('1/1/2012', periods = 100, freq = 'S')
# print(rng)
# ts = pd.Series(np.random.randint(0, 500, len(rng)), index = rng)
# print(ts.resample('5Min').sum())

# 时区表示
# rng = pd.date_range('1/1/2012 00:00', periods = 5, freq = 'D')
# ts = pd.Series(np.random.randn(len(rng)), rng)
# print(ts)
# ts_utc = ts.tz_localize('UTC')
# print(ts_utc)

# 转换到其它时区
# ts_utc = ts_utc.tz_convert('US/Eastern')
# print(ts_utc)

# 转换不同的时间跨度
# rng = pd.date_range('1/1/2012', periods = 5, freq = 'M')
# ts = pd.Series(np.random.randn(len(rng)), index = rng)
# print(ts)
# ps = ts.to_period()
# print(ps)
# ps = ps.to_timestamp()
# print(ps)

# 转换时段并且使用一些运算函数, 下例中, 我们转换年报11月到季度结束每日上午9点数据
# prng = pd.date_range('1990Q1', '2000Q4', freq = 'Q-NOV')
# print(prng)
# ts = pd.Series(np.random.randn(len(prng)), prng)
# ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
# print(ts.head())

# 分类
# 自版本0.15起, pandas可以在数据桢中包含分类. 完整的文档
# df = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6], 'raw_grade': ['a', 'b', 'b', 'a', 'a', 'c']})
# 转换原始类别为分类数据类型
# df['grade'] = df['raw_grade'].astype('category')
# print(df['grade'])

# 重命令分类为更有意义的名称 (分配到Series.cat.categories对应位置!)
# df['grade'].cat.categories = ['very good', 'good', 'very bad']
# print(df['grade'])

# 重排顺分类,同时添加缺少的分类(序列 .cat方法下返回新默认序列)
# df['grade'] = df['grade'].cat.set_categories(['very bad', 'bad', 'medium', 'good', 'very good'])
# print(df['grade'])

# 排列分类中的顺序,不是按词汇排列
# print(df.sort('grade'))

# 类别列分组,并且也显示空类别
# print(df.groupby('grade').size())

# 绘图
ts = pd.Series(np.random.randn(1000), index = pd.date_range('1/1/2000', periods = 1000))
ts = ts.cumsum()
ts.plot()
