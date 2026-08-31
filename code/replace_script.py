with open("docs/培训Notebook.md", "r", encoding="utf-8") as f:
    orig = f.read()

start_marker = "## 3. NumPy 与 Pandas 底层原理与内存模型"
end_marker = "# 7. Python 现代可视化工具流"

start_idx = orig.find(start_marker)
end_idx = orig.find(end_marker)

new_blk = """## 3. NumPy 与 Pandas 底层原理与内存模型

### 3.1 它们分别解决了数据分析什么痛点？

**NumPy 解决的痛点（纯数值与张量计算）：**
在科学计算场景下，原生 Python 的 `list` 存储机制会导致严重的内存碎片化（频繁指针解引用）和缓存未命中（Cache Miss），同时动态类型解释器导致循环极慢。NumPy 的诞生终结了这五大痛点：解决内存空间碎片化、消除解释器动态类型开销、填补多维张量代数的空白、终结切片的高昂拷贝代价，并打通底层 C/BLAS 和 SIMD 硬件加速生态。

**Pandas 解决的痛点（异构业务数据与关系代数）：**
真实世界的业务数据不是纯数学矩阵，而是包含字符串、时间戳、布尔值的异构表格，常存在缺失值和错位情况。Pandas 解决了：
1. 如何容纳异构数据类型（字符串、数值混合）。
2. 在连接表或对齐时如何依据业务“键（Label）”而非纯“位置（Index）”进行运算。
3. 如何像 SQL 一样使用 GroupBy、Join 等关系代数算子。
4. 如何优雅且不中断程序地处理 NaN 缺失数据。

### 3.2 核心设计哲学

#### 3.2.1 NumPy 的灵魂核心：`ndarray`
`ndarray`（N-Dimensional Array）的设计哲学是：**在动态语言中实现静态语言级别的内存布局。**
* **元数据解耦与视图（View）：** `ndarray` 将数据的 `Header`（包含 shape、dtype、strides）与底层的 `Data Buffer`（连续物理内存）彻底分离。这使得切片、转置、Reshape 都只需要极小开销修改 Header 且不需要复制物理内存。
* **齐次强类型与 SIMD：** 数据块内类型必须一致，使得 CPU 能够跨步寻址（索引 × 大小），并直接投喂给硬件向量化指令。
* **面向数组编程：** 把底层长循环隐式推给 C 语言层面，上层仅做“张量形态”的声明式表达。

#### 3.2.2 Pandas 的根本基石：`Series` 和 `DataFrame`
如果 NumPy 是“同质数学张量”，Pandas 则是“带标签的异构列式数据框”。
* **列式存储架构：** DataFrame 本质是多个一维 `Series` 的字典集合。在底层，同类型的列会被组织为内存连续的块。列式存储使得“求整列均值”等分析操作在物理上读取连续内存，极其高效。
* **标签驱动（Label-Based Alignment）：** 与 NumPy 严格的矩阵位置形状对齐不同，Series 和 DataFrame 的核心哲学在于“业务主键（Index）绑定”。两张表相加时，Pandas 会自动依据相同的 Index 标签进行匹配（如果没有则视为 NaN），哪怕它们数据行数或顺序完全不同。这规避了业务分析中最致命的错位问题。


## 4. NumPy 基础用法与核心设计验证（独立数值演示）

> *注：掌握 NumPy 基础应抛开业务包袱，关注纯粹的数值与内存机制。这里不使用复杂的出租车数据集，而是直接用数组来展示原理。*

### 4.1 数组初始化与随机张量生成
**场景与设计意图**：避免使用 Python List 转换带来极大的内存开销和卡顿，直接让底层 C API 连续分配好所需的物理内存。

```python
import numpy as np
rng = np.random.default_rng(42)

# 正态分布张量与线性分配
data_matrix = rng.normal(loc=0.0, scale=1.0, size=(1000, 1000)).astype(np.float32)
linear_seq = np.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0] 
```

### 4.2 索引机制与“零拷贝”视图 (View vs Copy)
**设计意图验证**：这是 NumPy 最关键也是最容易出 Bug 的机制。普通切片产生 View（视图，不会分配新内存，修改新变量会影响原数组），底层通过修改 Strides（步长）快速跳跃；但是高级（花式/布尔）索引会产生 Copy（迫使分配全新内存块）。

```python
grid = np.arange(12).reshape(3, 4)

# 1. 视图（View）演示（零物理内存分配）
slice_view = grid[0, 1:3]
slice_view[0] = 999 
# 此时 grid 的原值也会被同步修改，因为它们共享同一个 Data Buffer

# 2. 花式索引产生拷贝（Copy）
rows, cols = np.array([0, 2]), np.array([1, 3])
fancy_copy = grid[rows, cols]
fancy_copy[0] = -1 
# grid 保持不变，花式索引因为在内存上不连续，逼迫底层创建了新内存块
```

### 4.3 向量化与原地计算优化
**场景与设计意图**：超大数组做数学运算（如 `a + b + c`），默认总是返回全新建立的数组容器。这里用 `out=` 参数告诉 C 底层直接把结果倾倒进原有的“桶”里。

```python
a = np.ones(1000000, dtype=np.float32)
b = np.ones(1000000, dtype=np.float32)
out_buf = np.empty(1000000, dtype=np.float32) # 高性能抢占好空的物理内存

# 原地计算(In-place)：消除了 a+b 过程中向 OS 反复申请临时内存引发的 GC 抖动
np.add(a, b, out=out_buf)
```


## 5. 实战挑战：纽约出租车数据集的渐进式分析 (Pandas 实操)

> **实战基本题目背景**：我们抽取了 200W 行纽约出游记录（包含经纬度、时间、乘客数、总费用等）。
> **问题目标**：我们要循序渐进地解决 3 个问题：
> 1. 数据如此庞大，如何加载才不会把个人电脑内存吃光？里面如果有缺失值或脏数据（如车费出现完全不符合逻辑的数字），如何高速剔除？
> 2. 高阶规律如何发掘？比如不同出行人数分别平均花多少钱？怎样把另外一张“纽约气象日志”根据时间精确对齐拼接到打车记录上？
> 3. 行程两点的实际地理距离（曼哈顿距离、球面 Haversine 距离）该怎么算？如何将原本需要耗时数分钟的复杂 Python 数学循环优化压缩到 100 毫秒之内？

### 5.1 基础挑战：脏数据清洗与极速装载
**需求场景**：200W 数据一次性读取极耗内存，脏数据（负数车费、莫名其妙的 0 人订单）要过滤。

#### 5.1.1 解决思路一：列式下压 (Downcasting)
默认的 DataFrame 装载表格，所有的数字通通给 64 位（float64, int64），不仅大而且没必要。使用 `astype` 降级，可以压缩超过 50% 内存。

```python
import pandas as pd
df = pd.read_csv("../data/nyc_taxi_2M.csv")

# 观察初始内存用量
print(f"压缩前内存: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 底层思想应用：修改列类型，将所有经纬度和车费改成更轻的 float32，乘客数改成 int8
df['passenger_count'] = df['passenger_count'].astype(np.int8)
for col in ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount']:
    df[col] = df[col].astype(np.float32)
```

#### 5.1.2 解决思路二：向量化布尔掩码与缺失值填充
**设计意图**：在原生的 Python 人脑思维里，往往是用 `for row in data` 来写 if-else 进行过滤。在 Pandas 中，**严禁使用这种逐行迭代**。应当利用 SIMD 并行指令，用布尔向量掩码来批量切掉坏数据。

```python
# 高效保留车费 2~250 之间，且过滤缺失值
mask_valid = (
    df['fare_amount'].between(2.0, 250.0) & 
    (df['passenger_count'] > 0)
)
# 基于掩码切片（这是基于 loc 标签索引的高速访问）
clean_df = df.loc[mask_valid].copy()

# 缺失值填充语义：Pandas 设计将 NaN 看作核心一等公民
clean_df['passenger_count'].fillna(1, inplace=True)
```

### 5.2 高阶挑战：规律挖掘与复杂时空拼接
**需求场景**：单看每一行没意义。我们需要看群体规律，并且尝试跟异构的数据源进行集成组装。

#### 5.2.1 统计学算力：Split-Apply-Combine 模式
**场景与设计意图**：利用底层的 Hash 字典聚合建立极速的分组字典。

```python
# Groupby 分组聚合：快速得出不同乘坐人数的平均车费
fare_mean_by_passenger = clean_df.groupby('passenger_count')['fare_amount'].mean()

# 结合 Transform 算子：算出各自分组平均后，原班不动地拉长塞回原本 200W 行的数据里（避免了手动 Merge 拼表的麻烦）
clean_df['fare_mean'] = clean_df.groupby('passenger_count')['fare_amount'].transform('mean')
```

#### 5.2.2 容差对齐匹配 (Asof Merge)
**场景与设计意图**：我们拿到了一个每 15 分钟跳一次的纽约拥挤度/天气记录表。但打车的时间是随机的（比如 10:04、10:12），传统的 `pd.merge` 基于严丝合缝的等于符号，根本拼不上。此时，我们要利用时间序列“顺差对齐匹配”这个高级哲学解决。

```python
# 转成标准时间类型并进行排序（Asof 的刚性前置条件）
clean_df['pickup_datetime'] = pd.to_datetime(clean_df['pickup_datetime'])
clean_df = clean_df.sort_values('pickup_datetime')

# 编造一段模拟外部气象表
weather_df = pd.DataFrame({
    'time': pd.date_range("2023-01-01", periods=1000, freq="15min"),
    'congestion': np.random.rand(1000)
}).sort_values('time')

# Pandas 高阶用法：前向最近“非精确”匹配合并
merged_df = pd.merge_asof(
    clean_df, weather_df,
    left_on='pickup_datetime', right_on='time',
    direction='nearest',           # 找最近的那个气象记录
    tolerance=pd.Timedelta('10min') # 最多只能容忍 10 分钟内心跳没有记录
)
```

### 5.3 终极性能挑战：地理空间计算极限优化 (邪修法则)
**需求场景**：现在老板需要你计算上述 DataFrame 里这 200 万段行程的实际物理距离（不仅是直线，还要包含复杂的球面曲度 Haversine 距离）。这里每一点计算都牵扯大量 Numpy 三角函数（sin, cos, arcsin）。常规的 Pandas `apply` 或纯 Python 函数写出来速度令人崩溃。怎么通过深度底层技巧“卷出极限”？

#### 5.3.1 “邪修法则第一重”：自建领域专属访问器 (Custom Accessors)
**利用的 Python 特性**：Python 装饰器、对象挂载及魔术方法体系。
**为什么要用**：极高阶的数据团队里，你会发现大量属于特定公司的特有函数。如果随意四处 import 去调用，代码会杂乱无章。自定义访问器（Accessor）直接把复杂的 Numpy 计算打包寄生在了 `DataFrame` 本体身上，使得分析流变成丝滑优美的链式调用语法。

```python
@pd.api.extensions.register_dataframe_accessor("geo")
class TaxiGeoAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def get_manhattan_distance(self):
        # 利用 Pandas 列底层自动推及到 Numpy 张量加速算术的机制
        lat1, lon1 = self._df['pickup_latitude'], self._df['pickup_longitude']
        lat2, lon2 = self._df['dropoff_latitude'], self._df['dropoff_longitude']
        return (np.abs(lat2 - lat1) * 111.0) + (np.abs(lon2 - lon1) * 85.0)

# 优雅的调用方式：
distances = merged_df.geo.get_manhattan_distance()
```

#### 5.3.2 “邪修法则第二重”：Numba JIT 逃逸解释器
**解决的问题**：上面的 Manhattan 还是太简单。如果在 Haversine 这类公式中，存在诸如 `math.sin(dlat / 2.0)**2 + ...` 之类更长串联计算。此时，纯 NumPy 的底层问题暴露无遗：每一步算数都会在后台创建“临时矩阵内存大桶”，使得 200W 这个体量出现可怕的内存开销与 GC 清理停顿，更何况 Numpy 默认是单线程。我们需要绕开解释器。
**核心优化机制**：采用 Numba 的 JIT (Just-In-Time) 编译装饰器，它利用底层 LLVM 编译器直接将这段 Python 函数翻译成极速机器码；同时 `njit(parallel=True)` 将直接打通现代多核 CPU（C++ 级别的 OpenMP），使得巨大的 For 循环飞跃。

```python
import numba as nb
import math

@nb.njit(parallel=True, fastmath=True)
def fast_haversine_distance(lat1, lon1, lat2, lon2):
    n = len(lat1)
    out = np.empty(n, dtype=np.float32)
    # 利用 nb.prange 解锁真正的底层指令多线程加速（绕过 GIL 全局锁）
    for i in nb.prange(n):
        _lat1, _lon1 = math.radians(lat1[i]), math.radians(lon1[i])
        _lat2, _lon2 = math.radians(lat2[i]), math.radians(lon2[i])
        
        dlat = _lat2 - _lat1
        dlon = _lon2 - _lon1
        
        # 这些操作由于 Numba JIT，被直接展开成了原生底层标量指令流算数，不会生成任何额外空间！
        a = math.sin(dlat / 2.0)**2 + math.cos(_lat1) * math.cos(_lat2) * math.sin(dlon / 2.0)**2
        c = 2 * math.asin(math.sqrt(a))
        out[i] = 6371.0 * c # 乘地球半径
        
    return out

# 核心诀窍：务必用 `.values` 剥去 Pandas 外壳，仅仅将底层的 ndarray 即纯 Numpy 内存塞进 Numba！
lats1 = merged_df['pickup_latitude'].values
lons1 = merged_df['pickup_longitude'].values
lats2 = merged_df['dropoff_latitude'].values
lons2 = merged_df['dropoff_longitude'].values

# JIT 机制在首次执行时会有微弱耗时的“预热编译”，但编译后：
# 这多达几百万次的复杂平方根+三角函数求值，耗时可以直接被压缩至 10-50 毫秒数量级！
dists_fast = fast_haversine_distance(lats1, lons1, lats2, lons2)
```"""

new_full = orig[:start_idx] + new_blk + "\n\n" + orig[end_idx:]

with open("docs/培训Notebook.md", "w", encoding="utf-8") as f:
    f.write(new_full)
print("Updated Notebook Successfully.")
