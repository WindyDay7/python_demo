import re

with open("docs/培训Notebook.md", "r", encoding="utf-8") as f:
    content = f.read()

# I will place a marker where section 3 starts and where section 7 starts
start_idx = content.find("## 3. NumPy 与 Pandas 底层原理与内存模型")
end_idx = content.find("# 7. Python 现代可视化工具流")

new_content_block = """## 3. NumPy 与 Pandas 底层原理与内存模型

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

> *注：掌握 NumPy 基础应抛开业务包袱，关注纯粹的数值与内存机制。*

### 4.1 数组初始化与随机张量生成
避免使用 Python List 转换带来开销，直接在底层分配连续内存。

```python
import numpy as np
rng = np.random.default_rng(42)

# 正态分布张量与线性分配
data_matrix = rng.normal(loc=0.0, scale=1.0, size=(1000, 1000)).astype(np.float32)
linear_seq = np.linspace(0, 1, 5) # [0.0, 0.25, 0.5, 0.75, 1.0] 
```

### 4.2 索引机制与“零拷贝”视图 (View vs Copy)
**设计意图验证**：普通切片产生 View（修改新变量会影响原数组），高级（花式/布尔）索引产生 Copy。

```python
grid = np.arange(12).reshape(3, 4)

# 1. 视图（View）演示（零物理内存分配）
slice_view = grid[0, 1:3]
slice_view[0] = 999 
# 此时 grid 的原值也会被修改，因为它们共享 Data Buffer

# 2. 花式索引产生拷贝（Copy）
rows, cols = np.array([0, 2]), np.array([1, 3])
fancy_copy = grid[rows, cols]
fancy_copy[0] = -1 
# grid 保持不变，花式索引逼迫底层创建了新内存块
```

### 4.3 向量化与原地计算优化
```python
a = np.ones(1000000, dtype=np.float32)
b = np.ones(1000000, dtype=np.float32)
out_buf = np.empty(1000000, dtype=np.float32)

# 原地计算(In-place)：消除了 a+b 过程中向 OS 申请临时内存的开销（GC 友好）
np.add(a, b, out=out_buf)
```


## 5. 实战挑战：纽约出租车数据集的渐进式分析 (Pandas 实操)

> **实战基本题目背景**：我们抽取了 200W 行纽约出游记录（包含经纬度、时间、乘客数、总费用等）。
> **目标**：从基础的读取清理，到高阶规律挖掘，最终达成极端耗时情况下的极速运算机制。

### 5.1 基础挑战：脏数据清洗与极速装载
**需求场景**：200W 数据一次性使用 `pd.read_csv` 读取可能撑爆普通电脑内存，大量不合理数据（如车费负数）如何以“一次性向量过滤”而非粗笨 `for` 循环剔除？

#### 5.1.1 设计意图：列式下压 (Downcasting)
默认的 DataFrame 将生成 64 位的浮点和整型，占用巨量内存。使用 `astype` 降级可以压缩超过 50% 内存，契合大数据场景资源珍贵的瓶颈。

```python
import pandas as pd
df = pd.read_csv("../data/nyc_taxi_2M.csv")

# 观察内存用量
print(f"压缩前内存: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 底层思想应用：修改列类型
df['passenger_count'] = df['passenger_count'].astype(np.int8)
for col in ['pickup_longitude', 'pickup_latitude', 'fare_amount']:
    df[col] = df[col].astype(np.float32)
```

#### 5.1.2 向量化布尔掩码与缺失值填充
**设计意图**：利用 SIMD 并行指令批量进行条件判断。绝不要在 DataFrame 使用 `apply` 配合 `if-else`。

```python
# 高效保留车费 2~250 之间，且过滤缺失值
mask_valid = (
    df['fare_amount'].between(2.0, 250.0) & 
    (df['passenger_count'] > 0)
)
# 基于掩码创建分析副本
clean_df = df.loc[mask_valid].copy()

# 缺失值填充语义：Pandas 设计将 NaN 看作一等公民
clean_df['passenger_count'].fillna(1, inplace=True)
```

### 5.2 高阶挑战：规律挖掘与复杂时空拼接
**需求场景**：我们要看各个乘客数组的均费是多少？还要把 10 分钟一次的天气状态和零散打车记录完美拼接起来。

#### 5.2.1 统计学算力：Split-Apply-Combine 模式
**场景与设计意图**：利用底层的 Hash 字典聚合进行 O(M) 分组计算，而非反复查询 DataFrame。

```python
# 分组求车费平均
fare_mean_by_passenger = clean_df.groupby('passenger_count')['fare_amount'].mean()

# 结合 Transform，将算出来的分组特征原班不动地拉伸回去，填充到原表
clean_df['fare_mean'] = clean_df.groupby('passenger_count')['fare_amount'].transform('mean')
```

#### 5.2.2 容差对齐匹配 (Asof Merge)
**场景与设计意图**：两个异构时间源极少能完全“等值”，Pandas 提供了基于有序数据的邻近逼近匹配操作机制，对应时间序列的天然拉链需求。

```python
clean_df['pickup_datetime'] = pd.to_datetime(clean_df['pickup_datetime'])
clean_df = clean_df.sort_values('pickup_datetime')

# 模拟：某外部天气或者拥挤度表
weather_df = pd.DataFrame({
    'time': pd.date_range("2023-01-01", periods=1000, freq="15min"),
    'congestion': np.random.rand(1000)
}).sort_values('time')

# 前向最近匹配合并
merged_df = pd.merge_asof(
    clean_df, weather_df,
    left_on='pickup_datetime', right_on='time',
    direction='nearest', 
    tolerance=pd.Timedelta('10min') # 十分钟内容差匹配
)
```

### 5.3 终极性能挑战：地理空间计算极限优化 (邪修法则)
**需求场景**：根据经纬度，我们需要在每次调用中计算 200W 条行程的曼哈顿距离或者复杂的半正矢曲线距离（Haversine），这需要数百万次三角函数运算，常规方法极其缓慢或者代码极其冗余丑陋，我们需要极致榨干硬件性能与保持代码业务语义的高级组合技。

#### 5.3.1 “邪修法则第一重”：自建领域专属访问器 (Custom Accessors)
**利用的 Python 特性**：Python 的魔术方法和闭包封装。
**为什么要用**：将原本堆填在全局的凌乱长函数，优雅地“挂载”为 DataFrame 对象原生级别方法，极大增加业务脚本的易读性。

```python
@pd.api.extensions.register_dataframe_accessor("geo")
class TaxiGeoAccessor:
    def __init__(self, pandas_obj):
        self._df = pandas_obj

    def get_manhattan_distance(self):
        """完全解耦，随时通过 df.geo.get_manhattan_distance() 优雅调用"""
        lat1, lon1 = self._df['pickup_latitude'], self._df['pickup_longitude']
        lat2, lon2 = self._df['dropoff_latitude'], self._df['dropoff_longitude']
        # 利用 Pandas 底层自动推及到 Numpy 的向量算术
        return (np.abs(lat2 - lat1) * 111.0) + (np.abs(lon2 - lon1) * 85.0)

# 使用方式
distances = merged_df.geo.get_manhattan_distance()
```

#### 5.3.2 “邪修法则第二重”：Numba JIT 逃逸解释器
**解决的问题**：在计算复杂的 Haversine (球面弧长) 距离时，纯 NumPy 会反复申请内存引发抖动（$A^2+B^2 \dots$ 会依次产生极多中间矩阵），且无法充分运用所有 CPU 核心并行。
**底层思想**：使用 Numba （LLVM 技术），在 Python 运行前即时(JIT)将 Python 代码转译为与 C++ 媲美的直接机器码集，利用 `prange` 进行共享内存级别的多核并行。这是真正把动态化降维打击成纯 C 速度的工程利器。

```python
import numba as nb
import math

@nb.njit(parallel=True, fastmath=True)
def fast_haversine_distance(lat1, lon1, lat2, lon2):
    n = len(lat1)
    out = np.empty(n, dtype=np.float32)
    # 利用 nb.prange 解锁极速多线程操作
    for i in nb.prange(n):
        _lat1, _lon1 = math.radians(lat1[i]), math.radians(lon1[i])
        _lat2, _lon2 = math.radians(lat2[i]), math.radians(lon2[i])
        
        dlat = _lat2 - _lat1
        dlon = _lon2 - _lon1
        
        a = math.sin(dlat / 2.0)**2 + math.cos(_lat1) * math.cos(_lat2) * math.sin(dlon / 2.0)**2
        c = 2 * math.asin(math.sqrt(a))
        out[i] = 6371.0 * c
        
    return out

# 提取底层的 Numpy array 进行喂养，剥离 Pandas 的解析开销
lats1 = merged_df['pickup_latitude'].values
lons1 = merged_df['pickup_longitude'].values
lats2 = merged_df['dropoff_latitude'].values
lons2 = merged_df['dropoff_longitude'].values

# JIT 首次执行预热后，200万次复杂的三角函数求值耗时将低于 100 毫秒！
dists_fast = fast_haversine_distance(lats1, lons1, lats2, lons2)
```

"""

new_file_content = content[:start_idx] + new_content_block + "\n\n" + content[end_idx:]

with open("docs/培训Notebook.md", "w", encoding="utf-8") as f:
    f.write(new_file_content)

print("Done")
