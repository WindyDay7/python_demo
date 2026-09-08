# Python 数据数学分析核心课件

## 引言:从工程代码回归数据本质

### 为什么是 Python? 

在传统的软件后端开发中, 关注点通常是 **状态管理、网络 I/O、并发安全与面向对象的解耦抽象**. 然而, 当进入数据分析与科学计算领域时, 思维模型需要发生根本性转变:

```
[传统后端工程思维]                     [现代数据分析/矩阵思维]
对象封装(OOP)                    ->   数据与操作分离(Data-Oriented Design)
逐个遍历(for item in list)       ->   批量向量化计算(SIMD / Tensor Processing)
复杂控制流(if-else / 状态机)      ->   布尔掩码与索引选择(Mask & Fancy Indexing)
显式多线程/锁竞争                 ->   底层 C/Fortran/BLAS 内存连续并行计算
```

Python 之所以能够统治现代数据科学与 AI 领域, 并非因为其解释器运行效率高(相反, CPython 的解释器开销与 GIL 限制众所周知), 而是因为它的 **"胶水特性" 与 C-API 扩展能力**:

1. 上层提供极具表现力的动态语法和交互式 REPL 环境;
2. 下层无缝对接 OpenBLAS、MKL、CUDA 等底层硬件加速库;
3. 形成了从数据摄取(Pandas/Arrow)、处理(NumPy/SciPy)、可视化(Matplotlib/Seaborn)到模型训练(PyTorch/Scikit-learn)的完整闭环. 

> 为什么 Python 明明什么都能算, 我们还需要 NumPy 和 Pandas? 

> NumPy 和 Pandas 真正带来的价值, 是“多了一堆 API”, 还是“改变了数据和计算的抽象方式”? 

### 我们真正要学的不是 API

假设现在拿到 200 万条出租车订单. 

我们需要分析:

```text
1. 哪些订单不可信? 
2. 什么时候最忙? 
3. 什么时候流水最高? 
4. 哪些区域最热门? 
5. 哪些订单看起来异常? 
6. 为什么同一个计算, 有人的代码跑几十秒, 有人的代码只跑几秒? 
7. 怎么把这些分析变成公司里以后还能复用的工具? 
```

当然, 可以全部使用 Python:

```python
for row in rows:
    ...
```

但问题并不在于 Python **能不能做**. 

真正的问题是:

> **Python 原生的数据结构和逐对象执行模型, 并不是专门针对百万级同构数值计算和二维表格分析设计的.**

但是 Numpy 和 Pandas 是 Python 中专门为数据分析设计的

1. NumPy 的核心是同构 $N$ 维数组 `ndarray`; 官方文档将其定义为 NumPy 的核心多维数组结构, 并围绕数组提供高效的数学、逻辑、选择、排序、线性代数等运算. 
2. Pandas 则在数组之上增加了 **标签、索引、缺失值、异构列、自动对齐、分组、连接和时间序列** 等更贴近表格数据分析的抽象; `Index` 本身就是 Pandas 用于索引和对齐的轴标签对象. 

| 从哪里来                | 核心技术                                         | 关键机制                      | 最终走向        |
|:---:|:---:|:---:|:---:|
| **Python 原生对象**     | `list` / `dict` / Object                     | 灵活, 但对象开销大、逐元素处理           | ↓           |
| **NumPy**           | `ndarray`                                    | `dtype`、`shape`、`strides` | **高效数值计算**  |
| **NumPy 计算模型**      | UFunc / Vectorization / Broadcasting         | 批量计算、底层循环、形状扩展            | **性能提升**    |
| **Pandas**          | Series / DataFrame                           | Index、Alignment           | **结构化数据分析** |
| **Pandas 数据操作**     | `groupby` / `merge` / `resample` / `rolling` | 聚合、关联、时间序列、窗口计算           | **复杂数据处理**  |
| **业务分析**            | 数据清洗 → 分析 → 建模                               | 将工具应用于真实数据                | **业务结论**    |
| **性能工程**            | 内存、向量化、拷贝、计算瓶颈                               | 针对大规模数据优化                 | **更快、更省内存** |
| **领域工具 / Pipeline** | 专用工具、Accessor、Pipeline                       | 封装通用能力                    | **工程化应用**   |


一句话概括:
> **NumPy 解决“如何高效计算数组”; Pandas 解决“如何带着业务语义高效组织、查询、组合和分析表格数据”.**

### 出租车数据集简介

NYC TLC Yellow Taxi Trip Records 是纽约市出租车与豪华轿车委员会(NYC Taxi & Limousine Commission, TLC)公开发布的纽约市黄色出租车行程记录数据.

| 类别        | 字段                      | 含义                         | 典型值/单位                |
|:---:|:---:|:---:|:---:|
| **记录信息**  | `VendorID`              | 提供出租车计价/数据记录服务的技术供应商编号     | `1`, `2`, `6`, `7`    |
| **时间**    | `tpep_pickup_datetime`  | 计价器开始计费的时间, 即上车时间           | `2023-01-01 00:15:00` |
|           | `tpep_dropoff_datetime` | 计价器停止计费的时间, 即下车时间           | `2023-01-01 00:25:00` |
| **乘客/行程** | `passenger_count`       | 乘客人数                       | `1`, `2`, `3`...      |
|           | `trip_distance`         | 计价器记录的行程距离                 | 英里 mile               |
| **费率**    | `RatecodeID`            | 本次行程最终使用的费率类型              | `1`, `2`, `3`...      |
| **数据传输**  | `store_and_fwd_flag`    | 行程数据是否曾暂存在车载设备中后再上传        | `Y` / `N`             |
| **空间位置**  | `PULocationID`          | 上车所在的 Taxi Zone            | Zone ID               |
|           | `DOLocationID`          | 下车所在的 Taxi Zone            | Zone ID               |
| **支付**    | `payment_type`          | 支付方式                       | 信用卡、现金等               |
| **费用**    | `fare_amount`           | 计价器计算出的基础车费                | 美元                    |
|           | `extra`                 | 额外费用/附加费                   | 美元                    |
|           | `mta_tax`               | MTA 税费                     | 美元                    |
|           | `tip_amount`            | 小费金额                       | 美元                    |
|           | `tolls_amount`          | 行程产生的过路费                   | 美元                    |
|           | `improvement_surcharge` | Taxi Improvement Surcharge | 美元                    |
|           | `total_amount`          | 向乘客收取的总金额                  | 美元                    |
|           | `congestion_surcharge`  | 拥堵附加费                      | 美元                    |
|           | `airport_fee`           | JFK / LaGuardia 机场相关费用     | 美元                    |
|           | `cbd_congestion_fee`    | CBD 拥堵费, 2025 年起出现          | 美元                    |


#### 数据的格式

| 特性维度 | CSV | Parquet |
|:---:|:---:|:---:|
| **存储结构** | 行式存储(Row-oriented, 纯文本) | 列式存储(Columnar, 二进制) |
| **文件体积** | 较大(无内置压缩, 字符冗余高) | 极小(内置 Snappy/ZSTD 等高效压缩与字典编码) |
| **读取性能** | 较慢(需逐行扫描、切分字符串并推断类型) | 极快(支持列裁剪与分区下推) |
| **Schema & 类型** | 弱类型(无元数据, 需解析器自行推断) | 强类型(内嵌 Schema、数据类型及统计信息) |
| **可读性** | 极高(文本编辑器、Excel 直接打开修改) | 无法直接阅读(需专用解析引擎) |
| **I/O 优化** | 必须全表/全行扫描 | 仅读取查询所需的列和数据块(Row Groups) |


### 贯穿本节的七个问题

| 问题 | 业务问题 | 逐步引出的技术 |
|:---:|:---:|:---:|
| **Q1** | 这 200 万条数据可靠吗?  | `dtype`、缺失值、mask、`loc`、`query` |
| **Q2** | 一天中什么时候最忙?  | `datetime`、`groupby`、`agg` |
| **Q3** | 什么时间段的流水和单位运营时间收入最高?  | 向量化、`assign`、`groupby`、派生指标 |
| **Q4** | 哪些上下车区域最热门?  | `value_counts`、`merge`、连接校验 |
| **Q5** | 哪些订单和时间窗口可能异常?  | `where`、`select`、`quantile`、`transform`、`resample`、`rolling` |
| **Q6** | 为什么同一个分析可以相差一个数量级甚至更多?  | `Python loop`、`apply`、`NumPy`、`Numba`、`eval/query`、内存优化 |
| **Q7** | 如何把分析脚本变成公司可以重复使用的工具?  | `pipe`、纯函数、`Accessor`、领域 API |

### NumPy / Pandas 为什么存在

#### 问题

假设我们有 200 万个金额数据:

```python
prices = [...]
```

需要:

```text
全部上涨 5%
再增加 3 元服务费
过滤异常价格
求平均值
```

#### 直接使用 Python List 实现

```python
# 直接使用 List 存储与计算
result = []

for price in prices:
    result.append(price * 1.05 + 3)
```

问题是我们把 **200 万次循环控制** 交给了 Python 层. 具体来说, Python 原生 `List` 在这种大批量数值计算场景下存在两大明显短板:

**1. 存储方面**

Python `List` 并不是一块连续的数值缓冲区, 而是一个 **对象引用数组**. 它存储的每一个元素都是一个独立的 Python 对象(这里是 `float`), 每个对象都带有引用计数、类型指针等元信息. 以 200 万个金额为例:

```python
prices = [42.5, 17.3, 58.9, ...]   # 200 万个 float 对象
```

- `List` 本身只存 200 万个**指针**, 每个指针 8 字节(64 位系统); 
- 每一个 `float` 对象本身还要额外占用约 24 字节(CPython 对象头); 
- 于是 `List` 存储 200 万个浮点数的实际内存, 大约是相同规模 `ndarray` 的 **3~5 倍**.

```text
List:
  指针数组(8B × 200万) + 每个 float 对象(24B × 200万)
  ≈ 16 MB + 48 MB = 64 MB 左右

ndarray(float64):
  连续缓冲区(8B × 200万)
  ≈ 16 MB
```

内存翻倍, 还意味着缓存命中率更低、CPU 需要跨越更多随机内存地址.

**2. 计算方面**

```python
result = []
for price in prices:
    result.append(price * 1.05 + 3)
```

这段循环的执行, 每一步都发生在 **Python 解释器层**, 而不是底层机器码层:

- 每次 `for` 迭代都要经历字节码分发(bytecode dispatch); 
- 每次读取 `price` 都要做一次 **拆箱**(unboxing, 从 Python 对象还原成 C double); 
- 每次 `price * 1.05 + 3` 结果要重新 **装箱**(boxing)成新的 `float` 对象; 
- 每次 `append` 还要调用方法、检查容量、可能触发列表扩容与内存重分配.

当循环次数上升到 200 万, 这些**解释器级开销被放大 200 万倍**, 成为无可忽视的性能瓶颈. 这也是为什么同样的计算, 纯 Python 循环往往比 NumPy 向量化写法慢一到两个数量级——**真正慢的不是"算", 而是"每次都要让解释器来驱动"**.

#### 使用 Numpy 和 Pandas 实现

NumPy:
```python
result = prices * 1.05 + 3
```

Pandas:

```python
result = (
    df
    .query("fare_amount > 0")
    .groupby("pickup_hour")
    .agg(avg_fare=("fare_amount", "mean"))
)
```

### Numpy 和 Pandas 的特点

**NumPy 解决的痛点(纯数值与张量计算):**
在科学计算场景下, 原生 Python 的 `list` 存储机制会导致严重的内存碎片化(频繁指针解引用)和缓存未命中(Cache Miss), 同时动态类型解释器导致循环极慢. NumPy 的诞生终结了这五大痛点: 解决内存空间碎片化、消除解释器动态类型开销、填补多维张量代数的空白、终结切片的高昂拷贝代价, 并打通底层 C/BLAS 和 SIMD 硬件加速生态.

**Pandas 解决的痛点(异构业务数据与关系代数):**
真实世界的业务数据不是纯数学矩阵, 而是包含字符串、时间戳、布尔值的异构表格, 常存在缺失值和错位情况.Pandas 解决了:
1. 如何容纳异构数据类型(字符串、数值混合).
2. 在连接表或对齐时如何依据业务“键(Label)”而非纯“位置(Index)”进行运算.
3. 如何像 SQL 一样使用 GroupBy、Join 等关系代数算子.
4. 如何优雅且不中断程序地处理 NaN 缺失数据.

#### 核心设计哲学

##### NumPy 的灵魂核心:`ndarray`(N-dimensional array)
`ndarray`(N-Dimensional Array) 的本质是“同构多维连续内存缓冲区的视图抽象(N-dimensional Array / Tensor)”, 矩阵(Matrix)是它在 2 维空间下的一个特例. 它的设计哲学是:**在动态语言中实现静态语言级别的内存布局.**

* **元数据解耦与视图(View):** `ndarray` 将数据的 `Header`(包含 shape、dtype、strides)与底层的 `Data Buffer`(连续物理内存)彻底分离. 这使得切片、转置、Reshape 都只需要极小开销修改 Header 且不需要复制物理内存.
* **齐次强类型与 SIMD:** 数据块内类型必须一致, 使得 CPU 能够跨步寻址(索引 × 大小), 并直接投喂给硬件向量化指令.
* **面向数组编程:** 把底层长循环隐式推给 C 语言层面, 上层仅做“张量形态”的声明式表达.

##### Pandas 的根本基石:`Series` 和 `DataFrame`
如果 NumPy 是“同质数学张量”, Pandas 则是“带标签的异构列式数据框”.
* **列式存储架构:** DataFrame 本质是多个一维 `Series` 的字典集合. 在底层, 同类型的列会被组织为内存连续的块. 列式存储使得“求整列均值”等分析操作在物理上读取连续内存, 极其高效.
* **标签驱动(Label-Based Alignment):** 与 NumPy 严格的矩阵位置形状对齐不同, Series 和 DataFrame 的核心哲学在于“业务主键(Index)绑定”. 两张表相加时, Pandas 会自动依据相同的 Index 标签进行匹配(如果没有则视为 NaN), 哪怕它们数据行数或顺序完全不同. 这规避了业务分析中最致命的错位问题.

## 课程环境与初始化

### Numpy 与 Pandas 的版本

本课程 Numpy 与 Pandas 需要安装与配套的版本如下:

| Package    | Version |
|:---:|:---:|
| matplotlib |   3.9.4 |
| numba      |  0.60.0 |
| numpy      |   2.0.2 |
| pandas     |   2.3.3 |
| pyarrow    |  21.0.0 |
| jupyterlab |  4.5.10 |

## NumPy:真正理解 ndarray

通过这部分希望大家可以理解下面的问题:

```text
ndarray ≠ 更快的 list

ndarray =
    data buffer
    + dtype
    + shape
    + strides
```
能够解释:
                               
- `dtype`
- `shape`
- `ndim`
- `size`
- `itemsize`
- `nbytes`
- `strides`
- `flags`
- `view`
- `copy`
- `reshape`
- `transpose`
- `axis`

### ndarray 是什么

ndarray 是 Numpy 中的一个基础类型, 它底层的简化定义如下:

```C
typedef struct {
    PyObject_HEAD

    /* Number of dimensions */
    int nd;
    /* Shape of the array */
    npy_intp *dimensions;
    /* Strides of the array */
    npy_intp *strides;
    /* Pointer to the actual data */
    void *data;
    /* Data type descriptor */
    PyArray_Descr *descr;
    /* Base object, used for views */
    PyObject *base;
    /* Flags */
    int flags;
    /* Weak references */
    PyObject *weakreflist;
    /* ... other internal fields ... */

} PyArrayObject;
```

NumPy 官方 `ndarray` 文档把 `shape`、`strides`、`ndim`、`data`、`itemsize`、`nbytes`、`base`、`dtype` 等直接列为数组(ndarray)的核心属性; `strides` 表示沿每个维度前进一步需要跨过的字节数. 

| `ndarray` 核心组成  | 对应概念  | 含义                                    |
|:---:|:---:|:---:|
| **data buffer** | 数据缓冲区 | **真正存储数据的内存区域**                       |
| **dtype**       | 数据类型  | 描述**一个元素应该怎样解释**, 例如 `int64`、`float32` |
| **shape**       | 形状    | 描述数组的**逻辑维度**, 例如 `(3, 4)` 表示 3 行 4 列  |
| **strides**     | 步长    | 描述**逻辑坐标如何映射到实际内存地址**                 |

所以 `ndarray` 本质上可以理解为: 一块数据内存(Data Buffer) + 一组描述这块内存如何被解释和访问的元数据(dtype、shape、strides). 

```text
Python list

┌─────┐
│ ptr │──→ Python object
├─────┤
│ ptr │──→ Python object
├─────┤
│ ptr │──→ Python object
└─────┘

NumPy ndarray

metadata:
shape   = (4,)
dtype   = float64
strides = (8,)

data buffer:
┌────────┬────────┬────────┬────────┐
│ 1.2    │ 3.4    │ 5.6    │ 7.8    │
└────────┴────────┴────────┴────────┘
```

#### 新建一个 ndarray

我们创建一个最基本的 ndarray 如下: 

```python
a = np.array([1, 2, 3, 4], dtype=np.int32)

print("array   :", a)
print("dtype   :", a.dtype)
print("shape   :", a.shape)
print("ndim    :", a.ndim)
print("size    :", a.size)
print("itemsize:", a.itemsize)
print("nbytes  :", a.nbytes)
print("strides :", a.strides)
```

Numpy 内置了很多初始化函数: 

```python
m = np.arange(12, dtype=np.int32).reshape(3, 4)

print("shape   :", m.shape)
print("dtype   :", m.dtype)
print("strides :", m.strides)
print("flags:")
print(m.flags)
```

对于上面的 C-order `int32` 数组, 一个元素 4 bytes:

```text
shape = (3, 4)

内存:

0 1 2 3 | 4 5 6 7 | 8 9 10 11

列方向移动一步:4 bytes
行方向移动一步:4 × 4 = 16 bytes

strides = (16, 4)
```

常见创建函数:

```python
examples = {
    "array": np.array([1, 2, 3]),
    "arange": np.arange(0, 10, 2),
    "linspace": np.linspace(0, 1, 5),
    "zeros": np.zeros((2, 3)),
    "ones": np.ones((2, 3)),
    "empty": np.empty((2, 3)),
}

examples
```

#### ndarray 的内存占用

我们用一个简单的例子介绍 ndarray 在存储数值上对比 `List` 的优势, 假设我们比较一百万个整数:

```python
python_list = list(range(1_000_000))
numpy_array = np.arange(1_000_000, dtype=np.int64)

python_approx_bytes = (
    sys.getsizeof(python_list)
    + sum(sys.getsizeof(x) for x in python_list)
)

print(
    "Python list 近似占用:",
    f"{python_approx_bytes / 1024**2:,.2f} MB"
)

print(
    "NumPy 元素缓冲区:",
    f"{numpy_array.nbytes / 1024**2:,.2f} MB"
)
```

这里 Python 的统计是 CPython 对象大小近似, 而 `ndarray.nbytes` 统计的是元素缓冲区本身, 两者并不是完全相同口径; 这个实验的重点是理解:

> **Python list 存的是一组 Python 对象引用, 而 NumPy 可以把同类型数值紧密存储.**

但是不要为了“NumPy 看起来高级”把所有东西都放进 NumPy. 

例如:

```python
np.array(
    [
        {"name": "Alice"},
        {"name": "Bob"},
    ],
    dtype=object,
)
```

此时底层仍然绕不开 Python object, 大量 NumPy 数值计算优势会消失. 

#### ndarray 的维度 Axis 

在 ndarray 中, **`axis`(轴)的本质是多维数组维度的“索引编号”与“遍历方向”**. 它定义了多维数据在逻辑结构中的组织层级, 以及算子在内存中沿哪个方向进行遍历或折叠.

**1. `axis` 的本质**

- **维度的编号:** 一个 $N$ 维数组拥有从 `0` 到 $N-1$ 的轴(支持负数索引, `-1` 代表最后一维).
- **与 `shape` 的直接映射:** `ndarray.shape` 为一个元组 $(d_0, d_1, \dots, d_{N-1})$, 其中 `axis=i` 对应的就是维度大小 $d_i$.
- **与嵌套列表层级的映射:** 最外层的中括号对应 `axis=0`, 每往里深入一层括号, `axis` 编号加 1.
- **底层内存步长(Strides)的具象化:** 在默认的 C-order 连续内存中, `axis=0` 跨度最大(跳过一整个切片/行), `axis=-1` 跨度最小(内存地址连续相邻).

**2. 指定 `axis` 到底意味着什么?**

理解 `axis` 计算最关键的一句话:**“沿着指定的 `axis` 方向移动并折叠, 消除该维度”**.

**3. `axis` 的核心作用**

- **统计与聚合(Reduction):** 控制 `sum`、`mean`、`max`、`std` 等函数的计算方向, 决定保留哪些维度、压缩哪些维度.
- **拼接与扩展(Concatenation & Stacking):**
  - `np.concatenate([a, b], axis=0)`:沿垂直方向拼接(增加行数).
  - `np.concatenate([a, b], axis=1)`:沿水平方向拼接(增加列数).
- **排序与累加(Sorting & Scanning):** 如 `np.sort(arr, axis=1)` 仅对每一行内部的元素排序, 行与行之间互不干扰.
- **维度转换与重排(Transposition):** 在图像处理(如深度学习中 `(H, W, C)` 转 `(C, H, W)`)中, 通过 `np.transpose(arr, axes=(2, 0, 1))` 重新排列各轴顺序.

#### View、Copy 与 Strides

在 ndarray 中, **`Strides`(步长)是连接高维逻辑坐标与底层一维内存的“寻址导航器”, 而 `View`(视图)与 `Copy`(副本)是由此衍生的两种内存管理策略**. NumPy 之所以能在处理海量数据时保持极高的计算性能, 本质就在于将“数据实体”与“元数据(Metadata)”彻底解耦.

**为什么必须放在一起讲?**

`View` 的物理实现完全依赖于 `Strides` 的重新映射; 掌握了 `Strides` 的线性寻址逻辑, 就能一眼看穿为什么基础切片可以实现零拷贝($O(1)$ 复杂度), 而花式索引却必须退化为内存深拷贝($O(N)$ 复杂度).

**1. 本质**

- **双层架构解耦:** `ndarray` 由两部分构成: 保存纯字节流的**连续数据缓冲区(Data Buffer)\**与保存描述信息的\**元数据头(Metadata Header)**(包含 `shape`、`dtype`、`strides`、`data pointer`、`base` 等).
- **`Strides`(步长)的本质:** 一个元组 $(s_0, s_1, \dots, s_{N-1})$, 表示在内存字节流中沿 `axis=i` 移动一个索引单位需要跨越的**物理字节数(Bytes)**.
- **`View`(视图)的本质:** 创建了一个全新的元数据头, 其 `data pointer` 仍指向**同一个底层内存缓冲区**. 创建时间复杂度为 $O(1)$, 对视图的数据修改会直接同步到原数组(产生副作用).
- **`Copy`(副本)的本质:** 在堆内存中开辟了一块全新的独立数据缓冲区, 并将数据完整复制过去. 时间复杂度为 $O(N)$, 与原数组在物理内存上完全隔离.

**2. 原理**

- **内存线性寻址公式:** 任意多维逻辑索引 $(i_0, i_1, \dots, i_{N-1})$ 对应的底层字节偏移量为:

  $$\text{Byte Offset} = \sum_{k=0}^{N-1} (i_k \times s_k)$$

- **View vs Copy 的判定准则:**

  - **产生 View:** 只要操作后的新数据子集能够通过**单组等差步长(Strides)与基地址偏移**完全表达(如切片 `arr[::2]`、转置 `arr.T`、维度压缩 `arr.squeeze()`), NumPy 就仅修改元数据返回 View.
  - **强制 Copy:** 一旦目标元素的内存分布**无法用固定的线性步长表达**(如花式索引 `arr[[0, 2]]`、布尔掩码 `arr[arr > 5]`), 或显式调用 `.copy()`, NumPy 必须分配新内存并产生 Copy.

**3. 核心作用**

- **极速性能(Zero-Copy):** 切片、翻转、转置等操作耗时与数组大小完全无关(均在纳秒级完成), 避免大规模内存申请与数据搬运.
- **内存安全控制:** 显式调用 `.copy()` 切断与原数组的指针绑定, 防止多线程计算或数据预处理流水线中的意外内存污染.
- **内存连续性(Contiguity)管理:** ndarray 的数据是否按照 C 语言/行优先(row-major)的方式连续存储, 确保调用底层 BLAS/LAPACK 或 C/C++ 扩展时的 SIMD 向量化加速效率.
- **高级数据重组(Strided Tricks):** 通过手动操控 Strides, 无需分配额外内存即可实现图像滑动窗口(Sliding Window)、重叠切片或卷积核展开.


### ndarray 的用法

#### 广播机制(Broadcasting)

在 ndarray 的算术运算中, **`Broadcasting`(广播机制)是 NumPy 在不复制数据的前提下, 自动拉伸和对齐不同形状数组的“隐式虚拟扩展引擎”**. 它是向量化计算(Vectorization)的核心支柱, 让具有不同形状的张量能够在底层 C 语言级别无缝进行逐元素(Element-wise)运算.

**1. 本质**

- **零内存消耗的虚拟扩展:** 广播并不是在物理内存中真实复制数据填充数组, 而是通过**将对应拉伸维度的步长(Strides)设为 `0`**. 步长为 0 意味着逻辑索引在递增, 但底层指针停留在同一物理内存地址上, 实现 $O(1)$ 内存开销的“逻辑克隆”.
- **双重规约机制:** 广播定义了严格的维度靠右对齐与长度兼容法则, 将多维数组间的算术运算、比较运算和赋值操作彻底标准化.

**2. 原理(对齐双法则)**

广播在底层严格按照以下两个阶段执行判断与形状推导:

- **法则一: 维度补齐(靠右对齐, 左侧补 1)**
  - 如果两个数组的维度数(`ndim`)不同, NumPy 会在维度较少的数组 `shape` **左侧补 1**, 直到两者维度数量一致.
  - *例:* `shape=(3,)` 与 `shape=(4, 3)` 对齐时, 前者先被补齐为 `(1, 3)`.
- **法则二: 长度兼容(等于自身或等于 1)**
  - 从最后一个维度(Trailing Dimension, 即最右侧轴)开始向前逐轴比对. 对于每个轴, 两者的长度必须满足以下条件之一, 否则抛出 `ValueError`:
    1. 两个维度的长度相等; 
    2. 其中一个维度的长度为 `1`(该维度会被虚拟拉伸到与另一个维度相同).
- **输出形状判定:** 最终输出结果在每个轴上的长度, 等于参与运算各数组在该轴长度的**最大值**: $\max(d_{1,i}, d_{2,i})$.

**3. 核心作用**

- **消除显式 Python 循环:** 避免书写多重嵌套循环, 将计算下沉至底层 C 连续遍历, 极大激发 CPU 的 SIMD 向量化指令集性能.
- **内存极致优化:** 彻底淘汰 `np.tile` 或 `np.repeat` 等物理深拷贝复制数据的低效做法.
- **高维特征与批处理运算:** 在机器学习与图像处理中, 轻松实现全局特征去中心化、样本按权重缩放、网格生成(Meshgrid)以及成对距离矩阵(Pairwise Distance)计算.

#### Vectorization 与 UFunc(通用函数)

Vectorization(向量化)是 NumPy 高效计算的编程思想/使用方式, 而 UFunc(通用函数)是 ndarray 实现高效逐元素计算的核心机制之一. 它们结合“连续内存 + 固定 dtype + C 层循环 + CPU 向量化/SIMD”等一整套机制共同构建了 Numpy 的高性能计算体系. **`Vectorization`(向量化)是将原本由 Python 解释器执行的逐元素标量循环, 下沉至底层 C 语言连续内存块与 CPU SIMD 硬件指令的“去循环执行模式”; 而 `UFunc`(Universal Function, 通用函数)则是封装了这种高效 C 语言内核、并自带广播与高阶聚合方法的“可调用对象引擎”**. 

**1. 本质**

* **消除解释器开销(Bypassing Interpreter Overhead):** 摆脱 Python 在每次循环中反复进行的动态类型检查(Dynamic Type Resolution)、对象封包解包(Boxing/Unboxing)以及引用计数维护, 将计算交由预编译的 C 循环指针递增执行.
* **UFunc 对象的物理本质:** 一个内建了多重类型分发表(Type Dispatch Table)与连续内存迭代器(`NpyIter`)的 C 结构体对象(`PyUFuncObject`), 它同时管理着一元算子(如 `sin`、`exp`)与二元算子(如 `add`、`multiply`).
* **硬件级并行(SIMD 并发):** 向量化使连续内存数据能够直接加载进现代 CPU 的向量寄存器(如 AVX-512、AVX2、ARM NEON), 实现单指令周期内对多个浮点数的并发吞吐(SIMD, Single Instruction Multiple Data).


**2. 原理**

* **类型分发(Type Resolution):** 传入数组时, UFunc 根据输入的 `dtype` 匹配最精确的底层 C 函数指针(如 `int64` 走整数加法指令, `float64` 走双精度浮点流水线).
* **天然嵌入广播:** 任何多元 UFunc 都会自动调用广播协议, 无需额外处理维度即可自动扩展输入形状.
* **方法派生矩阵(UFunc Methods):** 任何**二元通用函数**(Binary UFunc, 如 `np.add`、`np.multiply`、`np.maximum` 等)都自动派生出 5 个高阶方法, 直接将简单的逐元素算子转化为强大的矩阵与序列计算引擎:
1. `.reduce()`:沿指定轴折叠压扁(累加/累乘/最值筛选); 
2. `.accumulate()`:沿指定轴保存中间累积状态(前缀和/前缀积); 
3. `.outer()`:两输入做笛卡尔积全组合外积运算; 
4. `.at()`:无缓冲的就地离散索引更新(支持重复索引累加); 
5. `.reduceat()`:按指定的索引切片边界分段折叠汇总.


**3. 核心作用**

* **极速性能吞吐:** 通常比原生 Python 的 `for` 循环与列表推导式带来 50 至 500 倍的性能提升.
* **高阶函数式表达:** 用 `.outer()` 或 `.reduce()` 替代复杂的多层循环与动态列表收集, 大幅精简工程代码.
* **解决原地累加数据冲突(Race Hazard):** `ufunc.at` 解决了常规花式索引(Fancy Indexing)在遇到重复索引时因写入缓冲(Buffering)导致累加丢失的致命问题.
* **灵活的接口封装:** 借助 `np.vectorize` 与 `np.frompyfunc`, 可将任意纯 Python 业务函数快速封装为兼容广播特性的伪向量化对象.

##  Pandas: 

我通过前面抛出的几个问题来解释 Pandas 的实际用法.

| 问题 | 业务问题 | 逐步引出的技术 |
|:---:|:---:|:---:|
| **Q1** | 这 200 万条数据可靠吗?  | `dtype`、缺失值、mask、`loc`、`query` |
| **Q2** | 一天中什么时候最忙?  | `datetime`、`groupby`、`agg` |
| **Q3** | 什么时间段的流水和单位运营时间收入最高?  | 向量化、`assign`、`groupby`、派生指标 |
| **Q4** | 哪些上下车区域最热门?  | `value_counts`、`merge`、连接校验 |
| **Q5** | 哪些订单和时间窗口可能异常?  | `where`、`select`、`quantile`、`transform`、`resample`、`rolling` |
| **Q6** | 为什么同一个分析可以相差一个数量级甚至更多?  | `Python loop`、`apply`、`NumPy`、`Numba`、`eval/query`、内存优化 |
| **Q7** | 如何把分析脚本变成公司可以重复使用的工具?  | `pipe`、纯函数、`Accessor`、领域 API |

基于问题 Q1-Q6 介绍 Pandas 的核心设计思想与用法.

### Pandas 的设计哲学

#### Series、DataFrame 和 Index

从 NumPy 到 Pandas 的设计思想, 其实是在回答一个非常现实的问题:

> **当数据不再只是“一坨数字”, 而是带着字段名、业务主键、缺失值、时间戳和连接关系的表格时, 我们还能够继续只靠 `ndarray` 吗?**

答案通常是不够.

NumPy 擅长的是 **同构数值数组** 的高性能计算; 而 Pandas 擅长的是 **带业务语义的异构表格数据** 的组织、筛选、对齐、聚合、连接与时间序列分析.

也就是说:

```text
NumPy 关心的是:
这块连续内存怎样更快地算? 

Pandas 关心的是:
这张表里的哪一列代表时间? 
哪一列代表金额? 
哪些行缺失? 
两张表如何按主键对齐? 
分组后如何再把结果贴回原表? 
```

这就是 Pandas 的设计出发点:

> **在 NumPy 的高性能数组之上, 增加标签、索引、缺失值语义、关系代数和时间序列能力, 让“数据分析代码”既能跑得够快, 又能写得像业务逻辑.**

#### Pandas 的三块根基: Series、DataFrame、Index

##### 1. `Series` 的本质

`Series` 可以理解为:

> **一列带标签的一维数组 = values + index**

它不是简单的一维 list, 也不是裸的 ndarray, 而是“数值/对象数组”与“行标签”绑定后的结果.

```python
s = pd.Series([10.0, 20.0, 30.0], index=["A", "B", "C"], name="fare")

print("Series 内容:\n", s)
print("values:", s.values)
print("index :", s.index)
print("name  :", s.name)
print("dtype :", s.dtype)
```

这意味着 `Series` 的运算不是单纯按位置发生, 而是可以按标签自动对齐:

```python
s1 = pd.Series([10, 20, 30], index=["A", "B", "C"])
s2 = pd.Series([1, 2, 3], index=["B", "C", "D"])

print("按索引对齐相加的结果:\n", s1 + s2)
```

输出会是:

```text
A     NaN
B    21.0
C    32.0
D     NaN
dtype: float64
```

因为 Pandas 默认认为:

> **业务上“B”只能和“B”对齐, “C”只能和“C”对齐, 没匹配到的就应当显式表现为缺失值 `NaN`.**

这就是 Pandas 跟 NumPy 在抽象层次上最本质的区别之一.

##### 2. `DataFrame` 的本质

`DataFrame` 可以理解为:

> **多个共享同一套行索引的 `Series` 组成的二维表结构**

它既可以看作“带列名和行索引的二维表”, 也可以看作“按列组织的一组一维数组”.

```python
df_demo = pd.DataFrame(
    {
        "fare_amount": [8.5, 12.0, 6.0],
        "passenger_count": [1, 2, 1],
        "pickup_hour": [8, 9, 8],
    },
    index=["trip_1", "trip_2", "trip_3"],
)

print("DataFrame 内容:\n", df_demo)
print("index   :", df_demo.index)
print("columns :", df_demo.columns)
print("各列 dtype:")
print(df_demo.dtypes)
```

从物理角度看, `DataFrame` 更接近 **列式存储** 而不是“一个个行对象”.

这件事非常关键, 因为数据分析的大量操作都是:

- 对某一列做过滤; 
- 对某几列做数值计算; 
- 对某一列做分组统计; 
- 对时间列做重采样; 
- 对主键列做连接.

列式组织意味着 Pandas 可以更高效地把这些操作下压到 NumPy 或扩展数组层面执行.

##### 3. `Index` 的本质

很多初学者把 `Index` 理解成“行号”, 这是不够的.

`Index` 的真正角色是:

> **Pandas 中“轴标签”的统一抽象, 用于定位、对齐、分组、切片、连接和时间序列操作.**

它既可以是默认的 `RangeIndex`, 也可以是:

- 字符串主键组成的 `Index`
- 时间戳组成的 `DatetimeIndex`
- 多层标签组成的 `MultiIndex`

例如:

```python
df_idx = pd.DataFrame(
    {
        "fare_amount": [10, 12, 8],
        "passenger_count": [1, 2, 1],
    },
    index=["order_001", "order_002", "order_003"],
)

print("按标签定位 order_002 这一行:\n", df_idx.loc["order_002"])
```

如果把时间列设置为索引:

```python
df_time = pd.DataFrame(
    {
        "pickup_count": [120, 150, 180],
    },
    index=pd.to_datetime([
        "2024-01-01 08:00:00",
        "2024-01-01 09:00:00",
        "2024-01-01 10:00:00",
    ]),
)

print("按时间区间切片(08:00 至 09:00):\n", df_time.loc["2024-01-01 08:00:00":"2024-01-01 09:00:00"])
```

此时 Pandas 就能直接支持基于时间的切片、重采样与滑窗.

#### Pandas 的设计哲学总结

把上面三者放在一起看, Pandas 的核心设计哲学可以概括为四句话:

**1. 在数组之上增加标签语义**

NumPy 的世界里, 重点是位置; Pandas 的世界里, 重点是字段名和标签.

**2. 在高性能基础上容纳异构数据**

现实业务表不是纯 `float64` 矩阵, 而是金额、字符串、时间、布尔、缺失值混合在一起.

**3. 把关系代数和时间序列作为一等公民**

`groupby`、`merge`、`resample`、`rolling` 在 Pandas 里不是边角料, 而是最核心的分析算子.

**4. 让分析代码表达业务意图而不是遍历细节**

不是反复写:

```python
for row in rows:
    ...
```

而是写成:

```python
(
    df
    .query("total_amount > 0")
    .assign(pickup_hour=lambda x: x["tpep_pickup_datetime"].dt.hour)
    .groupby("pickup_hour")
    .agg(order_count=("total_amount", "size"))
)
```

这类代码表达的是“业务分析动作”, 而不是“解释器怎样逐行跑”.


### Q1 :这 200 万条数据可靠吗? 

这个问题看起来很朴素, 但实际上是所有分析的起点.

如果数据本身就不可信:

- 类型错了; 
- 时间还没转成真正的时间戳; 
- 缺失值没有识别; 
- 坐标出现 0 或离谱异常值; 
- 金额出现负数或极端脏值; 

那么后面所有均值、分组、建模、可视化都会建立在错误基础上.

所以 Q1 的真正目标不是“会几个 API”, 而是建立一个习惯:

> **分析前先做数据可信度诊断(data validation / data quality check).**

#### 这一单元要解决的 API

| 技术 | 在这里解决什么问题 |
|:---:|:---:|
| `dtype` | 这一列到底是数值、字符串还是时间?  |
| 缺失值 | 哪些列有 `NaN` / `None` / 空洞?  |
| `mask` | 如何批量构造“可信/不可信”的布尔条件?  |
| `loc` | 如何按布尔条件精确筛出异常行?  |
| `query` | 如何用接近 SQL 的写法进行业务过滤?  |

#### 1. 本质

Q1 本质上做的是三件事:

1. 识别列的数据类型是否符合业务预期; 
2. 识别缺失值与异常值; 
3. 根据布尔规则把“可信数据子集”筛选出来.

Pandas 在这里的价值是:

> **把逐行检查, 变成整列向量化检查; 把“人工目测”, 变成“可重复执行的规则”.**

#### 2. 底层原理

这一类操作底层主要依赖三件事:

**第一, 每一列都有明确的 `dtype` 与缺失值表示语义**

Pandas 会根据列类型选择不同的存储与计算路径. 数值列可以直接批量比较; 对象列通常更贵; 时间列转换后才能启用 `.dt`、时间切片和重采样能力.

**第二, 布尔掩码本质上是一列 `True/False`**

例如:

```python
mask = df["fare_amount"] > 0
```

这个 `mask` 的本质是一个与 DataFrame 行数等长的布尔 `Series`, 它像 NumPy 里的布尔索引一样, 但带有索引语义.

**第三, `loc` 与 `query` 都是在做“基于条件的行过滤”**

- `loc` 更显式, 更适合组合多个条件和精确选列; 
- `query` 更接近声明式 DSL, 可读性更强, 在复杂业务表达里很方便.

#### 3. 数据分析中的作用与用法

在真实项目里, Q1 一般会固定形成一个“数据准入规则”阶段:

- 先看 `df.dtypes`
- 再看 `df.isna().sum()`
- 再看描述统计 `describe()` 或分位数
- 再写出异常规则 `mask`
- 最后筛出 `clean_df`

这一步非常像后端系统里的参数校验, 只不过对象从“一次请求”变成了“200 万条记录”.

如果把这道题真正落到 Pandas API 上, 可以按下面顺序思考:

1. 先用 `pd.read_parquet`、`shape`、`dtypes` 认清数据结构和字段类型; 
2. 再用 `isna`、数值比较、时间差计算等方式识别缺失值和异常值; 
3. 用多个布尔 `mask` 组合成“可信数据”的准入规则; 
4. 用 `loc` 抽查异常样本, 用 `query` 产出清洗后的可信子集; 
5. 最后用 `np.select` 把异常进一步分类成可统计、可追踪的标签.


#### 真正需要记住什么? 

不是“我会 `query` 和 `loc`”. 而是:

> **Pandas 的第一价值, 是让数据质量检查变成整列、批量、可复现的规则系统, 而不是人工抽样拍脑袋.**


### Pandas: GroupBy、Transform、Merge 对应着问题 Q2～Q4

Q2: 一天中什么时候最忙?  
Q3: 什么时间段的流水和单位运营时间收入最高?  
Q4: 哪些上下车区域最热门?  

Q2 到 Q4 其实对应了三类极其核心的分析动作:

- Q2:聚合统计
- Q3:分组后再回填到原表
- Q4:把不同来源的数据拼接起来

如果说 Q1 是“先把数据洗干净”, 那么 Q2～Q4 就是“开始真正做业务分析”.

#### GroupBy、Transform、Merge

##### 1. `groupby` 的本质

这里的 `groupby` 和 MySQL 中的 `groupby` 十分类似, 它的经典模型是:

> **Split - Apply - Combine**

也就是:

1. 按某个键把数据拆成多个组; 
2. 对每个组做计算; 
3. 再把结果组合起来.

例如“按小时统计订单量”:

```python
df.groupby("pickup_hour").size()
```

##### 2. `transform` 的本质

`transform` 很容易和 `agg` 混淆.

二者的关键区别是:

- `agg` 会把每组压缩成一个或少数几个结果; 
- `transform` 会把每组算出的结果, **按原行数广播回去**.

也就是说:

> **`transform` 适合做“分组特征回填”, 而 `agg` 适合做“分组汇总报表”.**

##### 3. `merge` 的本质

`merge` 的本质就是表与表之间基于键的连接.

它在 Pandas 中扮演的角色, 基本等价于 SQL 里的 Join:

- `left`
- `right`
- `inner`
- `outer`

而且 Pandas 额外强调:

> **连接不仅是“拼起来”, 更重要的是“验证有没有拼错”.**

这也是为什么 `validate=` 参数在教学里非常值得讲.

#### 核心特点: Index 对齐机制

Pandas 的底层灵魂是 **Index 对齐**, 这三个操作的核心差异, 本质上就是对 Index 的不同处理策略: 

1. **`groupby (+ agg)`: 索引升格与压缩(降维)**
* **特点**: 把指定的分组键提拔为**新的 Index**.
* **机制**: 原表的 $N$ 行被压缩为分组数 $K$ 行, 聚合结果与**新生成的 Group Index** 严格对齐.

2. **`transform`: 严格保留原索引(保维)**
* **特点**: 计算结果**强制与原表的原始 Index 保持对齐**并原路广播.
* **机制**: 输出始终保持 $N$ 行, 无需遍历也无需手动关联, 天然保证行索引与原表一行一对应.

3. **`merge`: 跨表索引重构与映射(改维)**
* **特点**: 基于键或 Index 寻找交集/并集, 并**重构生成一套全新的 Index**.
* **机制**: 处理跨表对齐, 根据匹配关系(1:1、1:N、N:N)对行和 Index 进行保留、丢弃或笛卡尔裂变.

#### 数据分析中的作用与用法

Q2～Q4 基本覆盖了报表分析最常见的三种输出:

- 一个按小时/按区域统计的汇总表; 
- 一张增加了组内特征的新表; 
- 两张业务表拼成的一张宽表.

#### Q2:一天中什么时候最忙? 

这个问题的解决思路是:
> 先把上车时间拆成小时, 再按小时把订单分组, 对每组统计订单量、总流水和平均客单价; 如果想看更细的模式, 再按“工作日 + 小时”做联合分组.可以按下面顺序思考:

1. 用 `.dt.hour`、`.dt.day_name()` 从时间列里提取业务维度; 
2. 用 `assign` 把这些派生列安全地加回 DataFrame; 
3. 用 `groupby(...).agg(...)` 输出小时级汇总报表; 
4. 如果要看二维热度分布, 就对多个键一起 `groupby`; 
5. 这一题的核心是 `groupby + agg`, `transform` 和 `merge` 在这里不是必须步骤.


#### Q3:什么时间段的流水和单位运营时间收入最高? 

这个问题相比 Q2 多了一个关键点:

> **不是只统计“多少单”, 而是构造新的业务指标后再分组比较.**

落到 API 上时, 关键不是直接拿原始列做聚合, 而是先把能回答业务问题的指标算出来.

可以按下面顺序思考:

1. 用 `assign` 一次性构造 `pickup_hour`、`revenue_per_minute`、`fare_per_mile` 等派生指标; 
2. 用 `groupby("pickup_hour").agg(...)` 把订单压缩成小时级报表; 
3. 用 `sort_values` 比较哪个小时总流水最高、哪个小时单位运营时间收入最高; 
4. 用 `transform("mean")` 把组内均值贴回原表, 再判断单笔订单相对所在小时平均水平是高还是低.


#### Q4:哪些上下车区域最热门? 

PULocationID:上车区域 ID(Pick-up Location ID, 关联 taxi_zone_lookup.csv 中的地理分区与行政区).
DOLocationID:下车区域 ID(Drop-off Location ID).

课堂里可以直接按 LocationID 统计热度, 再读取官方 `taxi_zone_lookup.csv`, 把 LocationID 连接成真实的 Borough、Zone 和 service_zone 信息.

这个问题的业务翻译是:

> 先分别统计上车区域和下车区域各自出现了多少次, 再把两张统计表按 `LocationID` 拼起来, 得到综合热度表.

使用这一节 API 解决时, 可以按下面顺序思考:

1. 用 `value_counts()` 先拿到每个 `LocationID` 的出现频次; 
2. 用 `rename_axis` 和 `reset_index` 把频次统计结果整理成标准 DataFrame; 
3. 用 `merge` 把频次表和区域维表拼起来, 补充区域标签; 
4. 再用一次 `merge` 把上车热度和下车热度合成一张综合热度表; 
5. 最后把综合热度表和真实的 taxi zone lookup 维表做 `merge`, 并用 `validate` 显式校验连接关系, 避免拼表时 silently 出错.


### Pandas: Resample、Rolling 与窗口思维对应着问题 Q5
哪些订单和时间窗口可能异常?  

Q5 的关键词不是“异常值”本身, 而是:

> **异常不能只看单条记录, 还要看它在时间窗口中的相对位置.** 这就是窗口思维.

一笔 80 美元的订单一定异常吗? 不一定. 但如果某个 15 分钟窗口里的订单金额整体突然抬升, 或某一单远高于其所在小时/窗口的分位数阈值, 那就值得关注.

#### Resample、Rolling 与窗口思维

##### 本质

`resample` 的本质是:

> **把时间轴重新切成固定频率的桶, 再在桶内聚合.**

`rolling` 的本质是:

> **让一个长度固定的滑动窗口沿着时间轴或序列逐步移动, 每移动一步都做一次局部统计.**

`transform` 在这里继续扮演“把组级统计量贴回原表”的角色.

##### 底层原理

这类 API 成立的前提是:

- 时间列必须是 `datetime64[ns]` 或 `DatetimeIndex`
- 数据最好按时间排序
- Pandas 通过时间桶切分、索引对齐和窗口边界管理来完成批量计算

其中:

- `resample("1h")` 是按小时分桶; 
- `rolling(24)` 是按固定观察点个数滑窗; 
- `rolling("2H")` 则是按时间跨度滑窗.

##### 数据分析中的作用与用法

在生产环境中, Q5 常见于:

- 监控指标异常波动; 
- 交易金额突增; 
- 某时间段订单量断崖变化; 
- 某笔订单相对同时段分布明显偏高.

如果把这个问题翻译成 API 的使用路径, 可以理解为:

> 先在单笔订单层面判断异常, 再把订单放回所属时间窗口, 观察它相对同小时和滑动窗口是否异常.

使用这一节 API 解决时, 可以按下面顺序思考:

1. 用 `assign` 构造 `revenue_per_minute` 这类更适合做异常判断的指标; 
2. 用 `quantile` 和 `np.select` 先做单笔订单的分位数异常分类; 
3. 用 `transform` 计算“所在小时”的中位数和 p95, 并把它们回填到每一行; 
4. 用 `where` 只保留异常值, 方便快速抽查; 
5. 用 `resample` 做小时级窗口汇总, 再用 `rolling` 构造动态阈值.


### Q6: 为什么同一个分析可以相差一个数量级甚至更多? 

这个问题的核心不是“谁会写更多 API”, 而是: **不同写法把计算放在了不同的执行层.**

#### Pandas 密集计算的几种执行方式

1. `Python loop`
最直接, 但循环控制、分支判断、类型解析都发生在 Python 解释器层, 开销最高.

2. `apply(axis=1)`
看起来像 Pandas 风格, 但本质仍是“逐行调用 Python 函数”; 每一行都可能被包装成临时 `Series`, 因此很多场景甚至比手写循环还慢.

3. `NumPy / Pandas vectorization`
把整列计算下沉到底层数组和 ufunc, 是数值计算的默认首选.只要表达式足够简单、分支不多, 通常都能跑得很快.

4. `Numba`
当计算已经超出“自然向量化”的舒适区, 例如有多层分支、反复更新、临时量较多时, 可以把数值循环 JIT 成机器码, 常常会比纯向量化更快.

#### Pandas 的内存与存储优化

1. 合适的 `dtype` 会直接影响内存占用、缓存命中率和整体吞吐.
2. 性能实验前应尽早裁剪列, 只读取真正要用的字段.
3. `Parquet` 是带 schema 的列式二进制格式, 通常比 `CSV` 更适合分析场景.
4. 做性能比较时, 应该尽量把 “I/O / 清洗 / 核心计算” 三件事拆开, 不要把它们混在一个 benchmark 里.


#### 这一节要让学员真正记住什么? 

Q6 不该被总结成“Numba 很高级”.

真正应该记住的是这条性能优化路线:

```text
先别写 Python loop
↓
尽量改成 Pandas / NumPy 向量化
↓
检查 dtype 和内存占用
↓
能用 query / eval 简化表达时就别手搓太多中间变量
↓
I/O 场景优先考虑 Parquet
↓
只有在热点逻辑无法自然向量化时, 再上 Numba
```

也就是:

> **性能优化首先是数据模型和执行模型的优化, 然后才是工具层面的优化.**


### 这一部分 Pandas 总结

如果把 Pandas 这一大段内容压缩成一句话, 那就是:

> **Pandas 不是“带表头的 Excel 替代品”, 而是一套把标签、缺失值、分组、连接、时间窗口和性能工程统一起来的表格分析计算模型.**

从 Q1 到 Q6, 学员真正应该建立起来的不是零散 API 记忆, 而是下面这条分析路径:

```text
Q1 数据是否可信? 
    -> dtype / missing / mask / loc / query

Q2 指标怎么聚合? 
    -> datetime / groupby / agg

Q3 聚合结果怎么回填成特征? 
    -> assign / transform / 派生指标

Q4 多张表怎么安全拼接? 
    -> value_counts / merge / validate

Q5 如何引入时间窗口思维? 
    -> quantile / where / resample / rolling

Q6 为什么性能差异会这么大? 
    -> loop / apply / NumPy / dtype / query/eval / Numba / Parquet
```


## Python 数据分析的扩展

前面的 Q1～Q6 主要解决的是:

- 如何理解数据;  
- 如何清洗数据;  
- 如何做分组、窗口和性能分析;  
- 如何在 Pandas / NumPy 里把分析问题写清楚.  

但在真实项目里, 课程通常还要再往前走一步:

1. 把一组稳定重复的分析动作沉淀成**领域 API**, 避免团队成员反复复制粘贴脚本; 
2. 把真正的热点数值循环下沉到**原生编译层**, 避免所有性能问题都卡在 Python 解释器上.

下面用两个非常典型的扩展方向来收尾:

1. `DataFrame Accessor`: 解决“怎么把分析脚本升级成可复用领域工具”; 
2. `Numba JIT`: 解决“怎么把复杂数值循环编译成机器码”.

### 自定义 Accessor(描述符模式与命名空间扩展)

#### 原理

Accessor 可以理解为: **在不修改 Pandas 源码的前提下, 给 `DataFrame` 注册一个自定义业务命名空间**.

例如, 当我们写:

```python
df.taxi.clean()
```

这里的 `taxi` 不是 DataFrame 的原生字段, 也不是普通列名, 而是我们通过:

```python
@pd.api.extensions.register_dataframe_accessor("taxi")
```

注册出来的一个领域入口.

从使用体验上看, 它很像一种“描述符式的命名空间扩展”:

1. Pandas 在 `DataFrame` 类上挂上一个叫 `taxi` 的访问入口; 
2. 当你访问 `df.taxi` 时, Pandas 会创建一个和当前 `df` 绑定的 accessor 对象; 
3. accessor 内部通常把原始 `DataFrame` 保存在 `self._obj` 中; 
4. 然后你就可以在这个对象上暴露 `clean()`、`hourly_kpis()`、`location_popularity()` 这类业务方法.

它的价值不在于“语法更酷”, 而在于:

1. 把零散脚本沉淀成领域 API; 
2. 让团队成员直接复用统一规则; 
3. 把字段校验、特征工程、清洗逻辑集中到一个地方维护; 
4. 让后续代码从“操作表”升级为“调用业务能力”.

#### 使用场景

Accessor 特别适合下面这类情况:

1. 同一份业务表反复做相同的字段校验、特征构造和清洗规则; 
2. 团队里已经形成稳定的分析词汇, 比如“可信订单”“小时 KPI”“区域热度”; 
3. 希望下游代码更接近业务表达, 而不是每次都重新手搓几十行 Pandas; 
4. 希望把分析脚本逐步升级成公司内部可复用的工具层.


### Numba JIT 加速与绕过 GIL(Native Compilation)

#### 原理

Numba 的核心不是“把 Pandas 变快”, 而是: **把已经整理好的数值循环编译成原生机器码**.

也就是说, Numba 最擅长的不是下面这些:

1. 读 CSV / Parquet; 
2. 做 `merge`; 
3. 做 `groupby`; 
4. 处理大量字符串、对象列和标签对齐.

这些事情仍然更适合交给 Pandas.

Numba 真正擅长的是:

1. 输入已经是 NumPy 数组; 
2. 计算逻辑里有大量数值循环; 
3. 中间有多层 `if/else` 分支; 
4. 重复更新很多次, 用纯向量化会写得非常绕, 或者会制造很多中间临时数组.

典型写法是:

```python
@njit
def kernel(...):
    ...
```

它的原理可以概括成三句话:

1. `@njit` 会让 Numba 尝试进入 `nopython` 模式, 把循环编译成原生代码; 
2. 编译成功后, 循环中的数值运算不再依赖 Python 解释器逐条调度; 
3. 如果再加上 `nogil=True`, 那么进入这段编译后代码时, 线程可以不一直持有 GIL; 如果加上 `parallel=True` 和 `prange`, 还可以在满足独立迭代的前提下做多线程并行.

要特别强调一点:

> **Numba 不是用来替代 Pandas 的, 它更像是“把 Pandas 清洗好的数组, 交给原生数值内核去算”.**

#### 使用场景

Numba 特别适合下面这类情况:

1. 行数很多, 且每行都要做复杂数值计算; 
2. 逻辑中有多层分支、重复更新、局部状态累积; 
3. 用纯向量化虽然能写, 但会出现很多 `np.where`、很多中间数组, 代码可读性和内存压力都很差; 
4. 每一行之间相互独立, 适合 `parallel=True` + `prange` 并行.



