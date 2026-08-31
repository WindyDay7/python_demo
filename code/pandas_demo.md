```python
import pandas as pd
import numpy as np

@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def distance_to_center(self, center_lat, center_lon):
        # 假设 df 中有 'lat' 和 'lon' 列
        # 这里用简化的欧几里得距离代替真实的哈弗斯因距离
        return np.sqrt((self._obj['lat'] - center_lat)**2 + 
                       (self._obj['lon'] - center_lon)**2)

# 学员完成后，可以这样极其优雅地调用：
df = pd.DataFrame({'lat': [39.9, 31.2], 'lon': [116.3, 121.5]})
print(df.geo.distance_to_center(35.0, 110.0))
```

## 解释

### 修饰器 @pd.api.extensions.register_dataframe_accessor("geo") 的作用

* **动态命名空间扩展**：这个修饰器是 Pandas 官方提供的一个扩展接口（Extension API）。它的作用就像是给 Pandas 的 `DataFrame` “无创植入”一个自定义的子模块。
* **注册专属属性**：通过传入字符串 `"geo"`，它会在所有的 Pandas DataFrame 实例上自动挂载一个名为 `.geo` 的属性。
* **避免污染与风险**：相比于直接去写一个继承自 `pd.DataFrame` 的子类（这种做法往往会破坏 Pandas 底层复杂的内部结构和方法链），使用修饰器注册访问器是一种安全、解耦且优雅的官方推荐做法。

### 组合（Composition）与委托（Delegation）的使用

* **非继承关系**：`GeoAccessor` 并不是 `DataFrame` 的子类，它们之间没有面向对象中的“血缘继承”关系，而是通过**组合**的方式结合在一起。
* **构造函数自动注入**：当你在代码中访问 `df.geo` 时，Pandas 在幕后会自动做两件事：
1. 实例化 `GeoAccessor` 类。
2. 把当前的 DataFrame 对象作为参数传进去，触发 `__init__(self, pandas_obj)`，并将其保存在 `self._obj` 中。

`df.geo` 触发 Pandas 的动态属性拦截，生成 `GeoAccessor` 实例。
当你写下 `df.geo` 时，你确实是在**访问一个属性**。

- 在 Pandas 内部，注册访问器（Accessor）时，Pandas 会利用 Python 的描述符机制（Descriptors，如 `@property`）在 DataFrame 类上动态绑定这个属性。
- 当你访问 `df.geo` 时，Pandas 触发了一段后台代码：
  1. 动态创建一个 `GeoAccessor` 类的实例。
  2. 把当前的 `df` 传进去。
  3. **把这个实例本身作为属性值返回给你**。

也就是说，`df.geo` 运行完之后，在内存中产生了一个 `GeoAccessor` 的对象实例。


###  

