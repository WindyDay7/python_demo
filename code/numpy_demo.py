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

arr = np.array([[1, 2, 3], 
                [4, 5, 6]], dtype=np.int64)