import pandas as pd
import random
import time

# 文件路径配置（请根据你实际存放的路径修改）
input_filename = "./data/train.csv"
output_filename = "./data/nyc_taxi_2M.csv"
target_samples = 2_000_000

start_time = time.time()

# ==========================================
# 步骤 1：极速统计总行数（不加载全量数据进内存）
# ==========================================
print("1. 正在扫描文件统计总行数 (大概需要 10~20 秒)...")
with open(input_filename, 'r', encoding='utf-8') as f:
    # 利用生成器表达式快速计数
    total_rows = sum(1 for _ in f)
    
print(f" -> 扫描完毕！文件总行数为: {total_rows:,} 行")

# ==========================================
# 步骤 2：生成要抽取的随机索引
# ==========================================
print(f"2. 正在从中随机挑选 {target_samples:,} 行...")
# 数据行索引从 1 到 total_rows-1（索引 0 是表头）
# 使用 set() 是为了利用哈希表带来 O(1) 的超快查询速度
keep_indices = set(random.sample(range(1, total_rows), target_samples))

# 必须把表头（第 0 行）也加进保留列表，否则 DataFrame 没有列名
keep_indices.add(0) 

# ==========================================
# 步骤 3：读取所需数据并跳过无关行
# ==========================================
print("3. 正在执行抽取 (这步通过底层过滤，可能需要 1~2 分钟，请耐心等待)...")
# skiprows 接受一个函数：如果行号 i 不在 keep_indices 里，这一行就不会被读进内存
df = pd.read_csv(input_filename, skiprows=lambda i: i not in keep_indices)

# ==========================================
# 步骤 4：保存为新的数据集供培训使用
# ==========================================
print("4. 正在保存抽样后的数据集到本地...")
df.to_csv(output_filename, index=False)

end_time = time.time()
print("\n🎉 抽取大功告成！")
print(f"✅ 文件已保存为: {output_filename}")
print(f"✅ 最终数据形状: {df.shape}")
print(f"✅ 整个过程总耗时: {end_time - start_time:.2f} 秒")