import numpy as np

# 加载 .npy 文件
data = np.load('data/set.000/coord.npy')

# 打印数据内容
print(data)

# 打印数据形状（几维、多少个原子等信息）
print("Shape:", data.shape)

# 打印数据类型
print("Data type:", data.dtype)
