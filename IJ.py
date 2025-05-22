import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 该代码用于生成石墨的晶体结构，并计算原子间的相互作用力
a1 = np.array([2.456, 0.0, 0.0])
a2 = np.array([-1.228, 2.126958, 0.0])
a3 = np.array([0.0, 0.0, 7.0])

# 基元原子（分数坐标）
basis_frac = np.array([
    [0.0, 0.0, 0.25],
    [0.0, 0.0, 0.75],
    [1/3, 2/3, 0.25],
    [2/3, 1/3, 0.75]
])

# 构建晶胞
def generate_graphite(nx, ny, nz):
    """
    生成石墨晶体结构的原子位置。
    nx, ny, nz: 晶胞在x, y, z方向上的重复次数
    返回值是晶格的初始位置以及初始时间
    """
    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = i*a1 + j*a2 + k*a3
                for frac in basis_frac:
                    xyz = pos + shift
                    positions.append(np.append(xyz, 0.0))  
    return np.array(positions)
def compute_accelerations(positions, epsilon=0.01, sigma=3.4, mass=12.0, cutoff=8.5):
    """
    给定所有原子的位置，返回对应加速度数组。
    这里使用的是Lennard-Jones势能函数来计算原子间的相互作用力。
    该函数计算每对原子之间的相互作用力，并返回加速度。
    """
    N = len(positions)
    accelerations = np.zeros((N, 3))  # 只计算 xyz 三维加速度

    # 提取位置部分（忽略时间维）
    pos = positions[:, :3]

    for i in range(N):
        for j in range(i + 1, N):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)

            if r < cutoff:
                r6 = (sigma / r) ** 6
                r12 = r6 ** 2
                force_scalar = 24 * epsilon * (2 * r12 - r6) / r**2
                force_vector = force_scalar * rij
                accelerations[i] += force_vector / mass
                accelerations[j] -= force_vector / mass  # Newton's Third Law

    return accelerations
def compute_new_positions(positions, accelerations, dt=0.01):
    """
    更新原子位置，目前打算使用verlrt方法
    """
    return new_positions
def boundary_conditions(positions, box_size):
    """
    处理周期性边界条件
    """
    return positions
#构建 3x3x2 石墨结构
#这里主要是为了方便可视化，nx, ny, nz的值可以根据需要进行调整
#检验函数generate_graphite有没有正确生成
#我们最终是要做一个连续播放的动画
#并且要模拟更多的原子
nx, ny, nz = 9, 9, 6
positions = generate_graphite(nx, ny, nz)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='black', s=40)


ax.set_title("Initial Graphite Structure", fontsize=14)
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")

plt.tight_layout()
plt.show()
