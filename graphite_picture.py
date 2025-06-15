import numpy as np
import matplotlib.pyplot as plt

# 基矢定义
a1 = np.array([2.456, 0.0])
a2 = np.array([-1.228, 2.126958])

# 基元原子（二维投影，忽略z坐标）
basis_frac_2d = np.array([
    [0.0, 0.0],    # A位
    [1/3, 2/3],    # B位
    [2/3, 1/3]     # A'位（等效于A位，用于显示周期性）
])

# 生成单层石墨烯的原子位置
def generate_graphene_2d(nx, ny):
    positions = []
    for i in range(nx):
        for j in range(ny):
            shift = i*a1 + j*a2
            for frac in basis_frac_2d[:2]:  # 只取前两个基元原子（避免重复）
                pos = frac[0] * a1 + frac[1] * a2
                positions.append(pos + shift)
    return np.array(positions)

# 生成3x3超胞的原子位置
positions = generate_graphene_2d(3, 3)

# 绘制原子和键
plt.figure(figsize=(8, 8))
plt.scatter(positions[:, 0], positions[:, 1], c='black', s=100, label='Carbon atoms')

# 绘制键（连接最近邻原子）
bond_threshold = 1.5  # Å
for i, pos1 in enumerate(positions):
    for j, pos2 in enumerate(positions[i+1:], i+1):
        dist = np.linalg.norm(pos1 - pos2)
        if dist < bond_threshold:
            plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'gray', linewidth=2)

# 标注基矢
plt.arrow(0, 0, a1[0], a1[1], head_width=0.2, color='red', label='a₁')
plt.arrow(0, 0, a2[0], a2[1], head_width=0.2, color='blue', label='a₂')

# 标注晶胞
cell_vertices = np.array([
    [0, 0],
    a1,
    a1 + a2,
    a2,
    [0, 0]
])
plt.plot(cell_vertices[:, 0], cell_vertices[:, 1], 'r--', alpha=0.5, label='Unit cell')

plt.xlabel('X (Å)')
plt.ylabel('Y (Å)')
plt.title('2D Graphene Lattice with Basis Vectors')
plt.axis('equal')
plt.legend()
plt.grid(alpha=0.3)
plt.show()