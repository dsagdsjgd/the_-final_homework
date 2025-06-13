import h5py
import numpy as np
import matplotlib.pyplot as plt

# === 设置参数 ===
r_max = 40      # 最大距离（Å）
dr = 0.1       # bin 宽度
bins = np.arange(0, r_max + dr, dr)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# === 读取轨迹文件 ===
with h5py.File("graphite_simulation.h5", "r") as f:
    data = f["trajetory"][:]  # shape: (N_total, 5)

positions = data[:, 1:4]  # 提取 [x, y, z]

# === 分别计算距离 ===
r_xy = np.linalg.norm(positions[:, :2], axis=1)  # sqrt(x^2 + y^2)
r_z = np.abs(positions[:, 2])                   # |z|
r_x = np.abs(positions[:, 0])                   # |x|
r_y = np.abs(positions[:, 1])                   # |y|

# === 统计粒子数 ===
counts_xy, _ = np.histogram(r_xy, bins=bins)
counts_z, _  = np.histogram(r_z,  bins=bins)
counts_x, _  = np.histogram(r_x,  bins=bins)
counts_y, _  = np.histogram(r_y,  bins=bins)

# === 绘图 ===
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(bin_centers, counts_xy, label="XY plane")
axs[0, 0].set_title("XY-plane Radial Distribution")
axs[0, 0].set_xlabel("r (Å)")
axs[0, 0].set_ylabel("Particle count")
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(bin_centers, counts_z, label="Z direction", color="orange")
axs[0, 1].set_title("Z-direction Distribution")
axs[0, 1].set_xlabel("z (Å)")
axs[0, 1].set_ylabel("Particle count")
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].plot(bin_centers, counts_x, label="X direction", color="green")
axs[1, 0].set_title("X-direction Distribution")
axs[1, 0].set_xlabel("x (Å)")
axs[1, 0].set_ylabel("Particle count")
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(bin_centers, counts_y, label="Y direction", color="red")
axs[1, 1].set_title("Y-direction Distribution")
axs[1, 1].set_xlabel("y (Å)")
axs[1, 1].set_ylabel("Particle count")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
