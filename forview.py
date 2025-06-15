import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# === 读取数据 ===
with h5py.File("graphite_simulation.h5", "r") as f:
    data = f["trajetory"][:]  


particle_ids = data[:, 0].astype(int)
positions = data[:, 1:4]
times = data[:, 4]

# === 晶格参数 ===
nx, ny, nz = 6, 6, 4

def func(i, j, k):
    return i * (ny * nz) + j * nz + k

# === 获取基本信息 ===
unique_times = np.unique(times)
unique_ids = np.unique(particle_ids)
N_particles = len(unique_ids)
N_steps = len(unique_times)

positions_reshaped = positions.reshape((N_steps, N_particles, 3))

# === 绘图初始化 ===
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

# === 设置颜色（示例：中心红色、最大编号绿色，其余黑色） ===
colors = np.full(N_particles, 'black')
center_index = func(nx // 2, ny // 2, nz // 2)
if center_index < N_particles:
    colors[center_index] = 'red'
if N_particles > 0:
    colors[-1] = 'green'

scat = ax.scatter(np.zeros(N_particles), np.zeros(N_particles), np.zeros(N_particles), s=30, c=colors)

ax.set_xlim(np.min(positions[:, 0]), np.max(positions[:, 0]))
ax.set_ylim(np.min(positions[:, 1]), np.max(positions[:, 1]))
ax.set_zlim(np.min(positions[:, 2]), np.max(positions[:, 2]))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Graphite Simulation')

# === 更新函数 ===
def update(frame):
    pos = positions_reshaped[frame]
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    ax.set_title(f"Time = {(unique_times[frame]*1e12):.3f} ps")
    return scat,

# === 创建动画 ===
ani = FuncAnimation(fig, update, frames=N_steps, interval=100, blit=False)

# === 保存为 GIF ===
ani.save("graphite_sim.gif", writer=PillowWriter(fps=10))  # 可调 fps

print("GIF saved as graphite_sim.gif")
plt.show()
