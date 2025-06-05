import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 读取数据
with h5py.File("graphite_simulation.h5", "r") as f:
    data = f["trajetory"][:]  # shape: (steps * N_particles, 5)

# 提取信息
particle_ids = data[:, 0].astype(int)
positions = data[:, 1:4]
times = data[:, 4]

unique_times = np.unique(times)
unique_ids = np.unique(particle_ids)
N_particles = len(unique_ids)
N_steps = len(unique_times)

# 为每个时间步准备数据索引
# 把数据重塑成 (steps, N_particles, 3)
positions_reshaped = positions.reshape((N_steps, N_particles, 3))

# 画图初始化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

scat = ax.scatter([], [], [], s=30)

ax.set_xlim(np.min(positions[:,0]), np.max(positions[:,0]))
ax.set_ylim(np.min(positions[:,1]), np.max(positions[:,1]))
ax.set_zlim(np.min(positions[:,2]), np.max(positions[:,2]))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Graphite Simulation')

def update(frame):
    pos = positions_reshaped[frame]
    scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
    ax.set_title(f"Time = {unique_times[frame]:.3f} ps")
    return scat,

ani = FuncAnimation(fig, update, frames=N_steps, interval=100, blit=False)

plt.show()
