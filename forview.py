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

def func(i, j, k):
    return i * (ny * nz) + j * (nz) + k
nx, ny, nz = 3, 6, 4

# 只选择边界内的原子，否则边界处横跳干扰视线
def select_in_boundary():
    in_boundary = []
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            for k in range(1, nz + 1):
                in_boundary.append(func(i, j, k))
    particle_ids = particle_ids[in_boundary]
    positions = positions[in_boundary]
    times = times[in_boundary]
#select_in_boundary()

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

# 给边界处原子染色，观察零温下原子运动是否符合微扰条件
min_atom_indices = func(nx//2, ny//2, nz//2)
max_atom_indices = nx - 1
colors = np.full(N_particles, 'black')
colors[min_atom_indices] = 'red'
colors[max_atom_indices] = 'green'

scat = ax.scatter(np.zeros(N_particles), np.zeros(N_particles), np.zeros(N_particles), s=30, c=colors)
# scat = ax.scatter([], [], [], s=30)

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