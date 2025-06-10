import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
# 石墨晶格基矢
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

def generate_graphite(nx, ny, nz):
    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = i*a1 + j*a2 + k*a3
                for frac in basis_frac:
                    pos = frac[0] * a1 + frac[1] * a2 + frac[2] * a3
                    xyz = pos + shift
                    positions.append(np.append(xyz, 0.0))  # 最后一位是占位时间（可扩展）
    return np.array(positions)

def compute_accelerations(positions, nx, ny, nz, epsilon=0.0067, sigma=3.4, mass=12.0, cutoff=8.5):
    N = len(positions)
    accelerations = np.zeros((N, 3))
    pos = positions[:, :3]

    def func(i, j, k):
        return i * (ny * nz) + j * (nz) + k

    for i in range(N):
        for j in range(i + 1, N):
            # 如果不是同一个原子
            if (i // (ny * nz) != j // (ny * nz) or
                i % (ny * nz) // nz != j % (ny * nz) // nz or
                i % (ny * nz) % nz != j % (ny * nz) % nz):

                rij = pos[i] - pos[j]
                r = np.linalg.norm(rij)
                if(r==0):
                    print("i,j:", i, j, "rij:", rij, "r:", r)
                # TODO:第3步为什么会出现r=0?
                if r < cutoff:
                    r6 = (sigma / r) ** 6
                    r12 = r6 ** 2
                    force_scalar = 4 * epsilon * (r12 - r6) / r**2
                    force_vector = force_scalar * rij
                    accelerations[i] += force_vector / mass
                    accelerations[j] -= force_vector / mass
    return accelerations

def compute_new_positions(positions, accelerations, prev_positions, dt=0.01):
    # N = len(positions)
    new_positions = np.copy(positions)
    if prev_positions is None:
        new_positions[:, :3] = positions[:, :3] + 0.5 * accelerations * dt**2
    else:
        new_positions[:, :3] = 2 * positions[:, :3] - prev_positions[:, :3] + accelerations * dt**2
    return new_positions

def boundary_conditions(positions,nx,ny,nz):
    def func(i,j,k):
        return i*(ny*nz) + j*(nz) + k
    for i in range(nx):
        for j in range(ny):
            positions[func(i,j,0), :] = positions[func(i,j,nz), :] + np.append(nz*a3, 0.0)
            positions[func(i,j,nz+1), :] = positions[func(i,j,1), :] - np.append(nz*a3, 0.0)
    for i in range(nx):
        for k in range(nz):
            positions[func(i,0,k), :] = positions[func(i,ny,k), :] + np.append(ny*a2, 0.0)
            positions[func(i,ny+1,k), :] = positions[func(i,1,k), :] - np.append(ny*a2, 0.0)
    for i in range(ny):
        for j in range(nz):
            positions[func(0,i,j), :] = positions[func(nx,j,i), :] + np.append(nx*a1, 0.0)
            positions[func(nx+1,i,j), :] = positions[func(1,i,j), :] - np.append(nx*a1, 0.0)
    return positions        
# 初始化结构
nx, ny, nz = 3, 6, 4
positions = generate_graphite(nx+2, ny+2, nz+2)
N_particles = len(positions)
particle_ids = np.arange(N_particles)  # 粒子编号

# 可视化初始结构

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='black', s=40)

bond_threshold = 1.7
N = len(positions)
for i in range(N):
    for j in range(i + 1, N):
        dist = np.linalg.norm(positions[i, :3] - positions[j, :3])
        if dist < bond_threshold:
            xs = [positions[i, 0], positions[j, 0]]
            ys = [positions[i, 1], positions[j, 1]]
            zs = [positions[i, 2], positions[j, 2]]
            ax.plot(xs, ys, zs, color='gray', linewidth=1)

ax.set_title("Initial Graphite Structure", fontsize=14)
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
plt.tight_layout()
# plt.show()

# 模拟参数
dt = 0.01
steps = 30
prev_positions = None
current_positions = positions.copy()

# 模拟主循环
trajectory = []

with h5py.File("graphite_simulation.h5", "w") as h5file:
    flat_data = h5file.create_dataset("trajetory", (steps*N_particles, 5), dtype='f8')
    

    # 模拟主循环
    for step in tqdm(range(steps)):
        acc = compute_accelerations(current_positions, nx, ny, nz)
        current_positions = boundary_conditions(current_positions, nx, ny, nz)
        new_positions = compute_new_positions(current_positions, acc, prev_positions, dt)
        time_value = step * dt
        # 写入数据：仅 xyz 坐标
        step_data = np.hstack([
            particle_ids.reshape(-1, 1),
            new_positions[:, :3],
            np.full((N_particles, 1), time_value)
        ])

        # 写入平铺数据
        flat_data[step * N_particles : (step + 1) * N_particles] = step_data

        prev_positions = current_positions.copy()
        current_positions = new_positions.copy()
