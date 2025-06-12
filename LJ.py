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

a = 2.456
c = 7.0
sigma_between_layer = np.sqrt((a**2)/3 + (c/2)**2)

def compute_accelerations(positions, nx, ny, nz, epsilon=0.0067, sigma=sigma_between_layer, mass=12.0, cutoff=8.5):
# def compute_accelerations(positions, nx, ny, nz, epsilon=0.0067, sigma=1.2633, mass=12.0, cutoff=8.5):
    N = len(positions)
    accelerations = np.zeros((N, 3))
    pos = positions[:, :3]

    for i in range(N):
        for j in range(i + 1, N):
            # 原子分数坐标(i,j,k)与positions索引一一对应，映射关系：index=i*(ny*nz)+j*(nz)+k
            rij = pos[i] - pos[j]
            if (i%nz == 0 and j%nz == nz-1): # z方向上都在边界上
                rij = rij + nz*a3
            if (i%(ny*nz)//nz == 0 and j%(ny*nz)//nz == ny-1): # y方向上
                rij = rij+ ny*a2
            if (i//(ny*nz) == 0 and j//(ny*nz) == nx-1): # x方向上
                rij = rij + nx*a1
            
            r = np.linalg.norm(rij)
            if r < cutoff:
                r6 = (sigma / r) ** 6
                r12 = r6 ** 2
                force_scalar = 24 * epsilon * (2*r12 - r6) / r**2
                force_vector = force_scalar * rij
                accelerations[i] += force_vector / mass
                accelerations[j] -= force_vector / mass
    return accelerations # eV/Å(amu)

def compute_new_positions(positions, accelerations, prev_positions, dt=0.01, mass=12.0):
    N = len(positions)
    new_positions = np.copy(positions)
    # Convert accelerations to appropriate units (eV/Å(amu) to Å/s²)
    eV = 1.602176634e-19  # J/eV
    amu = 1.66053906660e-27  # kg
    angstrom_to_meter = 1e-10  # Å to m
    a_unit = eV / amu / angstrom_to_meter**2  # Å/s²
    if prev_positions is None:
        # Initialize velocities using Maxwell-Boltzmann distribution
        temperature = 30  # Kelvin
        k_B = 1.380649e-23  # J/K
        mass_kg = mass * amu  # Convert atomic mass to kg
        std_dev = np.sqrt(k_B * temperature / mass_kg)  # Standard deviation for velocity distribution
        velocity = np.random.normal(0, std_dev, (N, 3))  # m/s
        # velocity = np.zeros((N, 3))
        new_positions[:, :3] = positions[:, :3] + velocity / amu * dt + 0.5 * accelerations * a_unit * dt**2
    else:
        new_positions[:, :3] = 2 * positions[:, :3] - prev_positions[:, :3] + accelerations * a_unit * dt**2
    return new_positions

def boundary_conditions(positions, box_size):
    wrapped_positions = np.copy(positions)
    for i in range(3):
        wrapped_positions[:, i] = np.mod(wrapped_positions[:, i], box_size[i])
    return wrapped_positions

# 初始化结构
nx, ny, nz = 6, 6, 4
positions = generate_graphite(nx, ny, nz)
N_particles = len(positions)
particle_ids = np.arange(N_particles)  # 粒子编号
box_size = np.array([
    nx * np.linalg.norm(a1),
    ny * np.linalg.norm(a2)* np.cos(np.pi/6),
    nz * np.linalg.norm(a3)
])
positions = boundary_conditions(positions, box_size)

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
plt.show()

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
        new_positions = compute_new_positions(current_positions, acc, prev_positions, dt)
        new_positions = boundary_conditions(new_positions, box_size)
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
