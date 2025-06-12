import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
# 石墨晶格基矢
a1 = np.array([2.456, 0.0, 0.0])
a2 = np.array([-1.228, 2.126958, 0.0])
a3 = np.array([0.0, 0.0, 7.0])
# 初始条件
nx, ny, nz = 6, 6, 4
# 模拟参数
dt = 1e-13 # 0.1 ps
steps = 30
# 是否写入文件
writetext = True

# 基元原子（分数坐标）
basis_frac = np.array([
    [0.0, 0.0, 0.25],
    [0.0, 0.0, 0.75],
    [1/3, 2/3, 0.25],
    [2/3, 1/3, 0.75]
])

# 一些参数
a = 2.456
c = 7.0
sigma_between_layer = np.sqrt((a**2)/3 + (c/2)**2)

#返回的boundary是N*5列表，除了原子序号还包含了原子是在哪个边界的信息
def generate_graphite(nx, ny, nz):
    positions = []
    boundary = []  # 存储边界原子序号的数组
    atom_index = 0  # 原子序号计数器
    
    # 计算模拟盒子尺寸
    box_size = np.array([
        nx * np.linalg.norm(a1),
        ny * np.linalg.norm(a2),
        nz * np.linalg.norm(a3)
    ])
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                shift = i*a1 + j*a2 + k*a3
                for l in range(len(basis_frac)):
                    pos = basis_frac[l][0] * a1 + basis_frac[l][1] * a2 + basis_frac[l][2] * a3
                    xyz = pos + shift
                    positions.append(np.append(xyz, 0.0))  # 最后一位是占位时间
                    
                    # 检查是否为边界原子
                    is_boundary = False
                    # 检查x方向边界
                    if i == 0 or i == nx-1:
                        is_boundary = True
                    # 检查y方向边界
                    if j == 0 and (l == 0 or l == 1):
                        is_boundary = True
                    if j == ny-1 and (l == 2 or l == 3):
                        is_boundary = True
                    # 检查z方向边界
                    if k == 0 and (l == 0 or l == 2):
                        is_boundary = True
                    if k == nz-1 and (l == 1 or l == 3):
                        is_boundary = True
                    
                    if is_boundary:
                        boundary.append((atom_index,i,j,k,l))
                    
                    atom_index += 1
    
    return np.array(positions), boundary

# boundary参数是一个一维列表，包含原子序号
def compute_accelerations(positions, boundary, nx, ny, nz, epsilon=0.0067, sigma=1.2633, mass=12.0, cutoff=3.5, boundarycondition = 1):
# def compute_accelerations(positions, boundary, nx, ny, nz, epsilon=0.0067, sigma=1.2633, mass=12.0, cutoff=4, boundarycondition = 1):
    if boundarycondition == 0 : # 无边界
        N = len(positions)
        accelerations = np.zeros((N, 3))
        forces = np.zeros((N, N, 3))  # 存储两两之间的力
        pos = positions[:, :3]

        for i in range(N):
            for j in range(i + 1, N):
                rij = pos[i] - pos[j] 
                r = np.linalg.norm(rij)
                if r < cutoff:
                    r6 = (sigma / r) ** 6
                    r12 = r6 ** 2
                    force_scalar = 24 * epsilon * (2*r12 - r6) / r**2
                    force_vector = force_scalar * rij
                    forces[i, j] = force_vector
                    forces[j, i] = -force_vector
                    accelerations[i] += force_vector / mass
                    accelerations[j] -= force_vector / mass
        return accelerations , forces # eV/Å(amu)
    if boundarycondition == 1 : # 周期性边界
        N = len(positions)
        accelerations = np.zeros((N, 3))
        forces = np.zeros((N, N, 3))  # 存储两两之间的力
        pos = positions[:, :3]
        periodicpos = []
        periodicpos.append(pos)
        periodicpos.append(pos+a1*nx)
        periodicpos.append(pos-a1*nx)
        periodicpos.append(pos+a2*ny)
        periodicpos.append(pos-a2*ny)
        periodicpos.append(pos+a3*nz)
        periodicpos.append(pos-a3*nz)

        for i in range(N):
            for j in range(i + 1, N):
                # 对pos[j]进行调整
                for k in range(7):
                    rij = pos[i] - periodicpos[k][j]
                    r = np.linalg.norm(rij)
                    if r < cutoff:
                        r6 = (sigma / r) ** 6
                        r12 = r6 ** 2
                        force_scalar = 4 * epsilon * (12 * r12 - 6 * r6) / r**2
                        force_vector = force_scalar * rij
                        forces[i, j] = force_vector
                        forces[j, i] = -force_vector
                        accelerations[i] += force_vector / mass
                        accelerations[j] -= force_vector / mass

            return accelerations, forces
    if boundarycondition == 2: # 固定边界
        N = len(positions)
        accelerations = np.zeros((N, 3))
        forces = np.zeros((N, N, 3))  # 存储两两之间的力
        pos = positions[:, :3]

        for i in range(N):
            for j in range(i + 1, N):
                rij = pos[i] - pos[j]
                r = np.linalg.norm(rij)
                if r < cutoff:
                    r6 = (sigma / r) ** 6
                    r12 = r6 ** 2
                    force_scalar = 4 * epsilon * (12 * r12 - 6 * r6) / r**2
                    force_vector = force_scalar * rij
                    forces[i, j] = force_vector
                    forces[j, i] = -force_vector
                    if i not in boundary:
                        accelerations[i] += force_vector / mass
                    if j not in boundary:
                        accelerations[j] -= force_vector / mass
        return accelerations , forces

# 使用verlet算法计算新位置
def compute_new_positions(positions, accelerations, prev_positions, dt, mass=12.0):
    N = len(positions)
    new_positions = np.copy(positions)
    # Convert accelerations to appropriate units (eV/Å(amu) to Å/s²)
    eV = 1.602176634e-19  # J/eV
    amu = 1.66053906660e-27  # kg
    angstrom = 1e-10  # Å to m
    a_unit = eV / amu / angstrom**2  # Å/s²
    if prev_positions is None:
        # Initialize velocities using Maxwell-Boltzmann distribution
        temperature = 10  # Kelvin
        k_B = 1.380649e-23  # J/K
        mass_kg = mass * amu  # Convert atomic mass to kg
        std_dev = np.sqrt(k_B * temperature / mass_kg)  # Standard deviation for velocity distribution
        velocity = np.random.normal(0, std_dev, (N, 3))  # m/s
        # velocity = np.zeros((N, 3))
        new_positions[:, :3] = positions[:, :3] + velocity / angstrom * dt + 0.5 * accelerations * a_unit * dt**2
    else:
        new_positions[:, :3] = 2 * positions[:, :3] - prev_positions[:, :3] + accelerations * a_unit * dt**2
    return new_positions

# 计算速度
def compute_velocity(current_positions, prev_positions, dt):
    if prev_positions is None:
        return np.zeros_like(current_positions[:, :3])
    return (current_positions[:, :3] - prev_positions[:, :3]) / dt

#边界条件处理出界原子
def boundary_conditions(positions, box_size):
    wrapped_positions = np.copy(positions)
    for i in range(3):
        wrapped_positions[:, i] = np.mod(wrapped_positions[:, i], box_size[i])
    return wrapped_positions

# 初始化结构

positions,boundary = generate_graphite(nx, ny, nz)
boundary_atom_indices = np.array([item[0] for item in boundary])
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
prev_positions = None
current_positions = positions.copy()

# 模拟主循环
trajectory = []

# 在模拟主循环之前添加文件打开操作 ， 打开可视化数据文件和txt数据文件
if writetext:
    output_file = open("dynamics_output.txt", "w")
    # 写入文件头
    output_file.write("# Step Time ParticleID X Y Z Vx Vy Vz Ax Ay Az\n")
    output_file.write("# -----------------------------------------------------------------\n")

with h5py.File("graphite_simulation.h5", "w") as h5file:
    flat_data = h5file.create_dataset("trajetory", (steps*N_particles, 5), dtype='f8')

    # 模拟主循环
    for step in tqdm(range(steps)):
        acc, forces = compute_accelerations(current_positions,boundary, nx, ny, nz)
        velocity = compute_velocity(current_positions, prev_positions, dt)
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
        if writetext:
            # 写入每个粒子的信息
            for i in range(N_particles):
                # 格式: 步数 时间 粒子ID x y z vx vy vz ax ay az
                output_file.write(
                    f"{step:5d} {time_value:8.4f} {i:5d} "
                    f"{new_positions[i,0]:10.5f} {new_positions[i,1]:10.5f} {new_positions[i,2]:10.5f} "
                    f"{velocity[i,0]:10.5f} {velocity[i,1]:10.5f} {velocity[i,2]:10.5f} "
                    f"{acc[i,0]:10.5f} {acc[i,1]:10.5f} {acc[i,2]:10.5f}\n"
                )

            output_file.write("# Forces (selected pairs) at step {}\n".format(step))
            if step % 50 == 0 :
                for i in range(N_particles):
                    for j in range(i+1, N_particles):
                        output_file.write(
                            f"# Force {i}-{j}: "
                            f"{forces[i,j,0]:12.8f} {forces[i,j,1]:12.8f} {forces[i,j,2]:12.8f} "
                            f"|r|={np.linalg.norm(new_positions[i,:3]-new_positions[j,:3]):6.3f} \n"
                        )
            output_file.write("# -------------------------------------------------\n")
        
        prev_positions = current_positions.copy()
        current_positions = new_positions.copy()

    if writetext:
        output_file.close()
