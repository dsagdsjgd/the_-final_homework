import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pandas as pd

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
                for frac in basis_frac:
                    pos = frac[0] * a1 + frac[1] * a2 + frac[2] * a3
                    xyz = pos + shift
                    positions.append(np.append(xyz, 0.0))  # 最后一位是占位时间
                    
                    # 检查是否为边界原子
                    is_boundary = False
                    # 检查x方向边界
                    if i == 0 or i == nx-1:
                        is_boundary = True
                    # 检查y方向边界
                    if j == 0 or j == ny-1:
                        is_boundary = True
                    # 检查z方向边界
                    if k == 0 or k == nz-1:
                        is_boundary = True
                    
                    if is_boundary:
                        boundary.append((atom_index,i,j,k))
                    
                    atom_index += 1
    
    return np.array(positions), boundary

def compute_accelerations(positions, boundary, nx, ny, nz, epsilon=0.0067, sigma=1.418, mass=12.0, cutoff=8.5):
    N = len(positions)
    accelerations = np.zeros((N, 3))
    forces = np.zeros((N, N, 3))  # 存储两两之间的力
    pos = positions[:, :3]

    for i in range(N):
        for j in range(i + 1, N):
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
                force_scalar = 4 * epsilon * (12 * r12 - 6 * r6) / r**2
                force_vector = force_scalar * rij
                forces[i, j] = force_vector
                forces[j, i] = -force_vector
                accelerations[i] += force_vector / mass
                accelerations[j] -= force_vector / mass

    return accelerations, forces


def compute_new_positions(positions, accelerations, prev_positions, dt=0.01):
    N = len(positions)
    new_positions = np.copy(positions)
    if prev_positions is None:
        new_positions[:, :3] = positions[:, :3] + 0.5 * accelerations * dt**2
    else:
        new_positions[:, :3] = 2 * positions[:, :3] - prev_positions[:, :3] + accelerations * dt**2
    new_positions[:,3:] += dt
    return new_positions

def boundary_conditions(positions, box_size):
    wrapped_positions = np.copy(positions)
    for i in range(3):
        wrapped_positions[:, i] = np.mod(wrapped_positions[:, i], box_size[i])
    return wrapped_positions

def compute_velocity(current_positions, prev_positions, dt):
    if prev_positions is None:
        return np.zeros_like(current_positions[:, :3])
    return (current_positions[:, :3] - prev_positions[:, :3]) / dt

# 初始化结构
nx, ny, nz = 3, 6, 4
positions,boundary = generate_graphite(nx, ny, nz)
N_particles = len(positions)
particle_ids = np.arange(N_particles)
box_size = np.array([
    nx * np.linalg.norm(a1),
    ny * np.linalg.norm(a2)* np.cos(np.pi/6),
    nz * np.linalg.norm(a3)
])
positions = boundary_conditions(positions, box_size)
print(boundary)

# 模拟参数
dt = 0.01
steps = 30
prev_positions = None
current_positions = positions.copy()

# 在模拟主循环之前添加文件打开操作
output_file = open("dynamics_output.txt", "w")

# 写入文件头
output_file.write("# Step Time ParticleID X Y Z Vx Vy Vz Ax Ay Az\n")
output_file.write("# -----------------------------------------------------------------\n")

# 修改后的模拟主循环
for step in tqdm(range(steps)):
    acc, forces = compute_accelerations(current_positions,boundary, nx, ny, nz)
    velocity = compute_velocity(current_positions, prev_positions, dt)
    new_positions = compute_new_positions(current_positions, acc, prev_positions, dt)
    new_positions = boundary_conditions(new_positions, box_size)
    time_value = step * dt
    
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

output_file.close()