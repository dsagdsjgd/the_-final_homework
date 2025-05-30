import numpy as np
import os
import time

# 常量定义
NUM_LAYERS = 3
ATOMS_PER_LAYER = 100
LATTICE_CONST = 1.42  # nm
INTERLAYER_DIST = 3.35  # nm
MASS = 12.01  # amu
EPSILON = 2.964e-3  # eV
SIGMA = 0.34  # nm
CUTOFF = 4
DT = 0.01  # ps
STEPS = 100

# 创建输出目录
output_dir = f"simulation_data_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

def save_as_txt(step, pos, vel, acc, output_dir):
    """直接将数据保存为易读的txt文件"""
    filename = os.path.join(output_dir, f"step_{step:04d}.txt")
    with open(filename, 'w') as f:
        f.write(f"Step {step} - Atomic Data (positions in nm, velocities in nm/ps, accelerations)\n")
        f.write("Index\tX\tY\tZ\tVx\tVy\tVz\tAx\tAy\tAz\n")
        for i in range(len(pos)):
            f.write(f"{i}\t")
            f.write("\t".join([f"{x:.6f}" for x in pos[i]]) + "\t")
            f.write("\t".join([f"{v:.6f}" for v in vel[i]]) + "\t")
            f.write("\t".join([f"{a:.6f}" for a in acc[i]]) + "\n")

def initialize_graphite(n_layers, n_atoms_per_layer, a, d):
    positions = []
    n_side = int(np.ceil(np.sqrt(n_atoms_per_layer)))

    for k in range(n_layers):
        for i in range(n_side):
            for j in range(n_side):
                if (i + j) % 2 == 0:
                    x = i * a
                    y = j * a * np.sqrt(3) / 2
                    z = k * d
                    positions.append([x, y, z])
                if len(positions) >= n_layers * n_atoms_per_layer:
                    return np.array(positions), np.zeros((len(positions), 3))
    return np.array(positions), np.zeros((len(positions), 3))

def compute_lj_forces(pos, sigma, epsilon, rc, step):
    N = len(pos)
    forces = np.zeros_like(pos)
    filename = os.path.join(output_dir, f"forces_step_{step:04d}.txt")
    
    # 先收集所有要写入的行
    lines = []
    lines.append(f"Step {step} - Interatomic Forces (distance in nm, force in eV/nm)\n")
    lines.append("Atom1\tAtom2\tDistance\tFx\tFy\tFz\tForce_Magnitude\n")
    
    pair_count = 0
    
    for i in range(N - 1):
        for j in range(i + 1, N):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            
            if r < rc:  # 只保留截断半径条件
                sr6 = (sigma / r) ** 6
                sr12 = sr6 ** 2
                f_scalar = 24 * epsilon / r * (2 * sr12 - sr6)
                fij = f_scalar * (rij / r)
                forces[i] += fij
                forces[j] -= fij
                
                force_mag = np.linalg.norm(fij)
                lines.append(
                    f"{i}\t{j}\t{r:.6f}\t"
                    f"{fij[0]:.6e}\t{fij[1]:.6e}\t{fij[2]:.6e}\t"
                    f"{force_mag:.6e}\n"
                )
                pair_count += 1
    
    # 一次性写入所有数据
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Step {step}: 写入 {pair_count} 对原子相互作用数据到 {filename}")
    return forces

def run_md():
    pos, vel = initialize_graphite(NUM_LAYERS, ATOMS_PER_LAYER, LATTICE_CONST, INTERLAYER_DIST)
    acc = np.zeros_like(pos)

    # 保存初始状态
    save_as_txt(0, pos, vel, acc, output_dir)

    for step in range(1, STEPS + 1):
        forces = compute_lj_forces(pos, SIGMA, EPSILON, CUTOFF, step)
        acc = forces / MASS

        # Velocity Verlet 积分
        vel += 0.5 * DT * acc
        pos += DT * vel
        forces_new = compute_lj_forces(pos, SIGMA, EPSILON, CUTOFF, step)
        acc_new = forces_new / MASS
        vel += 0.5 * DT * acc_new

        # 每10步保存一次
        if step % 10 == 0 or step == STEPS:
            save_as_txt(step, pos, vel, acc_new, output_dir)
            print(f"Step {step} saved to {output_dir}/step_{step:04d}.txt")

if __name__ == "__main__":
    run_md()
    print(f"所有数据已保存到 {output_dir} 目录，可直接用记事本打开！")