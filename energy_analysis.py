import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 参数 ===
mass_amu = 12.0
amu = 1.66053906660e-27
mass_kg = mass_amu * amu

epsilon = 0.0067  # eV
sigma = 1.2633    # Å
cutoff = 4.0      # Å

eV_to_J = 1.60218e-19
angstrom_to_meter = 1e-10

# 是否画图
doplot = True

# === 读取数据 ===
with h5py.File("graphite_simulation.h5", "r") as f:
    pos_data = f["trajetory"][:]           # [id, x, y, z, t]
    vel_data = f["velocity_traj"][:]        # [id, vx, vy, vz, t]

# === 解包 ===
particle_ids = pos_data[:, 0].astype(int)
positions = pos_data[:, 1:4]     # Å
times = pos_data[:, 4]           # s

velocities = vel_data[:, 1:4]   
vel_times = vel_data[:, 4]       # s

# === 基本信息 ===
unique_times = np.unique(times)
unique_ids = np.unique(particle_ids)
N_particles = len(unique_ids)
N_steps = len(unique_times)

# === 重塑位置、速度数组 ===
positions_reshaped = positions.reshape((N_steps, N_particles, 3))    # Å
velocities_reshaped = velocities.reshape((N_steps, N_particles, 3))  # m/s

# === 能量存储 ===
kinetic_energy_list = []
potential_energy_list = []
total_energy_list = []

for step in tqdm(range(N_steps)):
    pos = positions_reshaped[step]      # Å
    vel = velocities_reshaped[step]     # m/s

    # ---- 动能 ----
    speeds_sq = np.sum(vel**2, axis=1)  # 每粒子速度平方
    kinetic_energy = 0.5 * mass_kg * np.sum(speeds_sq) / eV_to_J  # 总动能 (eV)

    # ---- 势能 ----
    potential_energy = 0.0
    for i in range(N_particles):
        for j in range(i + 1, N_particles):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if 0.95 < r < cutoff:
                r6 = (sigma / r) ** 6
                r12 = r6 ** 2
                potential_energy += 4 * epsilon * (r12 - r6)

    # ---- 存储 ----
    kinetic_energy_list.append(kinetic_energy)
    potential_energy_list.append(potential_energy)
    total_energy_list.append(kinetic_energy + potential_energy)

# === 画图 ===
if (doplot):
    times_ps = unique_times * 1e12  # s -> ps

    plt.figure(figsize=(8, 5))
    plt.plot(times_ps, kinetic_energy_list, label="Kinetic Energy (eV)")
    plt.plot(times_ps, potential_energy_list, label="Potential Energy (eV)")
    plt.plot(times_ps, total_energy_list, label="Total Energy (eV)")
    plt.xlabel("Time (ps)")
    plt.ylabel("Energy (eV)")
    plt.title("Energy vs Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\nResults saved to heat_capacity_results.dat")