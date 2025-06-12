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

angstrom_to_meter = 1e-10
eV_to_J = 1.60218e-19

# === 读取数据 ===
with h5py.File("graphite_simulation.h5", "r") as f:
    data = f["trajetory"][:]  # shape: (steps * N_particles, 5)

particle_ids = data[:, 0].astype(int)
positions = data[:, 1:4]      # 单位：Å
times = data[:, 4]            # 单位：s

unique_times = np.unique(times)
unique_ids = np.unique(particle_ids)
N_particles = len(unique_ids)
N_steps = len(unique_times)

# === 重塑位置数组 ===
positions_reshaped = positions.reshape((N_steps, N_particles, 3))  # 单位：Å

# === 能量存储 ===
kinetic_energy_list = []
potential_energy_list = []
total_energy_list = []

prev_positions = None
prev_time = None

for step in tqdm(range(N_steps)):
    pos = positions_reshaped[step]            # 单位：Å
    time = unique_times[step]                 # 单位：s

    # ---- 动能 ----
    if prev_positions is not None:
        dt = time - prev_time                 # 单位：s
        velocities = (pos - prev_positions) * angstrom_to_meter / dt  # m/s
        speeds_sq = np.sum(velocities**2, axis=1)
        kinetic_energy = 0.5 * mass_kg * np.sum(speeds_sq) / eV_to_J  # 转 eV
    else:
        kinetic_energy = 0.0

    # ---- 势能（Lennard-Jones）----
    potential_energy = 0.0
    for i in range(N_particles):
        for j in range(i + 1, N_particles):
            rij = pos[i] - pos[j]             # 单位：Å
            r = np.linalg.norm(rij)
            if r < cutoff:
                r6 = (sigma / r) ** 6
                r12 = r6 ** 2
                potential_energy += 4 * epsilon * (r12 - r6)

    # ---- 存储 ----
    kinetic_energy_list.append(kinetic_energy)
    potential_energy_list.append(potential_energy)
    total_energy_list.append(kinetic_energy + potential_energy)

    prev_positions = pos.copy()
    prev_time = time

# === 画图 ===
times_ps = unique_times * 1e12  # 转为皮秒用于显示

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
