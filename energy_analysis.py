import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 常数与参数 ===
kB = 1.380649e-23      # J/K
T = 100
mass_amu = 12.0
amu = 1.66053906660e-27
mass_kg = mass_amu * amu

v_rms = np.sqrt(3 * kB * T / mass_kg)
v_max = 5 * v_rms   # m/s

epsilon = 0.0067  # eV
sigma = 1.2633    # Å
cutoff = 4.0      # Å
eV_to_J = 1.60218e-19

doplot = True  # 是否绘图

# === 读取数据 ===
with h5py.File("graphite_simulation.h5", "r") as f:
    pos_data = f["trajetory"][:]           # [id, x, y, z, t]
    vel_data = f["velocity_traj"][:]       # [id, vx, vy, vz, t]

# === 解包 ===
particle_ids = pos_data[:, 0].astype(int)
positions = pos_data[:, 1:4]     # Å
times_pos = pos_data[:, 4]       # s
velocities = vel_data[:, 1:4]    # m/s
times_vel = vel_data[:, 4]       # s


unique_times_pos = np.unique(times_pos)
unique_times_vel = np.unique(times_vel)
unique_ids = np.unique(particle_ids)
N_particles = len(unique_ids)
N_steps_pos = len(unique_times_pos)
N_steps_vel = len(unique_times_vel)


positions_reshaped = positions.reshape((N_steps_pos, N_particles, 3))    # Å
velocities_reshaped = velocities.reshape((N_steps_vel, N_particles, 3))  # m/s
# 计算动能
kinetic_energy_list = []
for step in tqdm(range(N_steps_vel)):
    vel = velocities_reshaped[step]
    v_magnitudes = np.linalg.norm(vel, axis=1)
    valid_mask = v_magnitudes <= v_max
    valid_velocities = vel[valid_mask]
    valid_speeds_sq = np.sum(valid_velocities**2, axis=1)
    N_valid = np.count_nonzero(valid_mask)
    scaling_factor = N_particles / N_valid if N_valid > 0 else 0
    kinetic_energy = (0.5 * mass_kg * np.sum(valid_speeds_sq) * scaling_factor / eV_to_J)
    kinetic_energy_list.append(kinetic_energy)

# 计算势能
potential_energy_list = []
for step in tqdm(range(N_steps_pos)):
    pos = positions_reshaped[step]
    potential_energy = 0.0
    for i in range(N_particles):
        for j in range(i + 1, N_particles):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if 0.95 < r < cutoff:
                r6 = (sigma / r) ** 6
                r12 = r6 ** 2
                potential_energy += 4 * epsilon * (r12 - r6)
    potential_energy_list.append(potential_energy)


times_vel_ps = unique_times_vel * 1e12  # s → ps
times_pos_ps = unique_times_pos * 1e12  # s → ps

kinetic_energy_interp = np.interp(times_pos_ps, times_vel_ps, kinetic_energy_list)


total_energy_list = kinetic_energy_interp + np.array(potential_energy_list)

if doplot:
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axs[0].plot(times_pos_ps, kinetic_energy_interp, color='tab:blue')
    axs[0].set_ylabel("Kinetic Energy (eV)")
    axs[0].set_title("Kinetic Energy vs Time")
    axs[0].grid(True)

    axs[1].plot(times_pos_ps, potential_energy_list, color='tab:orange')
    axs[1].set_ylabel("Potential Energy (eV)")
    axs[1].set_title("Potential Energy vs Time")
    axs[1].grid(True)

    axs[2].plot(times_pos_ps, total_energy_list, color='tab:green')
    axs[2].set_xlabel("Time (ps)")
    axs[2].set_ylabel("Total Energy (eV)")
    axs[2].set_title("Total Energy vs Time")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
