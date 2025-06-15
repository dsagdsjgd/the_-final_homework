import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 参数 ===
mass_amu = 12.0
amu = 1.66053906660e-27
mass_kg = mass_amu * amu

# epsilon = 0.0067  # eV
# sigma = 1.2633    # Å
# cutoff = 4.0      # Å

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

# === 根据分子动理论计算出的温度实验值 ===
temperature_list = []
# temperature_list = np.array(kinetic_energy_list) * eV_to_J * 2 / ((N_particles - 3) * 1.380649e-23)  # K
for step in tqdm(range(N_steps)):
    pos = positions_reshaped[step]      # Å
    vel = velocities_reshaped[step]     # m/s

    speeds_sq = np.sum(vel**2, axis=1)  # 每粒子速度平方
    temperature = (mass_kg * np.mean(speeds_sq)) / 1.380649e-23 / (3*N_particles-3)*N_particles  # K
    temperature_list.append(temperature)

if (doplot):
    times_ps = unique_times * 1e12  # s -> ps

    plt.figure(figsize=(8, 5))
    plt.plot(times_ps[4:], temperature_list[4:], label="Kinetic Energy (eV)")
    plt.xlabel("time (10fs)")
    plt.ylabel("temperature (K)")
    plt.title("temperature(K) calculated based on the molecular kinetic theory at 20 K")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("Kinetic temperature VS time (10fs).png")

print((mass_kg * 9e16) / 1.380649e-23 / (3*N_particles-3)*N_particles) # 43373348753114.875