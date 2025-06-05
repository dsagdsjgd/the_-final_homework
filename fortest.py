import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

steps = 10
N_particles = 10*10*10

with h5py.File("test.h5", "w") as f:
    flat_data = f.create_dataset("trajetory", (steps*N_particles, 5), dtype='f8')

    for i in tqdm(range(steps)):
        for j in range(N_particles):
            flat_data[i*N_particles + j, 0] = j
            flat_data[i*N_particles + j, 1:3] = np.random.rand(2) * 10  # 随机3维坐标
            flat_data[i*N_particles + j, 3] = 0
            flat_data[i*N_particles + j, 4] = i * 0.01  # 时间步长
