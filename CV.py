import numpy as np
import subprocess
import os
import h5py
from LJ import nx,ny,nz
from energy_analysis import total_energy_list

# 常数
kB = 8.617333262145e-5  # 玻尔兹曼常数，单位 eV/K

T_start = 1
T_end = 11
T_number = 5

def run_simulation(temperature):
    """运行LJ.py模拟并返回能量列表"""
    # 修改LJ.py中的温度参数（使用UTF-8编码）
    with open("LJ.py", "r", encoding='utf-8') as f:  # 明确指定编码
        lines = f.readlines()
    
    # 找到并修改温度参数
    for i, line in enumerate(lines):
        if "temperature =" in line:
            lines[i] = f"temperature = {temperature}  # Kelvin\n"
            break
    
    # 写回文件（同样使用UTF-8）
    with open("LJ.py", "w", encoding='utf-8') as f:
        f.writelines(lines)
    
    # 运行模拟
    process = subprocess.run(['python', 'LJ.py'], check=True)
    
    # 运行能量分析
    process = subprocess.run(['python', 'energy_analysis.py'], check=True)
    
    # 获取能量数据
    return total_energy_list

def calculate_heat_capacity(energy_list, temperature):
    """根据能量涨落计算热容"""
    energies = np.array(energy_list)
    mean_E = np.mean(energies)
    mean_E2 = np.mean(energies**2)
    
    variance = mean_E2 - mean_E**2
    Cv = variance / (kB * temperature**2)
    # 对粒子数取平均———得到单位原子热容
    N = nx*ny*nz
    Cv/=N
    # 单位原子热容换算成摩尔原子热容
    Cv_molar = Cv * 1.602176634e-19 * 6.02214076e23
    return Cv_molar

def main():
    # 设置温度范围 (根据LJ.py中的参数适当选择)
    temperatures = np.linspace(T_start, T_end, T_number)
    
    # 存储结果
    results = []
    
    for T in temperatures:
        print(f"\nRunning simulation at T = {T} K")
        
        # 运行模拟并获取能量数据
        print("Running MD simulation...")
        energy_list = run_simulation(T)
        
        # 计算热容
        print("Calculating heat capacity...")
        Cv = calculate_heat_capacity(energy_list, T)
        
        results.append((T, Cv))
        print(f"Temperature: {T} K, Heat Capacity: {Cv:.4e} (J/(mol*K)")
    
    # 保存结果
    np.savetxt('heat_capacity_results.dat', results, 
               header='Temperature(K) Heat_Capacity((J/(mol*K))', fmt='%.2f %.4e')
    print("\nResults saved to heat_capacity_results.dat")
    
    # 绘图
    try:
        import matplotlib.pyplot as plt
        temps = [r[0] for r in results]
        cvs = [r[1] for r in results]
        
        plt.figure(figsize=(8, 5))
        plt.plot(temps, cvs, 'o-')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Heat Capacity (J/(mol*K))')
        plt.title('Heat Capacity vs Temperature')
        plt.grid(True)
        plt.savefig('heat_capacity_plot.png')
        plt.close()
        print("Plot saved to heat_capacity_plot.png")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

if __name__ == "__main__":
    main()