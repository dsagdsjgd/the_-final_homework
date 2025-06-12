from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data
from ase.io import read
from ase.build import make_supercell
import numpy as np

# 原始晶格信息
cell = [
    [ 2.4560000896,  0.0000000000, 0.0000000000],
    [-1.2280000448,  2.1269584693, 0.0000000000],
    [ 0.0000000000,  0.0000000000, 7.0000000000]
]

# 原子坐标（以分数形式）
scaled_positions = [
    [0.000000000, 0.000000000, 0.250000000],
    [0.000000000, 0.000000000, 0.750000000],
    [0.333333343, 0.666666687, 0.250000000],
    [0.666666687, 0.333333343, 0.750000000]
]

from ase import Atoms
atoms = Atoms('C4',
              scaled_positions=scaled_positions,
              cell=cell,
              pbc=True)

# 构建 3x3x2 超晶胞
supercell = atoms.repeat((3, 3, 2))

# 导出为 LAMMPS 数据文件
write_lammps_data('graphite.data', supercell, atom_style='atomic')
print("✅ 已生成 graphite.data 文件")
