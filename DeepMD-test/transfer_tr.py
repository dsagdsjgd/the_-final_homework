import dpdata

# 读取 LAMMPS dump 文件
system = dpdata.System('dump.deepmd', fmt='lammps/dump')

# 设置原子类型（这里以单元素 Si 为例，多元素请修改）
system.data['type_map'] = ['C']

# 转换为 DeepMD 格式，输出到 data/ 目录
system.to('deepmd/npy', 'data')
