# final_homework
this is for final homework
# 目前的实现想法
总体上分为两个脚本。第一个脚本用来计算分子的运动轨迹，目的是生成一个数据文件，该文件中包括每一个粒子每一时刻的位置，文件总共有五列，第一列为粒子编号，第二，三，四列为坐标（如果选用其他坐标系可以随之变化），第五列为时间信息。第二个脚本负责绘图，目前是想将一个这样数据文件绘制成一个动画，绘制方法为每一个时刻绘制一张图，将绘制得到的图依次呈现即可。脚本IJ.py中我粗略写了一些可能用到的函数，可能还有没想到的，大概的实现过程就是给一个初始条件（粒子位置，初速度），根据粒子位置求加速度，根据加速度和过去的位置信息更新位置，如此循环。
# 脚本的解释
运行LJ.py可以得到graphite_simulation.h5  
确保目录下有graphite_simulation.h5，运行forview.py可以得到可视化动画，或者make view也可以得到相同结果  
确保目录下有graphite_simulation.h5，运行energy_annlysis.py,可以得到系统的能量  
确保目录下有graphite_simulation.h5，运行time_relation.py,可以得到速度关联函数  
确保目录下有graphite_simulation.h5，运行compute_rdf.py,可以得到径向分布函数  
DeepMD-test中有使用lammps进行的石墨分子动力学模拟，脚本为test.lmp,运行即可得到轨迹和能量文件，里面还有一个生成石墨晶格的脚本用于test.lmp  
input.json文件为配置好的训练文件，按照转换文件的脚本可以得到符合格式的训练文件进行训练。