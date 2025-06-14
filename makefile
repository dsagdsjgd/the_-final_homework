.PHONY: data view

# 生成 MD 数据
data:
	python3 LJ.py

# 查看结果，如果没有数据文件则先生成
view:
	@if [ ! -f graphite_simulation.h5 ]; then \
		echo "File graphite_simulation.h5 not found. Generating data..."; \
		python3 LJ.py; \
	else \
		echo "Found graphite_simulation.h5. Skipping data generation."; \
	fi
	python3 forview.py
energy:
	python3 energy_analysis.py
clean:
	rm -f graphite_simulation.h5
	rm -f dynamics_output.txt
	rm -f graphite_sim.gif
	rm -f graphite_simulation_2.h5
	echo "Cleaned up generated files."