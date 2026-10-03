# Run every command from the repository root.
PY ?= venv/bin/python

.PHONY: help venv train evaluate inspect sweep-both sweep-fc1 finetune export verify export-sparse sim-mac sim-sparse sim-mlp clean

help:
	@echo "Targets:"
	@echo "  venv        create the Python environment and install packages"
	@echo "  train       train the float32 baseline model"
	@echo "  evaluate    measure the baseline accuracy"
	@echo "  inspect     print weight ranges and near-zero statistics"
	@echo "  sweep-both  prune both layers, quantize, measure accuracy"
	@echo "  sweep-fc1   prune only fc1, quantize, measure accuracy"
	@echo "  finetune    prune fc1 and fine-tune (70/80/90 percent)"
	@echo "  export      integer inference + write hex files for Verilog"
	@echo "  verify      check the hex files against the golden vectors"
	@echo "  export-sparse  write the sparse (index, value) lists + golden dot products"
	@echo "  sim-mac     simulate the MAC unit testbench (needs iverilog)"
	@echo "  sim-sparse  simulate the sparse dot-product engine (needs iverilog)"
	@echo "  sim-mlp     simulate the whole network, all 100 golden images (needs iverilog)"

venv:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt

train:
	$(PY) software/train_baseline.py

evaluate:
	$(PY) software/evaluate_baseline.py

inspect:
	$(PY) software/inspect_weights.py

sweep-both:
	$(PY) software/sweep_prune_both_layers.py

sweep-fc1:
	$(PY) software/sweep_prune_fc1.py

finetune:
	$(PY) software/finetune_pruned.py

export:
	$(PY) software/export_int8.py

verify:
	$(PY) software/verify_export.py

export-sparse:
	$(PY) software/export_sparse.py

sim-mac:
	mkdir -p build
	iverilog -o build/sim_mac hardware/tb/tb_mac.v hardware/rtl/mac.v
	vvp build/sim_mac

sim-sparse:
	mkdir -p build
	iverilog -o build/sim_sparse hardware/tb/tb_sparse_dot.v hardware/rtl/sparse_dot.v hardware/rtl/mac.v
	vvp build/sim_sparse

sim-mlp:
	mkdir -p build
	iverilog -o build/sim_mlp hardware/tb/tb_sparse_mlp.v hardware/rtl/sparse_mlp.v hardware/rtl/sparse_dot.v hardware/rtl/mac.v
	vvp build/sim_mlp

clean:
	rm -rf build
