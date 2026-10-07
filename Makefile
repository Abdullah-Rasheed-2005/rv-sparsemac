SHELL := /bin/bash
# Run every command from the repository root.
PY ?= venv/bin/python

.PHONY: help venv train evaluate inspect sweep-both sweep-fc1 finetune export verify export-sparse sim-mac sim-sparse sim-mlp sim-zs sim-zs-edge sim-zs-budget sim-soc-budget sim-zs-fashion sim-soc-fashion fw sim-soc sim-soc-ws sim-pcpi sim-dense report clean

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
	@echo "  sim-zs      same, but also skipping zero pixels / hidden values + stress tests"
	@echo "  sim-zs-edge sim-zs on an artificial network with ties, saturation, empty columns"
	@echo "  sim-zs-budget  bounded run time: hard layer-1 cycle budget and constant time (budget-aware model)"
	@echo "  sim-soc-budget PicoRV32 + accelerator with the budget set by SMAC.CFG (budget-aware model)"
	@echo "  sim-zs-fashion / sim-soc-fashion  the same two with the Fashion-MNIST model (same hardware)"
	@echo "  sim-dense   same engine as sim-mlp but fed every weight (measured dense baseline)"
	@echo "  fw          compile the RISC-V firmware (needs gcc-riscv64-unknown-elf)"
	@echo "  report      run all hardware/RISC-V simulations and write the numbers into docs/results.md"
	@echo "  sim-pcpi    test the PCPI wrapper alone (handshake, foreign instructions, results)"
	@echo "  sim-soc     PicoRV32 + accelerator: run the firmware on all 100 images"
	@echo "  sim-soc-ws  same with the weight-skipping-only accelerator (for comparison)"

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

sim-zs:
	mkdir -p build
	iverilog -o build/sim_zs hardware/tb/tb_sparse_mlp_zs.v hardware/rtl/sparse_mlp_zs.v
	vvp build/sim_zs

sim-zs-edge:
	$(PY) software/make_edge_model.py
	mkdir -p build
	iverilog -DMEMDIR='"build/edge_mem"' -o build/sim_zs_edge hardware/tb/tb_sparse_mlp_zs.v hardware/rtl/sparse_mlp_zs.v
	vvp build/sim_zs_edge

# Bounded run time. Needs models/budget_fc1_<S>.pth and models/budget_order_<S>.txt
# (written by software/budget_pareto.py into build/, then copied into models/).
# docs/budget_pareto.md shows why 90 % pruned with a budget of 1000 cycles is used.
BUDGET ?= 1000
BUDGET_SPARSITY ?= 90
sim-zs-budget:
	$(PY) software/export_budget.py $(BUDGET) $(BUDGET_SPARSITY)
	mkdir -p build
	iverilog -o build/sim_zs_budget hardware/tb/tb_zs_budget.v hardware/rtl/sparse_mlp_zs.v
	vvp build/sim_zs_budget

# The same hardware with the Fashion-MNIST model (models/fashion_fc1_<S>.pth, software/fashion_pareto.py).
FASHION_BUDGET ?= 1850
FASHION_SPARSITY ?= 90
sim-zs-fashion:
	$(PY) software/export_budget.py $(FASHION_BUDGET) $(FASHION_SPARSITY) fashion
	mkdir -p build
	iverilog -DMEMDIR='"build/fashion_mem"' -o build/sim_zs_fashion hardware/tb/tb_zs_budget.v hardware/rtl/sparse_mlp_zs.v
	vvp build/sim_zs_fashion

# ---------------- RISC-V firmware (needs: sudo apt install gcc-riscv64-unknown-elf) ----------------
RVPREFIX ?= riscv64-unknown-elf-
RVFLAGS  ?= -march=rv32im -mabi=ilp32 -O2 -ffreestanding -nostdlib -fno-pic -fno-builtin -Wall -Wextra
FWDEFS   ?=

fw:
	mkdir -p build/fw
	$(RVPREFIX)gcc $(RVFLAGS) $(FWDEFS) -Ifirmware -Wl,-T,firmware/link.ld -Wl,--gc-sections \
		-o build/fw/fw.elf firmware/start.S firmware/main.c
	$(RVPREFIX)objcopy -O verilog --verilog-data-width=1 build/fw/fw.elf build/fw/fw.hex
	$(RVPREFIX)size build/fw/fw.elf

# the testbench reads the same addresses as the firmware (single source: firmware/layout.h)
build/layout.vh: firmware/layout.h
	mkdir -p build
	sed -n 's/^#define \([A-Z0-9_]*\)[ \t]*0x\([0-9A-Fa-f]*\).*/`define \1 32'"'"'h\2/p' $< > $@

SOC_SRC = hardware/tb/tb_soc.v hardware/rtl/sparsemac_pcpi.v hardware/rtl/sparse_mlp_zs.v \
          hardware/rtl/sparse_mlp.v hardware/rtl/sparse_dot.v hardware/rtl/mac.v \
          hardware/third_party/picorv32/picorv32.v

sim-pcpi:
	mkdir -p build
	iverilog -g2005 -o build/sim_pcpi hardware/tb/tb_sparsemac_pcpi.v hardware/rtl/sparsemac_pcpi.v \
		hardware/rtl/sparse_mlp_zs.v hardware/rtl/sparse_mlp.v hardware/rtl/sparse_dot.v hardware/rtl/mac.v
	vvp build/sim_pcpi
	iverilog -g2005 -Ptb_sparsemac_pcpi.ZERO_SKIP=0 -o build/sim_pcpi_ws hardware/tb/tb_sparsemac_pcpi.v hardware/rtl/sparsemac_pcpi.v \
		hardware/rtl/sparse_mlp_zs.v hardware/rtl/sparse_mlp.v hardware/rtl/sparse_dot.v hardware/rtl/mac.v
	vvp build/sim_pcpi_ws

sim-soc: fw build/layout.vh
	iverilog -g2005 -Ibuild -o build/sim_soc $(SOC_SRC)
	vvp build/sim_soc

# same system, but the accelerator skips only zero weights (sparse_mlp.v).
# The software baselines are the same as in sim-soc, so only 2 CSC images are run here (saves time).
sim-soc-ws: FWDEFS += -DNSW_CSC=2
sim-soc-ws: fw build/layout.vh
	iverilog -g2005 -Ibuild -DWEIGHT_SKIP_ONLY -o build/sim_soc_ws $(SOC_SRC)
	vvp build/sim_soc_ws

# PicoRV32 + accelerator with the budget-aware model: no budget, budget, budget + constant time.
# The weights, images and golden values come from build/budget_mem (software/export_budget.py).
sim-soc-budget: FWDEFS += -DBUDGET_RUN -DNSW_CSC=2
sim-soc-budget: fw build/layout.vh
	$(PY) software/export_budget.py $(BUDGET) $(BUDGET_SPARSITY)
	iverilog -g2005 -Ibuild -DBUDGET_RUN -DMEMDIR='"build/budget_mem"' -o build/sim_soc_budget $(SOC_SRC)
	vvp build/sim_soc_budget

sim-soc-fashion: FWDEFS += -DBUDGET_RUN -DNSW_CSC=2
sim-soc-fashion: fw build/layout.vh
	$(PY) software/export_budget.py $(FASHION_BUDGET) $(FASHION_SPARSITY) fashion
	iverilog -g2005 -Ibuild -DBUDGET_RUN -DMEMDIR='"build/fashion_mem"' -o build/sim_soc_fashion $(SOC_SRC)
	vvp build/sim_soc_fashion

# Runs every simulation, keeps the logs, then writes docs/results.md from the logs (takes several minutes).
# 'set -o pipefail' makes a failing simulation stop the report.
report:
	mkdir -p build/logs
	set -o pipefail; $(MAKE) --no-print-directory sim-mlp      2>&1 | tee build/logs/sim-mlp.log
	set -o pipefail; $(MAKE) --no-print-directory sim-zs       2>&1 | tee build/logs/sim-zs.log
	set -o pipefail; $(MAKE) --no-print-directory sim-zs-edge  2>&1 | tee build/logs/sim-zs-edge.log
	set -o pipefail; $(MAKE) --no-print-directory sim-dense    2>&1 | tee build/logs/sim-dense.log
	set -o pipefail; $(MAKE) --no-print-directory sim-pcpi     2>&1 | tee build/logs/sim-pcpi.log
	set -o pipefail; $(MAKE) --no-print-directory sim-soc      2>&1 | tee build/logs/sim-soc.log
	set -o pipefail; $(MAKE) --no-print-directory sim-soc-ws   2>&1 | tee build/logs/sim-soc-ws.log
	set -o pipefail; $(MAKE) --no-print-directory sim-zs-budget  2>&1 | tee build/logs/sim-zs-budget.log
	set -o pipefail; $(MAKE) --no-print-directory sim-soc-budget 2>&1 | tee build/logs/sim-soc-budget.log
	set -o pipefail; $(MAKE) --no-print-directory sim-zs-fashion  2>&1 | tee build/logs/sim-zs-fashion.log
	set -o pipefail; $(MAKE) --no-print-directory sim-soc-fashion 2>&1 | tee build/logs/sim-soc-fashion.log
	$(PY) scripts/write_results.py

clean:
	rm -rf build

sim-dense:
	$(PY) software/export_dense_lists.py
	mkdir -p build
	iverilog -o build/sim_dense hardware/tb/tb_dense_mlp.v hardware/rtl/sparse_mlp.v hardware/rtl/sparse_dot.v hardware/rtl/mac.v
	vvp build/sim_dense
