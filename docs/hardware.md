# Hardware design

This document explains the hardware in the order it was built. Every claim here
is checked by a simulation (`make help` lists them).

```
mac.v  ->  sparse_dot.v  ->  sparse_mlp.v        weights skipped            (output by output)
                         ->  sparse_mlp_zs.v     weights AND zeros skipped  (input by input)
                                                 + optional run-time budget / constant time (section 6)
                                   |
                                   v
                         sparsemac_pcpi.v  ->  PicoRV32 (custom instructions)  ->  firmware/main.c
```

## 1. Two ways to loop over a matrix

The layer computes `out[n] = bias[n] + sum over j of x[j] * w[n][j]`.
There are two loop orders, and they decide which zeros can be skipped.

**Output by output** (`sparse_mlp.v`). For each neuron `n`, walk its list of
nonzero weights. Zero weights cost nothing. But the cycle that handles weight
`w[n][j]` also has to read pixel `x[j]`, and if the pixel is zero the cycle is
wasted: the hardware only finds out after reading it.

**Input by input** (`sparse_mlp_zs.v`). For each nonzero input `j`, walk the
nonzero weights of column `j`, and add `x[j] * w[n][j]` into `acc[n]`.

```
for j in nonzero inputs:             <- zero pixels / zero hidden values never appear
    for (n, w) in column j:          <- zero weights never appear
        acc[n] += x[j] * w
```

Every cycle now does one useful multiply-accumulate. The price: all 64 sums of
layer 1 are alive at the same time (a 64 x 32-bit register array), and the
weights must be stored by column.

## 2. Weight storage: CSR and CSC

The same sparse matrix, written two ways (`software/export_sparse.py`):

| | CSR (by row = by neuron) | CSC (by column = by input) |
|---|---|---|
| files | `fcN_nz_ptr/idx/val.hex` | `fcN_csc_ptr/row/val.hex` |
| `ptr` | `ptr[n]` .. `ptr[n+1]-1` = entries of neuron `n` | `ptr[j]` .. `ptr[j+1]-1` = entries of input `j` |
| per entry | input index + weight | output neuron + weight |
| used by | `sparse_dot.v`, `sparse_mlp.v` | `sparse_mlp_zs.v` |

Example (3 inputs, 2 neurons): `W = [[5,0,0],[0,0,-2]]`.
CSC: `ptr = [0,1,1,2]`, `row = [0,1]`, `val = [5,-2]`. Column 1 is empty
(`ptr[1] == ptr[2]`), so input 1 is never worth a MAC, even when it is nonzero.

## 3. `sparse_mlp_zs.v` step by step

### 3.1 The list of nonzero inputs

Pixels arrive one per clock on the write port, in order 0..783. A pixel that is
zero is dropped; a nonzero pixel is appended to a list `(position, value)`.
Writing address 0 empties the list first, so the list is built at no extra
cycle cost and no dense image RAM exists. After layer 1, the finish sweep
builds the same kind of list from the nonzero hidden values.

### 3.2 Pipeline

```
 list RAM        pointer ROM        column ROMs          register file
+----------+    +-----------+      +-----------+        +--------------+
| (j, a)   | -> | ptr[j]    | ->   | (row, w)  |  ->    | acc[row] +=  |
| entry k  |    | ptr[j+1]  |      | entry e   |        |   a * w      |
+----------+    +-----------+      +-----------+        +--------------+
  front end: 3 cycles, runs one column ahead      issue stage: 1 entry/cycle   MAC stage
```

- **Front end** reads list entry `k`, then the two column pointers, and puts
  `{lo, hi, a}` into a descriptor register (3 cycles). It starts as soon as the
  previous descriptor is taken, so while one column is streamed the next one is
  prepared. A column of 3 or more entries hides this completely.
- **Issue stage** reads entries `lo .. hi-1`, one per clock. In the clock where
  it issues the last entry of a column it already takes the next descriptor, so
  there is no bubble between columns.
- **MAC stage** does `acc[row] <= acc[row] + a * w` in one clock. The memories
  are synchronous (data one clock after the address), like FPGA block RAM.

Why no hazard? Within a column all rows are different, and the read-modify-write
of `acc` finishes inside one clock, so the next entry always sees the new value.

### 3.3 `touched`: accumulators without clearing

`touched[n] = 1` means `acc[n]` holds a partial sum. The first MAC into a neuron
counts the old value as 0. Clearing all 64 accumulators between layers and
images would cost cycles; resetting a 64-bit flag vector costs none. A neuron
with no nonzero weight or no nonzero input is never touched and correctly ends
up as just its bias.

### 3.4 Finish sweep

After the last MAC has landed, one neuron per clock:

```
v = acc[n] + bias[n]
layer 1:  hidden = min( max(v,0) >> SHIFT , 127 )   -> hid[n], and (n, hidden) into the list if hidden != 0
layer 2:  out_valid pulse with the logit v; the maximum is tracked (first maximum wins ties)
```

### 3.5 Cycle model

```
cycles ~ (nonzero pixel x nonzero weight pairs in fc1)
       + (nonzero hidden x nonzero weight pairs in fc2)
       + about 64 + 10   (finish sweeps)
       + a few cycles of pipeline start-up per layer
```

`export_sparse.py` prints the pair count (the "useful MACs"); `make sim-zs`
prints the measured cycles and the difference.

### 3.6 Limits (the design relies on these)

| Limit | Why |
|---|---|
| pixels written in order 0..783 | the list is appended in write order |
| no pixel writes while busy | the list RAM is in use |
| image is consumed by a run | the list is reused for the hidden values |
| at most 16384 nonzero weights per layer, 1024 inputs, 64 neurons in fc1 | array sizes / index widths |
| `shift` below 32 | 5-bit shift amount |

## 4. Verification

| Test | What it proves |
|---|---|
| `make sim-zs` | 100 golden images bit-exact (logits, prediction, hidden values of 10 images); 24 more images against a dense reference model written as plain loops: all-zero, all-255, single pixels, random densities 5..100 %, a second `start` while busy, a start without pixels |
| `make sim-zs-edge` | the same on an artificial network with saturating hidden values, a fully pruned neuron, empty columns, unused hidden inputs and exactly tied logits |
| mutation checks (done by hand while developing) | removing the list reset, the `touched` clear, the saturation, the ReLU, the tie rule, the "empty column" case or the signed weight each makes at least one test fail |

The real MNIST model never produces ties or saturation on the 100 golden images,
which is why the edge network exists: without it a wrong tie rule or a missing
saturation would pass every golden test.

## 5. The RISC-V system

```
+-------------+  memory bus   +---------------------------+
|  PicoRV32   | <-----------> | RAM, console, result, exit | (testbench)
|  RV32IM     |               +---------------------------+
|             |  PCPI wires   +---------------------------+
|             | <-----------> | sparsemac_pcpi.v           |
+-------------+               |   -> sparse_mlp_zs.v       |
                              +---------------------------+
```

PicoRV32's PCPI works like this: the core fetches an instruction it cannot
execute, raises `pcpi_valid` and shows `pcpi_insn`, `pcpi_rs1`, `pcpi_rs2`. A
co-processor that knows the instruction raises `pcpi_wait` (otherwise the core
traps after 16 cycles), and finally gives one cycle of `pcpi_ready` with
`pcpi_wr = 1` and the result in `pcpi_rd`. The core writes `pcpi_rd` into `rd`.

### 5.1 The instructions

All are R-type, opcode `0x0B` (the RISC-V *custom-0* space), `funct7 = 0`:

| funct3 | Name | rs1 | rs2 | rd | Meaning |
|---|---|---|---|---|---|
| 0 | `SMAC.LDW` | pixel address (0,4,...,780) | 4 pixels, little-endian | 0 | write 4 pixels (4 clocks) |
| 1 | `SMAC.RUN` | - | - | predicted digit | start and wait for the result |
| 2 | `SMAC.LOGIT` | class 0..9 | - | logit (0 if rs1 > 9) | read a logit of the last run |
| 3 | `SMAC.CYC` | - | - | cycles | accelerator clocks of the last run |
| 4 | `SMAC.CFG` | layer-1 cycle budget (0 = no limit) | bit 0: constant time | 0 | bound the run time (section 6) |
| 5 | `SMAC.RUNM` | address of the image (784 bytes, normal pixel order) | - | predicted digit | fetch the image from memory, then run (section 5.5) |

`funct3` 6..7, `funct7 != 0` and other opcodes are not claimed, so the core
still traps on them. `make sim-pcpi` checks this.

In C they are written with the assembler directive `.insn` (`firmware/smac.h`):
`.insn r 0x0b, 1, 0, rd, rs1, rs2` means opcode `0x0b`, `funct3 = 1`,
`funct7 = 0`. No custom compiler is needed.

### 5.2 Memory map of the simulated system

Defined once in `firmware/layout.h`; the Makefile turns it into
`build/layout.vh` for the testbench.

| Address | Content |
|---|---|
| `0x00000000` | firmware (code, data, stack below 64 KB) |
| `0x00010000` | fc1 weights (dense int8, for the software baseline) |
| `0x0001C400` | fc2 weights |
| `0x0001C800` / `0x0001C900` | fc1 / fc2 bias (int32) |
| `0x0001CA00` | hidden shift |
| `0x0001CB00` | true labels |
| `0x00020000` | test images (784 bytes each) |
| `0x00040000` | the test images in normal pixel order, read by the accelerator (`SMAC.RUNM`) |
| `0x00034000` / `0x00034800` / `0x00037000` | fc1 by column (CSC): pointers (uint16), rows, weights |
| `0x00039800` / `0x00039900` / `0x00039C00` | fc2 by column (CSC): pointers (uint16), rows, weights |
| `0x10000000` | console (write a byte = print it) |
| `0x10000010` | result port (the firmware reports logits and predictions) |
| `0x10000020` | exit (write the exit code) |

The accelerator keeps its own weight memories (filled from `hardware/mem/*.hex`
by `$readmemh`); the dense and CSC weights in RAM exist only for the software baselines.

### 5.3 What the firmware does

1. For each of the 100 images: send the 196 pixel words with `SMAC.LDW`,
   `SMAC.RUN`, then ten `SMAC.LOGIT`. Then the same images again with one
   `SMAC.RUNM` per image instead of the 196 `SMAC.LDW` and the `SMAC.RUN`.
2. Run the same network in plain C on the CPU, in three ways, and compare the
   logits with the accelerator: dense (1 image), skipping zero pixels (2 images),
   and the accelerator's own algorithm with CSC weights (`sw_csc`, all 100 images).
3. Print cycle counts measured with `rdcycle`.

The testbench independently compares everything the firmware reports with
`golden_logits` / `golden_pred`.

### 5.4 Reading the numbers fairly

- The CPU's memory has one wait state per access, so the CPU runs at roughly a
  third of an instruction per clock. This is realistic for a small core with
  SRAM and is the same for every program.
- The dense software baseline is the obvious first program. The second one
  skips zero pixels. The third one (`sw_csc`) is the fair comparison: it runs the
  accelerator's algorithm on the accelerator's weight format, so the remaining
  speedup is what the hardware itself adds. Quote that number first.
- The dense and skip-zero-pixels programs are run on very few images (a dense
  image takes millions of simulated cycles), so those two numbers are rough.
  The CSC software is run on all 100 images.
- The accelerator numbers include the cost of sending the image through the
  custom instruction, which is a large part of the total (the CPU needs a load,
  an address calculation and the custom instruction for every 4 pixels). The
  compute alone is `SMAC.CYC`. Skipping all-zero words in the firmware was tried
  and made loading slower, because the extra test per word costs more than the
  skipped instructions save. The real fix is for the accelerator to read the
  image from RAM itself: `SMAC.RUNM`, section 5.5.
- These are simulation cycle counts of a design with simulation memories. They
  are not FPGA timing and not energy.

### 5.5 The accelerator loads the image itself (`SMAC.RUNM`)

Sending the pixels with `SMAC.LDW` costs more CPU time than the accelerator
needs for the whole network. `SMAC.RUNM` removes that: the CPU only passes the
address of the image, and the wrapper (`sparsemac_pcpi.v`) does the rest.

```
FETCH   196 bus reads, 4 pixels each, into a small buffer        2 cycles per word
ORDER   for k = 0..783: pixel order[k] from the buffer -> input k   1 cycle per pixel
RUN     as SMAC.RUN
```

- **Second bus master.** The wrapper has its own memory port (`ld_valid`,
  `ld_addr`, `ld_rdata`, `ld_ready`, read only). While PicoRV32 waits for a
  co-processor instruction it uses the bus only once (it prefetches the next
  instruction), so the system gives the bus to the wrapper whenever the CPU
  does not ask for it. The CPU has priority. The testbench counts the cycles the
  loader had to wait: one per instruction.
- **Order table.** `order.hex` says which pixel is input number `k`. For the
  normal model it is 0, 1, 2, ...; for the budget model it is the fixed
  most-useful-first order (section 6.3). The image in memory stays in its
  normal pixel order; no software has to reorder anything.
- **Same time for every image.** FETCH and ORDER do not depend on the pixel
  values, so `SMAC.RUNM` adds a constant to the run time. With a budget the
  whole instruction is bounded, and in constant-time mode the whole instruction
  takes the same number of cycles for every image.
- `SMAC.CYC` after `SMAC.RUNM` returns the cycles of the whole instruction.

Checked by `make sim-pcpi` (wrapper alone, with a small memory model),
`make sim-soc` and `make sim-soc-budget` (with PicoRV32).

## 6. Bounded run time

### 6.1 The problem

A zero-skipping accelerator is fast on average, but its run time depends on the
input: 2,163 cycles for an average digit, more than 10,000 for an image in which
every pixel is nonzero. That has three consequences:

- a real-time system must plan for the slowest input, not the average one;
- anybody who can choose the input can make the accelerator several times slower;
- the run time tells something about the image (a "1" has fewer nonzero pixels than an "8").

### 6.2 The rule

`sparse_mlp_zs.v` has two extra inputs, set from the CPU with `SMAC.CFG`:

```
budget      0 = no limit. Otherwise layer 1 may spend at most 'budget' cycles on columns.
            A column of n nonzero weights costs max(n, 3) cycles
            (the front end needs 3 cycles per column, see 3.2).
            A column is started only if it still fits.
            The first column that does not fit ends layer 1; the remaining inputs are dropped.
const_time  1 = 'done' comes exactly budget + PAD cycles after 'start' (PAD = 800).
```

The check sits in the front end, in the cycle where the two column pointers have
arrived: `used + max(len, 3) > budget` sets a flag `cut`, and `cut` stops the
fetching of further inputs. The columns already granted finish normally, so the
run phase of layer 1 takes at most `budget` plus a few start-up cycles. Layer 2
(at most 640 weights) and the two finish sweeps always run completely; they are
covered by `PAD`.

### 6.3 Why the order of the inputs matters

Inputs are visited in the order they were written. If the budget ends the layer,
the inputs that were not reached are lost, so the most useful inputs must come
first. Two things make this work:

1. A fixed order, chosen once from the weights: columns with the largest
   sum of |weight| per cycle first (`software/budget_lib.py`). No sorting in hardware.
   The order is built into the exported files (`software/export_budget.py`): input
   `k` of the hardware is pixel `order[k]`, for the weight columns and for the images.
2. Budget-aware fine-tuning (`software/budget_experiment.py`): the pruned network
   is trained a little more while its inputs are cut at random budgets, so it
   learns to decide from the first inputs. The pruning mask is kept.

For the 80 % pruned network and a budget of 1,700 MACs: about 81 % accuracy
without the order, about 95.7 % with the order, about 96.7 % with the order and
the fine-tuning (10,000 test images, `budget_experiment.py`; unlimited: about 97.0 %).

### 6.3b Which model, which budget

Pruning and the budget do different jobs: pruning lowers the average cost, the
budget bounds the worst case. `software/budget_pareto.py` combines them
([budget_pareto.md](budget_pareto.md)). The hardware simulations use the 90 %
pruned model with a budget of 1,000 cycles: 96.81 % accuracy against 96.85 %
without a budget, 857 layer-1 cycles on average, never more than 1,000.

`software/budget_study.py` ([budget_study.md](budget_study.md)) compares the
budget with two other ways to guarantee the same worst case, for the 80 %
model: pruning more (93.1 % at 1,700 cycles) and keeping a fixed subset of
pixels (79.7 %), against 96.7 % with the budget; three random seeds differ by
about 0.1 point at most.

### 6.4 Verification

| Test | What it proves |
|---|---|
| `make sim-zs-budget` | budget off: unchanged results. Budget on: 100 images bit-exact against golden values computed by `budget_lib.py`, every run below budget + 800 cycles, stress images (all zero, all 255, single pixels, random densities) against a reference model. Constant time: every image takes exactly the same number of cycles |
| `make sim-pcpi` | `SMAC.CFG` handshake, two different images take the same `SMAC.CYC` in constant time, and results are back to normal after `SMAC.CFG 0` |
| `make sim-soc-budget` | PicoRV32 sets the budget with `SMAC.CFG`; three passes over 100 images (no budget, budget, constant time) match the golden values |

The cycle-accurate behaviour was also modelled register by register in Python
before the Verilog was written; the simulation reproduced that model's numbers.

### 6.5 Limits

- With `SMAC.LDW` the bound covers only the accelerator, and the images must
  already be stored in the fixed order. With `SMAC.RUNM` (section 5.5) the image
  is fetched and ordered by the accelerator in a fixed number of cycles, so the
  bound and the constant time cover the whole instruction.
- Fetching and ordering take about as long as the computation for this small network.
- `out_valid` pulses still come at a data-dependent time in constant-time mode;
  only `done` (and therefore the end of `SMAC.RUN`) is constant. Power and
  electromagnetic side channels are not addressed.
- `PAD = 800` is a safe value for this network size, not a tight one.
- The model and budget were chosen by looking at test-set accuracy of several
  configurations, which is slightly optimistic. The combined experiment used one
  random seed per sparsity and twice as many fine-tuning epochs as the plain models.
- MNIST only.
