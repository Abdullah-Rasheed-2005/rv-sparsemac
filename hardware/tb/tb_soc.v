// tb_soc.v - PicoRV32 + sparse accelerator (PCPI) + RAM, running the firmware
//
//        +-----------+  native memory bus   +----------------------------+
//        | PicoRV32  | <------------------> | RAM 256 KB (byte array)    |
//        |           |                      | + 3 test "devices" (MMIO)  |
//        |           |  PCPI wires          +----------------------------+
//        |           | <------------------> sparsemac_pcpi -> accelerator
//        +-----------+
//
// What this testbench does
//   1. fills the RAM with the firmware (build/fw/fw.hex) and with the data files
//      of hardware/mem/ at the addresses given in firmware/layout.h,
//   2. releases the reset; the CPU runs the firmware,
//   3. prints everything the firmware writes to the console device,
//   4. collects the logits and predictions the firmware reports through the
//      result device and compares them with golden_logits / golden_pred,
//   5. ends when the firmware writes its exit code (0 = the firmware found no
//      difference between its software result and the accelerator result).
//
// Memory timing: every access takes 2 clock cycles (one wait state), like a
// simple synchronous SRAM. This slows the CPU (and so the software baseline)
// compared with zero-wait memory. The accelerator's own memories are not on
// this bus and have no wait state, so keep this in mind when reading speedups.
//
// Run from the repository root with `make sim-soc` (or `make sim-soc-ws`).
`timescale 1ns/1ps
`include "layout.vh"          // generated from firmware/layout.h by the Makefile
`ifndef MEMDIR
  `define MEMDIR "hardware/mem"
`endif

module tb_soc;
`ifdef WEIGHT_SKIP_ONLY
    localparam ZERO_SKIP = 0;
`else
    localparam ZERO_SKIP = 1;
`endif
    localparam MAX_IMG = 100;
`ifdef BUDGET_RUN
    localparam NPASS = 3;                 // no budget, budget, budget + constant time
`else
    localparam NPASS = 1;
`endif

    reg clk = 0;
    reg resetn = 0;
    always #5 clk = ~clk;                 // 100 MHz

    // ---------------- CPU ----------------
    wire        trap;
    wire        mem_valid, mem_instr;
    reg         mem_ready;
    wire [31:0] mem_addr, mem_wdata;
    wire [3:0]  mem_wstrb;
    reg  [31:0] mem_rdata;

    wire        pcpi_valid, pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_insn, pcpi_rs1, pcpi_rs2, pcpi_rd;

    picorv32 #(
        .ENABLE_COUNTERS     (1),
        .ENABLE_COUNTERS64   (1),
        .ENABLE_REGS_16_31   (1),
        .ENABLE_REGS_DUALPORT(1),
        .CATCH_MISALIGN      (1),
        .CATCH_ILLINSN       (1),
        .ENABLE_PCPI         (1),       // our co-processor
        .ENABLE_MUL          (0),
        .ENABLE_FAST_MUL     (1),       // 'mul' for the software baseline
        .ENABLE_DIV          (0),
        .ENABLE_IRQ          (0),
        .COMPRESSED_ISA      (0),
        .PROGADDR_RESET      (32'h0000_0000),
        .STACKADDR           (`STACK_TOP)
    ) cpu (
        .clk(clk), .resetn(resetn), .trap(trap),
        .mem_valid(mem_valid), .mem_instr(mem_instr), .mem_ready(mem_ready),
        .mem_addr(mem_addr), .mem_wdata(mem_wdata), .mem_wstrb(mem_wstrb),
        .mem_rdata(mem_rdata),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .irq(32'b0)
    );

    sparsemac_pcpi #(.ZERO_SKIP(ZERO_SKIP)) accel (
        .clk(clk), .resetn(resetn),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready)
    );

    // ---------------- RAM and test devices ----------------
    reg [7:0] ram [0:`RAM_SIZE-1];

    reg [31:0] res [0:NPASS*MAX_IMG*11-1];      // logits (10) + prediction, per image and pass
    integer    res_cnt = 0;
    reg [31:0] res_time [0:NPASS*MAX_IMG*11-1]; // clock cycle of every result word
    integer    cycle_no = 0;
    reg        exited = 0;
    reg [31:0] exit_code = 0;

    wire [31:0] a = {mem_addr[31:2], 2'b00};

    always @(posedge clk) begin
        cycle_no <= cycle_no + 1;
        mem_ready <= 1'b0;
        if (resetn && mem_valid && !mem_ready) begin
            mem_ready <= 1'b1;
            mem_rdata <= 32'd0;
            if (mem_addr < `RAM_SIZE) begin
                mem_rdata <= {ram[a + 3], ram[a + 2], ram[a + 1], ram[a]};
                if (mem_wstrb[0]) ram[a + 0] <= mem_wdata[7:0];
                if (mem_wstrb[1]) ram[a + 1] <= mem_wdata[15:8];
                if (mem_wstrb[2]) ram[a + 2] <= mem_wdata[23:16];
                if (mem_wstrb[3]) ram[a + 3] <= mem_wdata[31:24];
            end else if (|mem_wstrb) begin
                case (mem_addr)
                    `MMIO_CONSOLE: $write("%c", mem_wdata[7:0]);
                    `MMIO_RESULT: begin
                        if (res_cnt < NPASS*MAX_IMG*11) begin
                            res[res_cnt]      <= mem_wdata;
                            res_time[res_cnt] <= cycle_no;
                            res_cnt           <= res_cnt + 1;
                        end
                    end
                    `MMIO_EXIT: begin
                        exit_code <= mem_wdata;
                        exited    <= 1'b1;
                    end
                    default: $display("TB: write to unmapped address %h", mem_addr);
                endcase
            end
        end
    end

    // ---------------- load memories ----------------
    reg [31:0] tmpw [0:63];
    reg [15:0] tmpp [0:784];              // column pointers (16-bit words in the hex files)
    integer    nz1, nz2;                  // number of nonzero weights in fc1 / fc2
    reg [31:0] gold_log  [0:MAX_IMG*10-1];
    reg [7:0]  gold_pred [0:MAX_IMG-1];
    reg [31:0] gold_logb  [0:MAX_IMG*10-1];   // with the run-time budget (BUDGET_RUN only)
    reg [7:0]  gold_predb [0:MAX_IMG-1];
    integer i, j;

    initial begin
        for (i = 0; i < `RAM_SIZE; i = i + 1) ram[i] = 8'h00;

        $readmemh("build/fw/fw.hex",                 ram);
        $readmemh({`MEMDIR, "/fc1_weights.hex"},    ram, `W1_BASE,     `W1_BASE + 50176 - 1);
        $readmemh({`MEMDIR, "/fc2_weights.hex"},    ram, `W2_BASE,     `W2_BASE + 640 - 1);
        $readmemh({`MEMDIR, "/hidden_shift.hex"},   ram, `SHIFT_ADDR,  `SHIFT_ADDR);
        $readmemh({`MEMDIR, "/test_labels.txt"},    ram, `LABELS_BASE, `LABELS_BASE + MAX_IMG - 1);
        $readmemh({`MEMDIR, "/test_images.hex"},    ram, `IMAGES_BASE, `IMAGES_BASE + MAX_IMG*784 - 1);

        // the biases are 32-bit words in the hex files: store them little-endian
        $readmemh({`MEMDIR, "/fc1_bias.hex"}, tmpw, 0, 63);
        for (i = 0; i < 64; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                ram[`B1_BASE + 4*i + j] = tmpw[i][8*j +: 8];
        $readmemh({`MEMDIR, "/fc2_bias.hex"}, tmpw, 0, 9);
        for (i = 0; i < 10; i = i + 1)
            for (j = 0; j < 4; j = j + 1)
                ram[`B2_BASE + 4*i + j] = tmpw[i][8*j +: 8];

        // the weights stored by column (CSC), for the sparse software baseline.
        // Pointers are 16-bit words: store them little-endian, 2 bytes each.
        $readmemh({`MEMDIR, "/fc1_csc_ptr.hex"}, tmpp, 0, 784);
        nz1 = tmpp[784];
        for (i = 0; i < 785; i = i + 1) begin
            ram[`C1PTR_BASE + 2*i]     = tmpp[i][7:0];
            ram[`C1PTR_BASE + 2*i + 1] = tmpp[i][15:8];
        end
        if (nz1 > 10240) $display("ERROR: tb_soc: fc1 has %0d nonzero weights, the RAM area holds 10240", nz1);
        $readmemh({`MEMDIR, "/fc1_csc_row.hex"}, ram, `C1ROW_BASE, `C1ROW_BASE + nz1 - 1);
        $readmemh({`MEMDIR, "/fc1_csc_val.hex"}, ram, `C1VAL_BASE, `C1VAL_BASE + nz1 - 1);

        $readmemh({`MEMDIR, "/fc2_csc_ptr.hex"}, tmpp, 0, 64);
        nz2 = tmpp[64];
        for (i = 0; i < 65; i = i + 1) begin
            ram[`C2PTR_BASE + 2*i]     = tmpp[i][7:0];
            ram[`C2PTR_BASE + 2*i + 1] = tmpp[i][15:8];
        end
        if (nz2 > 640) $display("ERROR: tb_soc: fc2 has %0d nonzero weights, the RAM area holds 640", nz2);
        $readmemh({`MEMDIR, "/fc2_csc_row.hex"}, ram, `C2ROW_BASE, `C2ROW_BASE + nz2 - 1);
        $readmemh({`MEMDIR, "/fc2_csc_val.hex"}, ram, `C2VAL_BASE, `C2VAL_BASE + nz2 - 1);

        $readmemh({`MEMDIR, "/golden_logits.hex"}, gold_log);
        $readmemh({`MEMDIR, "/golden_pred.txt"},   gold_pred);
`ifdef BUDGET_RUN
        $readmemh({`MEMDIR, "/golden_logits_b.hex"}, gold_logb);
        $readmemh({`MEMDIR, "/golden_pred_b.txt"},   gold_predb);
        $readmemh({`MEMDIR, "/budget.hex"}, tmpp, 0, 0);
        ram[`BUDGET_ADDR]     = tmpp[0][7:0];
        ram[`BUDGET_ADDR + 1] = tmpp[0][15:8];
`endif

        if (ZERO_SKIP) $display("accelerator: sparse_mlp_zs (skips zero weights and zero activations)");
        else           $display("accelerator: sparse_mlp (skips zero weights only)");
        repeat (10) @(posedge clk);
        resetn = 1;
    end

    // ---------------- finish and check ----------------
    integer img, c, nimg, errors, pred_errors, span, ps, base;
    reg [31:0] exp_log;
    reg [7:0]  exp_pred;

    always @(posedge clk) begin
        if (resetn && trap) begin
            $display("\nFAILED: the CPU trapped (illegal instruction, bad address or misaligned access) at cycle %0d", cycle_no);
            $finish;
        end
        if (exited) begin
            $display("\n--- testbench ---");
            $display("firmware exit code: %0d", exit_code);
            nimg = res_cnt / 11 / NPASS;          // images per pass
            errors = 0; pred_errors = 0;
            // pass 0: no budget (golden_logits). passes 1 and 2: with the budget (golden_logits_b).
            for (ps = 0; ps < NPASS; ps = ps + 1) begin
                for (img = 0; img < nimg; img = img + 1) begin
                    base = (ps*nimg + img) * 11;
                    for (c = 0; c < 10; c = c + 1) begin
                        exp_log = (ps == 0) ? gold_log[img*10 + c] : gold_logb[img*10 + c];
                        if (res[base + c] !== exp_log) begin
                            errors = errors + 1;
                            if (errors <= 10)
                                $display("FAIL pass=%0d img=%0d class=%0d got=%0d expected=%0d", ps, img, c,
                                         $signed(res[base + c]), $signed(exp_log));
                        end
                    end
                    exp_pred = (ps == 0) ? gold_pred[img] : gold_predb[img];
                    if (res[base + 10] !== {24'd0, exp_pred}) begin
                        pred_errors = pred_errors + 1;
                        $display("FAIL pass=%0d img=%0d pred=%0d expected=%0d", ps, img, res[base + 10], exp_pred);
                    end
                end
            end
            if (NPASS > 1) $display("passes checked: %0d (no budget, budget, budget + constant time)", NPASS);
            $display("images reported by the CPU: %0d", nimg);
            $display("total simulated clock cycles: %0d", cycle_no);
            if (nimg > 1) begin
                span = res_time[(nimg-1)*11 + 10] - res_time[10];
                $display("average per image, whole loop (load + run + read): %0d cycles", span / (nimg - 1));
            end
            if (exit_code == 0 && errors == 0 && pred_errors == 0 && nimg > 0 && res_cnt == NPASS*nimg*11)
                $display("PASS: CPU + accelerator matched golden logits and predictions (%0d images, %0d logits)",
                         nimg, nimg*10);
            else
                $display("FAILED: exit code %0d, %0d logit errors, %0d prediction errors, %0d images",
                         exit_code, errors, pred_errors, nimg);
            $finish;
        end
    end

    // safety net
    initial begin
        #2000000000;
        $display("FAILED: simulation timeout (cycle %0d, %0d result words)", cycle_no, res_cnt);
        $finish;
    end
endmodule
