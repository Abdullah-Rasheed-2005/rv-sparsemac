// tb_sparsemac_pcpi.v - Test of the PCPI wrapper WITHOUT a CPU
//
// A small task plays the part of PicoRV32's PCPI side: it puts an instruction on
// the wires, keeps pcpi_valid high, and waits for pcpi_ready exactly as the core
// does. This lets us test things that a normal C program never does:
//
//   - the co-processor raises pcpi_wait at once for its own instructions
//     (PicoRV32 traps "illegal instruction" if nobody does within 16 cycles),
//   - it stays silent for instructions that are not its own (other opcode,
//     funct7 != 0, funct3 = 6..7), so the core can still trap on them,
//   - pcpi_ready is a single cycle, and only while pcpi_valid is high,
//   - SMAC.LOGIT with a number above 9 gives 0,
//   - SMAC.CYC equals the number of cycles the accelerator was busy,
//   - several images in a row, and a RUN right after a RUN,
//   - the result of the golden images through the PCPI wires.
//
// Run from the repository root: `make sim-pcpi` (needs `make export-sparse`).
`timescale 1ns/1ps
module tb_sparsemac_pcpi;
    parameter ZERO_SKIP = 1;

    reg clk = 0, resetn = 0;
    always #5 clk = ~clk;

    reg         pcpi_valid = 0;
    reg  [31:0] pcpi_insn = 0, pcpi_rs1 = 0, pcpi_rs2 = 0;
    wire        pcpi_wr, pcpi_wait, pcpi_ready;
    wire [31:0] pcpi_rd;

    // memory port of the image loader (SMAC.RUNM). The "memory" here is img_mem:
    // address a = byte a of the test images. Two clock cycles per access.
    wire        ld_valid;
    wire [31:0] ld_addr;
    reg  [31:0] ld_rdata = 0;
    reg         ld_ready = 0;

    sparsemac_pcpi #(.ZERO_SKIP(ZERO_SKIP)) dut (
        .clk(clk), .resetn(resetn),
        .pcpi_valid(pcpi_valid), .pcpi_insn(pcpi_insn),
        .pcpi_rs1(pcpi_rs1), .pcpi_rs2(pcpi_rs2),
        .pcpi_wr(pcpi_wr), .pcpi_rd(pcpi_rd),
        .pcpi_wait(pcpi_wait), .pcpi_ready(pcpi_ready),
        .ld_valid(ld_valid), .ld_addr(ld_addr), .ld_rdata(ld_rdata), .ld_ready(ld_ready)
    );

    // R-type instruction word: funct7 rs2 rs1 funct3 rd opcode
    function [31:0] insn(input [6:0] f7, input [2:0] f3, input [6:0] op);
        insn = {f7, 5'd0, 5'd0, f3, 5'd0, op};
    endfunction

    integer errors = 0;
    integer ready_cycles;       // how many cycles pcpi_ready was high in the last instruction
    integer wait_seen;          // pcpi_wait was high in the first cycle of the instruction
    integer total_cycles;       // cycles from pcpi_valid to pcpi_ready
    reg [31:0] result;
    reg        got_wr;

    // Execute one instruction the way the core does. Returns after ready (or timeout).
    task exec(input [31:0] ins, input [31:0] rs1, input [31:0] rs2);
        integer n;
        begin
            @(negedge clk);
            pcpi_insn = ins; pcpi_rs1 = rs1; pcpi_rs2 = rs2; pcpi_valid = 1;
            wait_seen = 0; ready_cycles = 0; total_cycles = 0; result = 0; got_wr = 0;
            n = 0;
            // sample just before each rising edge (like the core does)
            while (ready_cycles == 0 && n < 200000) begin
                #4;   // 1 ns before the posedge
                if (n == 0 && pcpi_wait) wait_seen = 1;
                if (pcpi_ready) begin
                    ready_cycles = ready_cycles + 1;
                    result = pcpi_rd; got_wr = pcpi_wr;
                end
                @(posedge clk); #1;
                n = n + 1;
                total_cycles = n;
            end
            // the core drops valid in the cycle after it sees ready
            @(negedge clk);
            if (pcpi_ready) begin
                errors = errors + 1;
                $display("FAIL: pcpi_ready still high one cycle after the core dropped valid");
            end
            pcpi_valid = 0;
            @(negedge clk);
            if (pcpi_ready || pcpi_wait) begin
                errors = errors + 1;
                $display("FAIL: pcpi_ready/pcpi_wait high while pcpi_valid is low");
            end
        end
    endtask

    // An instruction that must be ignored: ready/wait must never rise.
    task exec_foreign(input [31:0] ins, input [255:0] name);
        integer n;
        reg seen;
        begin
            @(negedge clk);
            pcpi_insn = ins; pcpi_rs1 = 32'd1; pcpi_rs2 = 32'd2; pcpi_valid = 1;
            seen = 0;
            for (n = 0; n < 40; n = n + 1) begin
                #4; if (pcpi_wait || pcpi_ready) seen = 1;
                @(posedge clk); #1;
            end
            @(negedge clk);
            pcpi_valid = 0;
            if (seen) begin
                errors = errors + 1;
                $display("FAIL: co-processor answered a foreign instruction (%0s, %h)", name, ins);
            end
            repeat (2) @(negedge clk);
        end
    endtask

    // ---------------- data ----------------
    reg [7:0]  img_mem  [0:78399];
    reg [31:0] gold_log [0:999];
    reg [7:0]  gold_pred [0:99];

    integer img, w, c, busy_cycles, e0;

    always @(posedge clk) begin
        ld_ready <= 1'b0;
        if (ld_valid && !ld_ready) begin
            ld_ready <= 1'b1;
            ld_rdata <= {img_mem[ld_addr + 3], img_mem[ld_addr + 2], img_mem[ld_addr + 1], img_mem[ld_addr]};
        end
    end
    reg [31:0] word;
    reg [31:0] pred_r, cyc_r;
    reg [31:0] ct1, ct2;        // SMAC.CYC of two different images in constant-time mode
    reg [31:0] lc1, lc2;        // cycles spent before the accelerator starts, two different images

    // count the cycles the accelerator is busy (hierarchical reference into the wrapper)
    always @(posedge clk) begin
        if (!resetn) busy_cycles <= 0;
        else if (dut.nn_start) busy_cycles <= 0;
        else if (dut.nn_busy) busy_cycles <= busy_cycles + 1;
    end

    task run_image(input integer n);
        integer w2, k;
        begin
            for (w2 = 0; w2 < 196; w2 = w2 + 1) begin
                word = 32'd0;
                for (k = 0; k < 4; k = k + 1) word = word | (img_mem[n*784 + w2*4 + k] << (8*k));
                exec(insn(7'd0, 3'd0, 7'h0b), w2*4, word);
                if (ready_cycles != 1 || !wait_seen || !got_wr || result !== 32'd0) begin
                    errors = errors + 1;
                    if (errors < 10) $display("FAIL: SMAC.LDW handshake (ready=%0d wait=%0d wr=%0d rd=%h)",
                                              ready_cycles, wait_seen, got_wr, result);
                end
            end
            exec(insn(7'd0, 3'd1, 7'h0b), 0, 0);                     // SMAC.RUN
            if (ready_cycles != 1 || !wait_seen || !got_wr) begin
                errors = errors + 1;
                $display("FAIL: SMAC.RUN handshake (ready=%0d wait=%0d wr=%0d)", ready_cycles, wait_seen, got_wr);
            end
            pred_r = result;
        end
    endtask

    integer pass_ct;
    initial begin
        $readmemh("hardware/mem/test_images.hex",   img_mem, 0, 78399);
        $readmemh("hardware/mem/golden_logits.hex", gold_log);
        $readmemh("hardware/mem/golden_pred.txt",   gold_pred);

        repeat (3) @(posedge clk);
        #1 resetn = 1;
        repeat (3) @(posedge clk);

        // ---- instructions that are not ours must be ignored ----
        exec_foreign(insn(7'd0, 3'd6, 7'h0b), "funct3=6");
        exec_foreign(insn(7'd0, 3'd7, 7'h0b), "funct3=7");
        exec_foreign(insn(7'd1, 3'd1, 7'h0b), "funct7=1");
        exec_foreign(insn(7'd0, 3'd1, 7'h2b), "custom-1 opcode");
        exec_foreign(insn(7'd1, 3'd0, 7'h33), "mul (opcode 0x33)");

        // ---- golden images through the PCPI wires ----
        pass_ct = 0;
        for (img = 0; img < 6; img = img + 1) begin
            run_image(img);
            // cycles measured by the wrapper vs. cycles the accelerator was busy
            exec(insn(7'd0, 3'd3, 7'h0b), 0, 0);                     // SMAC.CYC
            cyc_r = result;
            if (cyc_r !== busy_cycles) begin
                errors = errors + 1;
                $display("FAIL: SMAC.CYC = %0d but the accelerator was busy %0d cycles", cyc_r, busy_cycles);
            end
            if (img == 0) $display("image 0: SMAC.CYC = %0d, accelerator busy for %0d cycles", cyc_r, busy_cycles);

            for (c = 0; c < 10; c = c + 1) begin
                exec(insn(7'd0, 3'd2, 7'h0b), c, 0);                 // SMAC.LOGIT c
                if (result !== gold_log[img*10 + c]) begin
                    errors = errors + 1;
                    if (errors < 10) $display("FAIL img=%0d class=%0d got=%0d expected=%0d",
                                              img, c, $signed(result), $signed(gold_log[img*10 + c]));
                end
            end
            if (pred_r !== {24'd0, gold_pred[img]}) begin
                errors = errors + 1;
                $display("FAIL img=%0d pred=%0d expected=%0d", img, pred_r, gold_pred[img]);
            end else pass_ct = pass_ct + 1;
        end

        // ---- out-of-range logit numbers ----
        exec(insn(7'd0, 3'd2, 7'h0b), 10, 0);
        if (result !== 32'd0) begin errors = errors + 1; $display("FAIL: LOGIT 10 = %h, expected 0", result); end
        exec(insn(7'd0, 3'd2, 7'h0b), 32'hFFFFFFFF, 0);
        if (result !== 32'd0) begin errors = errors + 1; $display("FAIL: LOGIT -1 = %h, expected 0", result); end
        exec(insn(7'd0, 3'd2, 7'h0b), 32'h00000100, 0);
        if (result !== 32'd0) begin errors = errors + 1; $display("FAIL: LOGIT 256 = %h, expected 0", result); end

        // ---- SMAC.CFG: run-time budget and constant time ----
        exec(insn(7'd0, 3'd4, 7'h0b), 32'd500, 32'd1);           // budget 500 cycles, constant time
        if (ready_cycles != 1 || !wait_seen || !got_wr || result !== 32'd0) begin
            errors = errors + 1;
            $display("FAIL: SMAC.CFG handshake (ready=%0d wait=%0d wr=%0d rd=%h)", ready_cycles, wait_seen, got_wr, result);
        end
        if (ZERO_SKIP) begin
            run_image(3);  exec(insn(7'd0, 3'd3, 7'h0b), 0, 0);  ct1 = result;
            run_image(5);  exec(insn(7'd0, 3'd3, 7'h0b), 0, 0);  ct2 = result;
            $display("constant time (budget 500): SMAC.CYC = %0d and %0d for two different images", ct1, ct2);
            if (ct1 !== ct2) begin
                errors = errors + 1;
                $display("FAIL: constant time, the two images took a different number of cycles");
            end
            exec(insn(7'd0, 3'd4, 7'h0b), 32'd500, 32'd0);       // budget only
            run_image(3);  exec(insn(7'd0, 3'd3, 7'h0b), 0, 0);
            $display("budget 500, not constant time: SMAC.CYC = %0d", result);
            if (result > 32'd1300) begin
                errors = errors + 1;
                $display("FAIL: budget 500 but the run took %0d cycles (bound 1300)", result);
            end
        end
        exec(insn(7'd0, 3'd4, 7'h0b), 32'd0, 32'd0);             // back to no limit (image 7 below checks it)

        // ---- SMAC.RUNM: the wrapper fetches the image itself (order.hex of hardware/mem = normal order) ----
        for (img = 8; img < 12; img = img + 1) begin
            exec(insn(7'd0, 3'd5, 7'h0b), img * 784, 0);         // rs1 = address of the image
            if (ready_cycles != 1 || !wait_seen || !got_wr) begin
                errors = errors + 1;
                $display("FAIL: SMAC.RUNM handshake (ready=%0d wait=%0d wr=%0d)", ready_cycles, wait_seen, got_wr);
            end
            pred_r = result;
            exec(insn(7'd0, 3'd3, 7'h0b), 0, 0);                 // SMAC.CYC = whole instruction
            cyc_r = result;
            if (img == 8) lc1 = cyc_r - busy_cycles;
            if (img == 9) lc2 = cyc_r - busy_cycles;
            for (c = 0; c < 10; c = c + 1) begin
                exec(insn(7'd0, 3'd2, 7'h0b), c, 0);
                if (result !== gold_log[img*10 + c]) begin
                    errors = errors + 1;
                    if (errors < 10) $display("FAIL SMAC.RUNM img=%0d class=%0d got=%0d expected=%0d",
                                              img, c, $signed(result), $signed(gold_log[img*10 + c]));
                end
            end
            if (pred_r !== {24'd0, gold_pred[img]}) begin
                errors = errors + 1;
                $display("FAIL SMAC.RUNM img=%0d pred=%0d expected=%0d", img, pred_r, gold_pred[img]);
            end
        end
        $display("SMAC.RUNM: 4 images matched; fetching + ordering took %0d and %0d cycles for two different images", lc1, lc2);
        if (lc1 !== lc2) begin
            errors = errors + 1;
            $display("FAIL: the loading time of SMAC.RUNM depends on the image");
        end

        // ---- RUN right after RUN (no new pixels): zero-skip design consumes the image,
        //      the weight-skip design keeps it. Only check that it finishes and answers. ----
        exec(insn(7'd0, 3'd1, 7'h0b), 0, 0);
        if (ready_cycles != 1) begin errors = errors + 1; $display("FAIL: second RUN did not finish"); end

        // ---- reset in the middle of nothing: logits keep working afterwards ----
        run_image(7);
        if (pred_r !== {24'd0, gold_pred[7]}) begin
            errors = errors + 1; $display("FAIL: image 7 pred=%0d expected=%0d", pred_r, gold_pred[7]);
        end

        $display("%0d golden images matched through the PCPI wires (ZERO_SKIP=%0d)", pass_ct + 1, ZERO_SKIP);
        if (errors == 0) $display("PASS: sparsemac_pcpi handshake and results are correct");
        else             $display("FAILED: %0d errors", errors);
        $finish;
    end

    initial begin
        #2000000000;
        $display("FAILED: simulation timeout");
        $finish;
    end
endmodule
