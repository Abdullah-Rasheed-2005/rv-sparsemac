// tb_sparse_mlp_zs.v - Testbench for sparse_mlp_zs.v (zero skipping on both sides)
//
// Part 1: the 100 golden images
//   - every logit against golden_logits (bit for bit), the prediction against
//     golden_pred, the 64 hidden values of the first 10 images,
//   - cycles from start to done, compared with the "ideal" number of useful
//     MACs (nonzero pixel x nonzero weight pairs), so you can see how much the
//     control and pipeline cost on top of the real work.
//
// Part 2: stress tests that the golden images do not cover. A small dense
//   reference model (plain Verilog loops over the dense weight files) computes
//   the expected answer for
//     - an all-zero image and an all-255 image,
//     - a single pixel at address 0 and at address 783,
//     - random images with 5 / 20 / 50 / 100 % nonzero pixels,
//     - a second 'start' pulse while the accelerator is busy (must be ignored).
//   The images follow each other without a reset, so a leftover from the
//   previous image (for example a stale nonzero list) would show up as an error.
//
// Run from the repository root with `make sim-zs`. Needs `make export-sparse` first.
//
// `make sim-zs-edge` runs the same testbench on an artificial network
// (software/make_edge_model.py) with saturating hidden values, a fully pruned
// neuron, empty columns and tied logits. The files are taken from the folder
// given by the macro MEMDIR (default hardware/mem).
`timescale 1ns/1ps
`ifndef MEMDIR
  `define MEMDIR "hardware/mem"
`endif
module tb_sparse_mlp_zs;
    localparam NIMG = 100;
    localparam NHID_CHECK = 10;

    reg clk = 0, rst = 1, start = 0;
    reg        img_we = 0;
    reg [9:0]  img_waddr = 0;
    reg [7:0]  img_wdata = 0;

    wire        busy, done, out_valid;
    wire [3:0]  out_idx;
    wire signed [31:0] out_val;
    wire [3:0]  pred;

    sparse_mlp_zs #(
        .FC1_PTR   ({`MEMDIR, "/fc1_csc_ptr.hex"}),
        .FC1_ROW   ({`MEMDIR, "/fc1_csc_row.hex"}),
        .FC1_VAL   ({`MEMDIR, "/fc1_csc_val.hex"}),
        .FC1_BIAS  ({`MEMDIR, "/fc1_bias.hex"}),
        .FC2_PTR   ({`MEMDIR, "/fc2_csc_ptr.hex"}),
        .FC2_ROW   ({`MEMDIR, "/fc2_csc_row.hex"}),
        .FC2_VAL   ({`MEMDIR, "/fc2_csc_val.hex"}),
        .FC2_BIAS  ({`MEMDIR, "/fc2_bias.hex"}),
        .SHIFT_FILE({`MEMDIR, "/hidden_shift.hex"})
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .busy(busy), .done(done),
        .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
        .pred(pred)
    );

    always #5 clk = ~clk;   // 100 MHz clock

    // ---------------- golden data ----------------
    reg [7:0]  img_mem  [0:78399];
    reg [31:0] gold_log [0:NIMG*10-1];
    reg [7:0]  gold_pred [0:NIMG-1];
    reg [7:0]  gold_lab  [0:NIMG-1];
    reg [7:0]  gold_hid  [0:NHID_CHECK*64-1];

    // ---------------- dense reference model data ----------------
    reg [7:0]  w1m [0:50175];
    reg [7:0]  w2m [0:639];
    reg [31:0] b1m [0:63];
    reg [31:0] b2m [0:9];
    reg [7:0]  shm [0:0];

    reg [7:0]  cur_img [0:783];      // the image being tested
    integer    ref_hid [0:63];
    integer    ref_log [0:9];
    integer    ref_pred;
    integer    ideal_macs;           // nonzero pixel x nonzero weight pairs (both layers)

    // collect the logits that come out one by one
    reg [31:0] got_log [0:9];
    always @(posedge clk)
        if (out_valid) got_log[out_idx] <= out_val;

    // ---------------- reference model (dense, plain loops) ----------------
    task compute_ref;
        integer n, j, a, xi, wi, sh, best;
        begin
            sh = shm[0];
            ideal_macs = 0;
            for (n = 0; n < 64; n = n + 1) begin
                a = $signed(b1m[n]);
                for (j = 0; j < 784; j = j + 1) begin
                    xi = cur_img[j];
                    wi = $signed(w1m[n*784 + j]);
                    a = a + xi * wi;
                    if (xi != 0 && wi != 0) ideal_macs = ideal_macs + 1;
                end
                if (a < 0) a = 0;
                a = a >>> sh;
                if (a > 127) a = 127;
                ref_hid[n] = a;
            end
            best = 0;
            for (n = 0; n < 10; n = n + 1) begin
                a = $signed(b2m[n]);
                for (j = 0; j < 64; j = j + 1) begin
                    wi = $signed(w2m[n*64 + j]);
                    a = a + ref_hid[j] * wi;
                    if (ref_hid[j] != 0 && wi != 0) ideal_macs = ideal_macs + 1;
                end
                ref_log[n] = a;
                if (n == 0 || a > ref_log[best]) best = n;
            end
            ref_pred = best;
        end
    endtask

    // ---------------- drive one image ----------------
    integer cyc;
    task load_and_run;           // image in cur_img, wait for done, result in got_log / pred
        integer p;
        begin
            for (p = 0; p < 784; p = p + 1) begin
                @(negedge clk);
                img_we = 1; img_waddr = p; img_wdata = cur_img[p];
            end
            @(negedge clk);
            img_we = 0;

            @(negedge clk);
            start = 1;
            @(posedge clk);          // the accelerator accepts the start here
            #1 start = 0;
            cyc = 0;
            while (done !== 1'b1 && cyc < 100000) begin
                @(posedge clk); #1;
                cyc = cyc + 1;
            end
            @(posedge clk); #1;      // one more clock so the last logit is collected
        end
    endtask

    integer errors, hid_errors, pred_errors, correct, tot_cyc, tot_ideal, min_cyc, max_cyc;
    integer img, p, k, t, dens;
    integer seed;
    integer stress_err, stress_n;

    // compare the accelerator against the reference model (stress tests)
    task check_stress(input [8*24-1:0] name);
        integer c;
        begin
            compute_ref;
            stress_n = stress_n + 1;
            for (c = 0; c < 10; c = c + 1)
                if ($signed(got_log[c]) !== ref_log[c]) begin
                    stress_err = stress_err + 1;
                    if (stress_err <= 10)
                        $display("FAIL stress %0s class=%0d got=%0d expected=%0d",
                                 name, c, $signed(got_log[c]), ref_log[c]);
                end
            if (pred !== ref_pred[3:0]) begin
                stress_err = stress_err + 1;
                $display("FAIL stress %0s pred=%0d expected=%0d", name, pred, ref_pred);
            end
            for (c = 0; c < 64; c = c + 1)
                if (dut.hid[c] !== ref_hid[c][7:0]) begin
                    stress_err = stress_err + 1;
                    if (stress_err <= 10)
                        $display("FAIL stress %0s hidden %0d got=%0d expected=%0d",
                                 name, c, dut.hid[c], ref_hid[c]);
                end
        end
    endtask

    initial begin
        errors = 0; hid_errors = 0; pred_errors = 0; correct = 0;
        tot_cyc = 0; tot_ideal = 0; min_cyc = 1 << 30; max_cyc = 0;
        stress_err = 0; stress_n = 0; seed = 12345;

        $readmemh({`MEMDIR, "/test_images.hex"},    img_mem, 0, 78399);
        $readmemh({`MEMDIR, "/golden_logits.hex"},  gold_log);
        $readmemh({`MEMDIR, "/golden_pred.txt"},    gold_pred);
        $readmemh({`MEMDIR, "/test_labels.txt"},    gold_lab);
        $readmemh({`MEMDIR, "/golden_hidden.hex"},  gold_hid);
        $readmemh({`MEMDIR, "/fc1_weights.hex"},    w1m);
        $readmemh({`MEMDIR, "/fc2_weights.hex"},    w2m);
        $readmemh({`MEMDIR, "/fc1_bias.hex"},       b1m);
        $readmemh({`MEMDIR, "/fc2_bias.hex"},       b2m);
        $readmemh({`MEMDIR, "/hidden_shift.hex"},   shm);

        repeat (2) @(posedge clk);
        #1 rst = 0;

        // =============== Part 1: golden images ===============
        for (img = 0; img < NIMG; img = img + 1) begin
            for (p = 0; p < 784; p = p + 1) cur_img[p] = img_mem[img*784 + p];
            load_and_run;
            if (done !== 1'b1 && cyc >= 100000) begin
                errors = errors + 1;
                $display("FAIL: timeout on image %0d", img);
            end
            compute_ref;                         // only used for the ideal MAC count here
            tot_cyc   = tot_cyc + cyc;
            tot_ideal = tot_ideal + ideal_macs;
            if (cyc < min_cyc) min_cyc = cyc;
            if (cyc > max_cyc) max_cyc = cyc;

            for (k = 0; k < 10; k = k + 1) begin
                if (got_log[k] !== gold_log[img*10 + k]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("FAIL img=%0d class=%0d got=%0d expected=%0d",
                                 img, k, $signed(got_log[k]), $signed(gold_log[img*10 + k]));
                end
            end
            if ({4'b0000, pred} !== gold_pred[img]) begin
                pred_errors = pred_errors + 1;
                $display("FAIL img=%0d pred=%0d expected=%0d", img, pred, gold_pred[img]);
            end
            if ({4'b0000, pred} === gold_lab[img]) correct = correct + 1;

            if (img < NHID_CHECK) begin
                for (k = 0; k < 64; k = k + 1) begin
                    if (dut.hid[k] !== gold_hid[img*64 + k]) begin
                        hid_errors = hid_errors + 1;
                        if (hid_errors <= 10)
                            $display("FAIL hidden img=%0d neuron=%0d got=%0d expected=%0d",
                                     img, k, dut.hid[k], gold_hid[img*64 + k]);
                    end
                end
            end
        end

        $display("--- golden images ---");
        $display("cycles per image (start to done): average %0d, min %0d, max %0d",
                 tot_cyc / NIMG, min_cyc, max_cyc);
        $display("useful MACs per image (nonzero pixel x nonzero weight): average %0d", tot_ideal / NIMG);
        $display("overhead on top of the useful MACs: %0d cycles per image", (tot_cyc - tot_ideal) / NIMG);
        $display("accuracy on these %0d images: %0d%%", NIMG, correct * 100 / NIMG);

        // =============== Part 2: stress tests ===============
        // 1. all zero
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd0;
        load_and_run;  check_stress("all zero");
        $display("all-zero image: %0d cycles", cyc);
        // 2. all 255
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd255;
        load_and_run;  check_stress("all 255");
        $display("all-255 image:  %0d cycles (worst case, nothing to skip in the pixels)", cyc);
        // 3. all zero again right after a full image (the list must be emptied by address 0)
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd0;
        load_and_run;  check_stress("zero after full");
        // 4. single pixels
        cur_img[0] = 8'd255;
        load_and_run;  check_stress("pixel 0");
        cur_img[0] = 8'd0;  cur_img[783] = 8'd1;
        load_and_run;  check_stress("pixel 783");
        cur_img[783] = 8'd0; cur_img[400] = 8'd7;
        load_and_run;  check_stress("pixel 400");
        // 5. random images, 5 / 20 / 50 / 100 percent nonzero
        for (t = 0; t < 16; t = t + 1) begin
            case (t % 4)
                0: dens = 5;
                1: dens = 20;
                2: dens = 50;
                default: dens = 100;
            endcase
            for (p = 0; p < 784; p = p + 1) begin
                if (($random(seed) & 32'h7fffffff) % 100 < dens)
                    cur_img[p] = 1 + (($random(seed) & 32'h7fffffff) % 255);
                else
                    cur_img[p] = 8'd0;
            end
            load_and_run;  check_stress("random");
        end
        // 6. a second 'start' pulse while busy must be ignored
        for (p = 0; p < 784; p = p + 1) cur_img[p] = img_mem[7*784 + p];
        for (p = 0; p < 784; p = p + 1) begin
            @(negedge clk);
            img_we = 1; img_waddr = p; img_wdata = cur_img[p];
        end
        @(negedge clk); img_we = 0;
        @(negedge clk); start = 1;
        @(posedge clk); #1 start = 0;
        repeat (100) @(posedge clk);
        #1 start = 1;                            // while busy
        @(posedge clk); #1 start = 0;
        cyc = 100;
        while (done !== 1'b1 && cyc < 100000) begin
            @(posedge clk); #1; cyc = cyc + 1;
        end
        @(posedge clk); #1;
        check_stress("start while busy");

        // 7. a start without new pixels = all-zero image (the previous image is consumed)
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd0;
        @(negedge clk); start = 1;
        @(posedge clk); #1 start = 0;
        cyc = 0;
        while (done !== 1'b1 && cyc < 100000) begin
            @(posedge clk); #1; cyc = cyc + 1;
        end
        @(posedge clk); #1;
        check_stress("start without pixels");

        $display("--- stress tests: %0d images checked against the dense reference model ---", stress_n);

        if (errors == 0 && hid_errors == 0 && pred_errors == 0 && stress_err == 0)
            $display("PASS: sparse_mlp_zs matched golden logits and the reference model (%0d + %0d images)",
                     NIMG, stress_n);
        else
            $display("FAILED: %0d logit errors, %0d hidden errors, %0d prediction errors, %0d stress errors",
                     errors, hid_errors, pred_errors, stress_err);
        $finish;
    end

    // Safety net: stop if something hangs
    initial begin
        #3000000000;
        $display("FAILED: simulation timeout");
        $finish;
    end
endmodule
