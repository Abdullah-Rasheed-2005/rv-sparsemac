// tb_zs_budget.v - Testbench for the run-time budget of sparse_mlp_zs.v
//
// Uses the budget-aware model exported by software/export_budget.py into
// build/budget_mem/ (inputs already stored most-useful-first).
//
// Part 1: budget = 0 (no limit). The 100 test images must give golden_logits,
//         exactly as before: the budget logic must not change anything when off.
// Part 2: budget = B (from budget.hex). The 100 images must give
//         golden_logits_b / golden_pred_b, which software/budget_lib.py computed
//         with the same rule. The run time of every image is checked against
//         the bound  B + SLACK.
// Part 3: stress images with the budget (all zero, all 255, single pixels,
//         random densities), checked against a reference model written here as
//         plain loops. The all-255 image is the worst case of the unlimited
//         design; here it must also stay below B + SLACK.
// Part 4: const_time = 1. Every image (also all-zero and all-255) must finish
//         after exactly the same number of cycles, with the same results.
//
// Run from the repository root with `make sim-zs-budget`.
`timescale 1ns/1ps
`ifndef MEMDIR
  `define MEMDIR "build/budget_mem"
`endif
module tb_zs_budget;
    localparam NIMG  = 100;
    localparam PAD   = 800;      // same value as the parameter of the accelerator
    localparam SLACK = 800;      // finish sweeps + all of layer 2 + pipeline start-up

    reg clk = 0, rst = 1, start = 0;
    reg [15:0] budget = 16'd0;
    reg        const_time = 1'b0;
    reg        img_we = 0;
    reg [9:0]  img_waddr = 0;
    reg [7:0]  img_wdata = 0;

    wire        busy, done, out_valid;
    wire [3:0]  out_idx;
    wire signed [31:0] out_val;
    wire [3:0]  pred;

    sparse_mlp_zs #(
        .PAD       (PAD),
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
        .budget(budget), .const_time(const_time),
        .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .busy(busy), .done(done),
        .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
        .pred(pred)
    );

    always #5 clk = ~clk;   // 100 MHz clock

    // ---------------- data ----------------
    reg [7:0]  img_mem   [0:78399];
    reg [31:0] gold_log  [0:NIMG*10-1];     // no budget
    reg [31:0] gold_logb [0:NIMG*10-1];     // with the budget
    reg [7:0]  gold_predb [0:NIMG-1];
    reg [15:0] bud_mem   [0:0];

    reg [7:0]  w1m [0:50175];
    reg [7:0]  w2m [0:639];
    reg [31:0] b1m [0:63];
    reg [31:0] b2m [0:9];
    reg [7:0]  shm [0:0];
    integer    col_nnz [0:783];             // nonzero weights of every fc1 column

    reg [7:0]  cur_img [0:783];
    integer    ref_log [0:9];
    integer    ref_pred;
    reg        keep [0:783];                // reference model: input j is inside the budget
    integer    hidr [0:63];                 // reference model: hidden values

    reg [31:0] got_log [0:9];
    always @(posedge clk)
        if (out_valid) got_log[out_idx] <= out_val;

    // ---------------- reference model: dense loops + the budget rule ----------------
    // bud = 0 means no limit. Inputs are visited in the order 0, 1, 2, ...
    task compute_ref(input integer bud);
        integer n, j, a, xi, wi, sh, best, usedr, cst, stopped;
        begin
            sh = shm[0];
            usedr = 0; stopped = 0;
            for (j = 0; j < 784; j = j + 1) begin
                keep[j] = 1'b0;
                if (cur_img[j] != 8'd0 && !stopped) begin
                    cst = (col_nnz[j] < 3) ? 3 : col_nnz[j];
                    if (bud != 0 && usedr + cst > bud) stopped = 1;
                    else begin
                        usedr = usedr + cst;
                        keep[j] = 1'b1;
                    end
                end
            end
            for (n = 0; n < 64; n = n + 1) begin
                a = $signed(b1m[n]);
                for (j = 0; j < 784; j = j + 1)
                    if (keep[j]) begin
                        xi = cur_img[j];
                        wi = $signed(w1m[n*784 + j]);
                        a = a + xi * wi;
                    end
                if (a < 0) a = 0;
                a = a >>> sh;
                if (a > 127) a = 127;
                hidr[n] = a;
            end
            best = 0;
            for (n = 0; n < 10; n = n + 1) begin
                a = $signed(b2m[n]);
                for (j = 0; j < 64; j = j + 1) begin
                    wi = $signed(w2m[n*64 + j]);
                    a = a + hidr[j] * wi;
                end
                ref_log[n] = a;
                if (n == 0 || a > ref_log[best]) best = n;
            end
            ref_pred = best;
        end
    endtask

    // ---------------- drive one image (cur_img), wait for done ----------------
    integer cyc;
    task load_and_run;
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
            while (done !== 1'b1 && cyc < 200000) begin
                @(posedge clk); #1;
                cyc = cyc + 1;
            end
            @(posedge clk); #1;      // one more clock so the last logit is collected
        end
    endtask

    integer errors, stress_err, time_err, ct_err;
    integer img, p, k, t, dens, seed, n, j;
    integer tot_cyc, max_cyc, tot_free, max_free, ct_cyc, cut_images;
    integer B;

    // compare the accelerator with the reference model for the image in cur_img
    task check_ref(input [8*24-1:0] name, input integer bud);
        integer c;
        begin
            compute_ref(bud);
            for (c = 0; c < 10; c = c + 1)
                if ($signed(got_log[c]) !== ref_log[c]) begin
                    stress_err = stress_err + 1;
                    if (stress_err <= 10)
                        $display("FAIL %0s class=%0d got=%0d expected=%0d", name, c, $signed(got_log[c]), ref_log[c]);
                end
            if (pred !== ref_pred[3:0]) begin
                stress_err = stress_err + 1;
                $display("FAIL %0s pred=%0d expected=%0d", name, pred, ref_pred);
            end
        end
    endtask

    // every run with a budget must finish within B + SLACK cycles
    task check_time(input [8*24-1:0] name);
        begin
            if (cyc > B + SLACK) begin
                time_err = time_err + 1;
                $display("FAIL %0s: %0d cycles, the bound is %0d", name, cyc, B + SLACK);
            end
        end
    endtask

    initial begin
        errors = 0; stress_err = 0; time_err = 0; ct_err = 0; seed = 4711;
        tot_cyc = 0; max_cyc = 0; tot_free = 0; max_free = 0; cut_images = 0;

        $readmemh({`MEMDIR, "/test_images.hex"},     img_mem, 0, 78399);
        $readmemh({`MEMDIR, "/golden_logits.hex"},   gold_log);
        $readmemh({`MEMDIR, "/golden_logits_b.hex"}, gold_logb);
        $readmemh({`MEMDIR, "/golden_pred_b.txt"},   gold_predb);
        $readmemh({`MEMDIR, "/budget.hex"},          bud_mem);
        $readmemh({`MEMDIR, "/fc1_weights.hex"},     w1m);
        $readmemh({`MEMDIR, "/fc2_weights.hex"},     w2m);
        $readmemh({`MEMDIR, "/fc1_bias.hex"},        b1m);
        $readmemh({`MEMDIR, "/fc2_bias.hex"},        b2m);
        $readmemh({`MEMDIR, "/hidden_shift.hex"},    shm);
        B = bud_mem[0];

        for (j = 0; j < 784; j = j + 1) begin
            col_nnz[j] = 0;
            for (n = 0; n < 64; n = n + 1)
                if (w1m[n*784 + j] != 8'd0) col_nnz[j] = col_nnz[j] + 1;
        end

        repeat (2) @(posedge clk);
        #1 rst = 0;

        // =============== Part 1: budget off ===============
        budget = 16'd0; const_time = 1'b0;
        for (img = 0; img < NIMG; img = img + 1) begin
            for (p = 0; p < 784; p = p + 1) cur_img[p] = img_mem[img*784 + p];
            load_and_run;
            tot_free = tot_free + cyc;
            if (cyc > max_free) max_free = cyc;
            for (k = 0; k < 10; k = k + 1)
                if (got_log[k] !== gold_log[img*10 + k]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("FAIL (no budget) img=%0d class=%0d got=%0d expected=%0d",
                                 img, k, $signed(got_log[k]), $signed(gold_log[img*10 + k]));
                end
        end
        $display("--- no budget ---");
        $display("cycles per image (start to done): average %0d, max %0d", tot_free / NIMG, max_free);

        // =============== Part 2: with the budget ===============
        budget = B[15:0];
        for (img = 0; img < NIMG; img = img + 1) begin
            for (p = 0; p < 784; p = p + 1) cur_img[p] = img_mem[img*784 + p];
            load_and_run;
            check_time("golden image");
            tot_cyc = tot_cyc + cyc;
            if (cyc > max_cyc) max_cyc = cyc;
            t = 0;
            for (k = 0; k < 10; k = k + 1) begin
                if (got_log[k] !== gold_logb[img*10 + k]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("FAIL (budget) img=%0d class=%0d got=%0d expected=%0d",
                                 img, k, $signed(got_log[k]), $signed(gold_logb[img*10 + k]));
                end
                if (gold_logb[img*10 + k] !== gold_log[img*10 + k]) t = 1;
            end
            cut_images = cut_images + t;
            if ({4'b0000, pred} !== gold_predb[img]) begin
                errors = errors + 1;
                $display("FAIL (budget) img=%0d pred=%0d expected=%0d", img, pred, gold_predb[img]);
            end
        end
        $display("--- layer-1 budget = %0d cycles ---", B);
        $display("cycles per image (start to done): average %0d, max %0d, bound %0d", tot_cyc / NIMG, max_cyc, B + SLACK);
        $display("images whose logits are changed by the budget: %0d of %0d", cut_images, NIMG);

        // =============== Part 3: stress images with the budget ===============
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd0;
        load_and_run;  check_ref("all zero", B);  check_time("all zero");
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd255;
        load_and_run;  check_ref("all 255", B);   check_time("all 255");
        $display("all-255 image with the budget: %0d cycles", cyc);
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd0;
        cur_img[0] = 8'd255;
        load_and_run;  check_ref("pixel 0", B);   check_time("pixel 0");
        cur_img[0] = 8'd0;  cur_img[783] = 8'd1;
        load_and_run;  check_ref("pixel 783", B); check_time("pixel 783");
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
            load_and_run;  check_ref("random", B);  check_time("random");
        end
        // the all-255 image without a budget, for comparison
        budget = 16'd0;
        for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd255;
        load_and_run;  check_ref("all 255 free", 0);
        $display("all-255 image without a budget: %0d cycles", cyc);

        // =============== Part 4: constant time ===============
        budget = B[15:0]; const_time = 1'b1;
        ct_cyc = -1;
        for (img = 0; img < 22; img = img + 1) begin
            if (img < 20)
                for (p = 0; p < 784; p = p + 1) cur_img[p] = img_mem[img*784 + p];
            else if (img == 20)
                for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd0;
            else
                for (p = 0; p < 784; p = p + 1) cur_img[p] = 8'd255;
            load_and_run;
            check_ref("const time", B);
            if (ct_cyc < 0) ct_cyc = cyc;
            if (cyc != ct_cyc) begin
                ct_err = ct_err + 1;
                $display("FAIL const time: image %0d took %0d cycles, the first one took %0d", img, cyc, ct_cyc);
            end
        end
        const_time = 1'b0;
        $display("--- constant time ---");
        $display("every one of 22 images (20 test images, all zero, all 255): %0d cycles", ct_cyc);
        if (ct_cyc != B + PAD + 1) $display("note: expected budget + PAD + 1 = %0d", B + PAD + 1);

        if (errors == 0 && stress_err == 0 && time_err == 0 && ct_err == 0)
            $display("PASS: sparse_mlp_zs budget matched golden values, the reference model, the time bound and constant time");
        else
            $display("FAILED: %0d golden errors, %0d reference errors, %0d time-bound errors, %0d constant-time errors",
                     errors, stress_err, time_err, ct_err);
        $finish;
    end

    // Safety net: stop if something hangs
    initial begin
        #3000000000;
        $display("FAILED: simulation timeout");
        $finish;
    end
endmodule
