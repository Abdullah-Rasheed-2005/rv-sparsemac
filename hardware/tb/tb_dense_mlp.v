// tb_dense_mlp.v - Dense reference: the SAME sparse_mlp engine, fed full lists
//
// The weight lists in build/dense_mem/ hold every weight of every neuron,
// zeros included (software/export_dense_lists.py). The engine therefore skips
// nothing, and the cycle count is a MEASURED dense baseline built from the same
// hardware as the sparse designs. Only what is skipped differs.
//
// Runs all 100 golden test images and checks
//   - every one of the 1000 logits against golden_logits (bit for bit),
//   - the predicted digit against golden_pred,
//   - the 64 hidden values of the first 10 images against golden_hidden.
// It also adds up the clock cycles from start to done.
//
// Needs the files written by software/export_dense_lists.py.
// Run from the repository root with `make sim-dense`.
`timescale 1ns/1ps
module tb_dense_mlp;
    localparam NIMG = 100;
    localparam NHID_CHECK = 10;   // images with golden hidden values

    reg clk = 0, rst = 1, start = 0;
    reg        img_we = 0;
    reg [9:0]  img_waddr = 0;
    reg [7:0]  img_wdata = 0;

    wire        busy, done, out_valid;
    wire [3:0]  out_idx;
    wire signed [31:0] out_val;
    wire [3:0]  pred;

    sparse_mlp #(
        .DEPTH(65536),
        .FC1_PTR("build/dense_mem/fc1_nz_ptr.hex"),
        .FC1_IDX("build/dense_mem/fc1_nz_idx.hex"),
        .FC1_VAL("build/dense_mem/fc1_nz_val.hex"),
        .FC2_PTR("build/dense_mem/fc2_nz_ptr.hex"),
        .FC2_IDX("build/dense_mem/fc2_nz_idx.hex"),
        .FC2_VAL("build/dense_mem/fc2_nz_val.hex")
    ) dut (
        .clk(clk), .rst(rst), .start(start),
        .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .busy(busy), .done(done),
        .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
        .pred(pred)
    );

    always #5 clk = ~clk;   // 100 MHz clock

    reg [7:0]  img_mem  [0:78399];         // 100 images x 784 pixels
    reg [31:0] gold_log [0:NIMG*10-1];     // golden logits
    reg [7:0]  gold_pred [0:NIMG-1];       // golden predictions
    reg [7:0]  gold_lab  [0:NIMG-1];       // true labels
    reg [7:0]  gold_hid  [0:NHID_CHECK*64-1];

    // collect the logits that come out one by one
    reg [31:0] got_log [0:9];
    always @(posedge clk)
        if (out_valid) got_log[out_idx] <= out_val;

    integer img, p, k, cyc;
    integer errors, hid_errors, pred_errors, correct;
    integer tot_cyc;

    initial begin
        errors = 0; hid_errors = 0; pred_errors = 0; correct = 0; tot_cyc = 0;

        $readmemh("hardware/mem/test_images.hex",    img_mem, 0, 78399);
        $readmemh("hardware/mem/golden_logits.hex",  gold_log);
        $readmemh("hardware/mem/golden_pred.txt",    gold_pred);
        $readmemh("hardware/mem/test_labels.txt",    gold_lab);
        $readmemh("hardware/mem/golden_hidden.hex",  gold_hid);

        repeat (2) @(posedge clk);
        #1 rst = 0;

        for (img = 0; img < NIMG; img = img + 1) begin
            // 1. write the image into the accelerator
            for (p = 0; p < 784; p = p + 1) begin
                @(negedge clk);
                img_we = 1; img_waddr = p; img_wdata = img_mem[img*784 + p];
            end
            @(negedge clk);
            img_we = 0;

            // 2. start and wait for done
            @(negedge clk);
            start = 1;
            @(posedge clk);          // the accelerator accepts the start here
            #1 start = 0;
            cyc = 0;
            while (done !== 1'b1 && cyc < 1000000) begin
                @(posedge clk); #1;
                cyc = cyc + 1;
            end
            if (done !== 1'b1) begin
                errors = errors + 1;
                $display("FAIL: timeout on image %0d", img);
            end
            tot_cyc = tot_cyc + cyc;

            // 3. one more clock so the last logit is collected
            @(posedge clk); #1;

            // 4. compare
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

        $display("dense reference: every one of the %0d weights is processed", 784*64 + 64*10);
        $display("cycles per image (start to done): %0d", tot_cyc / NIMG);
        $display("(for comparison, inputs+5 cycles per neuron = %0d)", (784+5)*64 + (64+5)*10);
        $display("accuracy on these %0d images: %0d%%", NIMG, correct * 100 / NIMG);
        if (errors == 0 && hid_errors == 0 && pred_errors == 0)
            $display("PASS: dense reference matched golden logits and predictions (%0d images, %0d logits)",
                     NIMG, NIMG*10);
        else
            $display("FAILED: %0d logit errors, %0d hidden errors, %0d prediction errors",
                     errors, hid_errors, pred_errors);
        $finish;
    end

    // Safety net: stop if something hangs
    initial begin
        #900000000;
        $display("FAILED: simulation timeout");
        $finish;
    end
endmodule
