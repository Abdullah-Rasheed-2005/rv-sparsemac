// tb_sparse_dot.v - Testbench for sparse_dot.v
//
// Needs the files written by software/export_sparse.py (run `make export-sparse`
// once). Run from the repository root with `make sim-sparse`.
//
// What it checks, against golden values computed in Python:
//   1. fc1: 64 neurons x 10 images, raw dot product of the pixels
//   2. fc2: 10 neurons x 10 images, raw dot product of the hidden values
//   3. count = 0 gives result 0
//   4. a start pulse while the engine is busy is ignored
// It also adds up the clock cycles, so we can compare with the dense case.
`timescale 1ns/1ps
module tb_sparse_dot;
    localparam NIMG = 10;   // must match N_TEST in software/export_sparse.py

    reg clk = 0, rst = 1, start = 0;
    reg  [15:0] base = 0, count = 0;
    wire [15:0] nz_addr;
    wire [9:0]  x_addr;
    wire        busy, done;
    wire signed [31:0] result;

    // ---------------- memories (synchronous read, like block RAM) ----------------
    reg [15:0] idx_mem [0:65535];     // input index of each nonzero weight
    reg [7:0]  val_mem [0:65535];     // the weight values
    reg [15:0] ptr_mem [0:64];        // start of each neuron in the lists
    reg [7:0]  img_mem [0:78399];     // 100 test images, 784 pixels each
    reg [7:0]  x_mem   [0:1023];      // the input vector of the current layer
    reg [31:0] gold1   [0:NIMG*64-1]; // expected fc1 dot products
    reg [7:0]  hid_mem [0:NIMG*64-1]; // hidden values (input of fc2)
    reg [31:0] gold2   [0:NIMG*10-1]; // expected fc2 dot products

    reg [15:0] idx_word;
    reg [7:0]  w_word;
    reg [7:0]  x_data;
    always @(posedge clk) begin
        idx_word <= idx_mem[nz_addr];
        w_word   <= val_mem[nz_addr];
        x_data   <= x_mem[x_addr];
    end
    wire [9:0]         nz_idx = idx_word[9:0];
    wire signed [7:0]  nz_w   = w_word;

    sparse_dot dut (
        .clk(clk), .rst(rst), .start(start), .base(base), .count(count),
        .nz_addr(nz_addr), .nz_idx(nz_idx), .nz_w(nz_w),
        .x_addr(x_addr), .x_data(x_data),
        .busy(busy), .done(done), .result(result)
    );

    always #5 clk = ~clk;   // 100 MHz clock

    // ---------------- helpers ----------------
    integer errors = 0;
    integer cyc;                 // clock cycles from start to done
    reg signed [31:0] got;

    // Run one neuron and wait for done. Result goes to 'got', cycles to 'cyc'.
    task run_neuron(input [15:0] b, input [15:0] c);
        begin
            @(negedge clk);
            base = b; count = c; start = 1;
            @(posedge clk);          // the engine accepts the start here
            #1 start = 0;
            cyc = 0;
            while (done !== 1'b1 && cyc < 100000) begin
                @(posedge clk); #1;
                cyc = cyc + 1;
            end
            if (done !== 1'b1) begin
                errors = errors + 1;
                $display("FAIL: timeout, done never came");
            end
            got = result;
        end
    endtask

    integer img, n, p, nnz, cnt;
    integer tot1, tot2;

    initial begin
        tot1 = 0; tot2 = 0;
        for (p = 0; p < 1024; p = p + 1) x_mem[p] = 8'd0;

        $readmemh("hardware/mem/test_images.hex",      img_mem, 0, 78399);
        $readmemh("hardware/mem/golden_fc1_acc.hex",   gold1);
        $readmemh("hardware/mem/golden_hidden.hex",    hid_mem);
        $readmemh("hardware/mem/golden_fc2_acc.hex",   gold2);

        repeat (2) @(posedge clk);
        #1 rst = 0;

        // ================= fc1 =================
        $readmemh("hardware/mem/fc1_nz_ptr.hex", ptr_mem, 0, 64);
        nnz = ptr_mem[64];
        $readmemh("hardware/mem/fc1_nz_idx.hex", idx_mem, 0, nnz - 1);
        $readmemh("hardware/mem/fc1_nz_val.hex", val_mem, 0, nnz - 1);
        $display("fc1: %0d nonzero weights (dense would be %0d)", nnz, 64*784);

        for (img = 0; img < NIMG; img = img + 1) begin
            for (p = 0; p < 784; p = p + 1) x_mem[p] = img_mem[img*784 + p];
            for (n = 0; n < 64; n = n + 1) begin
                cnt = ptr_mem[n+1] - ptr_mem[n];
                run_neuron(ptr_mem[n], cnt);
                tot1 = tot1 + cyc;
                if (got !== gold1[img*64 + n]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("FAIL fc1 img=%0d neuron=%0d got=%0d expected=%0d",
                                 img, n, got, $signed(gold1[img*64 + n]));
                end
            end
        end

        // count = 0 must give 0
        run_neuron(16'd0, 16'd0);
        if (got !== 32'd0) begin
            errors = errors + 1;
            $display("FAIL: count=0 gave %0d", got);
        end

        // A start pulse while busy must be ignored.
        // x_mem still holds the last image, so neuron 0 of that image is checked.
        cnt = ptr_mem[1] - ptr_mem[0];
        if (cnt > 20) begin
            @(negedge clk);
            base = ptr_mem[0]; count = cnt; start = 1;
            @(negedge clk);
            start = 0;
            repeat (10) @(negedge clk);
            base = 16'd5; count = 16'd3; start = 1;      // this one must be ignored
            @(negedge clk);
            start = 0;
            while (done !== 1'b1) begin @(posedge clk); #1; end
            if (result !== gold1[(NIMG-1)*64]) begin
                errors = errors + 1;
                $display("FAIL: start while busy changed the result (%0d)", result);
            end
        end

        // ================= fc2 =================
        $readmemh("hardware/mem/fc2_nz_ptr.hex", ptr_mem, 0, 10);
        nnz = ptr_mem[10];
        $readmemh("hardware/mem/fc2_nz_idx.hex", idx_mem, 0, nnz - 1);
        $readmemh("hardware/mem/fc2_nz_val.hex", val_mem, 0, nnz - 1);
        $display("fc2: %0d nonzero weights (dense would be %0d)", nnz, 10*64);

        for (img = 0; img < NIMG; img = img + 1) begin
            for (p = 0; p < 64; p = p + 1) x_mem[p] = hid_mem[img*64 + p];
            for (n = 0; n < 10; n = n + 1) begin
                cnt = ptr_mem[n+1] - ptr_mem[n];
                run_neuron(ptr_mem[n], cnt);
                tot2 = tot2 + cyc;
                if (got !== gold2[img*10 + n]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("FAIL fc2 img=%0d neuron=%0d got=%0d expected=%0d",
                                 img, n, got, $signed(gold2[img*10 + n]));
                end
            end
        end

        // ================= report =================
        $display("cycles per image: fc1 = %0d, fc2 = %0d  (dense MACs: fc1 = %0d, fc2 = %0d)",
                 tot1 / NIMG, tot2 / NIMG, 64*784, 10*64);
        if (errors == 0)
            $display("PASS: sparse_dot matched the golden values (%0d fc1 + %0d fc2 neurons)",
                     NIMG*64, NIMG*10);
        else
            $display("FAILED: %0d errors", errors);
        $finish;
    end

    // Safety net: stop if something hangs
    initial begin
        #500000000;
        $display("FAILED: simulation timeout");
        $finish;
    end
endmodule
