// tb_mac.v - Testbench for mac.v
//
// Checks the MAC unit against a simple software model that is
// written inside this file (the "expected" variable).
`timescale 1ns/1ps
module tb_mac;
    reg clk = 0, rst = 1, clear = 0, en = 0;
    reg [7:0] x = 0;
    reg signed [7:0] w = 0;
    wire signed [31:0] acc;

    mac dut (.clk(clk), .rst(rst), .clear(clear), .en(en), .x(x), .w(w), .acc(acc));

    always #5 clk = ~clk;   // 100 MHz clock

    integer expected = 0;
    integer errors = 0;
    integer i;

    // Do one MAC and compare the result with the expected value
    task do_mac(input [7:0] xi, input signed [7:0] wi);
        integer a, b;
        begin
            x = xi; w = wi; en = 1;
            @(posedge clk); #1;
            en = 0;
            a = xi; b = wi;
            expected = expected + a * b;
            if (acc !== expected) begin
                errors = errors + 1;
                $display("FAIL: x=%0d w=%0d acc=%0d expected=%0d", xi, wi, acc, expected);
            end
        end
    endtask

    initial begin
        repeat (2) @(posedge clk);
        #1 rst = 0;

        // Test 1: biggest negative product
        do_mac(8'd255, -8'sd127);            // -32385
        // Test 2: biggest positive product (acc goes back to 0)
        do_mac(8'd255, 8'sd127);
        // Test 3: a zero input or a zero weight changes nothing
        do_mac(8'd0, 8'sd55);
        do_mac(8'd77, 8'sd0);

        // Test 4: clear
        clear = 1; @(posedge clk); #1; clear = 0; expected = 0;
        if (acc !== 0) begin errors = errors + 1; $display("FAIL: clear"); end

        // Test 5: worst case, 784 times (255 * -127)
        for (i = 0; i < 784; i = i + 1) do_mac(8'd255, -8'sd127);

        // Test 6: 1000 random pairs
        clear = 1; @(posedge clk); #1; clear = 0; expected = 0;
        for (i = 0; i < 1000; i = i + 1) do_mac($random, $random);

        if (errors == 0) $display("PASS: all MAC tests passed (acc=%0d)", acc);
        else             $display("FAILED: %0d errors", errors);
        $finish;
    end
endmodule
