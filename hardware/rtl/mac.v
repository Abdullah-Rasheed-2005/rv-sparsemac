// mac.v - Multiply-accumulate (MAC) unit
//
// Computes:  acc = acc + x * w
//   x   : unsigned 8-bit activation (pixel 0..255 or hidden value 0..127)
//   w   : signed 8-bit weight (-127..127)
//   acc : signed 32-bit accumulator
//
// Why 32 bits: the worst case for one neuron is 784 * 255 * 127,
// which is about 25.4 million and needs 26 bits. 32 bits is safe.
module mac (
    input  wire               clk,
    input  wire               rst,    // reset everything
    input  wire               clear,  // zero the accumulator (start a new neuron)
    input  wire               en,     // do one multiply-accumulate this cycle
    input  wire        [7:0]  x,      // activation, unsigned
    input  wire signed [7:0]  w,      // weight, signed
    output reg  signed [31:0] acc
);
    // Put a 0 bit in front of x so it becomes a 9-bit SIGNED number
    // with the same positive value. (200 stays 200, not -56.)
    wire signed [8:0]  x_s  = {1'b0, x};
    wire signed [16:0] prod = x_s * w;

    always @(posedge clk) begin
        if (rst || clear)
            acc <= 32'sd0;
        else if (en)
            acc <= acc + prod;
    end
endmodule
