// zs_nobudget.v - sparse_mlp_zs with the run-time budget switched off for good
//
// Used only by `make synth`: with budget = 0 and const_time = 0 as constants the
// synthesis tool removes the budget counter, the cut flag, the cycle counter and
// the padding state. The difference to sparse_mlp_zs with free inputs is what
// the bounded run time costs in hardware.
module zs_nobudget (
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    input  wire                 img_we,
    input  wire [9:0]           img_waddr,
    input  wire [7:0]           img_wdata,
    output wire                 busy,
    output wire                 done,
    output wire                 out_valid,
    output wire [3:0]           out_idx,
    output wire signed [31:0]   out_val,
    output wire [3:0]           pred
);
    sparse_mlp_zs u_nn (
        .clk(clk), .rst(rst), .start(start),
        .budget(16'd0), .const_time(1'b0),
        .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .busy(busy), .done(done),
        .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
        .pred(pred)
    );
endmodule
