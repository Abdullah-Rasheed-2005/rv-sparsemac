// sparse_dot.v - Dot product of ONE neuron that skips zero weights
//
// Computes:  result = sum over the nonzero weights of  x[idx] * w
//
// The weights are not stored as a dense row. They are stored as a list of
// (index, value) pairs that contains ONLY the nonzero weights, and one
// neuron owns a continuous piece of that list:
//
//     nz_idx[base .. base+count-1]   which input each weight belongs to
//     nz_val[base .. base+count-1]   the int8 weight itself
//
// A pruned neuron with 150 nonzero weights therefore needs 150 cycles
// instead of 784. (software/export_sparse.py writes these lists.)
//
// This module does NOT add the bias, ReLU or shift. The control FSM that
// comes later does that. It only returns the raw sum in 'result'.
//
// Memories are outside this module and are SYNCHRONOUS (the data appears
// one clock edge after the address), like FPGA block RAM:
//   - list memory : address nz_addr  -> nz_idx and nz_w
//   - input memory: address x_addr   -> x_data (unsigned 8-bit)
//
// Pipeline (one nonzero weight enters per cycle, one MAC per cycle):
//   cycle 0: nz_addr = base + i            (read the list)
//   cycle 1: nz_idx, nz_w arrive;  x_addr = nz_idx  (read the input)
//   cycle 2: x_data and the delayed weight arrive -> mac.v accumulates
//
// Handshake:
//   - Pulse 'start' for one cycle while 'busy' is 0, with 'base' and
//     'count' valid. A start while busy is ignored.
//   - 'busy' stays high while working.
//   - 'done' is high for ONE cycle when the answer is ready. 'result'
//     is valid from then on until the next start.
//   - count = 0 is allowed and gives result = 0.
//   - Time from start to done is about count + 3 clock cycles.
module sparse_dot #(
    parameter ADDR_W = 16,   // bits for list addresses (up to 65536 entries)
    parameter IDX_W  = 10    // bits for an input index (up to 1024 inputs)
) (
    input  wire                  clk,
    input  wire                  rst,

    input  wire                  start,
    input  wire [ADDR_W-1:0]     base,     // first list entry of this neuron
    input  wire [ADDR_W-1:0]     count,    // number of nonzero weights

    // nonzero list memory (synchronous read)
    output wire [ADDR_W-1:0]     nz_addr,
    input  wire [IDX_W-1:0]      nz_idx,
    input  wire signed [7:0]     nz_w,

    // input (activation) memory (synchronous read)
    output wire [IDX_W-1:0]      x_addr,
    input  wire [7:0]            x_data,

    output reg                   busy,
    output reg                   done,
    output wire signed [31:0]    result
);
    reg [ADDR_W-1:0] base_r;     // saved copy of base
    reg [ADDR_W-1:0] count_r;    // saved copy of count
    reg [ADDR_W-1:0] i;          // how many entries were requested so far
    reg              v1;         // pipeline stage 1 holds a valid entry
    reg              v2;         // pipeline stage 2 holds a valid entry
    reg signed [7:0] w_d;        // weight delayed to meet its input value

    wire accept = start && !busy;
    wire issue  = busy && (i < count_r);   // request one more list entry

    assign nz_addr = base_r + i;
    assign x_addr  = nz_idx;

    always @(posedge clk) begin
        if (rst) begin
            busy    <= 1'b0;
            done    <= 1'b0;
            base_r  <= {ADDR_W{1'b0}};
            count_r <= {ADDR_W{1'b0}};
            i       <= {ADDR_W{1'b0}};
            v1      <= 1'b0;
            v2      <= 1'b0;
            w_d     <= 8'sd0;
        end else begin
            done <= 1'b0;
            v1   <= issue;
            v2   <= v1;
            w_d  <= nz_w;

            if (accept) begin
                busy    <= 1'b1;
                base_r  <= base;
                count_r <= count;
                i       <= {ADDR_W{1'b0}};
            end else if (busy) begin
                if (issue)
                    i <= i + 1'b1;
                else if (!v1 && !v2) begin
                    // everything was requested and the pipeline is empty
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end

    // The MAC unit from mac.v. 'accept' clears the accumulator, v2 enables it.
    mac u_mac (
        .clk  (clk),
        .rst  (rst),
        .clear(accept),
        .en   (v2),
        .x    (x_data),
        .w    (w_d),
        .acc  (result)
    );
endmodule
