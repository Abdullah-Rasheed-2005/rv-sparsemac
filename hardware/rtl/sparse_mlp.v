// sparse_mlp.v - Full 784 -> 64 -> 10 inference with sparse weights
//
// This is the control FSM. It owns one sparse_dot engine and runs the whole
// network, one neuron after another:
//
//   layer 1 (fc1), for each of the 64 neurons:
//       acc    = sparse dot product of the 784 pixels  + bias1
//       hidden = min( max(acc, 0) >> SHIFT , 127 )      -> stored in hidden RAM
//   layer 2 (fc2), for each of the 10 neurons:
//       logit  = sparse dot product of the 64 hidden values + bias2
//   pred = index of the largest logit (the first one wins a tie)
//
// This is exactly the integer pipeline of software/export_int8.py, so the
// logits must match golden_logits bit for bit.
//
// How to use it:
//   1. Write the 784 pixels (0..255) through the image write port.
//   2. Pulse 'start' for one cycle while 'busy' is 0.
//   3. Every time a logit is ready, 'out_valid' is high for one cycle with
//      its class number in 'out_idx' and its value in 'out_val'.
//   4. 'done' is high for one cycle at the end, and 'pred' holds the answer.
//
// Memories: the weights, pointers and biases are ROMs that are filled from
// the hex files in hardware/mem/ when the simulation starts ($readmemh).
// They are simulation memories for now. Mapping them to FPGA block RAM is a
// later step. The files must come from software/export_sparse.py.
//
// Zero pixels and zero hidden values are NOT skipped yet. Only zero weights.
module sparse_mlp #(
    parameter FC1_PTR  = "hardware/mem/fc1_nz_ptr.hex",
    parameter FC1_IDX  = "hardware/mem/fc1_nz_idx.hex",
    parameter FC1_VAL  = "hardware/mem/fc1_nz_val.hex",
    parameter FC1_BIAS = "hardware/mem/fc1_bias.hex",
    parameter FC2_PTR  = "hardware/mem/fc2_nz_ptr.hex",
    parameter FC2_IDX  = "hardware/mem/fc2_nz_idx.hex",
    parameter FC2_VAL  = "hardware/mem/fc2_nz_val.hex",
    parameter FC2_BIAS = "hardware/mem/fc2_bias.hex",
    parameter SHIFT_FILE = "hardware/mem/hidden_shift.hex"
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 start,

    // write port of the input image RAM
    input  wire                 img_we,
    input  wire [9:0]           img_waddr,
    input  wire [7:0]           img_wdata,

    output reg                  busy,
    output reg                  done,

    output reg                  out_valid,   // one logit is ready
    output reg  [3:0]           out_idx,     // which class
    output reg  signed [31:0]   out_val,     // its value
    output reg  [3:0]           pred         // winning class, valid with done
);
    localparam DEPTH = 16384;   // entries in each nonzero list (2^14)

    // ---------------- ROMs ----------------
    reg [15:0] ptr1  [0:64];
    reg [15:0] idx1  [0:DEPTH-1];
    reg [7:0]  val1  [0:DEPTH-1];
    reg [31:0] bias1 [0:63];

    reg [15:0] ptr2  [0:10];
    reg [15:0] idx2  [0:DEPTH-1];
    reg [7:0]  val2  [0:DEPTH-1];
    reg [31:0] bias2 [0:9];

    reg [7:0]  shift_rom [0:0];

    // ---------------- RAMs ----------------
    reg [7:0]  img [0:1023];    // input image (784 used)
    reg [7:0]  hid [0:63];      // hidden values, output of layer 1

    integer n1, n2;
    initial begin
        $readmemh(FC1_PTR, ptr1, 0, 64);
        n1 = ptr1[64];
        if (n1 > DEPTH) $display("ERROR: sparse_mlp: fc1 has %0d nonzero weights, DEPTH is %0d", n1, DEPTH);
        $readmemh(FC1_IDX,  idx1, 0, n1 - 1);
        $readmemh(FC1_VAL,  val1, 0, n1 - 1);
        $readmemh(FC1_BIAS, bias1, 0, 63);

        $readmemh(FC2_PTR, ptr2, 0, 10);
        n2 = ptr2[10];
        if (n2 > DEPTH) $display("ERROR: sparse_mlp: fc2 has %0d nonzero weights, DEPTH is %0d", n2, DEPTH);
        $readmemh(FC2_IDX,  idx2, 0, n2 - 1);
        $readmemh(FC2_VAL,  val2, 0, n2 - 1);
        $readmemh(FC2_BIAS, bias2, 0, 9);

        $readmemh(SHIFT_FILE, shift_rom, 0, 0);
    end

    wire [4:0] shift = shift_rom[0][4:0];

    always @(posedge clk)
        if (img_we) img[img_waddr] <= img_wdata;

    // ---------------- FSM state ----------------
    localparam S_IDLE   = 2'd0;
    localparam S_LAUNCH = 2'd1;   // give the engine its neuron
    localparam S_WAIT   = 2'd2;   // wait for the dot product, then post-process

    reg [1:0]        state;
    reg              layer;       // 0 = fc1, 1 = fc2
    reg [6:0]        neuron;      // neuron number inside the layer
    reg signed [31:0] best_val;
    reg [3:0]        best_idx;

    // ---------------- sparse_dot engine ----------------
    wire [15:0] sd_base  = layer ? ptr2[neuron] : ptr1[neuron];
    wire [15:0] sd_next  = layer ? ptr2[neuron + 1] : ptr1[neuron + 1];
    wire [15:0] sd_count = sd_next - sd_base;
    wire        sd_start = (state == S_LAUNCH);

    wire [15:0]        nz_addr;
    wire [9:0]         x_addr;
    wire               sd_busy, sd_done;
    wire signed [31:0] result;

    // synchronous reads, one clock edge of delay, like block RAM
    reg [15:0] idx_word;
    reg [7:0]  w_word;
    reg [7:0]  x_data;
    always @(posedge clk) begin
        idx_word <= layer ? idx2[nz_addr[13:0]] : idx1[nz_addr[13:0]];
        w_word   <= layer ? val2[nz_addr[13:0]] : val1[nz_addr[13:0]];
        x_data   <= layer ? hid[x_addr[5:0]]    : img[x_addr];
    end
    wire [9:0]        nz_idx = idx_word[9:0];
    wire signed [7:0] nz_w   = w_word;

    sparse_dot u_dot (
        .clk(clk), .rst(rst),
        .start(sd_start), .base(sd_base), .count(sd_count),
        .nz_addr(nz_addr), .nz_idx(nz_idx), .nz_w(nz_w),
        .x_addr(x_addr), .x_data(x_data),
        .busy(sd_busy), .done(sd_done), .result(result)
    );

    // ---------------- post-processing of a finished dot product ----------------
    wire signed [31:0] b1 = bias1[neuron[5:0]];
    wire signed [31:0] b2 = bias2[neuron[3:0]];
    wire signed [31:0] acc_b = result + (layer ? b2 : b1);          // add the bias
    wire signed [31:0] relu  = acc_b[31] ? 32'sd0 : acc_b;          // ReLU
    wire [31:0]        shifted = relu >> shift;                      // divide by 2^SHIFT
    wire [7:0]         hval = (shifted > 32'd127) ? 8'd127 : shifted[7:0];   // saturate

    wire        better   = (neuron == 7'd0) || (acc_b > best_val);   // strict: first max wins
    wire [3:0]  next_idx = better ? neuron[3:0] : best_idx;

    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            layer     <= 1'b0;
            neuron    <= 7'd0;
            busy      <= 1'b0;
            done      <= 1'b0;
            out_valid <= 1'b0;
            out_idx   <= 4'd0;
            out_val   <= 32'sd0;
            pred      <= 4'd0;
            best_val  <= 32'sd0;
            best_idx  <= 4'd0;
        end else begin
            done      <= 1'b0;
            out_valid <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        busy   <= 1'b1;
                        layer  <= 1'b0;
                        neuron <= 7'd0;
                        state  <= S_LAUNCH;
                    end
                end

                S_LAUNCH: begin
                    // sd_start is high in this cycle, the engine takes it at this edge
                    state <= S_WAIT;
                end

                S_WAIT: begin
                    if (sd_done) begin
                        if (!layer) begin
                            // layer 1: store the hidden value
                            hid[neuron[5:0]] <= hval;
                            if (neuron == 7'd63) begin
                                layer  <= 1'b1;
                                neuron <= 7'd0;
                            end else begin
                                neuron <= neuron + 7'd1;
                            end
                            state <= S_LAUNCH;
                        end else begin
                            // layer 2: report the logit, keep track of the maximum
                            out_valid <= 1'b1;
                            out_idx   <= neuron[3:0];
                            out_val   <= acc_b;
                            best_val  <= better ? acc_b : best_val;
                            best_idx  <= next_idx;
                            if (neuron == 7'd9) begin
                                pred  <= next_idx;
                                busy  <= 1'b0;
                                done  <= 1'b1;
                                state <= S_IDLE;
                            end else begin
                                neuron <= neuron + 7'd1;
                                state  <= S_LAUNCH;
                            end
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
