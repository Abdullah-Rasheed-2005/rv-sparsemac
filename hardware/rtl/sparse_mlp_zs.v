// sparse_mlp_zs.v - Full 784 -> 64 -> 10 inference that skips BOTH kinds of zeros
//
// "zs" = zero skipping. Compared with sparse_mlp.v (which skips only zero
// WEIGHTS) this design also skips zero INPUTS: zero pixels in layer 1 and zero
// hidden values (after ReLU) in layer 2. The ports are identical to
// sparse_mlp.v, so the two modules can be swapped for each other.
//
// ---------------------------------------------------------------------------
// The idea (read this first)
// ---------------------------------------------------------------------------
// sparse_mlp.v works neuron by neuron ("output-stationary"): for one neuron it
// walks that neuron's list of nonzero weights. It has no cheap way to know that
// the pixel behind a weight is zero, so those cycles are wasted.
//
// This design turns the loops around ("input-stationary"):
//
//     for every NONZERO input j (value a):
//         for every NONZERO weight w[n][j] in column j:
//             acc[n] += a * w[n][j]
//
// Both loops contain only nonzero things, so every cycle does one useful
// multiply-accumulate. The number of cycles is about the number of
// (nonzero input, nonzero weight) pairs. Two things are needed:
//
//   1. A list of the nonzero inputs.   Built for free while the image is
//      written (pixels) or while layer 1 is finished (hidden values).
//   2. The weights stored by COLUMN (CSC format, see export_sparse.py):
//         fcN_csc_ptr.hex   column j owns entries ptr[j] .. ptr[j+1]-1
//         fcN_csc_row.hex   output neuron of each entry
//         fcN_csc_val.hex   the int8 weight of each entry
//
// Because one input now feeds MANY neurons, all 64 layer-1 sums must be alive
// at the same time. They live in a register array acc[0..63]. A flag
// touched[n] says "acc[n] already holds a partial sum"; if it is 0 the old
// value counts as 0. That replaces clearing 64 accumulators (no cycles lost).
//
// ---------------------------------------------------------------------------
// Pipeline (one MAC per cycle once it is running)
// ---------------------------------------------------------------------------
//   front end   : takes entry k of the nonzero-input list, reads the column
//                 pointers ptr[j], ptr[j+1]; fills the descriptor register
//                 {lo, hi, a}. Takes 3 cycles, but it works on the NEXT column
//                 while the current column is still being streamed.
//   issue stage : walks entries lo..hi-1 of the current column, one per cycle,
//                 and reads (row, weight) from the ROMs (1 cycle).
//   MAC stage   : acc[row] += a * weight.
//
// After the last MAC of a layer, the FINISH sweep goes over the neurons, one per
// cycle: add bias, then ReLU + shift + saturate (layer 1) or report the logit
// and track the maximum (layer 2). Layer 1 also builds the nonzero-input list
// for layer 2 in the same sweep.
//
// ---------------------------------------------------------------------------
// How to use it (same as sparse_mlp.v, with ONE extra rule)
// ---------------------------------------------------------------------------
//   1. Write the 784 pixels through the image write port IN ORDER 0,1,...,783.
//      Writing address 0 starts a new image (it empties the nonzero list).
//      Zero pixels are simply not stored. (The list is built on the fly, so
//      there is no dense image RAM any more.)
//   2. Pulse 'start' for one cycle while 'busy' is 0.
//   3. out_valid pulses once per logit (out_idx, out_val). 'done' pulses at the
//      end and 'pred' holds the answer (first maximum wins a tie).
//   Do not write pixels while busy. The pixels are consumed by a run: write the
//   image again before the next start. (A start without new pixels computes the
//   answer for an all-zero image.)
//
// The result is bit-exact with sparse_mlp.v and with golden_logits.
//
// ---------------------------------------------------------------------------
// Bounded run time (optional)
// ---------------------------------------------------------------------------
// The run time above depends on the image: more nonzero pixels, more cycles.
// Two inputs make it predictable:
//
//   budget (16 bit)  0 = no limit (the behaviour described above).
//                    Otherwise layer 1 may spend at most 'budget' cycles on
//                    columns. A column of n nonzero weights costs n cycles, but
//                    never less than 3 (the front end needs 3 cycles per column).
//                    A column is started only if it still fits into the budget.
//                    The first column that does not fit ENDS layer 1: the
//                    remaining inputs are dropped. Inputs are visited in the
//                    order they were written, so the most useful inputs must
//                    be written first (software/export_budget.py does that).
//                    Layer 2 is small and always runs completely.
//   const_time       1 = 'done' comes exactly budget + PAD cycles after 'start'
//                    for EVERY image (needs budget != 0). The run time then
//                    tells nothing about the image.
//
// The same rule is implemented in software/budget_lib.py (truncate with
// column_cycles), which produces the golden values for tb_zs_budget.v.
module sparse_mlp_zs #(
    parameter PAD      = 800,   // const_time: cycles after the budget (finish sweeps + all of layer 2)
    parameter FC1_PTR  = "hardware/mem/fc1_csc_ptr.hex",
    parameter FC1_ROW  = "hardware/mem/fc1_csc_row.hex",
    parameter FC1_VAL  = "hardware/mem/fc1_csc_val.hex",
    parameter FC1_BIAS = "hardware/mem/fc1_bias.hex",
    parameter FC2_PTR  = "hardware/mem/fc2_csc_ptr.hex",
    parameter FC2_ROW  = "hardware/mem/fc2_csc_row.hex",
    parameter FC2_VAL  = "hardware/mem/fc2_csc_val.hex",
    parameter FC2_BIAS = "hardware/mem/fc2_bias.hex",
    parameter SHIFT_FILE = "hardware/mem/hidden_shift.hex"
) (
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 start,

    input  wire [15:0]          budget,      // layer-1 cycle budget, 0 = no limit
    input  wire                 const_time,  // 1 = done exactly budget + PAD cycles after start

    // write port of the image (pixels must come in order 0..783)
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
    localparam DEPTH = 16384;   // entries in each column list (2^14)

    // ---------------- ROMs (filled by $readmemh in simulation) ----------------
    reg [15:0] cptr1 [0:784];          // fc1: 784 columns + 1
    reg [7:0]  crow1 [0:DEPTH-1];
    reg [7:0]  cval1 [0:DEPTH-1];
    reg [31:0] bias1 [0:63];

    reg [15:0] cptr2 [0:64];           // fc2: 64 columns + 1
    reg [7:0]  crow2 [0:DEPTH-1];
    reg [7:0]  cval2 [0:DEPTH-1];
    reg [31:0] bias2 [0:9];

    reg [7:0]  shift_rom [0:0];

    integer n1, n2;
    initial begin
        $readmemh(FC1_PTR, cptr1, 0, 784);
        n1 = cptr1[784];
        if (n1 > DEPTH) $display("ERROR: sparse_mlp_zs: fc1 has %0d nonzero weights, DEPTH is %0d", n1, DEPTH);
        $readmemh(FC1_ROW,  crow1, 0, n1 - 1);
        $readmemh(FC1_VAL,  cval1, 0, n1 - 1);
        $readmemh(FC1_BIAS, bias1, 0, 63);

        $readmemh(FC2_PTR, cptr2, 0, 64);
        n2 = cptr2[64];
        if (n2 > DEPTH) $display("ERROR: sparse_mlp_zs: fc2 has %0d nonzero weights, DEPTH is %0d", n2, DEPTH);
        $readmemh(FC2_ROW,  crow2, 0, n2 - 1);
        $readmemh(FC2_VAL,  cval2, 0, n2 - 1);
        $readmemh(FC2_BIAS, bias2, 0, 9);

        $readmemh(SHIFT_FILE, shift_rom, 0, 0);
    end

    wire [4:0] shift = shift_rom[0][4:0];

    // ---------------- RAMs and registers ----------------
    // nonzero-input list: (position, value) pairs. First the pixels of the
    // image, later (layer 2) the nonzero hidden values.
    reg [9:0]  al_idx [0:1023];
    reg [7:0]  al_val [0:1023];
    reg [10:0] a_cnt;                  // number of entries in the list

    reg [7:0]  hid [0:63];             // hidden values (kept for the testbench; the
                                       // hardware itself only needs the list)

    reg signed [31:0] acc [0:63];      // partial sums of the current layer
    reg [63:0]        touched;         // touched[n] = acc[n] holds a partial sum

    // ---------------- control state ----------------
    localparam S_IDLE = 2'd0;
    localparam S_RUN  = 2'd1;          // stream columns through the MAC
    localparam S_FIN  = 2'd2;          // finish sweep: bias, ReLU, shift, store
    localparam S_PAD  = 2'd3;          // const_time only: wait until budget + PAD cycles have passed

    reg [1:0]         state;
    reg               layer;           // 0 = fc1, 1 = fc2
    reg [6:0]         neuron;          // neuron counter of the finish sweep
    reg signed [31:0] best_val;
    reg [3:0]         best_idx;

    // front end
    reg [10:0] k;                      // next list entry to fetch
    reg [1:0]  f_st;                   // 0 = ready to fetch, 1 = list data in, 2 = pointers in
    reg [7:0]  a_hold;
    reg        desc_valid;             // a column descriptor is waiting
    reg [15:0] d_lo, d_hi;
    reg [7:0]  d_a;

    // run-time budget (layer 1)
    reg [16:0] used;                   // cycles already granted to columns of this layer
    reg        cut;                    // a column did not fit: no more columns in this layer
    reg [16:0] t_run;                  // clock cycles since start

    // issue stage
    reg [15:0] cur_e, cur_hi;          // entries still to issue: cur_e .. cur_hi-1
    reg [7:0]  cur_a;                  // activation of the current column

    // MAC stage inputs
    reg        v1;
    reg [7:0]  a1;

    // synchronous reads (data appears one clock edge after the address, like block RAM)
    reg [9:0]  aj_q;   reg [7:0] av_q;      // list entry k
    reg [15:0] plo_q, phi_q;                // ptr[j], ptr[j+1]
    reg [7:0]  erow_q, eval_q;              // column entry cur_e

    always @(posedge clk) begin
        aj_q   <= al_idx[k[9:0]];
        av_q   <= al_val[k[9:0]];
        plo_q  <= layer ? cptr2[aj_q[6:0]]            : cptr1[aj_q];
        phi_q  <= layer ? cptr2[aj_q[6:0] + 7'd1]     : cptr1[aj_q + 10'd1];
        erow_q <= layer ? crow2[cur_e[13:0]]          : crow1[cur_e[13:0]];
        eval_q <= layer ? cval2[cur_e[13:0]]          : cval1[cur_e[13:0]];
    end

    // ---------------- stage conditions ----------------
    wire        run         = (state == S_RUN);
    wire [15:0] cur_next    = cur_e + 16'd1;
    wire        issue       = run && (cur_e != cur_hi);
    wire        last_issue  = issue && (cur_next == cur_hi);
    // load the next descriptor when the current column is empty or being finished
    wire        take        = run && desc_valid && (!issue || last_issue);
    wire        fetch_ok    = run && (f_st == 2'd0) && ({1'b0, k} < {1'b0, a_cnt}) && (!desc_valid || take) && !cut;
    wire        drained     = run && (f_st == 2'd0) && ((k >= a_cnt) || cut) && !desc_valid && !issue && !v1;

    // ---------------- budget check (valid in front-end state 2, when the pointers have arrived) ----------------
    wire [15:0] col_len   = phi_q - plo_q;                                    // nonzero weights in this column
    wire [16:0] col_cost  = (col_len < 16'd3) ? 17'd3 : {1'b0, col_len};      // cycles this column costs
    wire [16:0] used_next = used + col_cost;
    wire        misfit    = (budget != 16'd0) && !layer && (used_next > {1'b0, budget});
    wire [16:0] t_target  = {1'b0, budget} + PAD;

    // ---------------- MAC stage (acc[row] += a * w) ----------------
    wire [5:0]          mrow = erow_q[5:0];
    wire signed [31:0]  mold = touched[mrow] ? acc[mrow] : 32'sd0;
    wire signed [8:0]   a_s  = {1'b0, a1};       // unsigned activation -> positive signed
    wire signed [7:0]   w_s  = eval_q;           // signed weight
    wire signed [16:0]  prod = a_s * w_s;

    // ---------------- finish sweep (one neuron per cycle) ----------------
    wire [5:0]          r6     = neuron[5:0];
    wire signed [31:0]  acc_r  = touched[r6] ? acc[r6] : 32'sd0;
    wire signed [31:0]  bias_r = layer ? bias2[neuron[3:0]] : bias1[r6];
    wire signed [31:0]  acc_b  = acc_r + bias_r;                      // add the bias
    wire signed [31:0]  relu   = acc_b[31] ? 32'sd0 : acc_b;          // ReLU
    wire        [31:0]  shifted = relu >> shift;                      // divide by 2^SHIFT
    wire        [7:0]   hval   = (shifted > 32'd127) ? 8'd127 : shifted[7:0];   // saturate

    wire        better   = (neuron == 7'd0) || (acc_b > best_val);    // strict: first max wins
    wire [3:0]  next_idx = better ? neuron[3:0] : best_idx;

    // ---------------- nonzero list write port (shared) ----------------
    // image load: only nonzero pixels are stored; address 0 restarts the list
    wire        ld_nz  = img_we && !busy && (img_wdata != 8'd0);
    wire [10:0] ld_pos = (img_waddr == 10'd0) ? 11'd0 : a_cnt;
    // finish sweep of layer 1: store the nonzero hidden values
    wire        fin_nz = (state == S_FIN) && !layer && (hval != 8'd0);

    wire        al_we = ld_nz | fin_nz;
    wire [9:0]  al_wa = fin_nz ? a_cnt[9:0]        : ld_pos[9:0];
    wire [9:0]  al_wi = fin_nz ? {4'b0000, r6}     : img_waddr;
    wire [7:0]  al_wv = fin_nz ? hval              : img_wdata;

    always @(posedge clk)
        if (al_we) begin
            al_idx[al_wa] <= al_wi;
            al_val[al_wa] <= al_wv;
        end

    // ---------------- main sequential block ----------------
    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            layer      <= 1'b0;
            neuron     <= 7'd0;
            busy       <= 1'b0;
            done       <= 1'b0;
            out_valid  <= 1'b0;
            out_idx    <= 4'd0;
            out_val    <= 32'sd0;
            pred       <= 4'd0;
            best_val   <= 32'sd0;
            best_idx   <= 4'd0;
            a_cnt      <= 11'd0;
            k          <= 11'd0;
            f_st       <= 2'd0;
            a_hold     <= 8'd0;
            desc_valid <= 1'b0;
            d_lo       <= 16'd0;
            d_hi       <= 16'd0;
            d_a        <= 8'd0;
            cur_e      <= 16'd0;
            cur_hi     <= 16'd0;
            cur_a      <= 8'd0;
            v1         <= 1'b0;
            a1         <= 8'd0;
            touched    <= 64'd0;
            used       <= 17'd0;
            cut        <= 1'b0;
            t_run      <= 17'd0;
        end else begin
            done      <= 1'b0;
            out_valid <= 1'b0;

            // cycle counter of the current run (set to 0 by start, see S_IDLE)
            if (state != S_IDLE) t_run <= t_run + 17'd1;

            // pipeline registers between issue stage and MAC stage
            v1 <= issue;
            a1 <= cur_a;

            // MAC stage
            if (v1) begin
                acc[mrow]     <= mold + prod;
                touched[mrow] <= 1'b1;
            end

            case (state)
                // -------------------------------------------------------
                S_IDLE: begin
                    // image load: keep the list length up to date
                    if (img_we && !busy)
                        a_cnt <= ld_nz ? (ld_pos + 11'd1) : ld_pos;

                    if (start) begin
                        busy       <= 1'b1;
                        layer      <= 1'b0;
                        neuron     <= 7'd0;
                        k          <= 11'd0;
                        f_st       <= 2'd0;
                        desc_valid <= 1'b0;
                        cur_e      <= 16'd0;
                        cur_hi     <= 16'd0;
                        touched    <= 64'd0;
                        used       <= 17'd0;
                        cut        <= 1'b0;
                        t_run      <= 17'd0;
                        state      <= S_RUN;
                    end
                end

                // -------------------------------------------------------
                S_RUN: begin
                    // ---- front end: list entry -> column pointers -> descriptor
                    case (f_st)
                        2'd0: if (fetch_ok) f_st <= 2'd1;           // address k is presented now
                        2'd1: begin a_hold <= av_q; f_st <= 2'd2; end // (j, a) arrived, pointers are read now
                        2'd2: begin                                  // pointers arrived
                            if (misfit) begin
                                cut        <= 1'b1;                  // over budget: this and all later columns are dropped
                            end else begin
                                d_lo       <= plo_q;
                                d_hi       <= phi_q;
                                d_a        <= a_hold;
                                desc_valid <= 1'b1;
                                used       <= used_next;
                            end
                            k          <= k + 11'd1;
                            f_st       <= 2'd0;
                        end
                        default: f_st <= 2'd0;
                    endcase

                    // ---- issue stage
                    if (issue) cur_e <= cur_next;
                    if (take) begin
                        cur_e      <= d_lo;
                        cur_hi     <= d_hi;
                        cur_a      <= d_a;
                        desc_valid <= 1'b0;
                    end

                    // ---- everything streamed and written back?
                    if (drained) begin
                        state  <= S_FIN;
                        neuron <= 7'd0;
                        if (!layer) a_cnt <= 11'd0;   // the list is rebuilt from the hidden values
                    end
                end

                // -------------------------------------------------------
                S_FIN: begin
                    if (!layer) begin
                        // layer 1: hidden value of neuron r6
                        hid[r6] <= hval;
                        if (hval != 8'd0) a_cnt <= a_cnt + 11'd1;     // (list write: see fin_nz)
                        if (neuron == 7'd63) begin
                            layer      <= 1'b1;
                            neuron     <= 7'd0;
                            k          <= 11'd0;
                            f_st       <= 2'd0;
                            desc_valid <= 1'b0;
                            cur_e      <= 16'd0;
                            cur_hi     <= 16'd0;
                            touched    <= 64'd0;
                            used       <= 17'd0;
                            cut        <= 1'b0;
                            state      <= S_RUN;
                        end else begin
                            neuron <= neuron + 7'd1;
                        end
                    end else begin
                        // layer 2: report the logit, keep track of the maximum
                        out_valid <= 1'b1;
                        out_idx   <= neuron[3:0];
                        out_val   <= acc_b;
                        best_val  <= better ? acc_b : best_val;
                        best_idx  <= next_idx;
                        if (neuron == 7'd9) begin
                            pred  <= next_idx;
                            a_cnt <= 11'd0;       // the image is consumed: a new start without
                                                  // new pixels sees an all-zero image
                            if (const_time && (budget != 16'd0)) begin
                                state <= S_PAD;   // the answer is ready, but 'done' waits
                            end else begin
                                busy  <= 1'b0;
                                done  <= 1'b1;
                                state <= S_IDLE;
                            end
                        end else begin
                            neuron <= neuron + 7'd1;
                        end
                    end
                end

                // -------------------------------------------------------
                S_PAD: begin
                    // const_time: every run ends at the same cycle, whatever the image was
                    if (t_run >= t_target) begin
                        busy  <= 1'b0;
                        done  <= 1'b1;
                        state <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
