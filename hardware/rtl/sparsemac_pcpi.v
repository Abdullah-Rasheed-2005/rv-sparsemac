// sparsemac_pcpi.v - The sparse accelerator as a PicoRV32 PCPI co-processor
//
// PicoRV32 has a "Pico Co-Processor Interface" (PCPI). When the core fetches an
// instruction it does not know, it puts the instruction word and the values of
// rs1 and rs2 on the PCPI wires and waits. A co-processor that recognises the
// instruction answers with a result (rd) when it is finished. This module
// recognises six custom instructions in the RISC-V "custom-0" opcode space
// (opcode 0x0B, funct7 = 0, R-type) and drives the accelerator with them.
//
//   31      25 24  20 19  15 14  12 11   7 6      0
//  +----------+------+------+------+------+--------+
//  | funct7=0 | rs2  | rs1  |funct3|  rd  | 0001011|      R-type, custom-0
//  +----------+------+------+------+------+--------+
//
//   funct3  name            what it does
//   ------  --------------  ----------------------------------------------------
//     0     SMAC.LDW        rs1 = pixel address (multiple of 4), rs2 = 4 pixels
//                           packed little-endian (pixel rs1 in bits 7:0).
//                           Writes the four pixels. rd gets 0.
//                           RULE: send the words in ascending order starting at
//                           address 0 (address 0 starts a new image).
//     1     SMAC.RUN        start the inference and WAIT until it is finished.
//                           rd = predicted digit (0..9).
//     2     SMAC.LOGIT      rd = logit number rs1 (0..9) of the last run
//                           (signed 32-bit). rd = 0 if rs1 > 9.
//     3     SMAC.CYC        rd = accelerator clock cycles of the last run,
//                           counted from the start edge to the done edge.
//
//     4     SMAC.CFG        bound the run time (zero-skipping design only).
//                           rs1[15:0] = layer-1 cycle budget, 0 = no limit.
//                           rs2[0]    = 1: constant time (every SMAC.RUN takes
//                           budget + PAD + 1 accelerator cycles). rd gets 0.
//                           The setting stays until the next SMAC.CFG or reset.
//
//     5     SMAC.RUNM       like SMAC.RUN, but the accelerator fetches the image
//                           ITSELF: rs1 = address of the 784 pixels in memory
//                           (a multiple of 4; pixels in their normal order).
//                           rd = predicted digit. No SMAC.LDW is needed.
//                           SMAC.CYC then returns the cycles of the WHOLE
//                           instruction (fetching + ordering + running).
//
//   funct3 6..7 and funct7 != 0 are NOT claimed: the core treats them as
//   illegal instructions (trap), so the opcode space stays free for later.
//
// How SMAC.RUNM gets the image (the "loader"):
//   1. FETCH  The wrapper is a second master on the memory bus (ld_* ports).
//             While the core waits for a co-processor instruction it does not
//             use the bus (after one early prefetch), so the wrapper reads the
//             196 image words one after another into a small buffer.
//   2. ORDER  The accelerator visits its inputs in the order they are written,
//             and a run-time budget needs the most useful inputs first. The
//             table order.hex says which pixel is input number k. The wrapper
//             walks k = 0..783, takes pixel order[k] from the buffer and writes
//             it as input k (one per clock).
//   3. RUN    as SMAC.RUN.
//   Steps 1 and 2 take the same number of cycles for every image.
//
// How the handshake works (PicoRV32 side):
//   * the core raises pcpi_valid and keeps rs1/rs2 stable until we answer,
//   * we must raise pcpi_wait within 16 cycles, otherwise the core traps with
//     "illegal instruction". We raise it at once and hold it until we answer,
//   * one cycle of pcpi_ready (with pcpi_wr = 1 and pcpi_rd) ends the instruction.
//
// ZERO_SKIP = 1 uses sparse_mlp_zs (skips zero weights AND zero activations),
// ZERO_SKIP = 0 uses sparse_mlp    (skips zero weights only).
// The weight memories are loaded by $readmemh from the folder MEMDIR (default
// hardware/mem), so the simulation must be started from the repository root.
`ifndef MEMDIR
  `define MEMDIR "hardware/mem"
`endif
module sparsemac_pcpi #(
    parameter ZERO_SKIP = 1
) (
    input  wire        clk,
    input  wire        resetn,

    input  wire        pcpi_valid,
    input  wire [31:0] pcpi_insn,
    input  wire [31:0] pcpi_rs1,
    input  wire [31:0] pcpi_rs2,
    output wire        pcpi_wr,
    output wire [31:0] pcpi_rd,
    output wire        pcpi_wait,
    output wire        pcpi_ready,

    // memory master of the image loader (SMAC.RUNM). Same handshake as the
    // PicoRV32 memory port: ld_valid stays high until ld_ready, the data is
    // taken in the cycle where ld_ready is high. Read only.
    output wire        ld_valid,
    output wire [31:0] ld_addr,
    input  wire [31:0] ld_rdata,
    input  wire        ld_ready
);
    // ---------------- instruction decode ----------------
    wire        op_match = pcpi_valid && (pcpi_insn[6:0] == 7'b0001011) && (pcpi_insn[31:25] == 7'b0000000);
    wire [2:0]  f3       = pcpi_insn[14:12];
    wire        is_ours  = op_match && (f3 <= 3'd5);       // funct3 = 0..5

    // ---------------- the accelerator ----------------
    reg         nn_start;
    reg         img_we;
    reg  [9:0]  img_waddr;
    reg  [7:0]  img_wdata;
    wire        nn_busy, nn_done, out_valid;
    wire [3:0]  out_idx;
    wire signed [31:0] out_val;
    wire [3:0]  nn_pred;
    wire        rst = !resetn;

    // run-time configuration, written by SMAC.CFG
    reg  [15:0] cfg_budget;
    reg         cfg_ct;

    generate
        if (ZERO_SKIP) begin : g_zs
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
            ) u_nn (
                .clk(clk), .rst(rst), .start(nn_start),
                .budget(cfg_budget), .const_time(cfg_ct),
                .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
                .busy(nn_busy), .done(nn_done),
                .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
                .pred(nn_pred)
            );
        end else begin : g_ws
            sparse_mlp #(
                .FC1_PTR   ({`MEMDIR, "/fc1_nz_ptr.hex"}),
                .FC1_IDX   ({`MEMDIR, "/fc1_nz_idx.hex"}),
                .FC1_VAL   ({`MEMDIR, "/fc1_nz_val.hex"}),
                .FC1_BIAS  ({`MEMDIR, "/fc1_bias.hex"}),
                .FC2_PTR   ({`MEMDIR, "/fc2_nz_ptr.hex"}),
                .FC2_IDX   ({`MEMDIR, "/fc2_nz_idx.hex"}),
                .FC2_VAL   ({`MEMDIR, "/fc2_nz_val.hex"}),
                .FC2_BIAS  ({`MEMDIR, "/fc2_bias.hex"}),
                .SHIFT_FILE({`MEMDIR, "/hidden_shift.hex"})
            ) u_nn (
                .clk(clk), .rst(rst), .start(nn_start),
                .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
                .busy(nn_busy), .done(nn_done),
                .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
                .pred(nn_pred)
            );
        end
    endgenerate

    // the ten logits of the last run
    reg signed [31:0] logits [0:9];
    always @(posedge clk)
        if (out_valid) logits[out_idx] <= out_val;

    // ---------------- small control FSM ----------------
    localparam W_IDLE  = 3'd0;  // waiting for one of our instructions
    localparam W_LD    = 3'd1;  // writing the four pixels of an SMAC.LDW
    localparam W_RUN   = 3'd2;  // accelerator is running (SMAC.RUN / SMAC.RUNM)
    localparam W_RESP  = 3'd3;  // answering the core
    localparam W_FETCH = 3'd4;  // SMAC.RUNM: reading the image words from memory
    localparam W_ORDER = 3'd5;  // SMAC.RUNM: writing the pixels in input order
    localparam W_GO    = 3'd6;  // SMAC.RUNM: start pulse for the accelerator

    reg [2:0]  st;
    reg [31:0] ld_word;
    reg [9:0]  ld_base;
    reg [1:0]  ld_cnt;
    reg [31:0] resp;
    reg [31:0] run_cnt;
    reg [31:0] cyc_last;

    wire [31:0] logit_sel = (pcpi_rs1 < 32'd10) ? logits[pcpi_rs1[3:0]] : 32'd0;

    // ---------------- image loader (SMAC.RUNM) ----------------
    reg [31:0] ibuf [0:195];            // the image, 4 pixels per word, as read from memory
    reg [15:0] order_rom [0:783];       // input k of the accelerator = pixel order_rom[k]
    initial $readmemh({`MEMDIR, "/order.hex"}, order_rom, 0, 783);

    reg [31:0] img_base;                // address of the image (rs1 of SMAC.RUNM)
    reg [7:0]  fw;                      // FETCH: word counter 0..195
    reg [9:0]  ka;                      // ORDER: next input number to look up
    reg        o_run;                   // ORDER: still presenting ka to the table
    // two pipeline stages: table output, then buffer output (both synchronous reads)
    reg        v1, v2;
    reg [9:0]  k1, k2;
    reg [9:0]  ord_q;                   // pixel number of input k1
    reg [1:0]  sel_q;                   // which byte of buf_q
    reg [31:0] buf_q;

    assign ld_valid = (st == W_FETCH);
    assign ld_addr  = img_base + {22'd0, fw, 2'b00};

    always @(posedge clk) begin
        if (st == W_FETCH && ld_ready) ibuf[fw] <= ld_rdata;
        ord_q <= order_rom[ka][9:0];
        buf_q <= ibuf[ord_q[9:2]];
        sel_q <= ord_q[1:0];
    end

    // The core may only be told "wait" while one of OUR instructions is active.
    assign pcpi_wait  = is_ours;
    assign pcpi_ready = (st == W_RESP) && pcpi_valid;
    assign pcpi_wr    = pcpi_ready;
    assign pcpi_rd    = resp;

    // start pulse and pixel write port are combinational outputs of the FSM state
    always @* begin
        nn_start  = 1'b0;
        img_we    = 1'b0;
        img_waddr = 10'd0;
        img_wdata = 8'd0;
        if (st == W_IDLE && is_ours && f3 == 3'd1 && !nn_busy)
            nn_start = 1'b1;
        if (st == W_GO)
            nn_start = 1'b1;
        if (st == W_LD) begin
            img_we    = 1'b1;
            img_waddr = ld_base + {8'b0, ld_cnt};
            img_wdata = ld_word >> (8 * ld_cnt);
        end
        if (st == W_ORDER && v2) begin
            img_we    = 1'b1;
            img_waddr = k2;                       // input number
            img_wdata = buf_q >> (8 * sel_q);     // pixel order_rom[k2]
        end
    end

    always @(posedge clk) begin
        if (!resetn) begin
            st       <= W_IDLE;
            ld_word  <= 32'd0;
            ld_base  <= 10'd0;
            ld_cnt   <= 2'd0;
            resp     <= 32'd0;
            run_cnt  <= 32'd0;
            cyc_last <= 32'd0;
            cfg_budget <= 16'd0;
            cfg_ct     <= 1'b0;
            img_base   <= 32'd0;
            fw         <= 8'd0;
            ka         <= 10'd0;
            o_run      <= 1'b0;
            v1         <= 1'b0;
            v2         <= 1'b0;
            k1         <= 10'd0;
            k2         <= 10'd0;
        end else begin
            // every cycle of SMAC.RUNM counts (W_RUN does its own counting below)
            if (st == W_FETCH || st == W_ORDER || st == W_GO) run_cnt <= run_cnt + 32'd1;

            case (st)
                W_IDLE: if (is_ours) begin
                    case (f3)
                        3'd0: begin                       // SMAC.LDW
                            ld_word <= pcpi_rs2;
                            ld_base <= pcpi_rs1[9:0];
                            ld_cnt  <= 2'd0;
                            st      <= W_LD;
                        end
                        3'd1: begin                       // SMAC.RUN
                            if (!nn_busy) begin
                                run_cnt <= 32'd0;
                                st      <= W_RUN;
                            end
                        end
                        3'd2: begin                       // SMAC.LOGIT
                            resp <= logit_sel;
                            st   <= W_RESP;
                        end
                        3'd3: begin                       // SMAC.CYC
                            resp <= cyc_last;
                            st   <= W_RESP;
                        end
                        3'd4: begin                       // SMAC.CFG
                            cfg_budget <= pcpi_rs1[15:0];
                            cfg_ct     <= pcpi_rs2[0];
                            resp       <= 32'd0;
                            st         <= W_RESP;
                        end
                        default: begin                    // SMAC.RUNM (funct3 = 5)
                            if (!nn_busy) begin
                                img_base <= {pcpi_rs1[31:2], 2'b00};
                                fw       <= 8'd0;
                                run_cnt  <= 32'd0;
                                st       <= W_FETCH;
                            end
                        end
                    endcase
                end

                W_FETCH: begin
                    // one word per bus access; ibuf[fw] is written in the block above
                    if (ld_ready) begin
                        if (fw == 8'd195) begin
                            ka    <= 10'd0;
                            o_run <= 1'b1;
                            v1    <= 1'b0;
                            v2    <= 1'b0;
                            st    <= W_ORDER;
                        end else begin
                            fw <= fw + 8'd1;
                        end
                    end
                end

                W_ORDER: begin
                    // ka -> (table) -> ord_q -> (buffer) -> buf_q -> pixel write
                    v1 <= o_run;
                    k1 <= ka;
                    v2 <= v1;
                    k2 <= k1;
                    if (o_run) begin
                        if (ka == 10'd783) o_run <= 1'b0;
                        else               ka    <= ka + 10'd1;
                    end
                    if (v2 && k2 == 10'd783) st <= W_GO;   // the last pixel is written in this cycle
                end

                W_GO: st <= W_RUN;                          // nn_start is high in this cycle

                W_LD: begin
                    ld_cnt <= ld_cnt + 2'd1;
                    if (ld_cnt == 2'd3) begin
                        resp <= 32'd0;
                        st   <= W_RESP;
                    end
                end

                W_RUN: begin
                    if (nn_done) begin
                        resp     <= {28'd0, nn_pred};
                        cyc_last <= run_cnt;
                        st       <= W_RESP;
                    end else begin
                        run_cnt <= run_cnt + 32'd1;
                    end
                end

                W_RESP: st <= W_IDLE;                    // ready was high in this cycle

                default: st <= W_IDLE;
            endcase
        end
    end
endmodule
