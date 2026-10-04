// sparsemac_pcpi.v - The sparse accelerator as a PicoRV32 PCPI co-processor
//
// PicoRV32 has a "Pico Co-Processor Interface" (PCPI). When the core fetches an
// instruction it does not know, it puts the instruction word and the values of
// rs1 and rs2 on the PCPI wires and waits. A co-processor that recognises the
// instruction answers with a result (rd) when it is finished. This module
// recognises four custom instructions in the RISC-V "custom-0" opcode space
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
//   funct3 4..7 and funct7 != 0 are NOT claimed: the core treats them as
//   illegal instructions (trap), so the opcode space stays free for later.
//
// How the handshake works (PicoRV32 side):
//   * the core raises pcpi_valid and keeps rs1/rs2 stable until we answer,
//   * we must raise pcpi_wait within 16 cycles, otherwise the core traps with
//     "illegal instruction". We raise it at once and hold it until we answer,
//   * one cycle of pcpi_ready (with pcpi_wr = 1 and pcpi_rd) ends the instruction.
//
// ZERO_SKIP = 1 uses sparse_mlp_zs (skips zero weights AND zero activations),
// ZERO_SKIP = 0 uses sparse_mlp    (skips zero weights only).
// The weight memories are loaded from hardware/mem/*.hex by $readmemh, so the
// simulation must be started from the repository root.
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
    output wire        pcpi_ready
);
    // ---------------- instruction decode ----------------
    wire        op_match = pcpi_valid && (pcpi_insn[6:0] == 7'b0001011) && (pcpi_insn[31:25] == 7'b0000000);
    wire [2:0]  f3       = pcpi_insn[14:12];
    wire        is_ours  = op_match && !f3[2];             // funct3 = 0..3

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

    generate
        if (ZERO_SKIP) begin : g_zs
            sparse_mlp_zs u_nn (
                .clk(clk), .rst(rst), .start(nn_start),
                .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
                .busy(nn_busy), .done(nn_done),
                .out_valid(out_valid), .out_idx(out_idx), .out_val(out_val),
                .pred(nn_pred)
            );
        end else begin : g_ws
            sparse_mlp u_nn (
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
    localparam W_IDLE = 2'd0;   // waiting for one of our instructions
    localparam W_LD   = 2'd1;   // writing the four pixels of an SMAC.LDW
    localparam W_RUN  = 2'd2;   // accelerator is running (SMAC.RUN)
    localparam W_RESP = 2'd3;   // answering the core

    reg [1:0]  st;
    reg [31:0] ld_word;
    reg [9:0]  ld_base;
    reg [1:0]  ld_cnt;
    reg [31:0] resp;
    reg [31:0] run_cnt;
    reg [31:0] cyc_last;

    wire [31:0] logit_sel = (pcpi_rs1 < 32'd10) ? logits[pcpi_rs1[3:0]] : 32'd0;

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
        if (st == W_LD) begin
            img_we    = 1'b1;
            img_waddr = ld_base + {8'b0, ld_cnt};
            img_wdata = ld_word >> (8 * ld_cnt);
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
        end else begin
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
                        default: begin                    // SMAC.CYC
                            resp <= cyc_last;
                            st   <= W_RESP;
                        end
                    endcase
                end

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
            endcase
        end
    end
endmodule
