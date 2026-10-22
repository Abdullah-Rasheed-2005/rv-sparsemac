// dp16kd_sim.v - Behavioural model of the ECP5 block RAM (DP16KD) for the
// gate-level simulation (make sim-gate).
//
// Why this file exists: the cell library of Yosys (share/ecp5/cells_sim.v)
// declares DP16KD only as an empty box, so a netlist with block RAMs cannot be
// simulated with it. This model covers what synth_ecp5 generates for this
// design: widths 1, 2, 4, 9, 18 on each port, REGMODE "NOREG" (data one clock
// after the address), the three write modes, chip selects and the output reset.
//
// Bit layout (the same as the INITVAL strings that Yosys writes):
//   the RAM holds 1024 words of 18 bits; INITVAL_xx holds words 16*xx .. 16*xx+15,
//   each in a 20-bit slot (the upper 2 bits of a slot are unused).
//   One 18-bit word = two 9-bit bytes. Narrow widths (1, 2, 4) use only the
//   lower 8 bits of each byte; the 9th bit is reached only at widths 9 and 18.
//   Width 18 uses address bits 13:4, width 9 bits 13:3, width 4 bits 13:2,
//   width 2 bits 13:1, width 1 bits 13:0. At width 18 the address bits 1:0
//   are the byte write enables (as in the Yosys mapping).
//
// The model is checked by the gate-level test itself: the ROM contents of the
// accelerator are read through it, so a wrong layout would give wrong logits.
`timescale 1ns/1ps
module DP16KD (
    input DIA17, DIA16, DIA15, DIA14, DIA13, DIA12, DIA11, DIA10, DIA9, DIA8, DIA7, DIA6, DIA5, DIA4, DIA3, DIA2, DIA1, DIA0,
    input ADA13, ADA12, ADA11, ADA10, ADA9, ADA8, ADA7, ADA6, ADA5, ADA4, ADA3, ADA2, ADA1, ADA0,
    input CEA, OCEA, CLKA, WEA, RSTA,
    input CSA2, CSA1, CSA0,
    output DOA17, DOA16, DOA15, DOA14, DOA13, DOA12, DOA11, DOA10, DOA9, DOA8, DOA7, DOA6, DOA5, DOA4, DOA3, DOA2, DOA1, DOA0,

    input DIB17, DIB16, DIB15, DIB14, DIB13, DIB12, DIB11, DIB10, DIB9, DIB8, DIB7, DIB6, DIB5, DIB4, DIB3, DIB2, DIB1, DIB0,
    input ADB13, ADB12, ADB11, ADB10, ADB9, ADB8, ADB7, ADB6, ADB5, ADB4, ADB3, ADB2, ADB1, ADB0,
    input CEB, OCEB, CLKB, WEB, RSTB,
    input CSB2, CSB1, CSB0,
    output DOB17, DOB16, DOB15, DOB14, DOB13, DOB12, DOB11, DOB10, DOB9, DOB8, DOB7, DOB6, DOB5, DOB4, DOB3, DOB2, DOB1, DOB0
);
    parameter DATA_WIDTH_A = 18;
    parameter DATA_WIDTH_B = 18;
    parameter REGMODE_A = "NOREG";
    parameter REGMODE_B = "NOREG";
    parameter RESETMODE = "SYNC";
    parameter ASYNC_RESET_RELEASE = "SYNC";
    parameter CSDECODE_A = "0b000";
    parameter CSDECODE_B = "0b000";
    parameter WRITEMODE_A = "NORMAL";
    parameter WRITEMODE_B = "NORMAL";
    parameter CLKAMUX = "CLKA";
    parameter CLKBMUX = "CLKB";
    parameter GSR = "ENABLED";
    parameter INIT_DATA = "STATIC";
    parameter WID = 0;

    parameter INITVAL_00 = 320'h0; parameter INITVAL_01 = 320'h0; parameter INITVAL_02 = 320'h0; parameter INITVAL_03 = 320'h0;
    parameter INITVAL_04 = 320'h0; parameter INITVAL_05 = 320'h0; parameter INITVAL_06 = 320'h0; parameter INITVAL_07 = 320'h0;
    parameter INITVAL_08 = 320'h0; parameter INITVAL_09 = 320'h0; parameter INITVAL_0A = 320'h0; parameter INITVAL_0B = 320'h0;
    parameter INITVAL_0C = 320'h0; parameter INITVAL_0D = 320'h0; parameter INITVAL_0E = 320'h0; parameter INITVAL_0F = 320'h0;
    parameter INITVAL_10 = 320'h0; parameter INITVAL_11 = 320'h0; parameter INITVAL_12 = 320'h0; parameter INITVAL_13 = 320'h0;
    parameter INITVAL_14 = 320'h0; parameter INITVAL_15 = 320'h0; parameter INITVAL_16 = 320'h0; parameter INITVAL_17 = 320'h0;
    parameter INITVAL_18 = 320'h0; parameter INITVAL_19 = 320'h0; parameter INITVAL_1A = 320'h0; parameter INITVAL_1B = 320'h0;
    parameter INITVAL_1C = 320'h0; parameter INITVAL_1D = 320'h0; parameter INITVAL_1E = 320'h0; parameter INITVAL_1F = 320'h0;
    parameter INITVAL_20 = 320'h0; parameter INITVAL_21 = 320'h0; parameter INITVAL_22 = 320'h0; parameter INITVAL_23 = 320'h0;
    parameter INITVAL_24 = 320'h0; parameter INITVAL_25 = 320'h0; parameter INITVAL_26 = 320'h0; parameter INITVAL_27 = 320'h0;
    parameter INITVAL_28 = 320'h0; parameter INITVAL_29 = 320'h0; parameter INITVAL_2A = 320'h0; parameter INITVAL_2B = 320'h0;
    parameter INITVAL_2C = 320'h0; parameter INITVAL_2D = 320'h0; parameter INITVAL_2E = 320'h0; parameter INITVAL_2F = 320'h0;
    parameter INITVAL_30 = 320'h0; parameter INITVAL_31 = 320'h0; parameter INITVAL_32 = 320'h0; parameter INITVAL_33 = 320'h0;
    parameter INITVAL_34 = 320'h0; parameter INITVAL_35 = 320'h0; parameter INITVAL_36 = 320'h0; parameter INITVAL_37 = 320'h0;
    parameter INITVAL_38 = 320'h0; parameter INITVAL_39 = 320'h0; parameter INITVAL_3A = 320'h0; parameter INITVAL_3B = 320'h0;
    parameter INITVAL_3C = 320'h0; parameter INITVAL_3D = 320'h0; parameter INITVAL_3E = 320'h0; parameter INITVAL_3F = 320'h0;

    // ---------------- storage: 1024 words of 18 bits ----------------
    reg [17:0] mem [0:1023];

    task load_row(input integer row, input [319:0] v);
        integer i;
        begin
            for (i = 0; i < 16; i = i + 1)
                mem[row * 16 + i] = v[i * 20 +: 18];
        end
    endtask

    initial begin
        load_row('h00, INITVAL_00); load_row('h01, INITVAL_01); load_row('h02, INITVAL_02); load_row('h03, INITVAL_03);
        load_row('h04, INITVAL_04); load_row('h05, INITVAL_05); load_row('h06, INITVAL_06); load_row('h07, INITVAL_07);
        load_row('h08, INITVAL_08); load_row('h09, INITVAL_09); load_row('h0A, INITVAL_0A); load_row('h0B, INITVAL_0B);
        load_row('h0C, INITVAL_0C); load_row('h0D, INITVAL_0D); load_row('h0E, INITVAL_0E); load_row('h0F, INITVAL_0F);
        load_row('h10, INITVAL_10); load_row('h11, INITVAL_11); load_row('h12, INITVAL_12); load_row('h13, INITVAL_13);
        load_row('h14, INITVAL_14); load_row('h15, INITVAL_15); load_row('h16, INITVAL_16); load_row('h17, INITVAL_17);
        load_row('h18, INITVAL_18); load_row('h19, INITVAL_19); load_row('h1A, INITVAL_1A); load_row('h1B, INITVAL_1B);
        load_row('h1C, INITVAL_1C); load_row('h1D, INITVAL_1D); load_row('h1E, INITVAL_1E); load_row('h1F, INITVAL_1F);
        load_row('h20, INITVAL_20); load_row('h21, INITVAL_21); load_row('h22, INITVAL_22); load_row('h23, INITVAL_23);
        load_row('h24, INITVAL_24); load_row('h25, INITVAL_25); load_row('h26, INITVAL_26); load_row('h27, INITVAL_27);
        load_row('h28, INITVAL_28); load_row('h29, INITVAL_29); load_row('h2A, INITVAL_2A); load_row('h2B, INITVAL_2B);
        load_row('h2C, INITVAL_2C); load_row('h2D, INITVAL_2D); load_row('h2E, INITVAL_2E); load_row('h2F, INITVAL_2F);
        load_row('h30, INITVAL_30); load_row('h31, INITVAL_31); load_row('h32, INITVAL_32); load_row('h33, INITVAL_33);
        load_row('h34, INITVAL_34); load_row('h35, INITVAL_35); load_row('h36, INITVAL_36); load_row('h37, INITVAL_37);
        load_row('h38, INITVAL_38); load_row('h39, INITVAL_39); load_row('h3A, INITVAL_3A); load_row('h3B, INITVAL_3B);
        load_row('h3C, INITVAL_3C); load_row('h3D, INITVAL_3D); load_row('h3E, INITVAL_3E); load_row('h3F, INITVAL_3F);
    end

    // position of the lowest data bit of the word at address 'ad' inside its 18-bit word
    function integer offs(input integer w, input [13:0] ad);
        case (w)
            18:      offs = 0;
            9:       offs = ad[3] * 9;
            4:       offs = ad[3] * 9 + ad[2] * 4;
            2:       offs = ad[3] * 9 + ad[2:1] * 2;
            default: offs = ad[3] * 9 + ad[2:0];
        endcase
    endfunction

    function [2:0] csval(input [8*5-1:0] s);      // "0b101" -> 3'b101
        csval = {s[16], s[8], s[0]};
    endfunction

    wire [13:0] ada = {ADA13, ADA12, ADA11, ADA10, ADA9, ADA8, ADA7, ADA6, ADA5, ADA4, ADA3, ADA2, ADA1, ADA0};
    wire [13:0] adb = {ADB13, ADB12, ADB11, ADB10, ADB9, ADB8, ADB7, ADB6, ADB5, ADB4, ADB3, ADB2, ADB1, ADB0};
    wire [17:0] dia = {DIA17, DIA16, DIA15, DIA14, DIA13, DIA12, DIA11, DIA10, DIA9, DIA8, DIA7, DIA6, DIA5, DIA4, DIA3, DIA2, DIA1, DIA0};
    wire [17:0] dib = {DIB17, DIB16, DIB15, DIB14, DIB13, DIB12, DIB11, DIB10, DIB9, DIB8, DIB7, DIB6, DIB5, DIB4, DIB3, DIB2, DIB1, DIB0};
    wire        sela = ({CSA2, CSA1, CSA0} == csval(CSDECODE_A));
    wire        selb = ({CSB2, CSB1, CSB0} == csval(CSDECODE_B));

    reg [17:0] doa = 18'd0, dob = 18'd0;
    assign {DOA17, DOA16, DOA15, DOA14, DOA13, DOA12, DOA11, DOA10, DOA9, DOA8, DOA7, DOA6, DOA5, DOA4, DOA3, DOA2, DOA1, DOA0} = doa;
    assign {DOB17, DOB16, DOB15, DOB14, DOB13, DOB12, DOB11, DOB10, DOB9, DOB8, DOB7, DOB6, DOB5, DOB4, DOB3, DOB2, DOB1, DOB0} = dob;

    // one clock edge of a port: read (old, new or no change), then write
    localparam [17:0] MASK_A = (18'd1 << DATA_WIDTH_A) - 18'd1;
    localparam [17:0] MASK_B = (18'd1 << DATA_WIDTH_B) - 18'd1;
    localparam KEEP_A = (WRITEMODE_A == "NORMAL"),  THRU_A = (WRITEMODE_A == "WRITETHROUGH");
    localparam KEEP_B = (WRITEMODE_B == "NORMAL"),  THRU_B = (WRITEMODE_B == "WRITETHROUGH");

    function [17:0] wr_word(input integer w, input [17:0] mask, input [13:0] ad,
                            input [17:0] old, input [17:0] di);
        integer o;
        begin
            o = offs(w, ad);
            if (w == 18)          // width 18: byte write enables on address bits 1:0
                wr_word = {ad[1] ? di[17:9] : old[17:9], ad[0] ? di[8:0] : old[8:0]};
            else
                wr_word = (old & ~(mask << o)) | ((di & mask) << o);
        end
    endfunction

    reg [17:0] olda, oldb, neww;
    always @(posedge CLKA or posedge RSTA)
        if (RSTA) doa <= 18'd0;
        else if (CEA && sela) begin
            olda = mem[ada[13:4]];
            if (WEA) begin
                neww = wr_word(DATA_WIDTH_A, MASK_A, ada, olda, dia);
                mem[ada[13:4]] = neww;
                if (!KEEP_A) doa <= ((THRU_A ? neww : olda) >> offs(DATA_WIDTH_A, ada)) & MASK_A;
            end else
                doa <= (olda >> offs(DATA_WIDTH_A, ada)) & MASK_A;
        end

    always @(posedge CLKB or posedge RSTB)
        if (RSTB) dob <= 18'd0;
        else if (CEB && selb) begin
            oldb = mem[adb[13:4]];
            if (WEB) begin
                neww = wr_word(DATA_WIDTH_B, MASK_B, adb, oldb, dib);
                mem[adb[13:4]] = neww;
                if (!KEEP_B) dob <= ((THRU_B ? neww : oldb) >> offs(DATA_WIDTH_B, adb)) & MASK_B;
            end else
                dob <= (oldb >> offs(DATA_WIDTH_B, adb)) & MASK_B;
        end
endmodule
