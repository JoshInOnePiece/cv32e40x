// -----------------------------------------------------------------------------
// rope_datapath.sv -- the BF16 rotation datapath (steps.md Milestone 4).
//
//       x (BF16)   y (BF16)      cos (BF16)   sin (BF16)
//          |          |              |            |
//          +----+-----+------+-------+------+-----+
//               |            |              |
//          [BF16 mul]   [BF16 mul]    [BF16 mul]   [BF16 mul]
//           x*cos        y*sin         x*sin        y*cos
//               |            |              |            |
//               +-----+------+              +-----+------+
//                     |                           |
//                [BF16 sub]                  [BF16 add]
//                     |                           |
//                  x' (BF16)                   y' (BF16)
//
// 4 BF16 multiplies, 2 BF16 adds, rounded RNE to BF16 at EVERY node (INV-1).
//
// Only ~2 rounding events on the data path, versus ~16 for a CORDIC implementation --
// which is why LUT+multiply is *more* accurate in BF16, not less.
//
// Flow control
// ------------
// Fixed-latency, fully-pipelined stream. `out_ready_i` is NOT accepted: the consumer
// must always be able to sink a result. rope_xif_coproc guarantees this with a credit
// counter, which is also what the XIF rules force it to do anyway (commit_valid and
// mem_result_valid have no ready signal, so the coprocessor must always be able to
// accept -- see steps.md Section 9.3).
//
// Latency is reported on `latency_o` so the testbench and the wrapper can size buffers
// without hard-coding a number that would silently rot if NumPipeRegs changed.
// -----------------------------------------------------------------------------

module rope_datapath
  import rope_pkg::*;
#(
  // Pipeline registers inside each CVFPU unit.
  parameter int unsigned NumPipeRegs = 1,
  // Register the LUT outputs (recommended: the ROM read is a wide logic cone).
  parameter bit          RegLut      = 1'b1,
  // INV-1: strict BF16. 16 is the ONLY value this RTL implements.
  //
  // SCOPE NOTE: steps.md Section 8.3 also asks for an AccWidth=32 build (BF16 in/out,
  // wide internal accumulate) to quantify what strict BF16 costs. That comparison IS
  // produced -- but in the golden model (model/rope_ref.py `rope_wide_acc`, reported by
  // model/characterize.py), not in RTL. Building it in hardware would need FP32 adders
  // plus BF16<->FP32 cast units, i.e. enabling FPnew's CONV opgroup and adding the very
  // conversion stages INV-1 forbids, for a path that is explicitly never shipped. The
  // parameter is kept so the intent is visible, and rejects 32 rather than silently
  // ignoring it. See docs/notes.md.
  parameter int unsigned AccWidth    = rope_pkg::AccWidthStrict
) (
  input  logic                 clk_i,
  input  logic                 rst_ni,
  input  logic                 flush_i,

  // ---- input: one BF16 pair plus its trigonometric coefficients ----------
  input  logic [15:0]          x_i,
  input  logic [15:0]          y_i,
  input  logic [15:0]          sin_i,
  input  logic [15:0]          cos_i,
  input  logic                 in_valid_i,
  output logic                 in_ready_o,

  // ---- output ------------------------------------------------------------
  output logic [15:0]          x_o,
  output logic [15:0]          y_o,
  output logic                 out_valid_o,

  output logic                 busy_o,
  // Total latency in cycles, for buffer sizing / assertions.
  output logic [7:0]           latency_o
);

  // Fail loudly at elaboration rather than quietly producing strict-BF16 results while
  // the caller believes it asked for wide accumulation.
  initial begin : check_acc_width
    if (AccWidth != rope_pkg::AccWidthStrict) begin
      $fatal(1, "rope_datapath: AccWidth=%0d is not implemented in RTL; only strict BF16 (%0d) is. The wide-accumulate comparison lives in model/characterize.py.",
             AccWidth, rope_pkg::AccWidthStrict);
    end
  end

  // ---------------------------------------------------------------------------
  // Stage 0: optionally register the operands and coefficients.
  //
  // Registering here is what protects the XIF timing budget: the coprocessor gets only
  // a small fraction of the cycle on paths through issue_req.rs*, because CV32E40X
  // sources those operands from its register-file bypass network. Zero combinational
  // logic may sit between the operand input and this flop (steps.md Section 9.3 rule 1).
  // ---------------------------------------------------------------------------

  logic [15:0] x_s0, y_s0, sin_s0, cos_s0;
  logic        valid_s0;

  if (RegLut) begin : gen_reg_lut
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        x_s0     <= '0;
        y_s0     <= '0;
        sin_s0   <= '0;
        cos_s0   <= '0;
        valid_s0 <= 1'b0;
      end else if (flush_i) begin
        valid_s0 <= 1'b0;
      end else begin
        x_s0     <= x_i;
        y_s0     <= y_i;
        sin_s0   <= sin_i;
        cos_s0   <= cos_i;
        // NOT `in_valid_i & in_ready_o`: see the in_ready_o comment below. Gating the
        // registered valid on in_ready_o creates a circular dependency through FPnew
        // (whose in_ready is itself gated by in_valid) that deadlocks at reset.
        valid_s0 <= in_valid_i;
      end
    end
  end else begin : gen_no_reg_lut
    assign x_s0     = x_i;
    assign y_s0     = y_i;
    assign sin_s0   = sin_i;
    assign cos_s0   = cos_i;
    assign valid_s0 = in_valid_i;
  end

  // ---------------------------------------------------------------------------
  // Stage 1: four BF16 multipliers
  // ---------------------------------------------------------------------------
  //
  //   p0 = x*cos    p1 = y*sin    p2 = x*sin    p3 = y*cos
  //
  // All four are identically configured, so they share latency and handshake timing.
  // That is asserted below rather than assumed.

  localparam int unsigned NumMul = 4;

  logic [NumMul-1:0][15:0] mul_a, mul_b;
  logic [NumMul-1:0][15:0] mul_r;
  logic [NumMul-1:0]       mul_in_ready, mul_out_valid, mul_busy;

  always_comb begin
    mul_a[0] = x_s0;  mul_b[0] = cos_s0;   // x*cos
    mul_a[1] = y_s0;  mul_b[1] = sin_s0;   // y*sin
    mul_a[2] = x_s0;  mul_b[2] = sin_s0;   // x*sin
    mul_a[3] = y_s0;  mul_b[3] = cos_s0;   // y*cos
  end

  for (genvar g = 0; g < NumMul; g++) begin : gen_mul
    rope_bf16_mul #(
      .NumPipeRegs ( NumPipeRegs )
    ) i_mul (
      .clk_i,
      .rst_ni,
      .a_i         ( mul_a[g]          ),
      .b_i         ( mul_b[g]          ),
      .in_valid_i  ( valid_s0          ),
      .in_ready_o  ( mul_in_ready[g]   ),
      .flush_i     ( flush_i           ),
      .result_o    ( mul_r[g]          ),
      .out_valid_o ( mul_out_valid[g]  ),
      // The consumer (the adder stage) is always ready, so no backpressure exists
      // inside the datapath. See the flow-control note in the header.
      .out_ready_i ( 1'b1              ),
      .busy_o      ( mul_busy[g]       )
    );
  end

  // ---------------------------------------------------------------------------
  // Stage 2: two BF16 adders
  // ---------------------------------------------------------------------------
  //
  //   x' = p0 - p1     (subtract, via FPnew's op_mod -- exact, free)
  //   y' = p2 + p3

  logic [1:0][15:0] add_r;
  logic [1:0]       add_in_ready, add_out_valid, add_busy;

  rope_bf16_add #(
    .NumPipeRegs ( NumPipeRegs ),
    .Sub         ( 1'b1        )
  ) i_sub_x (
    .clk_i,
    .rst_ni,
    .a_i         ( mul_r[0]         ),  // x*cos
    .b_i         ( mul_r[1]         ),  // y*sin
    .in_valid_i  ( mul_out_valid[0] ),
    .in_ready_o  ( add_in_ready[0]  ),
    .flush_i     ( flush_i          ),
    .result_o    ( add_r[0]         ),
    .out_valid_o ( add_out_valid[0] ),
    .out_ready_i ( 1'b1             ),
    .busy_o      ( add_busy[0]      )
  );

  rope_bf16_add #(
    .NumPipeRegs ( NumPipeRegs ),
    .Sub         ( 1'b0        )
  ) i_add_y (
    .clk_i,
    .rst_ni,
    .a_i         ( mul_r[2]         ),  // x*sin
    .b_i         ( mul_r[3]         ),  // y*cos
    .in_valid_i  ( mul_out_valid[2] ),
    .in_ready_o  ( add_in_ready[1]  ),
    .flush_i     ( flush_i          ),
    .result_o    ( add_r[1]         ),
    .out_valid_o ( add_out_valid[1] ),
    .out_ready_i ( 1'b1             ),
    .busy_o      ( add_busy[1]      )
  );

  assign x_o         = add_r[0];
  assign y_o         = add_r[1];
  assign out_valid_o = add_out_valid[0];

  // ---------------------------------------------------------------------------
  // in_ready: always 1, and why that is correct rather than lazy
  // ---------------------------------------------------------------------------
  //
  // FPnew does NOT expose an unconditional ready. fpnew_top drives
  //
  //     assign in_ready_o = in_valid_i & opgrp_in_ready[get_opgroup(op_i)];
  //
  // i.e. its ready is gated by its own valid. So `&mul_in_ready` reads as 0 whenever we
  // are not already presenting a valid input, and using it to qualify the input register
  // forms a combinational cycle: valid=0 -> ready=0 -> valid stays 0. That deadlocks the
  // datapath permanently at reset.
  //
  // The unconditional truth is that these units never stall in this configuration:
  // `out_ready_i` is tied to 1 on every instance, so every FPnew input-pipeline stage has
  // `inp_pipe_ready[i] = 1`, and each unit accepts an operand on every cycle it is
  // offered one. Hence the datapath can always accept, and in_ready_o is constant 1.
  //
  // That is an assumption about vendored code, so it is CHECKED every cycle by the
  // lockstep assertions below (which observe mul_in_ready while in_valid is asserted,
  // where the gating makes it a faithful view of the real internal readiness).
  assign in_ready_o = 1'b1;
  assign busy_o     = (|mul_busy) | (|add_busy);

  // Latency: the optional input register, plus the multiplier stage, plus the adder
  // stage. Reported rather than hard-coded so it tracks NumPipeRegs.
  assign latency_o = 8'(RegLut) + 8'(NumPipeRegs) + 8'(NumPipeRegs);

  // ---------------------------------------------------------------------------
  // Structural assertions
  // ---------------------------------------------------------------------------
  //
  // The datapath assumes the four multipliers (and the two adders) march in lockstep.
  // If a future CVFPU bump broke that, results would be silently paired with the wrong
  // operands, so check it every cycle rather than trusting the configuration.
`ifndef SYNTHESIS
  always @(posedge clk_i) begin
    if (rst_ni) begin
      assert (mul_out_valid == 4'b0000 || mul_out_valid == 4'b1111)
        else $fatal(1, "rope_datapath: multipliers out of lockstep (out_valid=%b)", mul_out_valid);
      assert (mul_in_ready == 4'b0000 || mul_in_ready == 4'b1111)
        else $fatal(1, "rope_datapath: multipliers out of lockstep (in_ready=%b)", mul_in_ready);
      // The in_ready_o = 1 justification: whenever a valid operand set is presented, all
      // four multipliers must actually accept it. If a future CVFPU bump introduced a
      // stall, this fires instead of silently dropping pairs.
      if (valid_s0) begin
        assert (mul_in_ready == 4'b1111)
          else $fatal(1, "rope_datapath: multiplier refused a valid input (in_ready=%b) -- the no-stall assumption behind in_ready_o=1 is broken", mul_in_ready);
      end
      assert (add_out_valid == 2'b00 || add_out_valid == 2'b11)
        else $fatal(1, "rope_datapath: adders out of lockstep (out_valid=%b)", add_out_valid);
      // The adder stage must never refuse a multiplier result: the datapath has no
      // internal backpressure, so a stall here would silently drop a pair.
      if (mul_out_valid[0]) begin
        assert (add_in_ready == 2'b11)
          else $fatal(1, "rope_datapath: adder stalled with a product pending -- data lost");
      end
    end
  end
`endif

endmodule : rope_datapath
