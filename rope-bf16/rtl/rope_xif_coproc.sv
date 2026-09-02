// -----------------------------------------------------------------------------
// rope_xif_coproc.sv -- CORE-V-XIF wrapper for the BF16 RoPE unit (Milestone 5).
//
//   ROPE.ROT rd, rs1, rs2        # custom-0, opcode 0x0B
//     rs1 = {x_bf16, y_bf16}     native BF16 pair in,  NO conversion
//     rs2 = {m[15:0], i[15:0]}   position and pair index (integers)
//     rd  = {x'_bf16, y'_bf16}   native BF16 pair out
//
// Single source pair, single destination: no dual writeback needed, X_RFW_WIDTH stays 32.
//
// Channels used: Issue, Commit, Result. Compressed is not needed. The memory channels
// belong to the streaming variant (Milestone 8) and are not driven here.
//
// The three protocol rules that steps.md Section 9.3 warns about, and how each is met:
//
//  1. REGISTER THE OPERAND INPUTS. The coprocessor gets only a small slice of the timing
//     budget on paths through issue_req.rs*, because CV32E40X sources those operands
//     straight from its register-file bypass network. So rs1/rs2 are flopped on arrival
//     and NO combinational logic touches them in the issue cycle. Specifically: the
//     phase multiply, the LUT read and the datapath all consume the *registered* copies.
//     Only the instruction word itself is decoded combinationally, because issue_resp
//     must answer in the issue cycle -- and the instruction word comes from the
//     instruction register, not the bypass network.
//
//  2. commit_valid HAS NO READY. It may arrive coincident with or after issue, and must
//     be observable at any time. State is tracked per `id` and a kill retires the
//     in-flight operation without writeback.
//
//  3. mem_result_valid HAS NO READY. Not applicable until Milestone 8 (no memory
//     transactions are issued), but the same "always able to accept" discipline is why
//     the result path is credit-controlled below.
//
// Ordering rule: a transaction with an earlier issued id must never depend on a later
// one. Results are returned strictly in issue order from a FIFO, so result_valid for an
// older id is never delayed waiting on a newer instruction's commit.
// -----------------------------------------------------------------------------

module rope_xif_coproc
  import rope_pkg::*;
#(
  // These five MUST match the core and the cv32e40x_if_xif instance, or elaboration
  // breaks in confusing ways (steps.md Section 9.5).
  parameter int unsigned X_NUM_RS      = 2,
  parameter int unsigned X_ID_WIDTH    = 4,
  parameter int unsigned X_MEM_WIDTH   = 32,
  parameter int unsigned X_RFR_WIDTH   = 32,
  parameter int unsigned X_RFW_WIDTH   = 32,

  parameter int unsigned NumPipeRegs   = 1,
  // Reassignable so it can move if it ever collides with another coprocessor.
  parameter logic [6:0]  RopeOpcode    = rope_pkg::RopeOpcodeCustom0
) (
  input  logic clk_i,
  input  logic rst_ni,

  cv32e40x_if_xif.coproc_issue  xif_issue_if,
  cv32e40x_if_xif.coproc_commit xif_commit_if,
  cv32e40x_if_xif.coproc_result xif_result_if
);

  localparam int unsigned NumIds = 1 << X_ID_WIDTH;

  // Depth of the result FIFO. It must cover the datapath latency so that every accepted
  // instruction has somewhere to land -- the datapath cannot be back-pressured.
  localparam int unsigned FifoDepth = 8;
  localparam int unsigned FifoPtrW  = $clog2(FifoDepth);

  // ---------------------------------------------------------------------------
  // Decode (combinational -- instruction word only, never the operands)
  // ---------------------------------------------------------------------------

  logic dec_is_rot;
  assign dec_is_rot = (rope_pkg::instr_opcode(xif_issue_if.issue_req.instr) == RopeOpcode)
                   && (rope_pkg::instr_funct3(xif_issue_if.issue_req.instr) == rope_pkg::RopeFunct3Rot)
                   && (rope_pkg::instr_funct7(xif_issue_if.issue_req.instr) == rope_pkg::RopeFunct7Rot);

  // Both source operands must be valid for us to accept.
  logic operands_ok;
  assign operands_ok = xif_issue_if.issue_req.rs_valid[0]
                     & xif_issue_if.issue_req.rs_valid[1];

  // Credit: only accept if the result FIFO has room for this instruction.
  logic [FifoPtrW:0] credit_q;
  logic              have_credit;
  assign have_credit = (credit_q < FifoDepth[FifoPtrW:0]);

  logic accept;
  assign accept = xif_issue_if.issue_valid & dec_is_rot & operands_ok & have_credit;

  // issue_ready deliberately does NOT depend on issue_valid: we are ready to answer any
  // issue request, and only withhold readiness for a ROPE.ROT we genuinely cannot take
  // yet (missing operands or no result-FIFO credit). A non-ROPE instruction is answered
  // immediately with accept=0 so the core is never stalled by us.
  assign xif_issue_if.issue_ready         = !dec_is_rot | (operands_ok & have_credit);
  assign xif_issue_if.issue_resp.accept    = accept;
  assign xif_issue_if.issue_resp.writeback = accept;      // ROPE.ROT always writes rd
  assign xif_issue_if.issue_resp.dualwrite = 1'b0;        // 32-bit result: no dual write
  assign xif_issue_if.issue_resp.dualread  = 3'b000;
  assign xif_issue_if.issue_resp.loadstore = 1'b0;        // no memory access (M8 only)
  assign xif_issue_if.issue_resp.ecswrite  = 1'b0;
  assign xif_issue_if.issue_resp.exc       = 1'b0;        // cannot raise an exception

  // ---------------------------------------------------------------------------
  // Rule 1: register the operands. Nothing combinational may consume rs* directly.
  // ---------------------------------------------------------------------------

  logic [15:0]            x_q, y_q;
  logic [rope_pkg::PosBits-1:0]     pos_q;
  logic [rope_pkg::PairIdxBits-1:0] pair_q;
  logic [X_ID_WIDTH-1:0]  id_q;
  logic [4:0]             rd_q;
  logic                   issued_q;

  logic [31:0] rs1_raw, rs2_raw;
  assign rs1_raw = xif_issue_if.issue_req.rs[0][31:0];
  assign rs2_raw = xif_issue_if.issue_req.rs[1][31:0];

  // Named intermediates: SystemVerilog does not allow bit-selecting a function call
  // result directly, and these also document the truncation of m and i to their
  // architectural widths.
  logic [15:0] rs2_pos_full, rs2_idx_full;
  assign rs2_pos_full = rope_pkg::rs2_pos(rs2_raw);
  assign rs2_idx_full = rope_pkg::rs2_pair_idx(rs2_raw);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      x_q      <= '0;
      y_q      <= '0;
      pos_q    <= '0;
      pair_q   <= '0;
      id_q     <= '0;
      rd_q     <= '0;
      issued_q <= 1'b0;
    end else begin
      issued_q <= accept;
      if (accept) begin
        // Straight slices of the operand words: no arithmetic in the issue cycle.
        x_q    <= rope_pkg::pair_x(rs1_raw);
        y_q    <= rope_pkg::pair_y(rs1_raw);
        pos_q  <= rs2_pos_full[rope_pkg::PosBits-1:0];
        pair_q <= rs2_idx_full[rope_pkg::PairIdxBits-1:0];
        id_q   <= xif_issue_if.issue_req.id;
        rd_q   <= rope_pkg::instr_rd(xif_issue_if.issue_req.instr);
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Phase generation and LUT lookup -- all from REGISTERED operands
  // ---------------------------------------------------------------------------

  logic [rope_pkg::PhaseBits-1:0] phase;

  rope_phase_gen i_phase_gen (
    .clk_i,
    .rst_ni,
    .pos_i       ( pos_q  ),
    .pair_idx_i  ( pair_q ),
    .phase_o     ( phase  ),
    // Sequential mode is unused by ROPE.ROT (it belongs to the streaming variant).
    .seq_clear_i ( 1'b0   ),
    .seq_step_i  ( 1'b0   ),
    .seq_idx_i   ( '0     ),
    .seq_phase_o (        )
  );

  logic [15:0] sin_val, cos_val;

  rope_sin_lut i_lut (
    .phase_i ( phase   ),
    .sin_o   ( sin_val ),
    .cos_o   ( cos_val )
  );

  // ---------------------------------------------------------------------------
  // Datapath
  // ---------------------------------------------------------------------------

  logic [15:0] res_x, res_y;
  logic        res_valid;
  logic        dp_in_ready, dp_busy;
  logic [7:0]  dp_latency;

  rope_datapath #(
    .NumPipeRegs ( NumPipeRegs ),
    .RegLut      ( 1'b1        )
  ) i_datapath (
    .clk_i,
    .rst_ni,
    .flush_i     ( 1'b0        ),
    .x_i         ( x_q         ),
    .y_i         ( y_q         ),
    .sin_i       ( sin_val     ),
    .cos_i       ( cos_val     ),
    .in_valid_i  ( issued_q    ),
    .in_ready_o  ( dp_in_ready ),
    .x_o         ( res_x       ),
    .y_o         ( res_y       ),
    .out_valid_o ( res_valid   ),
    .busy_o      ( dp_busy     ),
    .latency_o   ( dp_latency  )
  );

  // ---------------------------------------------------------------------------
  // In-flight tracking, so a kill can be honoured (rule 2)
  // ---------------------------------------------------------------------------
  //
  // A tag rides alongside the datapath in a shift register whose depth matches the
  // datapath latency. `killed` is looked up per id, so a commit_kill arriving at any
  // time after issue suppresses the writeback for exactly that instruction.

  localparam int unsigned TagDepth = 8;

  typedef struct packed {
    logic                  valid;
    logic [X_ID_WIDTH-1:0] id;
    logic [4:0]            rd;
  } tag_t;

  tag_t tag_pipe_q [TagDepth];

  // Per-id kill flags. Set by commit_kill, cleared when the id is retired or reissued.
  logic [NumIds-1:0] killed_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      killed_q <= '0;
    end else begin
      // commit_valid has no ready: sample it unconditionally, every cycle.
      if (xif_commit_if.commit_valid && xif_commit_if.commit.commit_kill) begin
        killed_q[xif_commit_if.commit.id] <= 1'b1;
      end
      // A newly accepted instruction reusing an id starts un-killed. Ordered after the
      // kill capture so a same-cycle kill of a different id is not lost.
      if (accept) begin
        killed_q[xif_issue_if.issue_req.id] <= 1'b0;
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int unsigned k = 0; k < TagDepth; k++) tag_pipe_q[k] <= '0;
    end else begin
      tag_pipe_q[0].valid <= issued_q;
      tag_pipe_q[0].id    <= id_q;
      tag_pipe_q[0].rd    <= rd_q;
      for (int unsigned k = 1; k < TagDepth; k++) begin
        tag_pipe_q[k] <= tag_pipe_q[k-1];
      end
    end
  end

  // The tag that belongs to the result emerging now. dp_latency counts the datapath's
  // own stages; the tag pipe is indexed to match, so this cannot drift if NumPipeRegs
  // changes.
  logic [7:0] tag_sel;
  assign tag_sel = dp_latency - 8'd1;

  tag_t cur_tag;
  always_comb begin
    cur_tag = '0;
    for (int unsigned k = 0; k < TagDepth; k++) begin
      if (k == tag_sel) cur_tag = tag_pipe_q[k];
    end
  end

  // ---------------------------------------------------------------------------
  // Result FIFO -- returns results strictly in issue order
  // ---------------------------------------------------------------------------

  typedef struct packed {
    logic [X_ID_WIDTH-1:0] id;
    logic [4:0]            rd;
    logic [31:0]           data;
  } fifo_entry_t;

  fifo_entry_t       fifo_q [FifoDepth];
  logic [FifoPtrW:0] wr_ptr_q, rd_ptr_q;
  logic              fifo_empty, fifo_push, fifo_pop;

  assign fifo_empty = (wr_ptr_q == rd_ptr_q);

  // Push a completed, non-killed result. A killed instruction consumes its credit but
  // produces no writeback.
  logic result_is_killed;
  assign result_is_killed = killed_q[cur_tag.id];

  assign fifo_push = res_valid & cur_tag.valid & ~result_is_killed;
  assign fifo_pop  = xif_result_if.result_valid & xif_result_if.result_ready;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_ptr_q <= '0;
      rd_ptr_q <= '0;
      for (int unsigned k = 0; k < FifoDepth; k++) fifo_q[k] <= '0;
    end else begin
      if (fifo_push) begin
        fifo_q[wr_ptr_q[FifoPtrW-1:0]].id   <= cur_tag.id;
        fifo_q[wr_ptr_q[FifoPtrW-1:0]].rd   <= cur_tag.rd;
        fifo_q[wr_ptr_q[FifoPtrW-1:0]].data <= rope_pkg::pack_pair(res_x, res_y);
        wr_ptr_q <= wr_ptr_q + 1'b1;
      end
      if (fifo_pop) begin
        rd_ptr_q <= rd_ptr_q + 1'b1;
      end
    end
  end

  // Credit accounting: one credit per accepted instruction, released when the
  // instruction leaves (writeback accepted, or retired as killed).
  logic retire_killed;
  assign retire_killed = res_valid & cur_tag.valid & result_is_killed;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      credit_q <= '0;
    end else begin
      credit_q <= credit_q + {{FifoPtrW{1'b0}}, accept}
                           - {{FifoPtrW{1'b0}}, (fifo_pop | retire_killed)};
    end
  end

  // ---------------------------------------------------------------------------
  // Result channel
  // ---------------------------------------------------------------------------

  assign xif_result_if.result_valid    = ~fifo_empty;
  assign xif_result_if.result.id       = fifo_q[rd_ptr_q[FifoPtrW-1:0]].id;
  assign xif_result_if.result.data     = fifo_q[rd_ptr_q[FifoPtrW-1:0]].data;
  assign xif_result_if.result.rd       = fifo_q[rd_ptr_q[FifoPtrW-1:0]].rd;
  assign xif_result_if.result.we       = 1'b1;
  assign xif_result_if.result.ecsdata  = 6'b0;
  assign xif_result_if.result.ecswe    = 3'b0;
  assign xif_result_if.result.exc      = 1'b0;
  assign xif_result_if.result.exccode  = 6'b0;
  assign xif_result_if.result.err      = 1'b0;
  assign xif_result_if.result.dbg      = 1'b0;

  // ---------------------------------------------------------------------------
  // Protocol assertions (steps.md Section 9.4)
  // ---------------------------------------------------------------------------
`ifndef SYNTHESIS
  // The datapath must never be back-pressured: the credit scheme exists to guarantee it.
  always @(posedge clk_i) begin
    if (rst_ni && issued_q) begin
      assert (dp_in_ready)
        else $fatal(1, "rope_xif_coproc: datapath stalled -- credit accounting is wrong");
    end
  end

  // The FIFO must never overflow, for the same reason.
  always @(posedge clk_i) begin
    if (rst_ni && fifo_push) begin
      assert ((wr_ptr_q - rd_ptr_q) < FifoDepth[FifoPtrW:0])
        else $fatal(1, "rope_xif_coproc: result FIFO overflow");
    end
  end

  // Every emerging result must carry a valid tag.
  always @(posedge clk_i) begin
    if (rst_ni && res_valid) begin
      assert (cur_tag.valid)
        else $fatal(1, "rope_xif_coproc: result with no matching tag (tag pipe misaligned)");
    end
  end

  // The tag pipeline must be deep enough for the configured latency.
  initial begin
    assert (TagDepth >= 1 + 2 * NumPipeRegs + 1)
      else $fatal(1, "rope_xif_coproc: TagDepth too small for NumPipeRegs=%0d", NumPipeRegs);
  end
`endif

endmodule : rope_xif_coproc
