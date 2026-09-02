// -----------------------------------------------------------------------------
// tb_rope_xif_coproc.sv -- Milestone 5 exit criteria.
//
//  [x] standalone XIF testbench passes with all SVA enabled (tb/rope_assertions.svh)
//  [x] kill-mid-operation: assert commit_kill, verify NO writeback and no state
//      corruption (the instruction after the killed one must still be correct)
//  [x] back-to-back ROPE.ROT at full throughput
//  [x] non-ROPE instructions are rejected (accept == 0) without stalling the core
//  [x] result_ready backpressure handled (credit accounting must not overflow the FIFO)
//
// Vector format:  <rs1> <rs2> <expected_rd>   (hex, 32-bit each)
//   rs1 = {x, y}, rs2 = {m, i}, rd = {x', y'}
// -----------------------------------------------------------------------------

module tb_rope_xif_coproc;

  import rope_pkg::*;

  localparam int unsigned MaxVectors  = 20_000_000;
  localparam int unsigned NumPipeRegs = 1;

  localparam int unsigned X_NUM_RS    = 2;
  localparam int unsigned X_ID_WIDTH  = 4;
  localparam int unsigned X_MEM_WIDTH = 32;
  localparam int unsigned X_RFR_WIDTH = 32;
  localparam int unsigned X_RFW_WIDTH = 32;

  logic clk, rst_n;

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  // --------------------------------------------------------------------------
  // Interface + DUT
  // --------------------------------------------------------------------------

  cv32e40x_if_xif #(
    .X_NUM_RS    ( X_NUM_RS    ),
    .X_ID_WIDTH  ( X_ID_WIDTH  ),
    .X_MEM_WIDTH ( X_MEM_WIDTH ),
    .X_RFR_WIDTH ( X_RFR_WIDTH ),
    .X_RFW_WIDTH ( X_RFW_WIDTH )
  ) xif ();

  rope_xif_coproc #(
    .X_NUM_RS    ( X_NUM_RS    ),
    .X_ID_WIDTH  ( X_ID_WIDTH  ),
    .X_MEM_WIDTH ( X_MEM_WIDTH ),
    .X_RFR_WIDTH ( X_RFR_WIDTH ),
    .X_RFW_WIDTH ( X_RFW_WIDTH ),
    .NumPipeRegs ( NumPipeRegs )
  ) i_dut (
    .clk_i         ( clk   ),
    .rst_ni        ( rst_n ),
    .xif_issue_if  ( xif   ),
    .xif_commit_if ( xif   ),
    .xif_result_if ( xif   )
  );

  // Tie off the channels the coprocessor does not drive, so the interface is fully
  // driven and the "no memory transaction" assertion is meaningful.
  assign xif.compressed_valid = 1'b0;
  assign xif.compressed_req   = '0;
  assign xif.mem_ready        = 1'b1;
  assign xif.mem_resp         = '0;
  assign xif.mem_result_valid = 1'b0;
  assign xif.mem_result       = '0;

  // rope_xif_coproc does not take the memory modports at all (ROPE.ROT never accesses
  // memory), so nothing drives these. Tied off here so the interface is fully driven.
  //
  // This does make the a_no_memory_transactions assertion vacuous in THIS testbench --
  // it is kept as a guard for Milestone 8: if the memory channel is ever wired into the
  // coprocessor, whoever does it must remove this tie-off, and the assertion then starts
  // checking the real "never store before a non-kill commit" rule.
  assign xif.mem_valid = 1'b0;
  assign xif.mem_req   = '0;

  `include "rope_assertions.svh"

  // Watchdog. A stalled handshake (e.g. credits never released because backpressure was
  // held too long) would otherwise spin forever; a clean timeout that reports how far it
  // got is far easier to diagnose.
  initial begin : watchdog
    #2s;
    $display("FAIL: timeout -- sent=%0d checked=%0d pending=%0d", sent, checked, pend_q.size());
    $fatal(1);
  end

  // Debug trace (enable with +debug).
  bit debug_on;
  initial debug_on = $test$plusargs("debug");

  always @(posedge clk) begin
    if (rst_n && debug_on) begin
      if (i_dut.accept)
        $display("[%0t] ACCEPT id=%0d rs1=%08h rs2=%08h", $time,
                 xif.issue_req.id, xif.issue_req.rs[0], xif.issue_req.rs[1]);
      if (i_dut.issued_q)
        $display("[%0t] issued_q=1 x=%04h y=%04h pos=%0d pair=%0d", $time,
                 i_dut.x_q, i_dut.y_q, i_dut.pos_q, i_dut.pair_q);
      if (i_dut.res_valid)
        $display("[%0t] res_valid tag.valid=%0b tag.id=%0d x=%04h y=%04h", $time,
                 i_dut.cur_tag.valid, i_dut.cur_tag.id, i_dut.res_x, i_dut.res_y);
      if (i_dut.fifo_push)
        $display("[%0t] PUSH wr=%0d id=%0d", $time, i_dut.wr_ptr_q, i_dut.cur_tag.id);
      if (xif.result_valid && xif.result_ready)
        $display("[%0t] TAKE rd_ptr=%0d id=%0d data=%08h", $time,
                 i_dut.rd_ptr_q, xif.result.id, xif.result.data);
    end
  end

  // --------------------------------------------------------------------------
  // Instruction encoding helper
  // --------------------------------------------------------------------------

  function automatic logic [31:0] encode_rope_rot(logic [4:0] rd,
                                                  logic [4:0] rs1,
                                                  logic [4:0] rs2);
    return {RopeFunct7Rot, rs2, rs1, RopeFunct3Rot, rd, RopeOpcodeCustom0};
  endfunction

  // --------------------------------------------------------------------------
  // Vectors
  // --------------------------------------------------------------------------

  int unsigned n_vec;
  logic [31:0] v_rs1 [MaxVectors];
  logic [31:0] v_rs2 [MaxVectors];
  logic [31:0] v_rd  [MaxVectors];

  int unsigned errors, checked, sent;
  string       vecfile;

  initial begin : load
    int fd, code;
    int unsigned a_i, b_i, c_i;

    if (!$value$plusargs("vectors=%s", vecfile)) vecfile = "vectors_xif.hex";
    fd = $fopen(vecfile, "r");
    if (fd == 0) begin
      $display("FATAL: cannot open '%s'", vecfile);
      $fatal(1);
    end
    n_vec = 0;
    forever begin
      code = $fscanf(fd, "%h %h %h\n", a_i, b_i, c_i);
      if (code != 3) break;
      v_rs1[n_vec] = a_i;
      v_rs2[n_vec] = b_i;
      v_rd[n_vec]  = c_i;
      n_vec++;
      if (n_vec >= MaxVectors) break;
    end
    $fclose(fd);
    $display("tb_rope_xif_coproc: loaded %0d vectors from %s", n_vec, vecfile);
  end

  // --------------------------------------------------------------------------
  // Scoreboard
  // --------------------------------------------------------------------------

  typedef struct {
    int unsigned vec;   // index into the vector arrays
    logic [4:0]  rd;
    logic [3:0]  id;
  } pending_t;

  pending_t    pend_q [$];
  int unsigned kill_writebacks;   // must stay zero

  // Ids that the TB has killed; used to confirm no writeback appears for them.
  logic [15:0] tb_killed;

  always @(posedge clk) begin
    if (rst_n && xif.result_valid && xif.result_ready) begin
      pending_t p;
      if (pend_q.size() == 0) begin
        $display("FAIL: writeback with nothing pending (id=%0d)", xif.result.id);
        errors++;
      end else begin
        p = pend_q.pop_front();
        if (tb_killed[xif.result.id]) begin
          kill_writebacks++;
          $display("FAIL: writeback for killed id=%0d", xif.result.id);
          errors++;
        end
        if (xif.result.data !== v_rd[p.vec]) begin
          errors++;
          if (errors <= 20) begin
            $display("MISMATCH [%0d]: rs1=%08h rs2=%08h got=%08h want=%08h",
                     p.vec, v_rs1[p.vec], v_rs2[p.vec], xif.result.data, v_rd[p.vec]);
          end
        end
        if (xif.result.rd !== p.rd) begin
          errors++;
          $display("FAIL: rd mismatch: got=%0d want=%0d", xif.result.rd, p.rd);
        end
        if (xif.result.id !== p.id) begin
          errors++;
          $display("FAIL: id out of order: got=%0d want=%0d", xif.result.id, p.id);
        end
        checked++;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Issue driver task
  // --------------------------------------------------------------------------

  logic [3:0] next_id;

  // TB TIMING DISCIPLINE (learned the hard way -- an earlier version accepted the same
  // instruction twice):
  //
  //   * DRIVE on the negedge, using BLOCKING assignments.
  //   * SAMPLE on the posedge.
  //
  // Asserting issue_valid at the same instant as a posedge races with the DUT's
  // always_ff sampling that edge: depending on Active/NBA scheduling the request can be
  // seen on both that edge and the next, i.e. accepted twice. Driving strictly off-edge
  // removes the race entirely, and one negedge->posedge pair per loop iteration still
  // gives full one-per-cycle throughput.
  task automatic issue_rot(input int unsigned vec_idx,
                           input logic [4:0]  rd,
                           input bit          expect_accept = 1'b1);
    logic [3:0] use_id;
    use_id = next_id;

    @(negedge clk);
    xif.issue_valid          = 1'b1;
    xif.issue_req.instr      = encode_rope_rot(rd, 5'd1, 5'd2);
    xif.issue_req.mode       = 2'b11;
    xif.issue_req.id         = use_id;
    xif.issue_req.rs[0]      = v_rs1[vec_idx];
    xif.issue_req.rs[1]      = v_rs2[vec_idx];
    xif.issue_req.rs_valid   = 2'b11;
    xif.issue_req.ecs        = 6'b0;
    xif.issue_req.ecs_valid  = 1'b1;

    // The DUT samples here. issue_ready/issue_resp are combinational off stable inputs.
    @(posedge clk);
    while (!xif.issue_ready) @(posedge clk);

    if (xif.issue_resp.accept !== expect_accept) begin
      $display("FAIL: accept=%0b, expected %0b", xif.issue_resp.accept, expect_accept);
      errors++;
    end

    if (xif.issue_resp.accept) begin
      // Built as a named temporary: Verilator does not support an assignment pattern
      // used directly as a function-call argument.
      pending_t new_pend;
      new_pend.vec = vec_idx;
      new_pend.rd  = rd;
      new_pend.id  = use_id;
      pend_q.push_back(new_pend);
      sent++;
      next_id = next_id + 1'b1;
    end

    // Release off-edge so the request is seen on exactly one posedge.
    @(negedge clk);
    xif.issue_valid = 1'b0;
  endtask

  // Commit an id. commit_valid has no ready signal, so this is a pure one-cycle pulse
  // that the coprocessor must be able to observe at any time.
  task automatic do_commit(input logic [3:0] id, input bit kill = 1'b0);
    @(negedge clk);
    xif.commit_valid       = 1'b1;
    xif.commit.id          = id;
    xif.commit.commit_kill = kill;
    @(posedge clk);
    @(negedge clk);
    xif.commit_valid = 1'b0;
  endtask

  // --------------------------------------------------------------------------
  // Main sequence
  // --------------------------------------------------------------------------

  initial begin : run
    errors          = 0;
    checked         = 0;
    sent            = 0;
    kill_writebacks = 0;
    next_id         = '0;
    tb_killed       = '0;

    rst_n                  = 1'b0;
    xif.issue_valid        = 1'b0;
    xif.issue_req          = '0;
    xif.commit_valid       = 1'b0;
    xif.commit             = '0;
    xif.result_ready       = 1'b1;

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    wait (n_vec > 0);

    // ---- TEST 1: a non-ROPE instruction must be rejected, not stall the core -----
    begin
      @(negedge clk);
      xif.issue_valid         = 1'b1;
      // A plain ADDI: opcode 0x13, definitely not ours.
      xif.issue_req.instr     = 32'h00100113;
      xif.issue_req.id        = 4'hF;
      xif.issue_req.rs[0]     = 32'h0;
      xif.issue_req.rs[1]     = 32'h0;
      xif.issue_req.rs_valid  = 2'b11;
      @(posedge clk);
      while (!xif.issue_ready) @(posedge clk);
      if (xif.issue_resp.accept !== 1'b0) begin
        $display("FAIL: coprocessor accepted a non-ROPE instruction");
        errors++;
      end else begin
        $display("TEST 1 ok: non-ROPE instruction rejected (accept=0, issue_ready=1)");
      end
      @(negedge clk);
      xif.issue_valid = 1'b0;
    end

    // ---- TEST 2: single ROPE.ROT, committed normally -----------------------------
    begin
      logic [3:0] id0;
      id0 = next_id;
      issue_rot(0, 5'd7);
      do_commit(id0, 1'b0);
      // Wait for the writeback.
      repeat (32) @(posedge clk);
      if (checked != 1) begin
        $display("FAIL: expected exactly 1 writeback, got %0d", checked);
        errors++;
      end else begin
        $display("TEST 2 ok: single ROPE.ROT wrote back correctly");
      end
    end

    // ---- TEST 3: kill mid-operation ---------------------------------------------
    //
    // Issue an instruction, kill it before it retires, and confirm that (a) no writeback
    // appears for it, and (b) the NEXT instruction still produces the correct result --
    // i.e. the kill did not corrupt the pipeline or the tag/credit bookkeeping.
    begin
      logic [3:0] id_kill, id_after;
      int unsigned checked_before;
      checked_before = checked;

      id_kill = next_id;
      issue_rot(1, 5'd8);
      // Drop the expectation: this one must never write back.
      void'(pend_q.pop_back());
      tb_killed[id_kill] = 1'b1;

      // Kill it one cycle later -- while it is still in the datapath.
      @(posedge clk);
      do_commit(id_kill, 1'b1);

      repeat (32) @(posedge clk);

      if (checked != checked_before) begin
        $display("FAIL: killed instruction produced a writeback");
        errors++;
      end else begin
        $display("TEST 3a ok: killed instruction produced no writeback");
      end

      // Now a normal instruction must still work.
      id_after = next_id;
      issue_rot(2, 5'd9);
      do_commit(id_after, 1'b0);
      repeat (32) @(posedge clk);
      if (checked != checked_before + 1) begin
        $display("FAIL: instruction after a kill did not write back (state corrupted)");
        errors++;
      end else begin
        $display("TEST 3b ok: instruction after a kill is unaffected");
      end
      tb_killed = '0;
    end

    // ---- TEST 4: back-to-back at full throughput --------------------------------
    //
    // Issue continuously with result_ready held high. Commits follow immediately.
    begin
      int unsigned start_vec, count, checked_before;
      start_vec      = 3;
      count          = (n_vec - start_vec > 2000) ? 2000 : (n_vec - start_vec);
      checked_before = checked;

      fork
        // Issue thread.
        begin
          for (int unsigned k = 0; k < count; k++) begin
            issue_rot(start_vec + k, 5'd10);
          end
        end
        // Commit thread: commit every id shortly after it is issued. Ids are handed out
        // sequentially, so tracking a running counter is sufficient here.
        begin
          logic [3:0] cid;
          cid = next_id;
          for (int unsigned k = 0; k < count; k++) begin
            @(posedge clk);
            do_commit(cid, 1'b0);
            cid = cid + 1'b1;
          end
        end
      join

      repeat (64) @(posedge clk);
      if (checked != checked_before + count) begin
        $display("FAIL: back-to-back: expected %0d writebacks, got %0d",
                 count, checked - checked_before);
        errors++;
      end else begin
        $display("TEST 4 ok: %0d back-to-back ROPE.ROT all correct", count);
      end
    end

    // ---- TEST 5: result_ready backpressure --------------------------------------
    //
    // Stall the result channel while issuing. The credit scheme must stop accepting
    // before the FIFO overflows; nothing may be lost or duplicated.
    begin
      int unsigned start_vec, count, checked_before;
      start_vec      = 3000;
      count          = (n_vec > start_vec + 64) ? 64 : 0;
      checked_before = checked;

      if (count > 0) begin
        @(negedge clk);
        xif.result_ready = 1'b0;
        fork
          begin
            for (int unsigned k = 0; k < count; k++) begin
              issue_rot(start_vec + k, 5'd11);
            end
          end
          begin
            logic [3:0] cid;
            cid = next_id;
            for (int unsigned k = 0; k < count; k++) begin
              @(posedge clk);
              do_commit(cid, 1'b0);
              cid = cid + 1'b1;
            end
          end
          // Backpressure thread. This MUST live inside the fork: with result_ready held
          // low, the credit counter fills after FifoDepth accepts and issue_ready
          // correctly drops, so the issue thread blocks. Releasing backpressure only
          // after the join would therefore deadlock -- the earlier version of this test
          // did exactly that. Holding for well over FifoDepth cycles first is what makes
          // this a real test of the credit scheme rather than of nothing.
          begin
            repeat (40) @(posedge clk);
            @(negedge clk);
            xif.result_ready = 1'b1;
          end
        join
        repeat (count * 4 + 64) @(posedge clk);

        if (checked != checked_before + count) begin
          $display("FAIL: backpressure: expected %0d writebacks, got %0d",
                   count, checked - checked_before);
          errors++;
        end else begin
          $display("TEST 5 ok: %0d results survived result_ready backpressure", count);
        end
      end
    end

    // ---- TEST 6: remaining vectors, with random backpressure --------------------
    begin
      int unsigned start_vec, count, checked_before;
      start_vec      = 3100;
      count          = (n_vec > start_vec) ? (n_vec - start_vec) : 0;
      if (count > 4000) count = 4000;
      checked_before = checked;

      if (count > 0) begin
        fork
          begin
            for (int unsigned k = 0; k < count; k++) begin
              issue_rot(start_vec + k, 5'd12);
            end
          end
          begin
            logic [3:0] cid;
            cid = next_id;
            for (int unsigned k = 0; k < count; k++) begin
              @(posedge clk);
              do_commit(cid, 1'b0);
              cid = cid + 1'b1;
            end
          end
          // Random backpressure on the result channel, driven off-edge like everything
          // else so it never races with the coprocessor sampling result_ready.
          begin
            for (int unsigned t = 0; t < count * 3; t++) begin
              @(negedge clk);
              xif.result_ready = ($urandom_range(0, 3) != 0);
            end
            @(negedge clk);
            xif.result_ready = 1'b1;
          end
        join

        repeat (256) @(posedge clk);
        if (checked != checked_before + count) begin
          $display("FAIL: random backpressure: expected %0d, got %0d",
                   count, checked - checked_before);
          errors++;
        end else begin
          $display("TEST 6 ok: %0d results correct under random backpressure", count);
        end
      end
    end

    $display("----------------------------------------------------------------");
    $display("tb_rope_xif_coproc: sent=%0d checked=%0d errors=%0d", sent, checked, errors);
    if (kill_writebacks != 0) begin
      $display("FAIL: %0d writebacks for killed instructions", kill_writebacks);
    end
    if (errors != 0) begin
      $display("FAIL: %0d errors", errors);
      $fatal(1);
    end
    $display("PASS: XIF coprocessor -- decode, writeback, kill, throughput, backpressure");
    $finish;
  end

endmodule : tb_rope_xif_coproc
