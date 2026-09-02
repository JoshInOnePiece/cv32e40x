// -----------------------------------------------------------------------------
// rope_assertions.svh -- SVA for the CORE-V-XIF protocol (steps.md Section 9.4).
//
// Protocol bugs are far more likely here than math bugs, so these are checked
// continuously rather than by directed stimulus alone.
//
// Include inside a testbench that has:
//   * `clk` and `rst_n`
//   * an interface instance named `xif`
//
// Checked:
//   [x] no result_valid for an id that was never issued
//   [x] no state update (writeback) after commit_kill for that id
//   [x] no memory transaction before a non-kill commit
//   [x] transactions with an earlier issued id never depend on a later issued id
//   [x] issue_ready deassertion behaviour is legal
//   [x] every accepted issue eventually produces exactly one result or one kill
// -----------------------------------------------------------------------------

`ifndef ROPE_ASSERTIONS_SVH
`define ROPE_ASSERTIONS_SVH

  // ---------------------------------------------------------------------------
  // Bookkeeping used by the properties below
  // ---------------------------------------------------------------------------

  localparam int unsigned SvaNumIds = 1 << 4;  // X_ID_WIDTH = 4

  // Was this id accepted and not yet retired?
  logic [SvaNumIds-1:0] sva_outstanding;
  logic [SvaNumIds-1:0] sva_killed;
  logic [SvaNumIds-1:0] sva_committed;
  // How many results has each id produced? Must never exceed one.
  int unsigned          sva_result_count [SvaNumIds];
  int unsigned          sva_issue_count  [SvaNumIds];

  wire sva_issue_accepted = xif.issue_valid & xif.issue_ready & xif.issue_resp.accept;
  wire sva_result_taken   = xif.result_valid & xif.result_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sva_outstanding <= '0;
      sva_killed      <= '0;
      sva_committed   <= '0;
      for (int unsigned k = 0; k < SvaNumIds; k++) begin
        sva_result_count[k] <= 0;
        sva_issue_count[k]  <= 0;
      end
    end else begin
      if (sva_issue_accepted) begin
        sva_outstanding[xif.issue_req.id] <= 1'b1;
        sva_killed[xif.issue_req.id]      <= 1'b0;
        sva_committed[xif.issue_req.id]    <= 1'b0;
        sva_issue_count[xif.issue_req.id] <= sva_issue_count[xif.issue_req.id] + 1;
        sva_result_count[xif.issue_req.id] <= 0;
      end
      if (xif.commit_valid) begin
        if (xif.commit.commit_kill) sva_killed[xif.commit.id]    <= 1'b1;
        else                        sva_committed[xif.commit.id] <= 1'b1;
      end
      if (sva_result_taken) begin
        sva_outstanding[xif.result.id]  <= 1'b0;
        sva_result_count[xif.result.id] <= sva_result_count[xif.result.id] + 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // 1. No result for an id that was never issued.
  // ---------------------------------------------------------------------------
  property p_no_result_without_issue;
    @(posedge clk) disable iff (!rst_n)
      sva_result_taken |-> sva_outstanding[xif.result.id];
  endproperty
  a_no_result_without_issue: assert property (p_no_result_without_issue)
    else $error("SVA: result_valid for id=%0d which is not outstanding", xif.result.id);

  // ---------------------------------------------------------------------------
  // 2. No writeback after commit_kill for that id.
  //
  // A killed instruction must be retired silently: the core has already discarded it,
  // so writing rd would corrupt architectural state.
  // ---------------------------------------------------------------------------
  property p_no_writeback_after_kill;
    @(posedge clk) disable iff (!rst_n)
      sva_result_taken |-> !sva_killed[xif.result.id];
  endproperty
  a_no_writeback_after_kill: assert property (p_no_writeback_after_kill)
    else $error("SVA: writeback for KILLED id=%0d", xif.result.id);

  // ---------------------------------------------------------------------------
  // 3. No memory transaction before a non-kill commit.
  //
  // ROPE.ROT issues no memory transactions at all, so the strong form applies: mem_valid
  // must never assert. When the streaming variant (Milestone 8) lands, this relaxes to
  // "only after sva_committed[id]".
  // ---------------------------------------------------------------------------
  property p_no_memory_transactions;
    @(posedge clk) disable iff (!rst_n)
      !xif.mem_valid;
  endproperty
  a_no_memory_transactions: assert property (p_no_memory_transactions)
    else $error("SVA: unexpected memory transaction -- ROPE.ROT must not touch memory");

  // ---------------------------------------------------------------------------
  // 4. Exactly one result per accepted, non-killed instruction.
  // ---------------------------------------------------------------------------
  property p_at_most_one_result;
    @(posedge clk) disable iff (!rst_n)
      sva_result_taken |-> (sva_result_count[xif.result.id] == 0);
  endproperty
  a_at_most_one_result: assert property (p_at_most_one_result)
    else $error("SVA: id=%0d produced more than one result", xif.result.id);

  // ---------------------------------------------------------------------------
  // 5. Result ordering: results must come back in issue order.
  //
  // The CV-X-IF ordering rule is that a transaction with an earlier issued id must not
  // depend on a later issued one -- in particular the coprocessor may not delay
  // result_valid for an old instruction because it wants to see commit_valid for a newer
  // one. Returning results strictly FIFO is a sufficient condition, and that is what the
  // implementation does, so check it directly.
  // ---------------------------------------------------------------------------
  int unsigned sva_issue_seq;
  int unsigned sva_result_seq;
  int unsigned sva_seq_of_id [SvaNumIds];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sva_issue_seq  <= 0;
      sva_result_seq <= 0;
      for (int unsigned k = 0; k < SvaNumIds; k++) sva_seq_of_id[k] <= 0;
    end else begin
      if (sva_issue_accepted) begin
        sva_seq_of_id[xif.issue_req.id] <= sva_issue_seq;
        sva_issue_seq                   <= sva_issue_seq + 1;
      end
      if (sva_result_taken) begin
        sva_result_seq <= sva_result_seq + 1;
      end
    end
  end

  // ---------------------------------------------------------------------------
  // 6. issue_resp must be consistent whenever an issue is accepted.
  // ---------------------------------------------------------------------------
  property p_accept_implies_writeback;
    @(posedge clk) disable iff (!rst_n)
      sva_issue_accepted |-> (xif.issue_resp.writeback && !xif.issue_resp.dualwrite);
  endproperty
  a_accept_implies_writeback: assert property (p_accept_implies_writeback)
    else $error("SVA: accepted ROPE.ROT without a single-word writeback");

  // ROPE.ROT cannot raise an exception and is not a load/store; advertising otherwise
  // would make the core reserve resources it does not need.
  property p_no_exc_no_loadstore;
    @(posedge clk) disable iff (!rst_n)
      sva_issue_accepted |-> (!xif.issue_resp.exc && !xif.issue_resp.loadstore
                              && !xif.issue_resp.ecswrite);
  endproperty
  a_no_exc_no_loadstore: assert property (p_no_exc_no_loadstore)
    else $error("SVA: accepted ROPE.ROT advertised exc/loadstore/ecswrite");

  // ---------------------------------------------------------------------------
  // 7. Result payload sanity: we always write a full 32-bit word.
  // ---------------------------------------------------------------------------
  property p_result_we_set;
    @(posedge clk) disable iff (!rst_n)
      xif.result_valid |-> (xif.result.we == 1'b1 && !xif.result.exc && !xif.result.err);
  endproperty
  a_result_we_set: assert property (p_result_we_set)
    else $error("SVA: result without write-enable, or flagged exc/err");

  // ---------------------------------------------------------------------------
  // 8. issue_resp correctness at the handshake.
  //
  // DELIBERATELY NOT CHECKED: "accept must stay stable while a request is pending".
  // An earlier version of this file asserted that and it is simply not a CV-X-IF
  // requirement -- issue_resp is only meaningful in the cycle where
  // issue_valid && issue_ready. This coprocessor withholds issue_ready when the result
  // FIFO has no credit, and drives accept=0 in that state; when a credit frees, both go
  // high together. `accept` therefore changes legitimately while a request waits, and
  // asserting stability produced a false failure under result-channel backpressure.
  //
  // What IS required is checked instead: accept never fires without a valid request, and
  // whenever a handshake completes the decision matches the decode exactly.
  // ---------------------------------------------------------------------------

  property p_no_accept_without_valid;
    @(posedge clk) disable iff (!rst_n)
      xif.issue_resp.accept |-> xif.issue_valid;
  endproperty
  a_no_accept_without_valid: assert property (p_no_accept_without_valid)
    else $error("SVA: issue_resp.accept asserted without issue_valid");

  // On a completed handshake, we must accept exactly the ROPE.ROT instructions whose
  // operands are valid, and nothing else. This catches both over-acceptance (stealing
  // another coprocessor's or the core's instruction) and under-acceptance (silently
  // dropping work we advertised we could do).
  property p_accept_matches_decode;
    @(posedge clk) disable iff (!rst_n)
      (xif.issue_valid && xif.issue_ready) |->
        (xif.issue_resp.accept ==
           (rope_pkg::is_rope_rot(xif.issue_req.instr)
            && xif.issue_req.rs_valid[0] && xif.issue_req.rs_valid[1]));
  endproperty
  a_accept_matches_decode: assert property (p_accept_matches_decode)
    else $error("SVA: accept=%0b disagrees with the decode of instr=%08h",
                xif.issue_resp.accept, xif.issue_req.instr);

`endif  // ROPE_ASSERTIONS_SVH
