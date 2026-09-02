// -----------------------------------------------------------------------------
// tb_rope_datapath.sv -- Milestone 4 exit criteria.
//
//  [x] bit-exact vs Model A on random (x, y, m, i) tuples -- ZERO mismatches
//  [x] directed cancellation tests (x ~ y with m*theta ~ 45 deg) included in the
//      vector set, so the hardest numerical corner is covered, not just averages
//  [x] specials (NaN / Inf / subnormal) propagate identically to the model
//  [x] back-to-back full-throughput operation (one pair per cycle after fill)
//
// This is the whole chain: integer phase generation -> LUT -> 4 muls -> 2 adds.
//
// Vector format:  <x> <y> <m> <i> <expected_x> <expected_y>   (hex)
// -----------------------------------------------------------------------------

module tb_rope_datapath;

  import rope_pkg::*;

  localparam int unsigned MaxVectors  = 20_000_000;
  localparam int unsigned NumPipeRegs = 1;

  logic clk, rst_n;

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  // --------------------------------------------------------------------------
  // DUT chain
  // --------------------------------------------------------------------------

  logic [PosBits-1:0]     pos;
  logic [PairIdxBits-1:0] pair_idx;
  logic [15:0]            x_in, y_in;
  logic                   in_valid, in_ready;

  logic [PhaseBits-1:0]   phase;
  logic [15:0]            sin_val, cos_val;

  logic [15:0]            x_out, y_out;
  logic                   out_valid;
  logic [7:0]             latency;

  rope_phase_gen i_phase_gen (
    .clk_i       ( clk      ),
    .rst_ni      ( rst_n    ),
    .pos_i       ( pos      ),
    .pair_idx_i  ( pair_idx ),
    .phase_o     ( phase    ),
    .seq_clear_i ( 1'b0     ),
    .seq_step_i  ( 1'b0     ),
    .seq_idx_i   ( '0       ),
    .seq_phase_o (          )
  );

  rope_sin_lut i_lut (
    .phase_i ( phase   ),
    .sin_o   ( sin_val ),
    .cos_o   ( cos_val )
  );

  rope_datapath #(
    .NumPipeRegs ( NumPipeRegs ),
    .RegLut      ( 1'b1        )
  ) i_dut (
    .clk_i       ( clk       ),
    .rst_ni      ( rst_n     ),
    .flush_i     ( 1'b0      ),
    .x_i         ( x_in      ),
    .y_i         ( y_in      ),
    .sin_i       ( sin_val   ),
    .cos_i       ( cos_val   ),
    .in_valid_i  ( in_valid  ),
    .in_ready_o  ( in_ready  ),
    .x_o         ( x_out     ),
    .y_o         ( y_out     ),
    .out_valid_o ( out_valid ),
    .busy_o      (           ),
    .latency_o   ( latency   )
  );

  // --------------------------------------------------------------------------
  // Vectors
  // --------------------------------------------------------------------------

  int unsigned n_vec;
  logic [15:0] v_x   [MaxVectors];
  logic [15:0] v_y   [MaxVectors];
  int unsigned v_m   [MaxVectors];
  int unsigned v_i   [MaxVectors];
  logic [15:0] v_ex  [MaxVectors];
  logic [15:0] v_ey  [MaxVectors];

  int unsigned errors, checked, sent;
  string       vecfile;

  initial begin : load
    int fd, code;
    int unsigned x_i2, y_i2, m_i2, i_i2, ex_i, ey_i;

    if (!$value$plusargs("vectors=%s", vecfile)) vecfile = "vectors_datapath.hex";
    fd = $fopen(vecfile, "r");
    if (fd == 0) begin
      $display("FATAL: cannot open '%s'", vecfile);
      $fatal(1);
    end
    n_vec = 0;
    forever begin
      code = $fscanf(fd, "%h %h %h %h %h %h\n", x_i2, y_i2, m_i2, i_i2, ex_i, ey_i);
      if (code != 6) break;
      v_x[n_vec]  = x_i2[15:0];
      v_y[n_vec]  = y_i2[15:0];
      v_m[n_vec]  = m_i2;
      v_i[n_vec]  = i_i2;
      v_ex[n_vec] = ex_i[15:0];
      v_ey[n_vec] = ey_i[15:0];
      n_vec++;
      if (n_vec >= MaxVectors) break;
    end
    $fclose(fd);
    $display("tb_rope_datapath: loaded %0d vectors from %s", n_vec, vecfile);
  end

  // --------------------------------------------------------------------------
  // Scoreboard: expected results ride a queue, popped in order as results emerge.
  // Using a queue rather than a hard-coded latency means the check stays valid if
  // NumPipeRegs changes.
  // --------------------------------------------------------------------------

  int unsigned exp_q [$];

  initial begin : stimulus
    errors  = 0;
    checked = 0;
    sent    = 0;

    rst_n    = 1'b0;
    in_valid = 1'b0;
    x_in     = '0;
    y_in     = '0;
    pos      = '0;
    pair_idx = '0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    wait (n_vec > 0);
    $display("tb_rope_datapath: datapath latency = %0d cycles", latency);

    // Drive one vector per cycle -- this is the full-throughput (back-to-back) test.
    for (int unsigned k = 0; k < n_vec; k++) begin
      x_in     <= v_x[k];
      y_in     <= v_y[k];
      pos      <= v_m[k][PosBits-1:0];
      pair_idx <= v_i[k][PairIdxBits-1:0];
      in_valid <= 1'b1;
      @(posedge clk);
      // in_ready is asserted every cycle in this configuration; if it ever were not,
      // hold the operands until it is.
      while (!in_ready) @(posedge clk);
      exp_q.push_back(k);
      sent++;
    end
    in_valid <= 1'b0;

    // Drain.
    repeat (latency + 16) @(posedge clk);

    $display("----------------------------------------------------------------");
    $display("tb_rope_datapath: sent=%0d checked=%0d errors=%0d", sent, checked, errors);
    if (checked != sent) begin
      $display("FAIL: checked %0d of %0d (results lost or pipeline not drained)", checked, sent);
      $fatal(1);
    end
    if (errors != 0) begin
      $display("FAIL: %0d mismatches vs Model A (steps.md requires ZERO)", errors);
      $fatal(1);
    end
    $display("PASS: datapath bit-exact vs Model A on all %0d vectors", checked);
    $finish;
  end

  // Watchdog. A stalled handshake or an empty vector file would otherwise spin forever,
  // which is much harder to diagnose than a clean timeout.
  initial begin : watchdog
    #500ms;
    $display("FAIL: timeout -- sent=%0d checked=%0d of %0d vectors", sent, checked, n_vec);
    $fatal(1);
  end

  always @(posedge clk) begin
    if (rst_n && out_valid) begin
      int unsigned idx;
      if (exp_q.size() == 0) begin
        $display("FAIL: result with no pending expectation at time %t", $time);
        errors++;
      end else begin
        idx = exp_q.pop_front();
        if (x_out !== v_ex[idx] || y_out !== v_ey[idx]) begin
          errors++;
          if (errors <= 20) begin
            $display("MISMATCH [%0d]: x=%04h y=%04h m=%0d i=%0d | got=(%04h,%04h) want=(%04h,%04h)",
                     idx, v_x[idx], v_y[idx], v_m[idx], v_i[idx],
                     x_out, y_out, v_ex[idx], v_ey[idx]);
          end
        end
        checked++;
      end
    end
  end

endmodule : tb_rope_datapath
