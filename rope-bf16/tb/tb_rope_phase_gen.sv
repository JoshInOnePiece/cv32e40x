// -----------------------------------------------------------------------------
// tb_rope_phase_gen.sv -- Milestone 2 exit criteria.
//
//  [x] matches the Python phase computation bit-exactly for m in [0, 8192], all i
//  [x] wraparound verified at m values that cross 2^32
//  [x] sequential and random-access modes agree for the same (m, i)
//
// Vector format:  <m> <i> <expected_phase>   (hex)
// -----------------------------------------------------------------------------

module tb_rope_phase_gen;

  import rope_pkg::*;

  localparam int unsigned MaxVectors = 2_000_000;

  logic clk, rst_n;

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  // --------------------------------------------------------------------------
  // DUT
  // --------------------------------------------------------------------------

  logic [PosBits-1:0]     pos;
  logic [PairIdxBits-1:0] pair_idx;
  logic [PhaseBits-1:0]   phase;

  logic                   seq_clear, seq_step;
  logic [PairIdxBits-1:0] seq_idx;
  logic [PhaseBits-1:0]   seq_phase;

  rope_phase_gen i_dut (
    .clk_i       ( clk       ),
    .rst_ni      ( rst_n     ),
    .pos_i       ( pos       ),
    .pair_idx_i  ( pair_idx  ),
    .phase_o     ( phase     ),
    .seq_clear_i ( seq_clear ),
    .seq_step_i  ( seq_step  ),
    .seq_idx_i   ( seq_idx   ),
    .seq_phase_o ( seq_phase )
  );

  // --------------------------------------------------------------------------
  // Vectors
  // --------------------------------------------------------------------------

  int unsigned n_vec;
  int unsigned v_m   [MaxVectors];
  int unsigned v_i   [MaxVectors];
  logic [31:0] v_exp [MaxVectors];

  int unsigned errors;
  string       vecfile;

  initial begin : load
    int fd, code;
    int unsigned m_i, i_i, p_i;

    if (!$value$plusargs("vectors=%s", vecfile)) vecfile = "vectors_phase.hex";
    fd = $fopen(vecfile, "r");
    if (fd == 0) begin
      $display("FATAL: cannot open '%s'", vecfile);
      $fatal(1);
    end
    n_vec = 0;
    forever begin
      code = $fscanf(fd, "%h %h %h\n", m_i, i_i, p_i);
      if (code != 3) break;
      v_m[n_vec]   = m_i;
      v_i[n_vec]   = i_i;
      v_exp[n_vec] = p_i;
      n_vec++;
      if (n_vec >= MaxVectors) break;
    end
    $fclose(fd);
    $display("tb_rope_phase_gen: loaded %0d vectors from %s", n_vec, vecfile);
  end

  // --------------------------------------------------------------------------
  // Checks
  // --------------------------------------------------------------------------

  initial begin : run
    int unsigned k;
    int unsigned seq_errors;
    logic [PhaseBits-1:0] expect_acc;

    errors     = 0;
    seq_errors = 0;

    rst_n     = 1'b0;
    pos       = '0;
    pair_idx  = '0;
    seq_clear = 1'b0;
    seq_step  = 1'b0;
    seq_idx   = '0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    wait (n_vec > 0);

    // ---- 1. Random-access mode against the golden model --------------------
    for (k = 0; k < n_vec; k++) begin
      pos      = v_m[k][PosBits-1:0];
      pair_idx = v_i[k][PairIdxBits-1:0];
      #1ns;  // settle the combinational path
      if (phase !== v_exp[k]) begin
        errors++;
        if (errors <= 20) begin
          $display("MISMATCH [%0d]: m=%0d i=%0d  got=%08h want=%08h",
                   k, v_m[k], v_i[k], phase, v_exp[k]);
        end
      end
    end
    $display("random-access: %0d vectors, %0d errors", n_vec, errors);

    // ---- 2. Sequential mode vs random access -------------------------------
    //
    // steps.md Section 6.3: the sequential accumulator must agree exactly with the
    // random-access multiply for every m, with ZERO drift. This is the property that
    // makes the integer accumulator preferable to the trigonometric recurrence.
    // NOTE ON TB TIMING: control signals are deasserted a delta *after* the clock edge
    // (`@(posedge clk); #1ns;`), never at zero delay. Deasserting at zero delay races
    // with the DUT's always_ff sampling the same edge, and the pulse can be missed.
    for (int unsigned pidx = 0; pidx < 4; pidx++) begin
      seq_idx   = pidx[PairIdxBits-1:0];
      pair_idx  = pidx[PairIdxBits-1:0];

      // Clear to position 0.
      seq_clear = 1'b1;
      @(posedge clk);
      #1ns;
      seq_clear = 1'b0;
      @(posedge clk);
      #1ns;

      for (int unsigned m = 0; m <= 4096; m++) begin
        // Compare the accumulator against the random-access result for this m.
        pos = m[PosBits-1:0];
        #1ns;
        if (seq_phase !== phase) begin
          seq_errors++;
          if (seq_errors <= 10) begin
            $display("SEQ MISMATCH: i=%0d m=%0d  seq=%08h rand=%08h",
                     pidx, m, seq_phase, phase);
          end
        end
        // Advance one token: hold seq_step across the edge, then release.
        seq_step = 1'b1;
        @(posedge clk);
        #1ns;
        seq_step = 1'b0;
      end
    end
    $display("sequential vs random-access: %0d errors", seq_errors);
    errors += seq_errors;

    // ---- 3. Explicit wraparound check --------------------------------------
    //
    // Walk the accumulator past 2^32 and confirm it wraps rather than saturating.
    begin
      logic [PhaseBits-1:0] prev;
      int unsigned wraps;
      wraps     = 0;
      seq_idx   = '0;
      seq_clear = 1'b1;
      @(posedge clk);
      #1ns;
      seq_clear = 1'b0;
      @(posedge clk);
      #1ns;
      prev = seq_phase;
      for (int unsigned s = 0; s < 64; s++) begin
        seq_step = 1'b1;
        @(posedge clk);
        #1ns;
        seq_step = 1'b0;
        if (seq_phase < prev) wraps++;  // the accumulator rolled over 2^32
        prev = seq_phase;
      end
      if (wraps == 0) begin
        $display("FAIL: accumulator never wrapped past 2^32 in 64 steps for i=0");
        errors++;
      end else begin
        $display("wraparound: observed %0d rollovers past 2^32 for i=0 (expected)", wraps);
      end
    end

    $display("----------------------------------------------------------------");
    if (errors != 0) begin
      $display("FAIL: %0d total errors", errors);
      $fatal(1);
    end
    $display("PASS: phase generator bit-exact (random-access, sequential, wraparound)");
    $finish;
  end

endmodule : tb_rope_phase_gen
