// -----------------------------------------------------------------------------
// tb_rope_bf16_unit.sv -- Milestone 1 exit criteria for the BF16 primitives.
//
// Reads a vector file produced by model/gen_vectors.py and checks rope_bf16_mul and
// rope_bf16_add bit-exactly against the golden model (Model A's per-node arithmetic).
//
// Vector file format, one test per line, all hex:
//     <op> <a> <b> <expected>
// where op is 0=MUL, 1=ADD, 2=SUB.
//
// Any mismatch is a hard failure: steps.md Section 5.3 demands ZERO mismatches.
// -----------------------------------------------------------------------------

module tb_rope_bf16_unit;

  localparam int unsigned NumPipeRegs = 0;
  localparam int unsigned MaxVectors  = 20_000_000;

  logic clk, rst_n;

  // Free-running clock.
  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  // --------------------------------------------------------------------------
  // DUTs -- one of each operation, all fed in lockstep
  // --------------------------------------------------------------------------

  logic [15:0] a, b;
  logic        in_valid;
  logic [2:0]  in_ready;
  logic [15:0] res_mul, res_add, res_sub;
  logic [2:0]  out_valid;

  rope_bf16_mul #(.NumPipeRegs(NumPipeRegs)) i_mul (
    .clk_i(clk), .rst_ni(rst_n),
    .a_i(a), .b_i(b),
    .in_valid_i(in_valid), .in_ready_o(in_ready[0]), .flush_i(1'b0),
    .result_o(res_mul), .out_valid_o(out_valid[0]), .out_ready_i(1'b1), .busy_o()
  );

  rope_bf16_add #(.NumPipeRegs(NumPipeRegs), .Sub(1'b0)) i_add (
    .clk_i(clk), .rst_ni(rst_n),
    .a_i(a), .b_i(b),
    .in_valid_i(in_valid), .in_ready_o(in_ready[1]), .flush_i(1'b0),
    .result_o(res_add), .out_valid_o(out_valid[1]), .out_ready_i(1'b1), .busy_o()
  );

  rope_bf16_add #(.NumPipeRegs(NumPipeRegs), .Sub(1'b1)) i_sub (
    .clk_i(clk), .rst_ni(rst_n),
    .a_i(a), .b_i(b),
    .in_valid_i(in_valid), .in_ready_o(in_ready[2]), .flush_i(1'b0),
    .result_o(res_sub), .out_valid_o(out_valid[2]), .out_ready_i(1'b1), .busy_o()
  );

  // --------------------------------------------------------------------------
  // Vectors
  // --------------------------------------------------------------------------

  int unsigned n_vec;
  logic [2:0]  v_op   [MaxVectors];
  logic [15:0] v_a    [MaxVectors];
  logic [15:0] v_b    [MaxVectors];
  logic [15:0] v_exp  [MaxVectors];

  int unsigned errors;
  int unsigned checked;
  int unsigned err_by_op [3];

  string vecfile;

  function automatic string op_name(logic [2:0] op);
    case (op)
      3'd0: return "MUL";
      3'd1: return "ADD";
      3'd2: return "SUB";
      default: return "???";
    endcase
  endfunction

  initial begin : load_vectors
    int fd, code;
    int unsigned op_i, a_i, b_i, e_i;

    if (!$value$plusargs("vectors=%s", vecfile)) vecfile = "vectors_bf16.hex";

    fd = $fopen(vecfile, "r");
    if (fd == 0) begin
      $display("FATAL: cannot open vector file '%s'", vecfile);
      $fatal(1);
    end

    n_vec = 0;
    forever begin
      code = $fscanf(fd, "%h %h %h %h\n", op_i, a_i, b_i, e_i);
      if (code != 4) break;
      v_op [n_vec] = op_i[2:0];
      v_a  [n_vec] = a_i[15:0];
      v_b  [n_vec] = b_i[15:0];
      v_exp[n_vec] = e_i[15:0];
      n_vec++;
      if (n_vec >= MaxVectors) break;
    end
    $fclose(fd);
    $display("tb_rope_bf16_unit: loaded %0d vectors from %s", n_vec, vecfile);
  end

  // --------------------------------------------------------------------------
  // Drive and check.
  //
  // The three units are identically configured, so their latencies match and a single
  // shared scoreboard index is valid. That assumption is checked, not assumed: see the
  // out_valid agreement assertion below.
  // --------------------------------------------------------------------------

  initial begin : stimulus
    int unsigned i;
    int unsigned sent;

    errors  = 0;
    checked = 0;
    for (int k = 0; k < 3; k++) err_by_op[k] = 0;

    rst_n    = 1'b0;
    in_valid = 1'b0;
    a        = '0;
    b        = '0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    repeat (2) @(posedge clk);

    // The vector load runs in a parallel initial block; make sure it finished.
    wait (n_vec > 0);

    sent = 0;
    for (i = 0; i < n_vec; i++) begin
      // Present the operands and hold until all three units accept.
      a        <= v_a[i];
      b        <= v_b[i];
      in_valid <= 1'b1;
      @(posedge clk);
      while (in_ready != 3'b111) @(posedge clk);
      sent++;
    end
    in_valid <= 1'b0;

    // Drain the pipelines.
    repeat (NumPipeRegs + 16) @(posedge clk);

    $display("----------------------------------------------------------------");
    $display("tb_rope_bf16_unit: sent=%0d checked=%0d errors=%0d", sent, checked, errors);
    $display("  MUL errors: %0d", err_by_op[0]);
    $display("  ADD errors: %0d", err_by_op[1]);
    $display("  SUB errors: %0d", err_by_op[2]);
    if (checked != n_vec) begin
      $display("FAIL: checked %0d of %0d vectors (pipeline drain problem)", checked, n_vec);
      $fatal(1);
    end
    if (errors != 0) begin
      $display("FAIL: %0d mismatches (steps.md requires ZERO)", errors);
      $fatal(1);
    end
    $display("PASS: all %0d vectors bit-exact", checked);
    $finish;
  end

  // Scoreboard: results come out in order, one per accepted input.
  int unsigned check_idx;

  initial check_idx = 0;

  always @(posedge clk) begin
    if (rst_n && out_valid[0]) begin
      logic [15:0] got;
      logic [15:0] want;
      logic [2:0]  op;

      want = v_exp[check_idx];
      op   = v_op[check_idx];

      case (op)
        3'd0: got = res_mul;
        3'd1: got = res_add;
        default: got = res_sub;
      endcase

      if (got !== want) begin
        errors++;
        err_by_op[op] = err_by_op[op] + 1;
        if (errors <= 20) begin
          $display("MISMATCH #%0d [%0d]: %s a=%04h b=%04h  got=%04h want=%04h",
                   errors, check_idx, op_name(op), v_a[check_idx], v_b[check_idx], got, want);
        end
      end
      checked++;
      check_idx++;
    end
  end

  // The shared scoreboard index is only sound if the three identically-configured units
  // really do produce their outputs on the same cycle. Assert it rather than trust it.
  always @(posedge clk) begin
    if (rst_n) begin
      assert (out_valid[0] == out_valid[1] && out_valid[1] == out_valid[2])
        else $fatal(1, "out_valid disagreement: %b -- latency assumption broken", out_valid);
      assert (in_ready[0] == in_ready[1] && in_ready[1] == in_ready[2])
        else $fatal(1, "in_ready disagreement: %b", in_ready);
    end
  end

endmodule : tb_rope_bf16_unit
