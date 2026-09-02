// -----------------------------------------------------------------------------
// tb_rope_sin_lut.sv -- Milestone 3 exit criteria.
//
//  [x] output matches Python lut_sin / lut_cos bit-exactly (exhaustive on the top 12
//      bits of the phase word -- i.e. every distinct table entry in every quadrant --
//      plus dense random coverage of the full 2^32 space)
//  [x] reflection symmetry verified at quadrant boundaries specifically
//  [x] sin^2 + cos^2 checked and its deviation from 1.0 reported
//
// Vector format:  <phase> <expected_sin> <expected_cos>   (hex)
// -----------------------------------------------------------------------------

module tb_rope_sin_lut;

  import rope_pkg::*;

  localparam int unsigned MaxVectors = 4_000_000;

  logic [PhaseBits-1:0] phase;
  logic [15:0]          sin_val, cos_val;

  rope_sin_lut i_dut (
    .phase_i ( phase   ),
    .sin_o   ( sin_val ),
    .cos_o   ( cos_val )
  );

  int unsigned n_vec;
  logic [31:0] v_ph  [MaxVectors];
  logic [15:0] v_sin [MaxVectors];
  logic [15:0] v_cos [MaxVectors];

  int unsigned errors;
  string       vecfile;

  // BF16 bit pattern -> real, decoded explicitly.
  //
  // The obvious `$bitstoshortreal({bits, 16'h0})` trick (exact, since BF16 is FP32's top
  // 16 bits) is not usable here: Verilator does not support promoting shortreal to real.
  // Decoding the fields by hand is exact for all normals and subnormals, which is all the
  // table contains.
  function automatic real bf16_to_real(logic [15:0] b);
    logic        sgn;
    logic [7:0]  exp;
    logic [6:0]  man;
    real         v;
    sgn = b[15];
    exp = b[14:7];
    man = b[6:0];
    if (exp == 8'd0) begin
      // Subnormal: value = man * 2^-133.
      v = real'(man) * $pow(2.0, -133.0);
    end else if (exp == 8'hFF) begin
      // Inf/NaN -- the midpoint-sampled quarter-wave table contains neither.
      v = 0.0;
    end else begin
      v = (1.0 + real'(man) / 128.0) * $pow(2.0, real'(int'(exp) - 127));
    end
    return sgn ? -v : v;
  endfunction

  initial begin : load
    int fd, code;
    int unsigned p_i, s_i, c_i;

    if (!$value$plusargs("vectors=%s", vecfile)) vecfile = "vectors_lut.hex";
    fd = $fopen(vecfile, "r");
    if (fd == 0) begin
      $display("FATAL: cannot open '%s'", vecfile);
      $fatal(1);
    end
    n_vec = 0;
    forever begin
      code = $fscanf(fd, "%h %h %h\n", p_i, s_i, c_i);
      if (code != 3) break;
      v_ph[n_vec]  = p_i;
      v_sin[n_vec] = s_i[15:0];
      v_cos[n_vec] = c_i[15:0];
      n_vec++;
      if (n_vec >= MaxVectors) break;
    end
    $fclose(fd);
    $display("tb_rope_sin_lut: loaded %0d vectors from %s", n_vec, vecfile);
  end

  initial begin : run
    int unsigned k;
    int unsigned sin_err, cos_err;

    errors  = 0;
    sin_err = 0;
    cos_err = 0;
    phase   = '0;

    wait (n_vec > 0);
    #1ns;

    // ---- 1. Bit-exact vs the golden model ---------------------------------
    for (k = 0; k < n_vec; k++) begin
      phase = v_ph[k];
      #1ns;
      if (sin_val !== v_sin[k]) begin
        sin_err++;
        if (sin_err <= 10)
          $display("SIN MISMATCH: phase=%08h got=%04h want=%04h", v_ph[k], sin_val, v_sin[k]);
      end
      if (cos_val !== v_cos[k]) begin
        cos_err++;
        if (cos_err <= 10)
          $display("COS MISMATCH: phase=%08h got=%04h want=%04h", v_ph[k], cos_val, v_cos[k]);
      end
    end
    $display("bit-exact check: %0d vectors, sin errors=%0d, cos errors=%0d",
             n_vec, sin_err, cos_err);
    errors += sin_err + cos_err;

    // ---- 2. Reflection symmetry at quadrant boundaries --------------------
    //
    // With midpoint sampling, quadrant 1 at index k must read exactly the same table
    // entry as quadrant 0 at index (N-1-k). Checked at the extreme indices, which is
    // precisely where edge sampling would have overflowed the table.
    begin
      int unsigned refl_err;
      logic [15:0] q0_val, q1_val;
      refl_err = 0;
      for (int unsigned k2 = 0; k2 < LutN; k2++) begin
        // Q0 at index (N-1-k2)
        phase = (32'd0 << QuadShift) | ((LutN - 1 - k2) << IdxShift);
        #1ns;
        q0_val = sin_val;
        // Q1 at index k2 -- must reflect to the same entry
        phase = (32'd1 << QuadShift) | (k2 << IdxShift);
        #1ns;
        q1_val = sin_val;
        if (q0_val !== q1_val) begin
          refl_err++;
          if (refl_err <= 5)
            $display("REFLECT MISMATCH: k=%0d Q0[N-1-k]=%04h Q1[k]=%04h", k2, q0_val, q1_val);
        end
      end
      if (refl_err == 0)
        $display("reflection symmetry: exact for all %0d indices (Q1[k] == Q0[N-1-k])", LutN);
      errors += refl_err;
    end

    // ---- 3. Quadrant sign structure ---------------------------------------
    //
    // Q2 must be the exact sign-bit flip of Q0, and Q3 of Q1. A sign-bit XOR is exact,
    // so this must hold bit-for-bit with no tolerance.
    begin
      int unsigned sign_err;
      logic [15:0] a_val, b_val;
      sign_err = 0;
      for (int unsigned k3 = 0; k3 < LutN; k3 += 7) begin
        phase = (32'd0 << QuadShift) | (k3 << IdxShift);
        #1ns; a_val = sin_val;
        phase = (32'd2 << QuadShift) | (k3 << IdxShift);
        #1ns; b_val = sin_val;
        if (b_val !== (a_val ^ 16'h8000)) begin
          sign_err++;
          if (sign_err <= 5)
            $display("SIGN MISMATCH Q2 vs Q0: k=%0d Q0=%04h Q2=%04h", k3, a_val, b_val);
        end
        phase = (32'd1 << QuadShift) | (k3 << IdxShift);
        #1ns; a_val = sin_val;
        phase = (32'd3 << QuadShift) | (k3 << IdxShift);
        #1ns; b_val = sin_val;
        if (b_val !== (a_val ^ 16'h8000)) begin
          sign_err++;
          if (sign_err <= 5)
            $display("SIGN MISMATCH Q3 vs Q1: k=%0d Q1=%04h Q3=%04h", k3, a_val, b_val);
        end
      end
      if (sign_err == 0)
        $display("quadrant signs: Q2 == ~Q0 and Q3 == ~Q1 exactly (sign-bit XOR)");
      errors += sign_err;
    end

    // ---- 4. The low 20 bits must be ignored -------------------------------
    //
    // phase[19:0] is reserved for future interpolation and must not affect the result
    // today. If it leaked into the index, results would depend on unused bits.
    begin
      int unsigned leak_err;
      logic [15:0] s_ref, c_ref;
      leak_err = 0;
      for (int unsigned t = 0; t < 4096; t += 37) begin
        phase = t << IdxShift;
        #1ns; s_ref = sin_val; c_ref = cos_val;
        for (int unsigned low = 1; low < 20; low++) begin
          phase = (t << IdxShift) | (32'd1 << (low - 1));
          #1ns;
          if (low <= IdxShift) begin
            if (sin_val !== s_ref || cos_val !== c_ref) begin
              leak_err++;
              if (leak_err <= 5)
                $display("LOW-BIT LEAK: top=%0d low bit %0d changed the result", t, low - 1);
            end
          end
        end
      end
      if (leak_err == 0)
        $display("phase[19:0] correctly ignored (reserved for interpolation)");
      errors += leak_err;
    end

    // ---- 5. sin^2 + cos^2, reported not asserted --------------------------
    //
    // This cannot be exactly 1.0 in BF16 and that is expected (steps.md 7.5). Report the
    // deviation; only a wild value would indicate a real bug.
    begin
      real  s_r, c_r, n_r, dev, max_dev, sum_sq;
      int unsigned cnt;
      max_dev = 0.0;
      sum_sq  = 0.0;
      cnt     = 0;
      for (int unsigned q = 0; q < 4; q++) begin
        for (int unsigned k4 = 0; k4 < LutN; k4 += 5) begin
          phase = (q << QuadShift) | (k4 << IdxShift);
          #1ns;
          s_r = bf16_to_real(sin_val);
          c_r = bf16_to_real(cos_val);
          n_r = s_r * s_r + c_r * c_r;
          dev = (n_r > 1.0) ? (n_r - 1.0) : (1.0 - n_r);
          if (dev > max_dev) max_dev = dev;
          sum_sq += dev * dev;
          cnt++;
        end
      end
      $display("sin^2+cos^2 over %0d points: max deviation = %f, RMS deviation = %f",
               cnt, max_dev, $sqrt(sum_sq / real'(cnt)));
      // A BF16 ulp near 1.0 is 2^-7 = 0.0078. Anything beyond a few ulp means the
      // quadrant or reflection logic is wrong, not merely quantised.
      if (max_dev > 4.0 * 0.0078125) begin
        $display("FAIL: sin^2+cos^2 deviates by more than 4 BF16 ulp -- logic bug, not quantisation");
        errors++;
      end
    end

    $display("----------------------------------------------------------------");
    if (errors != 0) begin
      $display("FAIL: %0d total errors", errors);
      $fatal(1);
    end
    $display("PASS: sine LUT bit-exact, reflection and quadrant signs exact");
    $finish;
  end

endmodule : tb_rope_sin_lut
