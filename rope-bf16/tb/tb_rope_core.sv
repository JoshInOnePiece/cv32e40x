// -----------------------------------------------------------------------------
// tb_rope_core.sv -- run the bare-metal C test on the integrated core.
//
// This is the Milestone 7 exit criterion -- the C test passing on the integrated core
// under simulation -- and it closes the loop end to end:
//
// (NB: no comment line here may begin with the word "verilator", case-insensitively:
//  the lexer would parse it as a pragma and fail with "Unknown verilator comment".)
//
//     golden model -> compiled-in vectors -> real RISC-V instructions -> the core ->
//     CORE-V-XIF -> the RoPE coprocessor -> writeback -> compared in software
//
// Memory map (must match sw/link.ld):
//     0x00000000  RAM, 256 KB, instructions and data
//     0x10000000  putchar   (write-only)
//     0x20000000  exit code (write-only; 0 = pass)
//
// The memories model OBI with always-grant and one cycle of response latency, which is
// fully pipelined: a request may be accepted on every cycle.
// -----------------------------------------------------------------------------

module tb_rope_core;

  // Overridable from the command line: -GUsePrefetch=1
  parameter bit UsePrefetch   = 1'b0;
  parameter bit UseValueCheck = 1'b1;

  localparam int unsigned MemWords  = 65536;             // 256 KB
  localparam logic [31:0] BootAddr  = 32'h0000_0080;     // matches sw/link.ld
  localparam logic [31:0] PutcharAddr = 32'h1000_0000;
  localparam logic [31:0] ExitAddr    = 32'h2000_0000;

  logic clk, rst_n;

  initial begin
    clk = 1'b0;
    forever #5ns clk = ~clk;
  end

  // --------------------------------------------------------------------------
  // Core + coprocessor
  // --------------------------------------------------------------------------

  // Instruction OBI
  logic        instr_req, instr_gnt, instr_rvalid;
  logic [31:0] instr_addr, instr_rdata;
  logic [ 1:0] instr_memtype;
  logic [ 2:0] instr_prot;
  logic        instr_dbg, instr_err;

  // Data OBI
  logic        data_req, data_gnt, data_rvalid;
  logic [31:0] data_addr, data_wdata, data_rdata;
  logic [ 3:0] data_be;
  logic        data_we;
  logic [ 1:0] data_memtype;
  logic [ 2:0] data_prot;
  logic        data_dbg;
  logic [ 5:0] data_atop;
  logic        data_err, data_exokay;

  logic [63:0] mcycle;
  logic        fencei_flush_req;
  logic        core_sleep;

  rope_cv32e40x_wrapper #(
    .UsePrefetch   ( UsePrefetch   ),
    .UseValueCheck ( UseValueCheck )
  ) i_dut (
    .clk_i               ( clk           ),
    .rst_ni              ( rst_n         ),
    .scan_cg_en_i        ( 1'b0          ),

    .boot_addr_i         ( BootAddr      ),
    .dm_exception_addr_i ( 32'h0000_0000 ),
    .dm_halt_addr_i      ( 32'h0000_0000 ),
    .mhartid_i           ( 32'h0         ),
    .mimpid_patch_i      ( 4'h0          ),
    .mtvec_addr_i        ( 32'h0000_0000 ),

    .instr_req_o         ( instr_req     ),
    .instr_gnt_i         ( instr_gnt     ),
    .instr_rvalid_i      ( instr_rvalid  ),
    .instr_addr_o        ( instr_addr    ),
    .instr_memtype_o     ( instr_memtype ),
    .instr_prot_o        ( instr_prot    ),
    .instr_dbg_o         ( instr_dbg     ),
    .instr_rdata_i       ( instr_rdata   ),
    .instr_err_i         ( instr_err     ),

    .data_req_o          ( data_req      ),
    .data_gnt_i          ( data_gnt      ),
    .data_rvalid_i       ( data_rvalid   ),
    .data_addr_o         ( data_addr     ),
    .data_be_o           ( data_be       ),
    .data_we_o           ( data_we       ),
    .data_wdata_o        ( data_wdata    ),
    .data_memtype_o      ( data_memtype  ),
    .data_prot_o         ( data_prot     ),
    .data_dbg_o          ( data_dbg      ),
    .data_atop_o         ( data_atop     ),
    .data_rdata_i        ( data_rdata    ),
    .data_err_i          ( data_err      ),
    .data_exokay_i       ( data_exokay   ),

    .mcycle_o            ( mcycle        ),
    .time_i              ( 64'h0         ),

    .irq_i               ( 32'h0         ),
    .wu_wfe_i            ( 1'b0          ),

    .clic_irq_i          ( 1'b0          ),
    .clic_irq_id_i       ( 5'h0          ),
    .clic_irq_level_i    ( 8'h0          ),
    .clic_irq_priv_i     ( 2'h0          ),
    .clic_irq_shv_i      ( 1'b0          ),

    .fencei_flush_req_o  ( fencei_flush_req ),
    // Acknowledge immediately: there are no caches to flush in this testbench.
    .fencei_flush_ack_i  ( fencei_flush_req ),

    .debug_req_i         ( 1'b0          ),
    .debug_havereset_o   (               ),
    .debug_running_o     (               ),
    .debug_halted_o      (               ),
    .debug_pc_valid_o    (               ),
    .debug_pc_o          (               ),

    .fetch_enable_i      ( 1'b1          ),
    .core_sleep_o        ( core_sleep    )
  );

  // --------------------------------------------------------------------------
  // Memory
  // --------------------------------------------------------------------------

  logic [31:0] mem [MemWords];

  string       hexfile;
  int unsigned exit_code;
  bit          saw_exit;

  initial begin : load_image
    int unsigned k;
    for (k = 0; k < MemWords; k++) mem[k] = 32'h0;

    if (!$value$plusargs("hex=%s", hexfile)) hexfile = "test_rope.hex";
    // No explicit offset: `objcopy -O verilog --verilog-data-width=4` emits an
    // "@<word-index>" record (0x20 for the 0x80 load address), and $readmemh honours it.
    // Passing a start address here as well would double-count the offset.
    $readmemh(hexfile, mem);
    $display("tb_rope_core: loaded %s", hexfile);
  end

  // ---- instruction port: always grant, one cycle of latency ----------------
  logic [31:0] instr_addr_q;
  logic        instr_pending_q;

  assign instr_gnt = 1'b1;
  assign instr_err = 1'b0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      instr_pending_q <= 1'b0;
      instr_addr_q    <= '0;
    end else begin
      instr_pending_q <= instr_req & instr_gnt;
      instr_addr_q    <= instr_addr;
    end
  end

  assign instr_rvalid = instr_pending_q;
  assign instr_rdata  = mem[instr_addr_q[31:2] & (MemWords - 1)];

  // ---- data port: always grant, one cycle of latency ----------------------
  logic [31:0] data_addr_q, data_wdata_q;
  logic [ 3:0] data_be_q;
  logic        data_we_q, data_pending_q;

  assign data_gnt    = 1'b1;
  assign data_err    = 1'b0;
  assign data_exokay = 1'b1;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_pending_q <= 1'b0;
      data_addr_q    <= '0;
      data_wdata_q   <= '0;
      data_be_q      <= '0;
      data_we_q      <= 1'b0;
    end else begin
      data_pending_q <= data_req & data_gnt;
      data_addr_q    <= data_addr;
      data_wdata_q   <= data_wdata;
      data_be_q      <= data_be;
      data_we_q      <= data_we;
    end
  end

  assign data_rvalid = data_pending_q;

  // Reads return RAM contents; the two device addresses read as zero.
  assign data_rdata = mem[data_addr_q[31:2] & (MemWords - 1)];

  // Exit register. This is deliberately the SOLE driver of exit_code/saw_exit, with
  // their initial values coming from the reset branch rather than from an initial block:
  // driving them from both an initial block and a clocked process is a multiple-driver
  // conflict, and an earlier version of this testbench did exactly that and always read
  // back the stale initial value (reporting a spurious failure even though the software
  // had written 0 and printed PASS).
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      exit_code <= 32'hFFFF_FFFF;
      saw_exit  <= 1'b0;
    end else if (data_pending_q && data_we_q
                 && ((data_addr_q & 32'hFFFF_F000) == ExitAddr)) begin
      exit_code <= data_wdata_q;
      saw_exit  <= 1'b1;
    end
  end

  // Writes: RAM with byte enables, plus the putchar device.
  always_ff @(posedge clk) begin
    if (rst_n && data_pending_q && data_we_q) begin
      if ((data_addr_q & 32'hFFFF_F000) == PutcharAddr) begin
        $write("%c", data_wdata_q[7:0]);
      end else if ((data_addr_q & 32'hFFFF_F000) == ExitAddr) begin
        // Handled by the exit register above.
      end else begin
        // Byte-enable-correct RAM write.
        if (data_be_q[0]) mem[data_addr_q[31:2] & (MemWords-1)][ 7: 0] <= data_wdata_q[ 7: 0];
        if (data_be_q[1]) mem[data_addr_q[31:2] & (MemWords-1)][15: 8] <= data_wdata_q[15: 8];
        if (data_be_q[2]) mem[data_addr_q[31:2] & (MemWords-1)][23:16] <= data_wdata_q[23:16];
        if (data_be_q[3]) mem[data_addr_q[31:2] & (MemWords-1)][31:24] <= data_wdata_q[31:24];
      end
    end
  end

  // --------------------------------------------------------------------------
  // Run control
  // --------------------------------------------------------------------------

  initial begin : run
    // exit_code / saw_exit are reset by the exit register above -- not initialised here,
    // so that clocked process remains their only driver.
    rst_n = 1'b0;
    repeat (20) @(posedge clk);
    rst_n = 1'b1;

    // Wait for the software to write the exit address.
    wait (saw_exit);
    repeat (10) @(posedge clk);

    $display("");
    $display("----------------------------------------------------------------");
    $display("tb_rope_core: exit_code=0x%08h after %0d cycles", exit_code, mcycle);
    if (exit_code == 32'hDEAD) begin
      $display("FAIL: the program trapped (see crt0.S trap_park)");
      $fatal(1);
    end
    if (exit_code != 0) begin
      $display("FAIL: software reported %0d mismatches", exit_code);
      $fatal(1);
    end
    $display("PASS: C test passed on the integrated core");
    $finish;
  end

  // Watchdog: a hang here usually means a bad memory map or a trap loop.
  initial begin : watchdog
    #50ms;
    $display("");
    $display("FAIL: timeout after %0d cycles (saw_exit=%0b)", mcycle, saw_exit);
    $fatal(1);
  end

endmodule : tb_rope_core
