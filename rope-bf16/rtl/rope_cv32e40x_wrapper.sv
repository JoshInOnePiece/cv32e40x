// -----------------------------------------------------------------------------
// rope_cv32e40x_wrapper.sv -- CV32E40X + CORE-V-XIF + BF16 RoPE coprocessor.
//
// Milestone 6 step 1-3: instantiate the core, the interface and the coprocessor, and
// connect all six XIF channel bundles. The compressed channel is tied off (ROPE.ROT has
// no compressed form) and the memory channels are left idle until the streaming variant
// (Milestone 8).
//
// The interface definition comes from the core's own cv32e40x_if_xif.sv rather than a
// hand-rolled copy, which keeps every struct type-compatible with the core by
// construction. Note this differs from steps.md Section 10.1 step 3, which suggests
// deps/core-v-xif/src/core_v_xif.sv: cv32e40x ships an equivalent definition, and using
// it keeps the project self-contained inside this repository with no external clone.
//
// The five XIF parameters are declared ONCE here as localparams and passed to both the
// core and the coprocessor. steps.md Section 9.5 is emphatic about this: a mismatch
// between the core, the interface instance and the coprocessor produces confusing
// elaboration-time breakage.
// -----------------------------------------------------------------------------

module rope_cv32e40x_wrapper
  import cv32e40x_pkg::*;
#(
  parameter rv32_e         RV32             = RV32I,
  parameter m_ext_e        M_EXT            = M,
  parameter bit            DEBUG            = 1,
  parameter int unsigned   NUM_MHPMCOUNTERS = 1,
  parameter int unsigned   NumPipeRegs      = 1
) (
  input  logic        clk_i,
  input  logic        rst_ni,
  input  logic        scan_cg_en_i,

  input  logic [31:0] boot_addr_i,
  input  logic [31:0] dm_exception_addr_i,
  input  logic [31:0] dm_halt_addr_i,
  input  logic [31:0] mhartid_i,
  input  logic [ 3:0] mimpid_patch_i,
  input  logic [31:0] mtvec_addr_i,

  // Instruction memory interface
  output logic        instr_req_o,
  input  logic        instr_gnt_i,
  input  logic        instr_rvalid_i,
  output logic [31:0] instr_addr_o,
  output logic [ 1:0] instr_memtype_o,
  output logic [ 2:0] instr_prot_o,
  output logic        instr_dbg_o,
  input  logic [31:0] instr_rdata_i,
  input  logic        instr_err_i,

  // Data memory interface
  output logic        data_req_o,
  input  logic        data_gnt_i,
  input  logic        data_rvalid_i,
  output logic [31:0] data_addr_o,
  output logic [ 3:0] data_be_o,
  output logic        data_we_o,
  output logic [31:0] data_wdata_o,
  output logic [ 1:0] data_memtype_o,
  output logic [ 2:0] data_prot_o,
  output logic        data_dbg_o,
  output logic [ 5:0] data_atop_o,
  input  logic [31:0] data_rdata_i,
  input  logic        data_err_i,
  input  logic        data_exokay_i,

  output logic [63:0] mcycle_o,
  input  logic [63:0] time_i,

  input  logic [31:0] irq_i,
  input  logic        wu_wfe_i,

  input  logic        clic_irq_i,
  input  logic [ 4:0] clic_irq_id_i,
  input  logic [ 7:0] clic_irq_level_i,
  input  logic [ 1:0] clic_irq_priv_i,
  input  logic        clic_irq_shv_i,

  output logic        fencei_flush_req_o,
  input  logic        fencei_flush_ack_i,

  input  logic        debug_req_i,
  output logic        debug_havereset_o,
  output logic        debug_running_o,
  output logic        debug_halted_o,
  output logic        debug_pc_valid_o,
  output logic [31:0] debug_pc_o,

  input  logic        fetch_enable_i,
  output logic        core_sleep_o
);

  // ---------------------------------------------------------------------------
  // XIF parameters -- locked on day one, shared by core, interface and coprocessor
  // ---------------------------------------------------------------------------

  localparam int unsigned X_NUM_RS    = 2;     // 3 if an R4-type op is added later
  localparam int unsigned X_ID_WIDTH  = 4;
  localparam int unsigned X_MEM_WIDTH = 32;    // exactly one interleaved BF16 pair/txn
  localparam int unsigned X_RFR_WIDTH = 32;
  localparam int unsigned X_RFW_WIDTH = 32;    // 32 suffices: no dual writeback

  // X_MISA / X_ECS_XS: ROPE.ROT advertises no MISA bit and never writes the extension
  // context status, so both stay zero.
  localparam logic [31:0] X_MISA      = 32'h0;
  localparam logic [ 1:0] X_ECS_XS    = 2'b00;

  // ---------------------------------------------------------------------------
  // Interface instance
  // ---------------------------------------------------------------------------

  cv32e40x_if_xif #(
    .X_NUM_RS    ( X_NUM_RS    ),
    .X_ID_WIDTH  ( X_ID_WIDTH  ),
    .X_MEM_WIDTH ( X_MEM_WIDTH ),
    .X_RFR_WIDTH ( X_RFR_WIDTH ),
    .X_RFW_WIDTH ( X_RFW_WIDTH ),
    .X_MISA      ( X_MISA      ),
    .X_ECS_XS    ( X_ECS_XS    )
  ) xif ();

  // ---------------------------------------------------------------------------
  // Core
  // ---------------------------------------------------------------------------

  cv32e40x_core #(
    .RV32             ( RV32             ),
    .M_EXT            ( M_EXT            ),
    .DEBUG            ( DEBUG            ),
    .X_EXT            ( 1'b1             ),   // enable the eXtension interface
    .X_NUM_RS         ( X_NUM_RS         ),
    .X_ID_WIDTH       ( X_ID_WIDTH       ),
    .X_MEM_WIDTH      ( X_MEM_WIDTH      ),
    .X_RFR_WIDTH      ( X_RFR_WIDTH      ),
    .X_RFW_WIDTH      ( X_RFW_WIDTH      ),
    .X_MISA           ( X_MISA           ),
    .X_ECS_XS         ( X_ECS_XS         ),
    .NUM_MHPMCOUNTERS ( NUM_MHPMCOUNTERS )
  ) i_core (
    .clk_i,
    .rst_ni,
    .scan_cg_en_i,

    .boot_addr_i,
    .dm_exception_addr_i,
    .dm_halt_addr_i,
    .mhartid_i,
    .mimpid_patch_i,
    .mtvec_addr_i,

    .instr_req_o,
    .instr_gnt_i,
    .instr_rvalid_i,
    .instr_addr_o,
    .instr_memtype_o,
    .instr_prot_o,
    .instr_dbg_o,
    .instr_rdata_i,
    .instr_err_i,

    .data_req_o,
    .data_gnt_i,
    .data_rvalid_i,
    .data_addr_o,
    .data_be_o,
    .data_we_o,
    .data_wdata_o,
    .data_memtype_o,
    .data_prot_o,
    .data_dbg_o,
    .data_atop_o,
    .data_rdata_i,
    .data_err_i,
    .data_exokay_i,

    .mcycle_o,
    .time_i,

    // All six XIF channel bundles.
    .xif_compressed_if ( xif ),
    .xif_issue_if      ( xif ),
    .xif_commit_if     ( xif ),
    .xif_mem_if        ( xif ),
    .xif_mem_result_if ( xif ),
    .xif_result_if     ( xif ),

    .irq_i,
    .wu_wfe_i,

    .clic_irq_i,
    .clic_irq_id_i,
    .clic_irq_level_i,
    .clic_irq_priv_i,
    .clic_irq_shv_i,

    .fencei_flush_req_o,
    .fencei_flush_ack_i,

    .debug_req_i,
    .debug_havereset_o,
    .debug_running_o,
    .debug_halted_o,
    .debug_pc_valid_o,
    .debug_pc_o,

    .fetch_enable_i,
    .core_sleep_o
  );

  // ---------------------------------------------------------------------------
  // Coprocessor
  // ---------------------------------------------------------------------------

  rope_xif_coproc #(
    .X_NUM_RS    ( X_NUM_RS    ),
    .X_ID_WIDTH  ( X_ID_WIDTH  ),
    .X_MEM_WIDTH ( X_MEM_WIDTH ),
    .X_RFR_WIDTH ( X_RFR_WIDTH ),
    .X_RFW_WIDTH ( X_RFW_WIDTH ),
    .NumPipeRegs ( NumPipeRegs )
  ) i_rope (
    .clk_i,
    .rst_ni,
    .xif_issue_if  ( xif ),
    .xif_commit_if ( xif ),
    .xif_result_if ( xif )
  );

  // ---------------------------------------------------------------------------
  // Unused channels
  // ---------------------------------------------------------------------------
  //
  // Compressed: ROPE.ROT has no 16-bit form, so we never accept a compressed offload.
  assign xif.compressed_ready = 1'b1;   // always able to answer
  assign xif.compressed_resp  = '0;     // accept = 0

  // Memory: ROPE.ROT performs no memory accesses. These become live in Milestone 8,
  // where the "always able to sink a returning load" rule (mem_result_valid has no
  // ready) must be honoured.
  assign xif.mem_valid = 1'b0;
  assign xif.mem_req   = '0;

endmodule : rope_cv32e40x_wrapper
