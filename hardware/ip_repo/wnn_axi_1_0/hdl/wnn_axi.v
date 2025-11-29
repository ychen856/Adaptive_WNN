`timescale 1 ns / 1 ps 

module wnn_axi # 
(
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 14,
    // BRAM-B address width
    parameter integer ADDR_BW              = 15,
    parameter integer C_S_AXIS_TDATA_WIDTH = 32
)
(
    input  wire                          s00_axi_aclk,
    input  wire                          s00_axi_aresetn,
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_awaddr,
    input  wire [2:0]                    s00_axi_awprot,
    input  wire                          s00_axi_awvalid,
    output wire                          s00_axi_awready,
    input  wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_wdata,
    input  wire [(C_S00_AXI_DATA_WIDTH/8)-1:0] s00_axi_wstrb,
    input  wire                          s00_axi_wvalid,
    output wire                          s00_axi_wready,
    output wire [1:0]                    s00_axi_bresp,
    output wire                          s00_axi_bvalid,
    input  wire                          s00_axi_bready,
    input  wire [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_araddr,
    input  wire [2:0]                    s00_axi_arprot,
    input  wire                          s00_axi_arvalid,
    output wire                          s00_axi_arready,
    output wire [C_S00_AXI_DATA_WIDTH-1:0] s00_axi_rdata,
    output wire [1:0]                    s00_axi_rresp,
    output wire                          s00_axi_rvalid,
    input  wire                          s00_axi_rready,
    
    // AXI Stream Slave Interface
    input wire  s_axis_aclk,
    input wire  s_axis_aresetn,
    output wire s_axis_tready,
    input wire [C_S_AXIS_TDATA_WIDTH-1 : 0] s_axis_tdata,
    input wire [(C_S_AXIS_TDATA_WIDTH/8)-1 : 0] s_axis_tstrb,
    input wire  s_axis_tlast,
    input wire  s_axis_tvalid,

    // BRAM-B native ports
    output wire [ADDR_BW-1:0]            bram_addr_b,
    output wire                          bram_en_b,
    output wire [15:0]                   bram_we_b,
    output wire [127:0]                  bram_din_b,
    input  wire [127:0]                  bram_dout_b
);

    // Internal signals
    wire [9:0] internal_ram_addr;
    wire [31:0] internal_ram_wdata;
    wire       internal_ram_wen;
    
    // Signals going between AXI-Lite slave and WNN core for control
    wire core_start;
    wire core_done;
    wire [$clog2(10)-1:0] core_predicted_class;
    
    // AXI LITE SLAVE
    wnn_axi_slave_lite_v1_0_S00_AXI #(
        .ADDR_BW           (ADDR_BW),
        .C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
    ) wnn_axi_slave_lite_v1_0_S00_AXI_inst (
        .core_start_pulse(core_start),
        .core_done_in(core_done),
        .core_pred_class_in(core_predicted_class),

        // Standard AXI connections
        .S_AXI_ACLK   (s00_axi_aclk),
        .S_AXI_ARESETN(s00_axi_aresetn),
        .S_AXI_AWADDR (s00_axi_awaddr),
        .S_AXI_AWPROT (s00_axi_awprot),
        .S_AXI_AWVALID(s00_axi_awvalid),
        .S_AXI_AWREADY(s00_axi_awready),
        .S_AXI_WDATA  (s00_axi_wdata),
        .S_AXI_WSTRB  (s00_axi_wstrb),
        .S_AXI_WVALID (s00_axi_wvalid),
        .S_AXI_WREADY (s00_axi_wready),
        .S_AXI_BRESP  (s00_axi_bresp),
        .S_AXI_BVALID (s00_axi_bvalid),
        .S_AXI_BREADY (s00_axi_bready),
        .S_AXI_ARADDR (s00_axi_araddr),
        .S_AXI_ARPROT (s00_axi_arprot),
        .S_AXI_ARVALID(s00_axi_arvalid),
        .S_AXI_ARREADY(s00_axi_arready),
        .S_AXI_RDATA  (s00_axi_rdata),
        .S_AXI_RRESP  (s00_axi_rresp),
        .S_AXI_RVALID (s00_axi_rvalid),
        .S_AXI_RREADY (s00_axi_rready)
    );

    // AXI STREAM LOADER
    axis_to_bram_loader #(
        .C_S_AXIS_TDATA_WIDTH(C_S_AXIS_TDATA_WIDTH),
        .RAM_ADDR_WIDTH(10)
    ) u_stream_loader (
        .S_AXIS_ACLK(s_axis_aclk), 
        .S_AXIS_ARESETN(s_axis_aresetn),
        .S_AXIS_TREADY(s_axis_tready),
        .S_AXIS_TDATA(s_axis_tdata),
        .S_AXIS_TSTRB(s_axis_tstrb),
        .S_AXIS_TLAST(s_axis_tlast),
        .S_AXIS_TVALID(s_axis_tvalid),
        // Connect outputs to internal wires to feed the WNN Core
        .ram_addr(internal_ram_addr),
        .ram_wdata(internal_ram_wdata),
        .ram_wen(internal_ram_wen)
    );

    // WNN CORE
    wnn_core #(
        .NUM_LUTS    (500),
        .ADDR_BITS   (6),
        .N_CLASSES   (10),
        .COUNT_BITS  (8),
        .INPUT_BITS  (25088) 
    ) u_wnn_core (
        .clk        (s_axis_aclk), 
        .rst_n      (s_axis_aresetn), 
        
        // Control signals from AXI
        .start      (core_start),
        .done       (core_done),
        .predicted_class(core_predicted_class),

        // DATA inputs from AXI Stream Loader
        .ivec_addr  (internal_ram_addr),
        .ivec_wdata (internal_ram_wdata),
        .ivec_wen   (internal_ram_wen),
        
        // Constant Masks
        .enable_mask      ({500{1'b1}}),
        .addr_mask        ({6{1'b1}}),

        // Weight BRAM Interface
        .bram_addr_b      (bram_addr_b),
        .bram_en_b        (bram_en_b),
        .bram_dout_b      (bram_dout_b)
    );
    
    assign bram_we_b = 16'h0000;
    assign bram_din_b = 128'h0;

endmodule