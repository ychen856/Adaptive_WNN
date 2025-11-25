`timescale 1 ns / 1 ps 

module wnn_axi # 
(
    // Parameters of Axi Slave Bus Interface S00_AXI
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 14,
    // BRAM-B address width
    parameter integer ADDR_BW              = 15
)
(
    // AXI4-Lite slave interface
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

    // BRAM-B native ports
    output wire [ADDR_BW-1:0]            bram_addr_b,
    output wire                          bram_en_b,
    output wire [15:0]                   bram_we_b,
    output wire [127:0]                  bram_din_b,
    input  wire [127:0]                  bram_dout_b
);

    // Instantiate AXI-lite slave wrapper + WNN core
    wnn_axi_slave_lite_v1_0_S00_AXI #(
        .ADDR_BW           (ADDR_BW),
        .C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
    ) wnn_axi_slave_lite_v1_0_S00_AXI_inst (
        // BRAM-B
        .bram_addr_b(bram_addr_b),
        .bram_en_b  (bram_en_b),
        .bram_we_b  (bram_we_b),
        .bram_din_b (bram_din_b),
        .bram_dout_b(bram_dout_b),

        // AXI4-Lite
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

endmodule
