`timescale 1 ns / 1 ps 

module wnn_axi_slave_lite_v1_0_S00_AXI # 
(
    parameter integer ADDR_BW           = 3,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 14
)
(
    output wire                         core_start_pulse,
    input  wire                         core_done_in,
    input  wire [3:0]                   core_pred_class_in,

    // Standard AXI Ports
    input wire                          S_AXI_ACLK,
    input wire                          S_AXI_ARESETN,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_AWADDR,
    input wire [2:0]                    S_AXI_AWPROT,
    input wire                          S_AXI_AWVALID,
    output wire                         S_AXI_AWREADY,
    input wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_WDATA,
    input wire [(C_S_AXI_DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input wire                          S_AXI_WVALID,
    output wire                         S_AXI_WREADY,
    output wire [1:0]                   S_AXI_BRESP,
    output wire                         S_AXI_BVALID,
    input  wire                         S_AXI_BREADY,
    input wire [C_S_AXI_ADDR_WIDTH-1:0] S_AXI_ARADDR,
    input wire [2:0]                    S_AXI_ARPROT,
    input wire                          S_AXI_ARVALID,
    output wire                         S_AXI_ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] S_AXI_RDATA,
    output wire [1:0]                    S_AXI_RRESP,
    output wire                          S_AXI_RVALID,
    input  wire                          S_AXI_RREADY
);
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_awaddr;
    reg                          axi_awready;
    reg                          axi_wready;
    reg [1:0]                    axi_bresp;
    reg                          axi_bvalid;
    reg [C_S_AXI_ADDR_WIDTH-1:0] axi_araddr;
    reg                          axi_arready;
    reg [C_S_AXI_DATA_WIDTH-1:0] axi_rdata;
    reg [1:0]                    axi_rresp;
    reg                          axi_rvalid;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;

    wire aw_hs = S_AXI_AWVALID & ~axi_awready;
    wire w_hs  = S_AXI_WVALID  & ~axi_wready;
    wire slv_reg_wren = aw_hs & w_hs;

    // Registers
    reg done_status;
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg1; // Result
    reg [C_S_AXI_DATA_WIDTH-1:0] slv_reg3; // Config

    wire [1:0] wr_addr_sel = S_AXI_AWADDR[3:2];

    // Write Logic
    always @(posedge S_AXI_ACLK) begin
        if (S_AXI_ARESETN && slv_reg_wren) begin
            case (S_AXI_AWADDR[C_S_AXI_ADDR_WIDTH-1:2])
                12'h003: slv_reg3 <= S_AXI_WDATA;
                default: ; 
            endcase
        end
    end

    // AXI Handshake Logic 
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            axi_awready <= 1'b0;
            axi_awaddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            axi_wready <= 1'b0;
            axi_bvalid <= 1'b0;
            axi_bresp  <= 2'b00;
            axi_arready <= 1'b0;
            axi_araddr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
            axi_rvalid <= 1'b0;
            axi_rresp  <= 2'b00;
        end else begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID) begin
                axi_awready <= 1'b1;
                axi_awaddr <= S_AXI_AWADDR;
            end else axi_awready <= 1'b0;

            if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID) axi_wready <= 1'b1;
            else axi_wready <= 1'b0;

            if (axi_awready && S_AXI_AWVALID && ~axi_bvalid && axi_wready && S_AXI_WVALID) begin
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b00;
            end else if (S_AXI_BREADY && axi_bvalid) axi_bvalid <= 1'b0;

            if (~axi_arready && S_AXI_ARVALID) begin
                axi_arready <= 1'b1;
                axi_araddr  <= S_AXI_ARADDR;
            end else axi_arready <= 1'b0;

            if (axi_arready && S_AXI_ARVALID && ~axi_rvalid) begin
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b00;
            end else if (axi_rvalid && S_AXI_RREADY) axi_rvalid <= 1'b0;
        end
    end

    // Read Mux
    always @(*) begin
        case (axi_araddr[7:2]) 
            6'h00: axi_rdata = {30'd0, done_status, 1'b0};
            6'h01: axi_rdata = slv_reg1;
            6'h03: axi_rdata = slv_reg3;
            default: axi_rdata = 32'h0;
        endcase
    end

    // CTRL Logic
    wire wr_ctrl         = slv_reg_wren && (wr_addr_sel == 2'h0) && S_AXI_WSTRB[0];
    wire start_req       = wr_ctrl && S_AXI_WDATA[0]; 
    wire clear_done_req  = wr_ctrl && S_AXI_WDATA[2];

    reg  start_pulse;
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) start_pulse <= 1'b0;
        else start_pulse <= start_req;
    end
    
    // OUTPUT: Send start pulse to the external core
    assign core_start_pulse = start_pulse;

    // Status Logic
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            done_status <= 1'b0;
            slv_reg1    <= 32'h0;
        end else begin
            if (clear_done_req) done_status <= 1'b0;
            
            // INPUT: Read done/result from external core
            if (core_done_in) begin
                done_status <= 1'b1;
                slv_reg1    <= {28'd0, core_pred_class_in};
            end
        end
    end

endmodule