`timescale 1 ns / 1 ps 

module wnn_axi_slave_lite_v1_0_S00_AXI # 
(
    parameter integer ADDR_BW           = 3,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 14
)
(
    output wire [ADDR_BW-1:0]           bram_addr_b,
    output wire                         bram_en_b,
    output wire [15:0]                  bram_we_b,   
    output wire [127:0]                 bram_din_b,  
    input  wire [127:0]                 bram_dout_b,

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
    output wire [1:0]                   S_AXI_RRESP,
    output wire                         S_AXI_RVALID,
    input  wire                         S_AXI_RREADY
);
    // AXI4-Lite signals
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

    // -----------------------------------------
    // Write Logic (Direct to Core RAM)
    // -----------------------------------------
    // Define signals to pass to Core
    reg [9:0]  core_ivec_addr;
    reg [31:0] core_ivec_wdata;
    reg        core_ivec_wen;

    always @(posedge S_AXI_ACLK) begin
        // Defaults 
        core_ivec_wen <= 1'b0; 
        core_ivec_addr <= 10'd0;
        core_ivec_wdata <= 32'd0;

        if (S_AXI_ARESETN && slv_reg_wren) begin
            case (S_AXI_AWADDR[C_S_AXI_ADDR_WIDTH-1:2])
                12'h000: ; // CTRL
                12'h001: ; // Output
                12'h003: slv_reg3 <= S_AXI_WDATA; // Config
                default: begin
                    // Input Vector Range: 0x40 (16) to 800
                    if (S_AXI_AWADDR[C_S_AXI_ADDR_WIDTH-1:2] >= 12'd16 && 
                        S_AXI_AWADDR[C_S_AXI_ADDR_WIDTH-1:2] < 12'd800) begin
                         // Pass write to Core RAM
                         core_ivec_addr  <= S_AXI_AWADDR[C_S_AXI_ADDR_WIDTH-1:2] - 12'd16;
                         core_ivec_wdata <= S_AXI_WDATA;
                         core_ivec_wen   <= 1'b1; 
                    end
                end
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

    // Instantiate WNN core
    wire                       core_done;
    wire [$clog2(10)-1:0]      core_pred_class;
    wire                       core_bram_en;
    wire [ADDR_BW-1:0]         core_bram_addr;
    wire [127:0]               core_bram_dout;
    
    assign bram_en_b      = core_bram_en;
    assign bram_addr_b    = core_bram_addr;
    assign bram_we_b      = 16'h0000;
    assign bram_din_b     = 128'h0;
    assign core_bram_dout = bram_dout_b;

    wnn_core #(
        .NUM_LUTS    (500),
        .ADDR_BITS   (6),
        .N_CLASSES   (10),
        .COUNT_BITS  (12),
        .INPUT_BITS  (25088) 
    ) u_wnn_core (
        .clk        (S_AXI_ACLK),
        .rst_n      (S_AXI_ARESETN), 
        .start      (start_pulse),
        .done       (core_done),
        
        // NEW: Passing write signals to core
        .ivec_addr  (core_ivec_addr),
        .ivec_wdata (core_ivec_wdata),
        .ivec_wen   (core_ivec_wen),
        
        .enable_mask      ({500{1'b1}}),
        .addr_mask        ({6{1'b1}}),
        .predicted_class  (core_pred_class),
        .bram_addr_b      (core_bram_addr),
        .bram_en_b        (core_bram_en),
        .bram_dout_b      (core_bram_dout)
    );

    // Status Logic
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            done_status <= 1'b0;
            slv_reg1    <= 32'h0;
        end else begin
            if (clear_done_req) done_status <= 1'b0;
            if (core_done) begin
                done_status <= 1'b1;
                slv_reg1    <= {28'd0, core_pred_class};
            end
        end
    end

endmodule