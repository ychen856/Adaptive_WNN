`timescale 1ns / 1ps

module axis_to_bram_loader #(
    parameter C_S_AXIS_TDATA_WIDTH = 32,
    parameter RAM_ADDR_WIDTH = 10 
)(
    input wire  S_AXIS_ACLK,
    input wire  S_AXIS_ARESETN,
    output wire S_AXIS_TREADY,
    input wire [C_S_AXIS_TDATA_WIDTH-1 : 0] S_AXIS_TDATA,
    input wire [(C_S_AXIS_TDATA_WIDTH/8)-1 : 0] S_AXIS_TSTRB, 
    input wire  S_AXIS_TLAST,
    input wire  S_AXIS_TVALID,

    // OUTPUTS to WNN CORE
    output reg [RAM_ADDR_WIDTH-1:0] ram_addr,
    output reg [31:0]               ram_wdata,
    output reg                      ram_wen
);

    // Always ready to accept data
    assign S_AXIS_TREADY = 1'b1;

    // Internal Counter
    reg [RAM_ADDR_WIDTH-1:0] addr_counter;

    always @(posedge S_AXIS_ACLK) begin
        if (!S_AXIS_ARESETN) begin
            addr_counter <= 0;
            ram_wen      <= 0;
            ram_addr     <= 0;
            ram_wdata    <= 0;
        end else begin
            // Default: Stop writing
            ram_wen <= 0;

            if (S_AXIS_TVALID && S_AXIS_TREADY) begin
                ram_wen      <= 1'b1;
                ram_addr     <= addr_counter;
                ram_wdata    <= S_AXIS_TDATA; 

                if (S_AXIS_TLAST) begin
                    addr_counter <= 0;
                end else begin
                    addr_counter <= addr_counter + 1;
                end
            end
        end
    end

endmodule