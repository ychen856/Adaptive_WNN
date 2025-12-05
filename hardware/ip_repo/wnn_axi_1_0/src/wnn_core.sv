`timescale 1 ns / 1 ps 

module wnn_core #( 
    parameter int NUM_LUTS    = 250, 
    parameter int ADDR_BITS   = 6,
    parameter int N_CLASSES   = 10,
    parameter int COUNT_BITS  = 8,
    parameter int INPUT_BITS  = 25088  
)(
    input  logic                       clk,
    input  logic                       rst_n,
    input  logic                       start,
    output logic                       done,

    // RAM Write Interface
    input  logic [9:0]                 ivec_addr,
    input  logic [31:0]                ivec_wdata,
    input  logic                       ivec_wen,

    input  logic [NUM_LUTS-1:0]        enable_mask,
    input  logic [ADDR_BITS-1:0]       addr_mask,

    output logic [$clog2(N_CLASSES)-1:0] predicted_class,

    // Weight BRAM Interface
    output logic [ADDR_BITS+$clog2(NUM_LUTS)-1:0] bram_addr_b,
    output logic                       bram_en_b,
    input  logic [N_CLASSES*COUNT_BITS-1:0] bram_dout_b
);

    // Internal Input RAM (Distributed RAM / LUTRAM)
    // Stores 784 x 32-bit words = 25,088 bits
    (* ram_style = "distributed" *) logic [31:0] input_ram [0:783];

    // Synchronous Write
    always_ff @(posedge clk) begin
        if (ivec_wen) begin
            input_ram[ivec_addr] <= ivec_wdata;
        end
    end

    // Parameters & Logic
    localparam int LUT_IDX_BITS   = $clog2(NUM_LUTS);
    localparam int SCORE_BITS     = COUNT_BITS + LUT_IDX_BITS + 2;
    localparam int MAX_ADDR_BITS  = ADDR_BITS;
    localparam int MAX_FLAT_BITS  = NUM_LUTS * ADDR_BITS;
    localparam int BIT_IDX_BITS   = $clog2(INPUT_BITS);
    localparam int CLASS_BITS     = $clog2(N_CLASSES);

    typedef enum logic [3:0] { 
        S_IDLE,
        S_PREP_ADDR, 
        S_BUILD_ADDR,
        S_WAIT_1,
        S_WAIT_2,
        S_ACCUM,
        S_FIND_MAX,
        S_DONE
    } state_t;

    state_t                  state;
    logic [LUT_IDX_BITS-1:0] lut_idx;
    logic [15:0]             bit_offset;
    logic [ADDR_BITS-1:0]    eff_addr;
    
    // Registers for Sequential Fetch
    logic [BIT_IDX_BITS-1:0] cached_bit_sels [MAX_ADDR_BITS];
    logic [2:0]              addr_build_idx; 
    logic [ADDR_BITS-1:0]    tmp_addr_accum;

    // ROMs
    (* rom_style = "distributed" *) logic [ADDR_BITS-1:0]     addr_bits_rom [0:NUM_LUTS-1];
    (* rom_style = "distributed" *) logic [BIT_IDX_BITS-1:0]  kept_bits_rom [0:MAX_FLAT_BITS-1];

    initial begin
        $readmemh("addr_bits.mem",  addr_bits_rom);
        $readmemh("kept_bits.mem",  kept_bits_rom);
    end

    // Data Path Signals
    logic [SCORE_BITS-1:0]   scores [N_CLASSES];
    logic [COUNT_BITS-1:0]   class_count [N_CLASSES];
    logic [SCORE_BITS-1:0]   current_max_score;
    logic [CLASS_BITS-1:0]   current_max_idx;
    logic [CLASS_BITS-1:0]   search_idx;

    genvar gi;
    generate
        for (gi = 0; gi < N_CLASSES; gi++) begin : UNPACK
            assign class_count[gi] = bram_dout_b[gi*COUNT_BITS +: COUNT_BITS];
        end
    endgenerate

    // FSM
    integer k;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state             <= S_IDLE;
            lut_idx           <= '0;
            bit_offset        <= '0;
            bram_addr_b       <= '0;
            bram_en_b         <= 1'b0;
            done              <= 1'b0;
            predicted_class   <= '0;
            current_max_score <= '0;
            current_max_idx   <= '0;
            search_idx        <= '0;
            addr_build_idx    <= '0;
            tmp_addr_accum    <= '0;
            for (k = 0; k < N_CLASSES; k++) scores[k] <= '0;
        end
        else begin
            case (state)
                S_IDLE: begin
                    bram_en_b  <= 1'b0;
                    done       <= 1'b0;
                    bit_offset <= '0;
                    if (start) begin
                        lut_idx <= '0;
                        for (k = 0; k < N_CLASSES; k++) scores[k] <= '0;
                        state <= S_PREP_ADDR;
                    end
                end

                S_PREP_ADDR: begin
                     if (!enable_mask[lut_idx]) begin
                        bit_offset <= bit_offset + addr_bits_rom[lut_idx];
                        if (lut_idx == NUM_LUTS-1) begin
                            current_max_score <= scores[0];
                            current_max_idx   <= '0;
                            search_idx        <= 'd1;
                            state             <= S_FIND_MAX;
                        end else begin
                            lut_idx <= lut_idx + 1;
                            state   <= S_PREP_ADDR; 
                        end
                    end
                    else begin
                        // Load the bit indices for this LUT
                        for (int i = 0; i < MAX_ADDR_BITS; i++) begin
                             cached_bit_sels[i] <= kept_bits_rom[bit_offset + i];
                        end
                        // Prepare for sequential fetch
                        addr_build_idx <= '0;
                        tmp_addr_accum <= '0;
                        state          <= S_BUILD_ADDR;
                    end
                end

                // Fetch bits one by one from RAM
                S_BUILD_ADDR: begin
                    if (addr_build_idx < addr_bits_rom[lut_idx]) begin
                        // Get the global bit index
                        automatic logic [BIT_IDX_BITS-1:0] global_idx = cached_bit_sels[addr_build_idx];
                        // Calculate RAM Word Address (idx / 32) and Bit Offset (idx % 32)
                        automatic logic [9:0]  word_addr = global_idx[14:5]; // global_idx / 32
                        automatic logic [4:0]  bit_pos   = global_idx[4:0];  // global_idx % 32
                        
                        // Read from RAM (Distributed RAM is instant read in same cycle logic)
                        automatic logic target_bit = input_ram[word_addr][bit_pos];

                        // 4. Shift into accumulator
                        tmp_addr_accum <= (tmp_addr_accum << 1) | target_bit;
                        
                        addr_build_idx <= addr_build_idx + 1;
                    end else begin
                        // Done building address
                        eff_addr    <= tmp_addr_accum & addr_mask;
                        bram_addr_b <= {lut_idx, (tmp_addr_accum & addr_mask)};
                        bram_en_b   <= 1'b0; 
                        state       <= S_WAIT_1;
                    end
                end

                S_WAIT_1: begin
                    bram_en_b <= 1'b1; // Enable Weight BRAM
                    state     <= S_WAIT_2;
                end
                
                S_WAIT_2: begin
                    bram_en_b <= 1'b0;
                    state     <= S_ACCUM;
                end

                S_ACCUM: begin
                    for (k = 0; k < N_CLASSES; k++) begin
                        scores[k] <= scores[k] + class_count[k];
                    end

                    bit_offset <= bit_offset + addr_bits_rom[lut_idx];
                    if (lut_idx == NUM_LUTS-1) begin
                        state <= S_FIND_MAX;
                        search_idx <= '0;
                        current_max_score <= '0; 
                    end
                    else begin
                        lut_idx <= lut_idx + 1;
                        state   <= S_PREP_ADDR; 
                    end
                end

                S_FIND_MAX: begin
                    if (search_idx == 0) begin
                        current_max_score <= scores[0];
                        current_max_idx   <= '0;
                        search_idx        <= search_idx + 1;
                    end
                    else begin
                        if (scores[search_idx] > current_max_score) begin
                            current_max_score <= scores[search_idx];
                            current_max_idx   <= search_idx;
                        end

                        if (search_idx == N_CLASSES-1) begin
                            state <= S_DONE;
                        end else begin
                            search_idx <= search_idx + 1;
                        end
                    end
                end

                S_DONE: begin
                    predicted_class <= current_max_idx;
                    done            <= 1'b1;
                    if (start) begin
                        lut_idx <= '0;
                        bit_offset <= '0;
                        for (k = 0; k < N_CLASSES; k++) scores[k] <= '0;
                        done  <= 1'b0;
                        state <= S_PREP_ADDR; 
                    end
                end
            endcase
        end
    end

endmodule