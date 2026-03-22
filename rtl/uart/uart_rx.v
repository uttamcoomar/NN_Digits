// uart_rx.v — 8N1 UART receiver
`default_nettype none

module uart_rx #(
    parameter CLK_HZ = 25_000_000,
    parameter BAUD   = 115_200
)(
    input  wire       clk, rst,
    input  wire       rx_pin,
    output reg  [7:0] rx_data,
    output reg        rx_valid
);
    localparam CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam HALF_BIT     = CLKS_PER_BIT / 2;
    localparam CTR_W        = $clog2(CLKS_PER_BIT + 1);

    reg rx_meta, rx_sync;
    always @(posedge clk) begin rx_meta <= rx_pin; rx_sync <= rx_meta; end

    localparam S_IDLE=0, S_START=1, S_DATA=2, S_STOP=3;
    reg [1:0]         state;
    reg [CTR_W-1:0]   ctr;
    reg [2:0]         bit_idx;
    reg [7:0]         shift;

    always @(posedge clk) begin
        rx_valid <= 0;
        if (rst) begin state <= S_IDLE; ctr <= 0; end
        else case (state)
            S_IDLE:  if (!rx_sync) begin ctr <= 1; state <= S_START; end
            S_START: if (ctr == HALF_BIT) begin
                        if (!rx_sync) begin ctr<=1; bit_idx<=0; state<=S_DATA; end
                        else state <= S_IDLE;
                     end else ctr <= ctr+1;
            S_DATA:  if (ctr == CLKS_PER_BIT) begin
                        ctr <= 1; shift[bit_idx] <= rx_sync;
                        if (bit_idx==7) begin bit_idx<=0; state<=S_STOP; end
                        else bit_idx <= bit_idx+1;
                     end else ctr <= ctr+1;
            S_STOP:  if (ctr == CLKS_PER_BIT) begin
                        if (rx_sync) begin rx_data<=shift; rx_valid<=1; end
                        state<=S_IDLE; ctr<=0;
                     end else ctr <= ctr+1;
        endcase
    end
endmodule

`default_nettype wire
