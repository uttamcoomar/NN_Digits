// uart_tx.v — 8N1 UART transmitter
`default_nettype none

module uart_tx #(
    parameter CLK_HZ = 25_000_000,
    parameter BAUD   = 115_200
)(
    input  wire       clk, rst,
    input  wire [7:0] tx_data,
    input  wire       tx_start,
    output reg        tx_busy,
    output reg        tx_pin
);
    localparam CLKS_PER_BIT = CLK_HZ / BAUD;
    localparam CTR_W        = $clog2(CLKS_PER_BIT + 1);

    localparam S_IDLE=0, S_START=1, S_DATA=2, S_STOP=3;
    reg [1:0]       state;
    reg [CTR_W-1:0] ctr;
    reg [2:0]       bit_idx;
    reg [7:0]       shift;

    always @(posedge clk) begin
        if (rst) begin state<=S_IDLE; tx_pin<=1; tx_busy<=0; end
        else case (state)
            S_IDLE: begin tx_pin<=1; tx_busy<=0;
                if (tx_start) begin shift<=tx_data; ctr<=1;
                    tx_busy<=1; tx_pin<=0; state<=S_START; end end
            S_START: if (ctr==CLKS_PER_BIT) begin
                        ctr<=1; bit_idx<=0; tx_pin<=shift[0]; state<=S_DATA;
                     end else ctr<=ctr+1;
            S_DATA:  if (ctr==CLKS_PER_BIT) begin
                        ctr<=1;
                        if (bit_idx==7) begin tx_pin<=1; state<=S_STOP; end
                        else begin bit_idx<=bit_idx+1; tx_pin<=shift[bit_idx+1]; end
                     end else ctr<=ctr+1;
            S_STOP:  if (ctr==CLKS_PER_BIT) begin
                        tx_busy<=0; ctr<=0; state<=S_IDLE;
                     end else ctr<=ctr+1;
        endcase
    end
endmodule

`default_nettype wire
