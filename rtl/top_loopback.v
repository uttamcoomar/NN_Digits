// top_loopback.v — UART echo test
// Receives any byte and immediately echoes it back.
// If this works, UART hardware is fine and the problem is in the MLP.
// If this also times out, the problem is UART pin mapping or clock.

`default_nettype none

module top (
    input  wire       clk_25mhz,
    input  wire       btn_pwr,
    input  wire       ftdi_txd,
    output wire       ftdi_rxd,
    output wire [3:0] led
);
    wire clk = clk_25mhz;
    wire rst = ~btn_pwr;

    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(.CLK_HZ(25_000_000), .BAUD(115_200)) u_rx (
        .clk(clk), .rst(rst),
        .rx_pin(ftdi_txd),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );

    wire tx_busy;
    uart_tx #(.CLK_HZ(25_000_000), .BAUD(115_200)) u_tx (
        .clk(clk), .rst(rst),
        .tx_data(rx_data),
        .tx_start(rx_valid),
        .tx_busy(tx_busy),
        .tx_pin(ftdi_rxd)
    );

    // LED shows last received byte lower 4 bits
    reg [3:0] led_r;
    always @(posedge clk)
        if (rx_valid) led_r <= rx_data[3:0];
    assign led = led_r;

endmodule

`default_nettype wire
