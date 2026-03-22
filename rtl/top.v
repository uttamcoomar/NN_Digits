// top.v — ULX3S top-level (25 MHz, 115200 baud)
`default_nettype none

module top #(
    parameter CLK_HZ = 25_000_000,
    parameter BAUD   = 115_200
)(
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

    uart_rx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_rx (
        .clk(clk), .rst(rst),
        .rx_pin(ftdi_txd),
        .rx_data(rx_data), .rx_valid(rx_valid)
    );

    reg [7:0] pix_buf [0:783];
    reg [9:0] wr_ptr;
    reg       start;

    always @(posedge clk) begin
        start <= 0;
        if (rst) wr_ptr <= 0;
        else if (rx_valid) begin
            pix_buf[wr_ptr] <= rx_data;
            if (wr_ptr == 783) begin wr_ptr<=0; start<=1; end
            else wr_ptr <= wr_ptr+1;
        end
    end

    wire [9:0] pix_addr;
    wire [3:0] result;
    wire       done;

    reg [7:0] pix_data;
    always @(posedge clk) pix_data <= pix_buf[pix_addr];

    mlp_accel u_mlp (
        .clk(clk), .rst(rst),
        .start(start), .done(done),
        .result(result),
        .pix_addr(pix_addr),
        .pix_data(pix_data)
    );

    reg [3:0] result_r;
    reg       tx_start_r;
    always @(posedge clk) begin
        tx_start_r <= 0;
        if (done) begin result_r<=result; tx_start_r<=1; end
    end

    wire tx_busy;
    uart_tx #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) u_tx (
        .clk(clk), .rst(rst),
        .tx_data({4'b0, result_r}),
        .tx_start(tx_start_r),
        .tx_busy(tx_busy),
        .tx_pin(ftdi_rxd)
    );

    assign led = result_r;

endmodule

`default_nettype wire
