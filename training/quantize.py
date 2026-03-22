"""
quantize.py — Export trained PyTorch MLP weights as $readmemh hex files.

Run AFTER train.py:
    python train.py        # produces weights/mlp_mnist.pt
    python quantize.py     # produces ../rtl/hex/*.hex

Fixed-point formats
-------------------
  Weights : Q4.12  signed 16-bit   scale = 4096
  Biases  : Q8.8   signed 16-bit   scale = 256
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64,  32)
        self.fc3 = nn.Linear(32,  10)
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def to_fixed16(arr, frac_bits):
    scale   = 2 ** frac_bits
    clipped = np.clip(arr * scale, -(2**15), 2**15 - 1)
    return clipped.round().astype(np.int16)


def write_hex(arr_i16, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in arr_i16.flatten():
            f.write(f"{int(v) & 0xFFFF:04x}\n")
    print(f"  {path}  ({arr_i16.size} entries)")


def main():
    pt_file  = "weights/mlp_mnist.pt"
    hex_dir  = "../rtl/hex"

    if not os.path.exists(pt_file):
        print(f"ERROR: {pt_file} not found — run train.py first")
        sys.exit(1)

    model = MLP()
    model.load_state_dict(torch.load(pt_file, map_location="cpu"))
    model.eval()

    layers = [
        (model.fc1.weight, model.fc1.bias, "l1"),
        (model.fc2.weight, model.fc2.bias, "l2"),
        (model.fc3.weight, model.fc3.bias, "l3"),
    ]

    print("Exporting hex files...")
    for W_t, b_t, tag in layers:
        W = W_t.detach().numpy().astype(np.float64)
        b = b_t.detach().numpy().astype(np.float64)
        write_hex(to_fixed16(W, 12), f"{hex_dir}/weight_{tag}.hex")
        write_hex(to_fixed16(b,  8), f"{hex_dir}/bias_{tag}.hex")

    print("Done. Hex files in", hex_dir)


if __name__ == "__main__":
    main()
