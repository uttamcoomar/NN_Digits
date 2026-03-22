import serial, numpy as np, time
from torchvision import datasets, transforms

ds  = datasets.MNIST("training/data", train=False, download=True,
                     transform=transforms.ToTensor())
ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=10)
time.sleep(0.5)

correct = 0
n = 100

for i in range(n):
    img, label = ds[i]
    pix = (img.squeeze().numpy()*255).astype(np.uint8).flatten()
    ser.reset_input_buffer()
    ser.write(pix.tobytes())
    resp = ser.read(1)
    if resp:
        pred = resp[0]
        ok = (pred == label)
        correct += ok
        print(f"[{i+1:3d}] label={label} fpga={pred} {'✓' if ok else '✗'}")
    else:
        print(f"[{i+1:3d}] TIMEOUT")

print(f"\nAccuracy: {correct}/{n} = {correct}%")
ser.close()
