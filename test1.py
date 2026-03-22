import serial, numpy as np, time
from torchvision import datasets, transforms

ds = datasets.MNIST("training/data", train=False, download=True,
                    transform=transforms.ToTensor())
img, label = ds[0]
pix = (img.squeeze().numpy()*255).astype(np.uint8).flatten()

ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=10)
time.sleep(0.5)
ser.reset_input_buffer()

print(f"Sending 784 bytes (label={label})...")
ser.write(pix.tobytes())

resp = ser.read(1)
if resp:
    print(f"FPGA={resp[0]}  {'PASS' if resp[0]==label else 'FAIL'}")
else:
    print("TIMEOUT")
ser.close()
