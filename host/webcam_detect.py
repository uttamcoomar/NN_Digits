"""
webcam_detect.py — Point your laptop camera at a handwritten digit,
                   send it to the ECP5 FPGA, display the prediction.

Dependencies:
    pip install opencv-python pyserial numpy

Usage:
    python3 webcam_detect.py --port /dev/ttyUSB0

Controls (in the OpenCV window):
    SPACE  — capture current frame and run inference
    Q      — quit
"""

import argparse
import time
import sys

import cv2
import numpy as np
import serial


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Crop the central 60% square, Otsu-threshold to get digit on black background
    (MNIST style: white digit on black), resize to 28x28.
    Returns uint8 array shape (784,).
    """
    h, w  = frame.shape[:2]
    side  = int(min(h, w) * 0.6)
    y0    = (h - side) // 2
    x0    = (w - side) // 2
    crop  = frame[y0:y0+side, x0:x0+side]
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.resize(bw, (28, 28), interpolation=cv2.INTER_AREA).flatten().astype(np.uint8)


def infer_fpga(ser: serial.Serial, pixels: np.ndarray) -> tuple[int, float]:
    """Send 784 bytes to FPGA, receive 1 byte result."""
    ser.reset_input_buffer()
    t0 = time.perf_counter()
    ser.write(pixels.tobytes())
    resp = ser.read(1)
    ms = (time.perf_counter() - t0) * 1000
    if not resp:
        raise TimeoutError("No response — check FPGA is programmed and port is correct")
    return resp[0], ms


def main():
    ap = argparse.ArgumentParser(description="Webcam digit -> ECP5 FPGA inference")
    ap.add_argument("--port",    default="/dev/ttyUSB0")
    ap.add_argument("--baud",    type=int, default=115200)
    ap.add_argument("--cam",     type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
        time.sleep(0.1)
        print(f"Opened {args.port} at {args.baud} baud")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        ser.close()
        sys.exit(1)

    print("Hold a handwritten digit inside the GREEN box.")
    print("SPACE = infer   Q = quit\n")

    overlay = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pixels = preprocess(frame)

        # Draw crop guide
        h, w  = frame.shape[:2]
        side  = int(min(h, w) * 0.6)
        y0    = (h - side) // 2
        x0    = (w - side) // 2
        cv2.rectangle(frame, (x0, y0), (x0+side, y0+side), (0, 200, 0), 2)
        cv2.putText(frame, "SPACE=infer  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
        if overlay:
            cv2.putText(frame, overlay,
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 100), 3)

        # Small preview of exactly what gets sent to FPGA
        preview = cv2.resize(pixels.reshape(28, 28), (280, 280),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Camera", frame)
        cv2.imshow("28x28 (what FPGA sees)", cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            try:
                digit, ms = infer_fpga(ser, pixels)
                overlay = f"Digit: {digit}  ({ms:.0f} ms)"
                print(overlay)
            except TimeoutError as e:
                overlay = "Timeout!"
                print(e)

    cap.release()
    cv2.destroyAllWindows()
    ser.close()


if __name__ == "__main__":
    main()
