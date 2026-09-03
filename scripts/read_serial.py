import serial, time, os, sys

port = os.environ.get("ESP32_PORT") or next(
    (os.path.join("/dev", f) for f in os.listdir("/dev") if f.startswith("cu.usbmodem")), None
)
if not port:
    sys.exit("No /dev/cu.usbmodem* device found. Plug in the board or set ESP32_PORT.")

ser = serial.Serial(port, 115200, timeout=1)
# Trigger a hardware reset via RTS (same trick esptool uses for hard_reset)
ser.setDTR(False)
ser.setRTS(True)
time.sleep(0.1)
ser.setRTS(False)
time.sleep(0.1)

end = time.time() + 22
buf = b""
while time.time() < end:
    data = ser.read(4096)
    if data:
        buf += data
ser.close()
print(buf.decode(errors="replace"))
