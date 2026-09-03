import subprocess, sys, os, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL = os.path.join(REPO_ROOT, "barista-qa", "model", "model.bin")
PORT = os.environ.get("ESP32_PORT") or next(
    (os.path.join("/dev", f) for f in os.listdir("/dev") if f.startswith("cu.usbmodem")), None
)
if not PORT:
    sys.exit("No /dev/cu.usbmodem* device found. Plug in the board or set ESP32_PORT.")
ESPTOOL = os.path.expanduser("~/.platformio/packages/tool-esptoolpy/esptool.py")
PY = os.path.expanduser("~/.platformio/penv/bin/python3")
BASE_OFFSET = 0x210000
CHUNK_SIZE = 512 * 1024  # 512KB per chunk, smaller = less to lose per glitch

data = open(MODEL, "rb").read()
total = len(data)
print(f"Total size: {total} bytes, chunk size: {CHUNK_SIZE}")

offset = 0
chunk_dir = os.path.join(SCRIPT_DIR, "model_chunks")
os.makedirs(chunk_dir, exist_ok=True)

idx = 0
while offset < total:
    chunk = data[offset:offset+CHUNK_SIZE]
    chunk_path = os.path.join(chunk_dir, f"chunk_{idx}.bin")
    with open(chunk_path, "wb") as f:
        f.write(chunk)
    flash_addr = BASE_OFFSET + offset
    print(f"\n=== chunk {idx}: {len(chunk)} bytes @ 0x{flash_addr:x} ===")

    for attempt in range(1, 6):
        result = subprocess.run(
            [PY, ESPTOOL, "--chip", "esp32s3", "--port", PORT, "--baud", "460800",
             "write_flash", hex(flash_addr), chunk_path],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"  attempt {attempt}: OK")
            break
        else:
            print(f"  attempt {attempt}: FAILED ({result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown'})")
            time.sleep(2)
    else:
        print(f"Chunk {idx} failed after 5 attempts. Aborting.")
        sys.exit(1)

    offset += CHUNK_SIZE
    idx += 1

print("\nAll chunks flashed successfully.")
