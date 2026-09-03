# ESP32-S3 Offline Tiny LLM + Web UI

A small language model running fully on-device on an ESP32-S3 N16R8 (16MB
flash, 8MB octal PSRAM), with a minimal web page (served by the ESP32 itself
over WiFi) for typing a prompt and getting a generated reply back — no cloud,
no internet required at inference time. Mostly a "how far can this chip
actually go" experiment.

Two model variants, in separate PlatformIO projects:

## `tinystories-260k/` — confirmed working end-to-end

- Engine: [llama2.c](https://github.com/karpathy/llama2.c) (Karpathy, MIT),
  ported to ESP32 by [DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm),
  adapted here to drop the OLED/ESP-DSP dependencies and add WiFi + a web UI.
- Model: 260K-parameter checkpoint trained on the
  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  dataset. Generates short story-style text continuations — not factual
  Q&A.
- Status: boots, joins WiFi, serves the page, generates text. Verified via
  serial monitor and a live request.
- Extras: status LED on GPIO38 (fast blink while connecting to WiFi, slow
  blink when idle/connected, fast blink while generating — see `main.cpp`),
  dual-core split (helper matmul/forward tasks pinned to Core 1, the
  generate() call runs from a task pinned to Core 0), per-token `vTaskDelay`
  yield so the Task Watchdog never starves.

Build/flash:
```bash
cd tinystories-260k
pio run --target upload      # firmware
pio run --target uploadfs    # model + tokenizer -> SPIFFS partition
```

## `barista-qa/` — bigger model, actually answers questions (in progress)

- Engine: [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) (MIT), the
  "Barista" model — an 8.9M-parameter model using Per-Layer Embeddings (most
  weights sit in a flash-mapped lookup table, only ~450 bytes read per token)
  so it fits despite being much bigger than the TinyStories checkpoint.
- Model: trained specifically to answer **espresso/coffee questions** (not
  general knowledge) — a genuinely different, QA-capable model rather than
  just story completion. Weights: `slvDev/esp32-ai-barista` on Hugging Face,
  SHA-256 verified at download time
  (`1359a1cb74de4143d630c2c192990de814cd47255bcdfa9cc135f07ef0a39fc4`).
- `src/runtime/`, `src/generated/` are copied/generated from the upstream
  project's tools (`generate_vocab_headers.py`, `generate_tokenizer_header.py`
  — both pure Python stdlib, no extra deps needed).
- `main.cpp` merges the upstream `answer()` inference loop with our own WiFi +
  web server layer, capturing generated output into a buffer for the HTTP
  response instead of only printing to serial.
- Status: **compiles cleanly** (781KB firmware, fits the 2MB app partition).
  Model partition flashing was flaky over the direct USB-CDC link at full
  speed, so `scripts/flash_model_chunked.py` flashes it in 512KB chunks with
  retries — that script exists but the model partition had not been fully
  verified as correctly flashed and boot-tested end-to-end before this was
  moved into version control. Treat this project as "builds, not yet fully
  hardware-validated."

Build/flash:
```bash
cd barista-qa
pio run                                    # compile-only check
python3 ../scripts/flash_model_chunked.py  # flash model.bin to the model partition (chunked, retries)
pio run --target upload                    # flash firmware
python3 ../scripts/read_serial.py          # watch boot output
```

## Hardware

- Board: ESP32-S3 N16R8 (16MB flash, 8MB octal PSRAM), native USB.
- Onboard RGB LED (GPIO38 per this board's actual wiring, confirmed by
  probing — silkscreen/generic docs suggested other pins that didn't work)
  turned out to be dead on the physical board used during development; an
  external LED on GPIO38 is used as the status indicator instead.

## Setup

Both sketches read WiFi credentials from two constants near the top of
`src/main.cpp`:

```cpp
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
```

Fill in your own network's name/password locally before flashing. Don't
commit real credentials back to this repo — it's public.

## Known caveats

- The engines here only support plain fp32 (tinystories) or the Barista
  project's own int4/int8 quantized format — there's no generic quantized
  llama2.c support, which is why a bigger vanilla llama2.c checkpoint (15M+
  params, ~60MB fp32) doesn't fit in 16MB of flash regardless of latency
  tolerance. The Barista PLE approach exists specifically to route around
  that ceiling.

## Credits

- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) — MIT
- [DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm)
- [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) — MIT
- [TinyStories dataset](https://arxiv.org/abs/2305.07759) — Eldan & Li,
  Microsoft Research

## License

MIT — see [LICENSE](LICENSE). Vendored files from the projects above keep
their original license/attribution.
