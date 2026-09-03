#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include "llm.h"

// ---- WiFi credentials ----
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// ---- Model files (flashed to SPIFFS via `pio run --target uploadfs`) ----
static char CHECKPOINT_PATH[] = "/spiffs/stories260K.bin";
static char TOKENIZER_PATH[] = "/spiffs/tok512.bin";

// ---- Generation settings ----
static const int MAX_STEPS = 80;          // cap generated length
static const int MAX_PROMPT_LEN = 200;    // cap input length
static const float TEMPERATURE = 0.8f;
static const float TOP_P = 0.9f;

static Transformer transformer;
static Tokenizer tokenizer;
static Sampler sampler;
static WebServer server(80);

static char response_buf[4096];
static char prompt_buf[MAX_PROMPT_LEN + 1];
static float last_tok_per_sec = 0.0f;

static SemaphoreHandle_t genDoneSem;

static void on_generate_complete(float tks) {
  last_tok_per_sec = tks;
}

// ---- Status LED (external white LED wired to GPIO38) ----
#define LED_STATUS_PIN 38

enum LedMode { LED_MODE_CONNECTING, LED_MODE_CONNECTED_IDLE, LED_MODE_THINKING, LED_MODE_ERROR };
static volatile LedMode currentLedMode = LED_MODE_CONNECTING;

static void led_task(void *param) {
  pinMode(LED_STATUS_PIN, OUTPUT);
  bool state = false;
  unsigned long lastToggle = 0;
  for (;;) {
    unsigned long interval;
    switch (currentLedMode) {
      case LED_MODE_CONNECTING:     interval = 120;  break; // fast blink while joining WiFi
      case LED_MODE_CONNECTED_IDLE: interval = 1000; break; // slow blink = online & idle
      case LED_MODE_THINKING:       interval = 150;  break; // fast blink = generating
      case LED_MODE_ERROR:          interval = 400;  break;
      default:                      interval = 500;  break;
    }
    unsigned long now = millis();
    if (now - lastToggle >= interval) {
      state = !state;
      digitalWrite(LED_STATUS_PIN, state ? HIGH : LOW);
      lastToggle = now;
    }
    vTaskDelay(pdMS_TO_TICKS(20));
  }
}

// ---- Generation task, pinned to Core 0 ----
// llm.c's helper matmul/forward tasks are hard-pinned to Core 1, so running
// the top-level generate() call from a task pinned to Core 0 lets each
// matmul split genuinely across both cores instead of competing on one.
static void generate_task(void *param) {
  unsigned long long seed = (unsigned long long)millis() ^ (unsigned long long)micros();
  build_sampler(&sampler, transformer.config.vocab_size, TEMPERATURE, TOP_P, seed);
  generate(&transformer, &tokenizer, &sampler, prompt_buf, MAX_STEPS, on_generate_complete);
  xSemaphoreGive(genDoneSem);
  vTaskDelete(NULL);
}

const char INDEX_HTML[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESP32 Tiny LLM</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; background:#111; color:#eee; }
  h1 { font-size: 1.3rem; }
  textarea { width: 100%; box-sizing: border-box; font-size: 1rem; padding: 10px; border-radius: 8px; border: 1px solid #444; background:#1c1c1c; color:#eee; }
  button { margin-top: 10px; padding: 10px 20px; font-size: 1rem; border-radius: 8px; border: none; background: #4a9eff; color: white; cursor: pointer; }
  button:disabled { background: #555; }
  #output { white-space: pre-wrap; margin-top: 16px; padding: 12px; background: #1c1c1c; border-radius: 8px; min-height: 60px; border: 1px solid #333; }
  #status { color: #888; font-size: 0.85rem; margin-top: 6px; }
</style>
</head>
<body>
<h1>ESP32-S3 Offline Tiny LLM</h1>
<p style="color:#aaa; font-size:0.9rem;">Runs fully on-device (no internet). Tiny model trained on children's stories &mdash; expect short story-style continuations, not factual answers.</p>
<textarea id="prompt" rows="3" placeholder="Type a story starter, e.g. 'Once upon a time'"></textarea><br>
<button id="sendBtn" onclick="send()">Send</button>
<div id="status"></div>
<div id="output"></div>
<script>
async function send() {
  const prompt = document.getElementById('prompt').value;
  const btn = document.getElementById('sendBtn');
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  btn.disabled = true;
  status.textContent = 'Generating on-device... (this can take a while on an ESP32)';
  output.textContent = '';
  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'prompt=' + encodeURIComponent(prompt)
    });
    const text = await res.text();
    output.textContent = text;
    status.textContent = res.headers.get('X-Tok-Per-Sec')
      ? ('Done. ' + res.headers.get('X-Tok-Per-Sec') + ' tok/s') : 'Done.';
  } catch (e) {
    status.textContent = 'Error: ' + e;
  }
  btn.disabled = false;
}
</script>
</body>
</html>
)HTML";

static void handle_root() {
  server.send(200, "text/html", INDEX_HTML);
}

static void handle_generate() {
  String prompt = server.hasArg("prompt") ? server.arg("prompt") : "";
  if (prompt.length() > MAX_PROMPT_LEN) {
    prompt = prompt.substring(0, MAX_PROMPT_LEN);
  }
  prompt.toCharArray(prompt_buf, sizeof(prompt_buf));

  g_llm_output_buf = response_buf;
  g_llm_output_cap = sizeof(response_buf);
  g_llm_output_len = 0;
  response_buf[0] = '\0';

  currentLedMode = LED_MODE_THINKING;

  // Run the heavy computation on a task pinned to Core 0, and wait here
  // (yielding, not busy-spinning) until it signals completion.
  xTaskCreatePinnedToCore(generate_task, "GenTask", 8192, NULL, 5, NULL, 0);
  xSemaphoreTake(genDoneSem, portMAX_DELAY);

  currentLedMode = LED_MODE_CONNECTED_IDLE;

  server.sendHeader("X-Tok-Per-Sec", String(last_tok_per_sec, 2));
  server.send(200, "text/plain", response_buf);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("\nBooting ESP32 Tiny LLM...");

  genDoneSem = xSemaphoreCreateBinary();

  xTaskCreatePinnedToCore(led_task, "LedTask", 2048, NULL, 1, NULL, 1);

  if (!SPIFFS.begin(false)) {
    Serial.println("SPIFFS mount failed! Did you run `pio run --target uploadfs`?");
    currentLedMode = LED_MODE_ERROR;
  }

  Serial.printf("PSRAM found: %s, free PSRAM: %u bytes, free heap: %u bytes\n",
                psramFound() ? "yes" : "no", ESP.getFreePsram(), ESP.getFreeHeap());

  Serial.println("Loading model...");
  build_transformer(&transformer, CHECKPOINT_PATH);
  build_tokenizer(&tokenizer, TOKENIZER_PATH, transformer.config.vocab_size);
  build_sampler(&sampler, transformer.config.vocab_size, TEMPERATURE, TOP_P, 42);
  Serial.println("Model loaded.");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to WiFi \"%s\"", WIFI_SSID);
  currentLedMode = LED_MODE_CONNECTING;
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(400);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    currentLedMode = LED_MODE_CONNECTED_IDLE;
    Serial.print("Connected! Open this in your laptop's browser: http://");
    Serial.println(WiFi.localIP());
  } else {
    currentLedMode = LED_MODE_ERROR;
    Serial.println("WiFi connection failed. Check SSID/password and retry.");
  }

  server.on("/", HTTP_GET, handle_root);
  server.on("/generate", HTTP_POST, handle_generate);
  server.begin();
  Serial.println("HTTP server started.");
}

void loop() {
  server.handleClient();
}
