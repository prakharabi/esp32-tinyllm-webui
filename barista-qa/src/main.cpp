// ESP32-S3 Barista over WiFi: a browser page replaces the Serial console from
// the original sketch. Inference core (PLE transformer, int8-staged dual-core
// matvec, BPE tokenizer) is unchanged from slvDev/esp32-ai's esp32_barista.ino.

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_timer.h"

#define LLM_INT8_ACT 1
#define BARISTA_DUAL_CORE 1
#include "runtime/llm.h"
#include "runtime/bpe_tokenizer.h"
#include "generated/tokenizer_encoder.h"
#include "generated/barista_words.h"
#include "generated/barista_out2in.h"

// ---- WiFi credentials ----
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

static Model model;
static Scratch scratch;
static BpeTokenizer tokenizer;
static bool ready = false;
static WebServer server(80);

// ---- dual-core per-layer matvec (unchanged from esp32_barista.ino) --------
static TaskHandle_t worker_h, main_h;
static const QT *job_t; static const int8_t *job_xq; static float job_xs;
static float *job_y; static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}
static void layer_matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);
  job_t=t; job_xq=xq; job_xs=xs; job_y=y; job_split=t->rows/2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

static int sram_fallbacks = 0;
static void *ps(size_t n) { return heap_caps_malloc(n, MALLOC_CAP_SPIRAM); }
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) { Serial.printf("PSRAM alloc failed: %s (%u)\n", what, (unsigned)n); while (1) delay(1000); }
  return p;
}
#define BARISTA_SCRATCH_IN_SRAM 1
static void *sram_or_ps(size_t n, const char *what) {
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (p) return p;
  ++sram_fallbacks;
  Serial.printf("SRAM full, %s -> PSRAM (%u B)\n", what, (unsigned)n);
  return ps_or_die(n, what);
}

static void copy_norms_to_sram() {
  Cfg *c = &model.c; int D = c->dim, L = c->n_layers, P = c->ple_dim;
  size_t before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  const float **vecs[3 * 32 + 2]; int n_vec = 0, sizes[3 * 32 + 2];
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  int moved = 0;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = heap_caps_malloc(bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!dst) { ++sram_fallbacks; continue; }
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
    ++moved;
  }
  Serial.printf("norms in SRAM: %d/%d vectors, %u B\n", moved, n_vec,
                (unsigned)(before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
}

static void alloc_scratch() {
  Cfg *c=&model.c; int D=c->dim,L=c->n_layers,P=c->ple_dim,F=c->ffn,S=c->seq_len;
  size_t before = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  scratch.x=(float*)sram_or_ps(D*4,"x");            scratch.h=(float*)sram_or_ps((F>D?F:D)*4,"h");
  scratch.qkv=(float*)sram_or_ps(3*D*4,"qkv");      scratch.att=(float*)sram_or_ps(D*4,"att");
  scratch.g1=(float*)sram_or_ps(F*4,"g1");          scratch.g2=(float*)sram_or_ps((P>F?P:F)*4,"g2");
  scratch.ple=(float*)sram_or_ps(L*P*4,"ple");      scratch.tmpP=(float*)sram_or_ps(L*P*4,"tmpP");
  scratch.trow=(float*)sram_or_ps(L*P*4,"trow");
  scratch.logits=(float*)sram_or_ps(model.out_vocab*4,"logits");
  scratch.scores=(float*)sram_or_ps(S*4,"scores");
  scratch.kcache=(float*)ps_or_die((size_t)L*S*D*4,"kcache");
  scratch.vcache=(float*)ps_or_die((size_t)L*S*D*4,"vcache");
  Serial.printf("scratch in SRAM: %u B\n",
                (unsigned)(before - heap_caps_get_free_size(MALLOC_CAP_INTERNAL)));
  copy_norms_to_sram();
  Serial.printf("sram free %.0f KB\n",
                heap_caps_get_free_size(MALLOC_CAP_INTERNAL)/1024.0);
}

// ---- Status LED (external white LED wired to GPIO38), PWM-smoothed --------
#define LED_STATUS_PIN 38
#define LEDC_CHANNEL 0
#define LEDC_FREQ 5000
#define LEDC_RES_BITS 8

enum LedMode { LED_MODE_CONNECTING, LED_MODE_CONNECTED_IDLE, LED_MODE_THINKING, LED_MODE_ERROR };
static volatile LedMode currentLedMode = LED_MODE_CONNECTING;

static void led_task(void *param) {
  ledcSetup(LEDC_CHANNEL, LEDC_FREQ, LEDC_RES_BITS);
  ledcAttachPin(LED_STATUS_PIN, LEDC_CHANNEL);
  float phase = 0.0f;
  const float tickMs = 20.0f;
  for (;;) {
    float periodMs; uint8_t maxDuty; // dimmed: well under full 255 duty
    switch (currentLedMode) {
      case LED_MODE_CONNECTING:     periodMs = 500.0f;  maxDuty = 110; break;
      case LED_MODE_CONNECTED_IDLE: periodMs = 2200.0f; maxDuty = 85;  break;
      case LED_MODE_THINKING:       periodMs = 450.0f;  maxDuty = 130; break;
      case LED_MODE_ERROR:          periodMs = 250.0f;  maxDuty = 130; break;
      default:                      periodMs = 1000.0f; maxDuty = 85;  break;
    }
    phase += (2.0f * (float)M_PI * tickMs) / periodMs;
    if (phase > 2.0f * (float)M_PI) phase -= 2.0f * (float)M_PI;
    float brightness01 = (sinf(phase) + 1.0f) * 0.5f; // smooth 0..1 breathing
    ledcWrite(LEDC_CHANNEL, (uint8_t)(brightness01 * maxDuty));
    vTaskDelay(pdMS_TO_TICKS((int)tickMs));
  }
}

// ---- Generation, adapted from esp32_barista.ino's answer() ----------------
// Captures pieces into a buffer for the HTTP response instead of Serial, and
// reports resource usage (time, pieces, heap/psram) alongside the answer.
#define BARISTA_ANSWER_ROOM 56
#define BARISTA_MAX_PIECES 48

struct GenStats {
  int pieces;
  double ms;
  double pieces_per_sec;
  size_t free_heap;
  size_t free_psram;
};

// Result codes for ask(): distinguish "model declined" from "model answered"
enum AskResult { ASK_OK, ASK_NOT_ASCII, ASK_TOO_LONG };

static AskResult ask(const char *question, char *out_buf, size_t out_cap, GenStats *stats) {
  uint16_t ids[BTK_MAX_INPUT_BYTES];
  int n = bpe_encode_ascii(&tokenizer, question, ids, (int)(sizeof(ids)/sizeof(ids[0])));
  if (n == BTK_ERR_NOT_ASCII) return ASK_NOT_ASCII;
  if (n <= 0 || n > model.c.seq_len - BARISTA_ANSWER_ROOM) return ASK_TOO_LONG;

  int64_t t0 = esp_timer_get_time();
  int pos = 0;
  for (int i = 0; i < n; i++) {
    llm_forward(&model, ids[i], pos++, &scratch);
    vTaskDelay(1); // yield periodically so the idle task never starves (no WDT trips)
  }
  llm_forward(&model, BARISTA_OUT2IN[BARISTA_BOS], pos++, &scratch);

  size_t out_len = 0;
  out_buf[0] = '\0';
  int pieces_out = 0;
  for (int step = 0; step < BARISTA_MAX_PIECES && pos < model.c.seq_len; step++) {
    int best = 0;
    for (int k = 1; k < BARISTA_WORD_COUNT; k++)
      if (scratch.logits[k] > scratch.logits[best]) best = k;
    if (best == BARISTA_EOS) break;
    const char *w = BARISTA_WORDS[best];
    bool punct = (w[1] == '\0' && strchr(".,:;?", w[0]) != NULL);
    if (pieces_out && !punct && out_len + 1 < out_cap) out_buf[out_len++] = ' ';
    size_t wlen = strlen(w);
    if (out_len + wlen < out_cap) { memcpy(out_buf + out_len, w, wlen); out_len += wlen; }
    out_buf[out_len] = '\0';
    pieces_out++;
    llm_forward(&model, BARISTA_OUT2IN[best], pos++, &scratch);
    vTaskDelay(1); // one yield per generated piece
  }
  double ms = (esp_timer_get_time() - t0) / 1000.0;

  stats->pieces = pieces_out;
  stats->ms = ms;
  stats->pieces_per_sec = ms > 0 ? pieces_out * 1000.0 / ms : 0;
  stats->free_heap = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
  stats->free_psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
  return ASK_OK;
}

// ---- Web UI -----------------------------------------------------------------
static char answer_buf[2048];
static char question_buf[BTK_MAX_INPUT_BYTES];

const char INDEX_HTML[] PROGMEM = R"HTML(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ESP32 Barista</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; background:#111; color:#eee; }
  h1 { font-size: 1.3rem; }
  textarea { width: 100%; box-sizing: border-box; font-size: 1rem; padding: 10px; border-radius: 8px; border: 1px solid #444; background:#1c1c1c; color:#eee; }
  button { margin-top: 10px; padding: 10px 20px; font-size: 1rem; border-radius: 8px; border: none; background: #4a9eff; color: white; cursor: pointer; }
  button:disabled { background: #555; }
  #output { white-space: pre-wrap; margin-top: 16px; padding: 12px; background: #1c1c1c; border-radius: 8px; min-height: 40px; border: 1px solid #333; }
  #stats { color: #888; font-size: 0.8rem; margin-top: 8px; white-space: pre-wrap; }
  #status { color: #888; font-size: 0.85rem; margin-top: 6px; }
</style>
</head>
<body>
<h1>ESP32-S3 Offline Barista</h1>
<p style="color:#aaa; font-size:0.9rem;">8.9M-parameter model running fully on-device (no internet). Trained on espresso Q&amp;A &mdash; ask about grind, dose, extraction, milk, etc.</p>
<textarea id="prompt" rows="2" placeholder="e.g. Why is my espresso sour?"></textarea><br>
<button id="sendBtn" onclick="send()">Send</button>
<div id="status"></div>
<div id="output"></div>
<div id="stats"></div>
<script>
async function send() {
  const prompt = document.getElementById('prompt').value;
  const btn = document.getElementById('sendBtn');
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  const stats = document.getElementById('stats');
  btn.disabled = true;
  status.textContent = 'Generating on-device...';
  output.textContent = '';
  stats.textContent = '';
  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'question=' + encodeURIComponent(prompt)
    });
    const data = await res.json();
    if (data.error) {
      output.textContent = '(' + data.error + ')';
    } else {
      output.textContent = data.answer;
      stats.textContent =
        `${data.pieces} pieces in ${data.ms.toFixed(0)} ms (${data.pieces_per_sec.toFixed(1)} pieces/s)\n` +
        `free heap: ${(data.free_heap/1024).toFixed(1)} KB   free psram: ${(data.free_psram/1024/1024).toFixed(2)} MB`;
    }
    status.textContent = 'Done.';
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

static String json_escape(const char *s) {
  String out;
  for (const char *p = s; *p; p++) {
    if (*p == '"' || *p == '\\') out += '\\';
    if (*p == '\n') { out += "\\n"; continue; }
    out += *p;
  }
  return out;
}

static void handle_generate() {
  String q = server.hasArg("question") ? server.arg("question") : "";
  if (q.length() > BTK_MAX_INPUT_BYTES - 1) q = q.substring(0, BTK_MAX_INPUT_BYTES - 1);
  q.toCharArray(question_buf, sizeof(question_buf));

  currentLedMode = LED_MODE_THINKING;
  GenStats stats;
  AskResult r = ask(question_buf, answer_buf, sizeof(answer_buf), &stats);
  currentLedMode = LED_MODE_CONNECTED_IDLE;

  if (r == ASK_NOT_ASCII) {
    server.send(200, "application/json", "{\"error\":\"ASCII only\"}");
    return;
  }
  if (r == ASK_TOO_LONG) {
    server.send(200, "application/json", "{\"error\":\"question too long\"}");
    return;
  }

  String json = "{";
  json += "\"answer\":\"" + json_escape(answer_buf) + "\",";
  json += "\"pieces\":" + String(stats.pieces) + ",";
  json += "\"ms\":" + String(stats.ms, 1) + ",";
  json += "\"pieces_per_sec\":" + String(stats.pieces_per_sec, 2) + ",";
  json += "\"free_heap\":" + String((unsigned long)stats.free_heap) + ",";
  json += "\"free_psram\":" + String((unsigned long)stats.free_psram);
  json += "}";
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 BARISTA (WiFi) ===");

  xTaskCreatePinnedToCore(led_task, "LedTask", 2048, NULL, 1, NULL, 1);

  Serial.printf("PSRAM free: %u bytes, heap free: %u bytes\n",
                ESP.getFreePsram(), ESP.getFreeHeap());

  if (bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE, &tokenizer)) {
    Serial.println("tokenizer load failed"); currentLedMode = LED_MODE_ERROR; while (1) delay(1000);
  }

  const esp_partition_t *part = esp_partition_find_first(
      ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
  if (!part) { Serial.println("no model partition"); currentLedMode = LED_MODE_ERROR; while (1) delay(1000); }
  const void *base; spi_flash_mmap_handle_t h;
  if (esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &base, &h) != ESP_OK) {
    Serial.println("mmap failed"); currentLedMode = LED_MODE_ERROR; while (1) delay(1000);
  }
  if (llm_load((const uint8_t*)base, &model)) {
    Serial.println("bad model"); currentLedMode = LED_MODE_ERROR; while (1) delay(1000);
  }
  Cfg *c = &model.c;
  Serial.printf("model: Vin=%d Vout=%d D=%d L=%d H=%d F=%d P=%d\n",
                c->vocab, model.out_vocab, c->dim, c->n_layers, c->n_heads, c->ffn, c->ple_dim);
  if (model.out_vocab != BARISTA_WORD_COUNT) {
    Serial.printf("word table mismatch: model %d vs table %d\n", model.out_vocab, BARISTA_WORD_COUNT);
    currentLedMode = LED_MODE_ERROR; while (1) delay(1000);
  }
  if ((int)tokenizer.active_vocab > c->vocab) {
    Serial.println("tokenizer/model mismatch"); currentLedMode = LED_MODE_ERROR; while (1) delay(1000);
  }
  {
    uint16_t max_in = 0;
    for (int k = 0; k < BARISTA_WORD_COUNT; k++)
      if (BARISTA_OUT2IN[k] > max_in) max_in = BARISTA_OUT2IN[k];
    if ((int)max_in + 1 != c->vocab) {
      Serial.println("out2in/model mismatch"); currentLedMode = LED_MODE_ERROR; while (1) delay(1000);
    }
  }

  alloc_scratch();

  int staged = llm_stage_core_int8_alloc(&model, ps);
  {
    void *b = ps(llm_stage_int8_bytes(&model.out_head));
    if (b) { llm_stage_int8(&model.out_head, b); ++staged; }
  }
  Serial.printf("int8-staged %d tensors | psram free %.2f MB\n",
                staged, heap_caps_get_free_size(MALLOC_CAP_SPIRAM)/1048576.0);

  int dual_core_active = 0;
#if BARISTA_DUAL_CORE
  main_h = xTaskGetCurrentTaskHandle();
  if (xTaskCreatePinnedToCore(worker_main, "mv", 4096, NULL, 2, &worker_h, 0) == pdPASS) {
    model.layer_matvec = layer_matvec_par;
    if (model.out_head.w8) model.head_matvec = layer_matvec_par;
    dual_core_active = 1;
  } else {
    Serial.println("dual-core worker failed; running single core");
  }
#endif
  Serial.printf("config: dual_core_active=%d\n", dual_core_active);

  ready = true;

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
  Serial.println("READY>");
}

void loop() {
  server.handleClient();
}
