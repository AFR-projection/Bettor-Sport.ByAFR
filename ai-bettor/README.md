# AI Bettor

Autonomous AI sports betting analysis agent berbasis LLM + quantitative engine.

Sistem ini **TIDAK mengklaim bisa memprediksi kemenangan dengan pasti**.
Target: mencari **probabilistic edge**, **positive expected value**, dan **risk-adjusted opportunity**.
Jika tidak ada edge yang cukup kuat, agent memilih **NO BET**.

## Pipeline

```
DATA → VALIDATION → NORMALIZATION → STATISTICAL MODEL → SIMULATION
→ MARKET ANALYSIS → PROBABILITY → EV → RISK → MULTI-AGENT REVIEW
→ BETTOR BRAIN → BET / NO BET → TELEGRAM
```

Bukan: `LLM → TEBAKAN → BET`.

## Agents

| Agent | Tugas |
|---|---|
| DATA SCOUT | Fetch fixtures & odds dari The Odds API, validasi, normalisasi, data quality score |
| QUANT ANALYST | Implied probability, edge, EV, confidence score |
| MARKET ANALYST | Bandingkan odds antar bookmaker, best price, consensus, line movement |
| SIMULATION ANALYST | Monte Carlo (default 20,000 runs, reproducible via RANDOM_SEED) |
| RISK MANAGER | Uncertainty, exposure, drawdown, correlation — bisa **veto** (NO BET) |
| BETTOR BRAIN | Final decision BET / NO BET + stake (modified Kelly) |

## Quick Start (lokal)

### 1. Konfigurasi

```bash
cp deploy/env.production.example .env
# isi API_TOKEN, DATABASE_URL (Neon), dan API key
```

Untuk development cepat cukup dua baris ini di `.env`:

```
DATABASE_URL=sqlite:///./ai_bettor_dev.db
THE_ODDS_API_KEYS=<key kamu>
```

Tanpa `API_TOKEN`, API terbuka penuh (aman untuk localhost, **tidak** untuk VPS —
lihat [Deploy ke VPS](#deploy-ke-vps)).

### 2. Instalasi & Jalankan (tanpa Docker)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux: . .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + pytest; produksi: requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Buka `http://localhost:8000` (dashboard) atau `/docs` (Swagger).

Skema database dibuat otomatis saat startup (`init_db()`: `create_all` +
penambahan kolom baru lewat `ALTER TABLE`), jadi tidak ada langkah migrasi wajib.

### 3. Docker

```bash
cp deploy/env.production.example .env   # wajib: DATABASE_URL + API_TOKEN
docker compose up -d --build
docker compose logs -f
```

Compose hanya menjalankan **satu** service (aplikasi). Database ada di Neon, dan
port aplikasi di-bind ke `127.0.0.1:8000` supaya hanya bisa diakses lewat Nginx.

### 4. Test

```bash
python -m pytest -q          # 274 test
```

## Database: Neon (serverless PostgreSQL)

1. Buat project di [neon.tech](https://neon.tech), lalu ambil **Connection string
   → Pooled connection** (host-nya mengandung `-pooler`).
2. Masukkan ke `.env`:

```
DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx-pooler.REGION.aws.neon.tech/ai_bettor?sslmode=require
```

Yang ditangani otomatis oleh `backend/database/session.py`:

- `postgres://` dan `postgresql://` diubah ke `postgresql+psycopg2://`
  (SQLAlchemy 2 menolak skema `postgres://` mentah).
- `sslmode=require` ditambahkan kalau belum ada — TLS wajib di Neon. Host lokal
  (`localhost`, `db`, `127.0.0.1`) dikecualikan.
- Pool kecil dan di-recycle (default 280 detik) dengan `pool_pre_ping`, jadi
  koneksi yang sudah ditutup proxy Neon diganti, bukan meledak di tengah request.

**Tidak ada lagi fallback diam-diam ke SQLite.** Kalau Postgres tidak bisa
dihubungi, aplikasi berhenti dengan pesan jelas — dulu ia lanjut jalan dengan
database kosong, yang terlihat seperti semua data hilang. Untuk kerja offline:
`ALLOW_SQLITE_FALLBACK=true`.

Cek database mana yang benar-benar terpakai lewat `GET /health` →
`database_info` (`provider`, `host`, `ssl`; password tidak pernah ikut).

### Migrasi (opsional)

`alembic/` sekarang berisi satu revisi awal yang benar-benar jalan dan dibuat
dari metadata model (12 tabel, termasuk `system_settings` dan
`predictions.pick_score`). URL diambil dari `DATABASE_URL` aplikasi, jadi tidak
ada URL kedua yang bisa ketinggalan:

```bash
alembic upgrade head                          # atau: alembic -x url=... upgrade head
alembic revision --autogenerate -m "add X"    # untuk perubahan skema berikutnya
```

Satu hal khusus Neon: **migrasi memakai koneksi direct, bukan pooled.** Endpoint
pooled (`-pooler`) itu PgBouncer transaction mode — session state tidak bertahan
antar statement, jadi migrasi bisa gagal dengan pesan yang sama sekali tidak
menyebut pooling (`prepared statement "s0" already exists`, `SET search_path`
yang hilang di statement berikutnya, write yang masuk ke transaksi read-only).
Isi `DATABASE_URL_UNPOOLED` dengan URL yang sama tanpa `-pooler`:

```
DATABASE_URL=postgresql://…@ep-xxxx-pooler.REGION.aws.neon.tech/ai_bettor?sslmode=require
DATABASE_URL_UNPOOLED=postgresql://…@ep-xxxx.REGION.aws.neon.tech/ai_bettor?sslmode=require
```

Aplikasi tetap memakai `DATABASE_URL` (pooled) — hanya alembic yang beralih.
Kalau `DATABASE_URL_UNPOOLED` dibiarkan kosong, migrasi jalan lewat URL pooled
dan menulis warning, bukan gagal diam-diam.

## Deploy ke VPS

### 1. Keamanan (lakukan sebelum port 8000 terbuka)

API ini bisa menulis API key, membakar kuota The Odds API, dan mengubah
pembukuan bankroll. Karena itu dua nilai ini wajib di `.env`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # → API_TOKEN
```

```
API_TOKEN=<hasil di atas>
ALLOWED_ORIGINS=https://bettor.example.com
```

- Semua endpoint minta token lewat header `X-API-Token` atau
  `Authorization: Bearer <token>`. Perbandingan token memakai
  `hmac.compare_digest`.
- Yang tetap publik hanya: `/`, `/index.html`, `/favicon.ico`, `/health`, dan
  `/auth/check`. `/docs` dan `/openapi.json` **ikut terkunci**.
- Dashboard meminta token sekali, menyimpannya di `localStorage`, dan membuka
  form lagi otomatis kalau token diganti di server (401).
- `API_TOKEN` dan `ALLOWED_ORIGINS` sengaja hanya dari environment, tidak bisa
  diubah dari dashboard — sesi yang sudah pegang token tidak bisa merotasi token
  atau melebarkan CORS.

### 2. Jalankan aplikasi

Pilih salah satu:

**Docker (disarankan)**

```bash
git clone <repo> /opt/ai-bettor && cd /opt/ai-bettor
cp deploy/env.production.example .env && nano .env
docker compose up -d --build
```

**systemd (tanpa Docker)**

```bash
sudo cp deploy/ai-bettor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ai-bettor
journalctl -u ai-bettor -f
```

Keduanya menjalankan **satu worker** dengan sengaja: scheduler otomatis hidup di
dalam proses, worker kedua berarti dua scheduler menabrak database yang sama.

### 3. Nginx + HTTPS

```bash
sudo cp deploy/nginx.conf /etc/nginx/sites-available/ai-bettor
sudo ln -s /etc/nginx/sites-available/ai-bettor /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d bettor.example.com
```

Config-nya sudah termasuk redirect HTTP→HTTPS, security header, CSP, rate limit
(10 r/s), dan `proxy_read_timeout 600s` — satu `POST /automation/trigger`
menjalankan siklus penuh dan itu hitungan menit, bukan detik.

### 4. Smoke test setelah deploy

```bash
curl -s https://bettor.example.com/health | jq '.status, .database_info, .auth'
curl -s -o /dev/null -w '%{http_code}\n' https://bettor.example.com/predictions            # 401
curl -s -o /dev/null -w '%{http_code}\n' -H "X-API-Token: $TOKEN" \
     https://bettor.example.com/predictions                                                # 200
```

Firewall: buka 80/443 saja. Port 8000 tidak perlu terbuka ke internet.


## API Endpoints

Semua endpoint di bawah butuh token kalau `API_TOKEN` di-set, kecuali yang
ditandai **publik**.

| Endpoint | Deskripsi |
|---|---|
| `GET /health` | **Publik.** Status services + `database_info` + status auth |
| `GET /auth/check` | **Publik.** Cek apakah token yang dikirim valid |
| `GET /metrics` | Metrics sistem |
| `GET /matches` / `GET /matches/{id}` | Match list / detail |
| `GET /odds` | Odds snapshots |
| `POST /scan` | Jalankan full scan pipeline |
| `POST /analyze` | Analisis kuantitatif |
| `POST /simulate` | Monte Carlo simulation |
| `POST /risk-assess` | Risk assessment |
| `POST /decide` | Final decision (BET/NO BET) |
| `GET /agents` / `GET /agents/status` | Agent status (dari backend, bukan fake) |
| `GET /predictions` / `GET /predictions/{id}` | Predictions (termasuk `pick_score`) |
| `GET /performance` | Performance metrics |
| `GET /bankroll` / `GET /bets` / `POST /bets/settle` | Paper betting / bankroll |
| `POST /backtest` | Backtesting framework |
| `GET /scoring/thresholds` | Threshold gate skor pick |
| `GET /logs` | System logs |
| `GET /settings` | Settings (API keys di-mask, router status) |
| `PUT /settings` | Simpan settings (multi-key Odds API, OpenRouter, Telegram, strategi) |
| `POST /settings/reload` | Baca ulang settings dari database |
| `GET /settings/odds-api/status` | Status router multi-key (health per key) |
| `POST /settings/test-odds-key` | Test koneksi satu The Odds API key |
| `POST /settings/test-openrouter` / `POST /settings/test-telegram` | Test integrasi |
| `GET /automation/status` | Status scheduler (alive, cycle terakhir, next run) |
| `POST /automation/trigger` | Jalankan satu siklus sekarang |
| `POST /automation/toggle` | Nyalakan/matikan scheduler (persisten) |

## Multi-Key Router (The Odds API)

Sistem mendukung **banyak API key** dengan **auto failover** — kelola langsung dari
halaman **Settings** di dashboard:

- Key **A** kena rate limit (429) → otomatis beralih ke key **B** → lalu **C**, dst.
- Key kena **401/403** → dinonaktifkan permanen sampai di-reset/test ulang.
- Key kena **5xx / timeout** → cooldown 30 detik, lalu dicoba lagi.
- Router memutar (round-robin) antar key sehat; status per key (requests, failures,
  cooldown, last error) tampil live di Settings.
- Keys tersimpan di database (tabel `system_settings`), **tidak pernah** ditampilkan
  penuh di UI — hanya di-mask (mis. `abc1...6KEY`).

Juga tersedia di `.env` untuk key tunggal: `THE_ODDS_API_KEY`. Jika diisi, otomatis
didaftarkan sebagai key pertama di router.

## Konfigurasi (.env)

Daftar lengkap dengan penjelasan ada di `deploy/env.production.example`. Yang
paling penting:

```
# Akses (wajib di VPS)
API_TOKEN=
ALLOWED_ORIGINS=https://bettor.example.com

# Database (Neon)
DATABASE_URL=postgresql://USER:PASSWORD@ep-xxxx-pooler.REGION.aws.neon.tech/ai_bettor?sslmode=require
DATABASE_URL_UNPOOLED=          # opsional, host tanpa -pooler; dipakai alembic
ALLOW_SQLITE_FALLBACK=false

# Integrasi
THE_ODDS_API_KEYS=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/auto
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Runtime
TIMEZONE=Asia/Jakarta
BETTING_MODE=PAPER
MONTE_CARLO_SIMULATIONS=20000
RANDOM_SEED=42
AGENT_SCAN_INTERVAL_SECONDS=900
MIN_EDGE=0.02
MIN_EV=0.02
MIN_CONFIDENCE=60
```

Sebagian besar nilai strategi/scanning di atas hanya **nilai awal**: setelah
tersimpan di database lewat halaman Settings, nilai database yang menang dan
perubahan langsung berlaku tanpa restart. `API_TOKEN`, `ALLOWED_ORIGINS`,
`DATABASE_URL`, dan `ALLOW_SQLITE_FALLBACK` sengaja hanya dari environment.

## Prinsip

1. Data integrity: tidak pernah mengarang odds/statistik/probability.
2. LLM hanya reasoning layer — probability berasal dari statistical engine.
3. NO BET adalah fitur utama: data buruk / edge kecil / EV negatif / risiko tinggi = NO BET.
4. Minimum acceptable odds: jika odds bergerak melewati threshold, pick menjadi `VALUE_INVALIDATED`.
5. PAPER mode default — tidak ada transaksi uang sungguhan otomatis.
6. Semua secret hanya dari `.env`, tidak pernah di-hardcode, tidak pernah
   dikembalikan penuh oleh API (selalu di-mask).
7. Gagal berisik, jangan gagal diam-diam: database tak terjangkau menghentikan
   proses, scan tanpa API key melaporkan `no_api_key`, bukan "0 match".
8. Probabilistic analysis. Tidak ada jaminan hasil.

## Struktur

```
backend/
├── agents/          # 6 AI agents
├── core/            # probability engine, market math (de-vig), EV, Monte Carlo
├── integrations/    # The Odds API (multi-key router), OpenRouter, Telegram
├── services/        # pipeline, scoring, backtest, paper betting, settings
├── database/        # SQLAlchemy models + session (Neon/SQLite)
├── security.py      # token guard + CORS helper
└── tests/           # 274 automated tests
frontend/            # dashboard satu file (dark theme, di-serve FastAPI)
alembic/             # migrasi (opsional; init_db() sudah menangani skema)
deploy/              # nginx.conf, systemd unit, env.production.example
docker/              # Dockerfile (multi-stage, non-root, healthcheck)
```