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

## Quick Start

### 1. Konfigurasi

```bash
cp .env.example .env
# isi API key
```

### 2. Instalasi & Jalankan (tanpa Docker)

```bash
pip install -r requirements.txt  # atau lihat pyproject.toml
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Buka `http://localhost:8000` (dashboard) atau `/docs` (Swagger).

> Jika PostgreSQL tidak tersedia, sistem otomatis fallback ke SQLite lokal
> untuk development. Untuk production set `DATABASE_URL` ke PostgreSQL.

### 3. Docker

```bash
docker-compose up --build
```

### 4. Test

```bash
python -m pytest backend/tests -v
```

## API Endpoints

| Endpoint | Deskripsi |
|---|---|
| `GET /health` | Health check + status services |
| `GET /metrics` | Metrics sistem |
| `GET /matches` / `GET /matches/{id}` | Match list / detail |
| `GET /odds` | Odds snapshots |
| `POST /scan` | Jalankan full scan pipeline |
| `POST /analyze` | Analisis kuantitatif |
| `POST /simulate` | Monte Carlo simulation |
| `POST /risk-assess` | Risk assessment |
| `POST /decide` | Final decision (BET/NO BET) |
| `GET /agents` | Agent status (dari backend, bukan fake) |
| `GET /predictions` / `GET /predictions/{id}` | Predictions |
| `GET /performance` | Performance metrics |
| `GET /bankroll` / `GET /bets` | Paper betting / bankroll |
| `POST /backtest` | Backtesting framework |
| `GET /logs` | System logs |
| `GET /settings` | Settings (API keys di-mask, router status) |
| `PUT /settings` | Simpan settings (multi-key Odds API, OpenRouter, Telegram, strategi) |
| `GET /settings/odds-api/status` | Status router multi-key (health per key) |
| `POST /settings/test-odds-key` | Test koneksi satu The Odds API key |
| `POST /settings/test-openrouter` | Test koneksi OpenRouter |

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

```
THE_ODDS_API_KEY=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DATABASE_URL=postgresql://localhost:5432/ai_bettor
TIMEZONE=Asia/Jakarta
MONTE_CARLO_SIMULATIONS=20000
RANDOM_SEED=42
ODDS_POLL_INTERVAL_SECONDS=300
MIN_EDGE=0.01
MIN_EV=0.01
MIN_CONFIDENCE=60
BETTING_MODE=PAPER
```

## Prinsip

1. Data integrity: tidak pernah mengarang odds/statistik/probability.
2. LLM hanya reasoning layer — probability berasal dari statistical engine.
3. NO BET adalah fitur utama: data buruk / edge kecil / EV negatif / risiko tinggi = NO BET.
4. Minimum acceptable odds: jika odds bergerak melewati threshold, pick menjadi `VALUE_INVALIDATED`.
5. PAPER mode default — tidak ada transaksi uang sungguhan otomatis.
6. Semua secret hanya dari `.env`, tidak pernah di-hardcode.
7. Probabilistic analysis. Tidak ada jaminan hasil.

## Struktur

```
backend/
├── agents/          # 6 AI agents
├── models/          # probability engine, EV, Monte Carlo
├── integrations/    # The Odds API, OpenRouter, Telegram
├── services/        # pipeline, scoring, backtest, paper betting
├── database/        # SQLAlchemy models + session
├── api/             # endpoints
└── tests/           # automated tests
frontend/            # dashboard (dark theme, served by FastAPI)
migrations/          # alembic migrations
docker/              # Dockerfiles
```