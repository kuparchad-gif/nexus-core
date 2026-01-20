# No-Model Trading & Monitoring Suite (Open Source)

**Status:** initial functional scaffold — model-free (rule/heuristic), fully wired for **Qdrant** (local + cloud replication), with a **paper broker**, **backtester**, and a **Streamlit monitoring dashboard** using high-quality Plotly charts.

> **Disclaimer:** Educational/engineering project only. Not investment advice. No performance guarantees.

## Highlights
- **No-Model Strategy**: deterministic rules (momentum / breakout / volatility filters) — no ML required.
- **Backtesting**: walk-forward style runs with realistic fees & slippage knobs.
- **Paper Broker**: simulate orders, PnL, exposure, compliance checks.
- **Monitoring Dashboard**: Streamlit + Plotly for rich interactive charts (equity curve, drawdown, heatmaps, signal tapes).
- **Qdrant Integration**: store feature vectors, signals, and snapshots locally and replicate to Qdrant Cloud.
- **Config-Driven**: edit `config/config.yaml` to change universe, fees, risk caps, schedule.
- **Open Source**: MIT license.

---

## Quick Start

### 1) Install
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys (Qdrant cloud optional)
```

### 2) (Optional) Start local Qdrant
```bash
docker compose up -d qdrant
```

### 3) Bootstrap collections (runs against local and cloud if credentials present)
```bash
python scripts/bootstrap_qdrant_collections.py
```

### 4) Ingest → Features → Backtest → Store → Dashboard
```bash
# 4.1 Ingest & feature engineering
python -m src.app ingest --symbols AAPL MSFT SPY QQQ --start 2018-01-01 --end 2025-10-01

# 4.2 Run backtest
python -m src.app backtest --config config/config.yaml

# 4.3 Store snapshots to Qdrant (local + cloud)
python -m src.app store

# 4.4 Launch dashboard
streamlit run src/monitor/dashboard.py
```

### 5) Paper trade loop (demo)
```bash
python -m src.app trade --paper --rebalance daily
```

---

## What’s inside

```
nomodel-trading-suite/
  ├─ config/
  │   └─ config.yaml
  ├─ src/
  │   ├─ app.py                  # CLI orchestrator (ingest/feature/backtest/store/trade)
  │   ├─ config.py               # env + global config loader
  │   ├─ data/ingest.py          # yfinance-based data pull
  │   ├─ features/engineer.py    # indicators, rolling stats, regime flags
  │   ├─ strategy/no_model.py    # rule-based signals (no ML)
  │   ├─ portfolio/positioning.py# weights, vol targeting, caps
  │   ├─ backtest/sim.py         # simulator & metrics
  │   ├─ exec/broker_paper.py    # paper broker (orders, fills, PnL)
  │   ├─ storage/qdrant_store.py # local+cloud upsert, replication
  │   ├─ utils/plots.py          # Plotly charts (equity, drawdown, heatmaps)
  │   └─ monitor/dashboard.py    # Streamlit dashboard
  ├─ scripts/
  │   ├─ bootstrap_qdrant_collections.py
  │   ├─ start_dashboard.sh
  │   └─ start_dashboard.ps1
  ├─ docker-compose.yml          # local Qdrant service
  ├─ requirements.txt
  ├─ .env.example
  ├─ LICENSE (MIT)
  └─ README.md
```

---

## Notes
- The deterministic "no-model" core can be swapped later: keep the same interfaces and plug your ML model in place of `strategy/no_model.py` if desired.
- Qdrant vectors store engineered features + normalized signal snapshots so you can later run vector search & similarity analytics.
- Designed to be extended; PRs welcome.

*Generated 2025-10-16 23:04:00 UTC*


---
## New: Live loop, Scheduler, and Model (optional)

### Train a local CPU-friendly model (LightGBM)
```bash
python -m src.app train --adapter lightgbm
```

### Start a simple live loop (intraday, yfinance fallback)
```bash
python -m src.app live --poll 60 --lookback 120
```

### Start scheduler (nightly pipeline at 23:30 UTC, Mon–Fri)
```bash
python -m src.app schedule
# keep process running (use a service/PM2/systemd on your machine)
```

### Use no-ML baseline
```bash
python -m src.app train --adapter nomodel
```

**Desktop vs Server model:**  
- **Desktop CPU model**: LightGBM is fast, low-memory, easy to ship and retrain locally.  
- **Hybrid**: keep LightGBM on desktop for low-latency; add a remote model server (future) and call it via a tool/REST for heavier experiments. This repo’s interfaces let you swap adapters later.


---
## Deluxe Pack Additions (Web3 Wallet + API)

- **Wallet tab** with local encrypted seed, EVM testnet address + balance, QR receive, guarded send (Are you sure?).
- **Network directory** (`config/networks.yaml`) and confirmation lock to prevent accidental network mismatch.
- **FastAPI server** (`api/server.py`) exposing `/health` and `/payout` (simulated).
- **One-liner launcher**:
```bash
python start_suite.py
```

### Environment
Fill `.env` with `MASTER_PASSWORD`, set `TEST_MODE=1` to prevent mainnet broadcasts, and configure testnet RPCs via `WEB3_RPC_*` variables.

### Web3 Support
- EVM networks via `web3.py` for **balance** and **(test-mode) signing**.
- Other chains present as placeholders; extend as needed.
