<p align="center">
  <h1 align="center">🌊 BlueByte AI</h1>
  <p align="center">
    <strong>A Unified Data Platform for Oceanographic, Fisheries & Molecular Biodiversity Insight</strong>
  </p>
  <p align="center">
    <em>Built for Smart India Hackathon 2026 — AI, Data Science & Intelligent Automation</em>
  </p>
  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#repository-layout">Repository Layout</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#tech-stack">Tech Stack</a> •
    <a href="#team">Team</a>
  </p>
</p>

---

## Overview

India's Exclusive Economic Zone spans over 2 million sq. km, and the oceanographic, fisheries, and molecular biodiversity data collected across it is scattered across incompatible formats and agency-specific silos (INCOIS, CMFRI, ICAR, OBIS, NOAA, and others). **BlueByte AI** pulls that data into one platform — a PostGIS-backed spatial data lake fed by a real-time ZeroMQ streaming pipeline, exposed through a FastAPI backend, and explored via an interactive React dashboard.

The platform currently ingests real, publicly available marine datasets (OBIS species occurrences + NOAA ERDDAP sea-surface temperature and chlorophyll data), simulates live buoy and vessel telemetry for demo purposes, and layers classic spatial/graph algorithms and a heterogeneous GNN on top for prediction and analysis.

## Architecture

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│  frontend/react_app (Vite)  │      │  frontend/vanilla_app         │
│  React 18 + TS + Tailwind   │      │  Plain HTML/CSS/JS dashboard  │
│  Leaflet map, Recharts      │      │  (lightweight alternative)    │
└──────────────┬───────────────┘      └───────────────┬────────────┘
               │ REST + WebSocket                      │
┌──────────────▼──────────────────────────────────────▼────────────┐
│                     server/api  (FastAPI)                         │
│  routes: ocean · predictions · alerts · chat (GraphRAG)           │
│  websocket_manager.py  ·  zmq_bridge.py (bridges ZMQ → WS clients)│
└──────────────┬─────────────────────────────────────────────────────┘
               │
┌──────────────▼─────────────┐   ┌───────────────────────────────────┐
│  server/broker (ZeroMQ)    │   │  server/algorithms                │
│  stream_broker.py — ingest │   │  kdtree.py — spatial indexing      │
│  telemetry_publisher.py    │   │  clustering.py — hotspot detection │
│  vessel_streamer.py — AIS  │   │  pathfinding.py — route optimization│
│  in-stream Z-score anomaly │   └───────────────────────────────────┘
│  detection                 │
└──────────────┬─────────────┘
               │
┌──────────────▼─────────────────────────────────────────────────────┐
│  db/  —  PostgreSQL + PostGIS + TimescaleDB                        │
│  schema_postgis.sql · etl_pipeline.py · stream_loader.py           │
│  graph_bridge.py (feeds ml/gnn_engine) · seed_data.py              │
└──────────────┬─────────────────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────────────────┐
│  ml/gnn_engine — heterogeneous GNN (PyTorch Geometric, optional)   │
│  Species ↔ OceanGrid ↔ eDNAMarker link prediction, with a          │
│  dependency-free fallback model when torch/PyG aren't installed    │
└──────────────┬─────────────────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────────────────┐
│  data/real_datasets — OBIS + NOAA ERDDAP fetch script and CSVs     │
└──────────────────────────────────────────────────────────────────────┘
```

## Repository Layout

```
Bluebyte/
├── start.py                     # One-command dev launcher (init DB → seed → run API)
├── requirements.txt              # Core backend dependencies
│
├── server/
│   ├── api/
│   │   ├── main.py               # FastAPI app, CORS, lifespan, static frontend mount
│   │   ├── websocket_manager.py  # WebSocket connection registry
│   │   ├── zmq_bridge.py         # Bridges ZMQ pub/sub streams to WebSocket clients
│   │   └── routes/
│   │       ├── ocean.py          # Buoy readings, grids, spatial queries, species
│   │       ├── predictions.py    # PFZ predictions, species-in-grid, routes, biodiversity map
│   │       ├── alerts.py         # Recent/active alerts, alert creation, stats
│   │       └── chat.py           # GraphRAG-style chatbot endpoint
│   ├── broker/
│   │   ├── stream_broker.py      # ZeroMQ PUB/SUB core: ingest, Z-score anomaly detection, fan-out
│   │   ├── telemetry_publisher.py# Simulated buoy telemetry (6 Indian Ocean coordinates)
│   │   ├── vessel_streamer.py    # Simulated AIS vessel tracks incl. IUU-risk vessels
│   │   └── test_subscriber.py    # Terminal client for watching live streams
│   └── algorithms/
│       ├── kdtree.py             # Spatial indexing
│       ├── clustering.py         # Hotspot / density clustering
│       ├── pathfinding.py        # Vessel route optimization
│       └── test_algorithms.py
│
├── db/
│   ├── schema_postgis.sql + schema_postgis_addendum.sql
│   ├── docker-compose.db.yml     # Postgres + PostGIS + TimescaleDB (timescale/timescaledb-ha)
│   ├── connection.py             # asyncpg connection pool
│   ├── etl_pipeline.py           # Harmonization / load pipeline
│   ├── stream_loader.py          # Persists live stream data to the DB
│   ├── graph_bridge.py           # Exports relational data into the GNN's graph format
│   ├── queries.py, seed_data.py
│
├── ml/gnn_engine/
│   ├── graph_builder.py          # Builds the heterogeneous graph (Species, OceanGrid, eDNAMarker)
│   ├── model.py                  # HeteroGNN (GATConv via PyTorch Geometric) + dependency-free fallback
│   ├── train.py, predict.py
│
├── data/real_datasets/
│   ├── fetch_real_data.py        # Pulls real data from OBIS + NOAA ERDDAP
│   └── README.md                 # Data dictionary and source list
│
├── frontend/
│   ├── react_app/                # Primary dashboard: React 18 + TypeScript + Vite + Tailwind + Leaflet
│   └── vanilla_app/               # Lightweight HTML/CSS/JS dashboard alternative
│
├── docs/
│   ├── EXECUTION_PLAN.md
│   ├── INNOVATION.md
│   └── ui_ux_design.md
│
└── LICENSE                       # MIT
```

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ (for the React frontend)
- Docker (for the PostGIS/TimescaleDB database) — or your own PostgreSQL instance with PostGIS

### 1. Clone and install backend dependencies
```bash
git clone https://github.com/shaurya212121/Bluebyte.git
cd Bluebyte
pip install -r requirements.txt
```

### 2. Start the database
```bash
docker compose -f db/docker-compose.db.yml up -d
```
This spins up PostgreSQL 16 with PostGIS + TimescaleDB and auto-applies `db/schema_postgis.sql` and `db/schema_postgis_addendum.sql` on first start. Set `BLUEBYTE_DATABASE_URL` if you're not using the default local credentials (see `db/connection.py`).

### 3. Launch the API server
```bash
python start.py
```
This initializes the database, seeds sample data, and starts the FastAPI server at `http://localhost:8000` (interactive docs at `/docs`).

### 4. (Optional) Enable live streaming
The dashboard works with seeded/static data out of the box. To see live telemetry and anomaly alerts, run these in separate terminals:
```bash
python server/broker/stream_broker.py
python server/broker/telemetry_publisher.py
# optionally also:
python server/broker/vessel_streamer.py
```
See `server/broker/README.md` for the full socket contract.

### 5. Run the React dashboard (dev mode)
```bash
cd frontend/react_app
npm install
npm run dev
```
In production, `server/api/main.py` also serves the built React app directly from `frontend/react_app`, so a separate frontend server isn't required once you run `npm run build`.

### 6. (Optional) Refresh real datasets
```bash
python data/real_datasets/fetch_real_data.py
```
Pulls current species occurrence and satellite data from OBIS and NOAA ERDDAP.

### 7. (Optional) Enable the GNN engine
The heterogeneous GNN (`ml/gnn_engine`) uses PyTorch Geometric when available and otherwise falls back to a lightweight heuristic model. To enable full GNN mode:
```bash
pip install -r ml/gnn_engine/requirements.txt
```

## API Reference

All routes are mounted under `/api/v1` (interactive Swagger UI at `/docs`):

| Route | Description |
|---|---|
| `GET /api/v1/ocean-data/readings` | Buoy sensor readings |
| `GET /api/v1/ocean-data/grids` | Ocean grid cells |
| `GET /api/v1/ocean-data/spatial-query` | Spatial (PostGIS) queries |
| `GET /api/v1/ocean-data/species` | List all tracked species |
| `GET /api/v1/ocean-data/species/match` | Match species to given environmental conditions |
| `GET /api/v1/predictions/pfz` | Potential Fishing Zone predictions |
| `GET /api/v1/predictions/species/{grid_id}` | Predicted species presence for a grid |
| `GET /api/v1/predictions/route` | Optimal vessel route |
| `GET /api/v1/predictions/biodiversity-map` | Biodiversity map data |
| `GET /api/v1/alerts/recent` | Recent alerts |
| `GET /api/v1/alerts/active` | Currently active alerts |
| `POST /api/v1/alerts` | Create an alert |
| `GET /api/v1/alerts/stats` | Alert statistics |
| `POST /api/v1/chat` | GraphRAG-style chatbot over the platform's data |
| `WS /ws` | Live telemetry & alert stream (see `websocket_manager.py`) |
| `GET /health` | Health check |

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 18, TypeScript, Vite, Tailwind CSS, Leaflet, Recharts, Framer Motion (+ a plain HTML/CSS/JS alternative) |
| **Backend** | Python, FastAPI, Uvicorn, WebSockets |
| **Streaming** | ZeroMQ (PUB/SUB), in-stream Z-score anomaly detection |
| **Database** | PostgreSQL + PostGIS + TimescaleDB, asyncpg |
| **ML** | PyTorch + PyTorch Geometric (heterogeneous GNN with GATConv), with a dependency-free fallback model |
| **Algorithms** | Custom KD-tree spatial indexing, clustering, and pathfinding |
| **Data Sources** | OBIS (species occurrences), NOAA CoastWatch ERDDAP (SST, chlorophyll-a) |

## Key Data Sources

| Source | Type | License |
|---|---|---|
| [OBIS](https://api.obis.org) | Fish/species occurrence records | CC0 Public Domain |
| [NOAA CoastWatch ERDDAP](https://coastwatch.pfeg.noaa.gov/erddap) | Sea surface temperature, chlorophyll-a | US Public Domain |
| [INCOIS](https://incois.gov.in/) | Buoy climatology / ocean advisories | Open Government Data |

See `data/real_datasets/README.md` for the full column reference and species list.

## Team

| Name | Role | Core Responsibility |
|---|---|---|
| **Shaurya** | Network & Streaming Engineer | ZeroMQ ingestion broker & high-throughput sockets |
| **Pranshu** | Backend & Socket Bridge | FastAPI gateway & real-time WebSockets |
| **Jaanya** | AI & GNN Specialist | Marine knowledge graph & GNN species-link prediction |
| **Vivaan** | Database Architect | PostGIS spatial data lake & time-series layer |
| **Dheer** | Algorithms & DSA Specialist | Spatial indexing (KD-tree) & vessel route optimization |
| **Diyan** | UI/UX & GIS Frontend | Interactive ocean GIS dashboard & visualizations |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [INCOIS](https://incois.gov.in/) — Ocean data services
- [OBIS](https://obis.org/) — Marine species occurrence data
- [NOAA CoastWatch](https://coastwatch.pfeg.noaa.gov/) — Satellite ocean data
- [Smart India Hackathon](https://www.sih.gov.in/) — Platform and problem statement

---

<p align="center">
  <strong>Built for Smart India Hackathon 2026</strong><br>
  <em>AI, Data Science & Intelligent Automation Track</em>
</p>
