# NYC Smart City Nexus

**Energy, Waste & Urban Intelligence Platform**

> Spark Hack NYC 2026 | Environmental Impact Track | NVIDIA GB10

A decision-support platform that merges NYC's energy and waste data into one unified intelligence layer. Built to run entirely on-device using NVIDIA RAPIDS, NIMs, and cuOpt on the Acer Veriton GN100.

---

## Quick Start

```bash
# Run the full pipeline
python smart-city-management/build_silver_layer.py
python smart-city-management/build_bridge_layer.py
python smart-city-management/build_spatial_layer.py
python smart-city-management/build_gold_layer.py
python smart-city-management/add_missing_outputs.py
```

---

## Project Structure

```
smart-city-management/
├── build_silver_layer.py     # Raw → Silver (12 Parquet datasets)
├── build_bridge_layer.py    # Create property_to_bbl bridge
├── build_spatial_layer.py  # cuSpatial feature joins
├── build_gold_layer.py     # → Gold (3 tables + handoffs)
├── add_missing_outputs.py   # depots + site_profiles
├── README.md
├── design doc/              # Technical design docs
├── frontend/                # Dashboard UI
├── AI/                     # LLM ranker
└── data/
    ├── raw/                # 12 raw CSVs from NYC Open Data
    ├── silver/              # Cleaned Parquet (partitioned)
    ├── gold/                # Unified tables + handoffs
    │   ├── route_inputs/    # cuOpt inputs
    │   ├── dispatch/        # BESS dispatch
    │   └── nim/            # NIM site batches
    └── dictionaries/        # Data documentation
```

---

## Datasets (12 sources from NYC Open Data)

### Energy

| Status | ID | Dataset | Rows | Source |
|--------|-----|---------|------|--------|
| ✅ | E1 | Energy Cost Savings Program | 2,363 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/Value-of-Energy-Cost-Savings-Program-Savings-for-B/bug8-9f3g) |
| ✅ | E3 | Electric Consumption & Cost | 553,666 | [NYC Open Data](https://data.cityofnewyork.us/Housing-Development/Electric-Consumption-And-Cost-2010-Sep-2025/jr24-e7cr) |
| ✅ | E4 | NYC EV Fleet Station Network | 1,638 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/NYC-EV-Fleet-Station-Network/fc53-9hrv) |
| ✅ | E5 | Municipal Solar Readiness (LL24) | 4,270 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/City-of-New-York-Municipal-Solar-Readiness-Assessment/cfz5-6fvh) |
| ✅ | E7 | LL84 Monthly Energy Data | ~2.2M | [NYC Open Data](https://data.cityofnewyork.us/Environment/Local-Law-84-Monthly-Data-Calendar-Year-/fvp3-gcb2) |
| ✅ | E10 | LL84 Annual Benchmarking | 103,259 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/NYC-Building-Energy-and-Water-Data-Disclosure/5zyy-y8am) |

### Waste

| Status | ID | Dataset | Rows | Source |
|--------|-----|---------|------|--------|
| ✅ | W1 | DSNY Monthly Tonnage | 24,883 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/DSNY-Monthly-Tonnage-Data/ebb7-mvp5) |
| ✅ | W2 | 311 Service Requests (DSNY) | 1,556,494 | [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9) |
| ✅ | W3 | Litter Basket Inventory | 20,413 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/DSNY-Litter-Basket-Inventory/8znf-7b2c) |
| ✅ | W7 | Food Scrap Drop-Off Locations | 591 | [NYC Open Data](https://data.cityofnewyork.us/Environment/Food-Scrap-Drop-Off-Locations-in-NYC/if26-z6xq) |
| ✅ | W8 | Disposal Facility Locations | 39 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/Location-of-Disposal-Facilities/ufxk-pq9j) |
| ✅ | W12 | Waste Characterization | 2,112 | [NYC Open Data](https://data.cityofnewyork.us/City-Government/DSNY-Waste-Characterization-2023-Main-Sort-Results/bpea-2i5q) |

---

## Pipeline Outputs

### Silver Layer (`data/silver/`)

| File | Description | Partitioned |
|------|-------------|-------------|
| `E3_electric_consumption.parquet` | NYCHA electric usage | No |
| `E4_ev_fleet_stations.parquet` | EV charging locations | No |
| `E5_solar_readiness.parquet` | Solar-ready buildings | No |
| `E5_solar_readiness_enriched.parquet` | + cuSpatial features | No |
| `E7_ll84_monthly.parquet` | Monthly energy | ✅ calendar_year |
| `E10_ll84_benchmarking.parquet` | Annual benchmarking | No |
| `property_to_bbl.parquet` | Bridge table (48,310 unique) | No |
| `W1_dsny_monthly_tonnage.parquet` | District tonnage | No |
| `W2_311_dsny.parquet` | 311 complaints | ✅ created_year |
| `W7_food_scrap_dropoffs.parquet` | Compost sites | No |
| `W8_disposal_facilities.parquet` | Transfer facilities | No |

### Gold Layer (`data/gold/`)

| File | Description |
|------|-------------|
| `unified_sites.parquet` | Gold Table 1: E5 + E10 + spatial features |
| `district_waste.parquet` | Gold Table 2: W1 aggregated by district |
| `time_series.parquet` | Gold Table 3: E7 monthly energy |
| `route_inputs/depots.parquet` | cuOpt depot locations (39) |
| `route_inputs/demand_nodes.parquet` | cuOpt demand nodes (59 districts) |
| `nim/site_batches.jsonl` | NIM scoring batch (20 sites) |
| `dispatch/site_profiles.parquet` | BESS dispatch profiles (4,268 sites) |

---

## Interface Contracts (GB10 Runbook)

### Join Rules

```
E5.bbl ← E10.bbl (via property_to_bbl)
E7.property_id → E10.property_id → E10.bbl → E5.bbl
```

### cuSpatial Features

From `E5` building points:
- EV ports within 500m
- EV ports within 1km
- Compost sites within 1km
- Nearest transfer facility distance

### Complaint Schema

Map both `Problem`/`Problem Detail` and `Complaint Type`/`Descriptor` into one canonical category column.

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Data Processing | RAPIDS cuDF + cuSpatial | GPU-accelerated data processing |
| Route Optimization | RAPIDS cuOpt | Waste collection vehicle routing |
| LLM Inference | NVIDIA NIMs (Llama 3.1 8B) | Local site scoring |
| Simulation | Custom Python (GPU-vectorized) | BESS dispatch |
| Dashboard | Streamlit + Folium | Interactive maps, charts |

---

## Running the Next Stages

### cuOpt (Route Optimization)
```python
import cuopt

# Load demand nodes and depots
demand_nodes = pd.read_parquet("data/gold/route_inputs/demand_nodes.parquet")
depots = pd.read_parquet("data/gold/route_inputs/depots.parquet")

# Solve route
# (See: https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
```

### NIM (Local LLM Scoring)
```bash
# Run NIM container
docker run -d --gpus=1 --name nimLlama \
  -v ./smart-city-management/data/gold/nim:/app/nim \
  nvcr.io/nim/nvidia/llama-3.1-nim:8b-instruct-q4

# Score sites
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @nim/site_batches.jsonl
```

### BESS Dispatch
```python
# Load site profiles and time series
profiles = pd.read_parquet("data/gold/dispatch/site_profiles.parquet")
time_series = pd.read_parquet("data/gold/time_series.parquet")

# Run dispatch simulation
# (See: design doc/BESS_simulation.md)
```

## Future Things to Implement

### Local LLM Scoring (Qwen 3 80B)

The mini workstation already has the Qwen 3 80B model activated and available for local inference. For future local LLM scoring of sites, we can leverage this model directly:

```bash
# Using the local Qwen 3 80B endpoint (already running on the workstation)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-80b",
    "messages": [{"role": "user", "content": "Score this site for solar potential..."}]
  }'
```

The Qwen 3 80B provides significantly more capacity than Llama 3.1 8B NIM for nuanced site scoring, natural language reasoning about building characteristics, and batch processing of site evaluations.

---

## Team

Built for [Spark Hack NYC 2026](https://luma.com/spark-hack-nyc) — Environmental Impact Track.

**NVIDIA GB10**: 128 GB unified memory, 1 PFLOP FP4 on Acer Veriton GN100

## Resources 

[Spark Hack Notion Page](https://concrete-panther-c83.notion.site/Spark-Hack-Series-New-York-Presented-by-NVIDIA-Acer-and-Antler-611f567d17cc8207881a0162071001c8)

##Finetuning sources
https://www.kaggle.com/datasets/mostafaabla/garbage-classification
https://universe.roboflow.com/garbage-detection-czeg5/garbage_detection-wvzwv
nvidia--NVIDIA-Nemotron-Nano-12B-v2-VL-FP8 model fine tuned with LoRA
both data sets were partitioned into organics, electronic, plastic, others in folders

