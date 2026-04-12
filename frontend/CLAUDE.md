# NYC Smart City Nexus — Frontend

Next.js 16 app for the NYC Smart City Nexus hackathon project (Spark Hack NYC 2026).
Interactive mission-control UI for city energy and waste planners.

> **Legacy Vite app** lives in `frontend/reactjs/` — do not delete, used as source reference.

## Commands

```bash
npm run dev     # dev server (http://localhost:3000)
npm run build   # production build
npm run start   # serve production build
npm run lint    # ESLint
```

## Tech Stack

| Layer | Choice |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack) |
| UI | React 19 — JSX components in `src/components/`, TSX pages in `src/app/` |
| Styling | Tailwind CSS v4 via `@tailwindcss/postcss` — `@import "tailwindcss"` in `globals.css` |
| Fonts | Michroma (`--font-display`), IBM Plex Mono (`--font-mono-ui`) via `next/font/google` |
| Scroll | Lenis smooth scroll — wrapped in `LenisProvider` in `layout.tsx` |
| Map | MapLibre GL JS 5 + react-map-gl 8 (`import Map from 'react-map-gl/maplibre'`) |
| Map tiles | CARTO Light Matter raster tiles (`CityMap.jsx`) + OpenFreeMap `liberty` vector tiles (`upload/page.tsx`) — both free, no API key |
| Charts | Custom hand-rolled SVG — no recharts |
| State | React Context (`DashboardContext`) — no Zustand/Redux |

## Routes

| Route | File | Description |
|---|---|---|
| `/` | `src/app/page.tsx` | Landing page — `SmartCityManagmentLanding` component |
| `/map` | `src/app/map/page.tsx` | Full NYC Smart City dashboard (map + all 6 sections) |
| `/upload` | `src/app/upload/page.tsx` | Result viewer — main image area (placeholder when empty) + MapLibre 3D map inset pinned to bottom-right, comment icon |

## Project Structure

```
src/
├── app/
│   ├── layout.tsx              # Root layout — fonts, LenisProvider
│   ├── globals.css             # Tailwind base import + global resets
│   ├── page.tsx                # Landing page
│   ├── map/
│   │   └── page.tsx            # Dashboard page — wraps DashboardProvider + all sections
│   └── upload/
│       └── page.tsx            # Result viewer page
├── components/
│   ├── lenis-provider.tsx      # Lenis smooth scroll context provider
│   ├── theradyme-landing.tsx   # Landing page component — "Begin →" opens EntryForm popup (photo upload, site name, location, borough, date/time, notes); "Save entry ↗" navigates to /upload
│   ├── Map/
│   │   ├── CityMap.jsx         # MapLibre GL map — 3 layer modes + district/building overlays
│   │   ├── MapLegend.jsx       # Dynamic legend per viewMode
│   │   └── MapControls.jsx     # Live geocoder search (currently tied to ENERGY_SITES — not yet updated)
│   ├── Sidebar.jsx             # View toggle pills, nav, borough/score/budget/EJ filters
│   ├── Header.jsx              # Breadcrumb, status pills
│   ├── KPIRibbon.jsx           # 4 KPI cards, switches content with viewMode
│   ├── RankingsTable.jsx       # Top 10 districts table with collapsible building rows
│   ├── SiteDetail.jsx          # 3-col detail panel — renders based on selectedItem type from context
│   ├── FlowVisualization.jsx   # SVG Sankey flows (Energy + Waste side by side) — each band has SVG <animate> opacity pulse with staggered delays to simulate left-to-right flow
│   ├── Simulation.jsx          # Tabbed: BESS Dispatch / Waste Forecast / Scenario Planner
│   └── BoroughComparison.jsx   # Horizontal grouped bar + 5 borough summary cards
├── context/
│   └── DashboardContext.jsx    # All shared state (viewMode, selectedId, selectedItem, borough, filters, scenario)
├── hooks/
│   └── useViewMode.js          # Re-exports useDashboard from context
└── data/
    ├── top50_scored_xgboost.json       # Real XGBoost-scored site data (50 sites)
    ├── top10_district_analysis.json    # Top 10 districts by solar potential — full data including
    │                                   #   buildings (10 per district), centroid lat/lng,
    │                                   #   buildings_summary, waste, complaints, ai_analysis.
    │                                   #   Each building has latitude + longitude fields.
    │                                   #   Single source of truth for district map layers and table.
    ├── highlight_buildings.json        # 8 hand-picked priority buildings — CAPITALIZED field keys:
    │                                   #   Site, Address, Borough, Agency, "Environmental Justice Area",
    │                                   #   lat, lon, bbl, energy_score, waste_score, nexus_score,
    │                                   #   recommended_bess_kwh, estimated_annual_savings_usd,
    │                                   #   top_recommendation, rank
    ├── districtBuildings.js    # DEPRECATED — superseded by top10_district_analysis.json
    ├── top10Districts.js       # DEPRECATED — superseded by top10_district_analysis.json
    ├── sites.js                # Imports JSON, transforms → ENERGY_SITES, WASTE_SITES, NEXUS_SITES,
    │                           #   BOROUGH_DATA, KPI arrays, score helpers
    ├── districts.js            # Borough + EJ GeoJSON polygons for MapLibre layers
    └── chartData.js            # Hourly dispatch, SOC, waste forecast, diversion gap, scenario
```

## Next.js Conventions

- All dashboard components use `'use client'` (hooks + browser APIs)
- Data files (`src/data/`) are plain JS modules — no `'use client'` needed
- Import alias `@/` maps to `src/` (configured in `tsconfig.json`)
- Pages are `.tsx`; dashboard components are `.jsx` (JSX only, no TypeScript types enforced in components)
- MapLibre CSS must be imported at the page/component level: `import 'maplibre-gl/dist/maplibre-gl.css'`
- `upload/page.tsx` imports MapLibre CSS directly (not via a layout) because it is the only non-dashboard page using a map

## Shared State (DashboardContext)

All dashboard components read from `useDashboard()`. Never use local state for cross-component concerns.

| Field | Type | Description |
|---|---|---|
| `viewMode` | `'energy' \| 'waste' \| 'nexus'` | Drives KPIs, table schema, map layers |
| `selectedId` | `number` | 0-based index into `ENERGY_SITES` (legacy, used by energy mode) |
| `selectedItem` | `{ type, data } \| null` | Currently selected map item — type is `'building'`, `'district'`, or `'highlight'` |
| `borough` | `string` | Filter — `'All Boroughs'` or a borough name |
| `minScore` | `number` | Filter — hide sites below this nexus score |
| `ejPriority` | `boolean` | EJ toggle |
| `simTab` | `'bess' \| 'waste' \| 'scenario'` | Active simulation tab |
| `scenario` | `object` | Scenario Planner slider values |

- `selectItem(type, data)` — called from `CityMap.jsx` `onMapClick` when user clicks a building dot, district centroid, or star icon
- `changeView(v)` also auto-switches `simTab`

## Map Click → SiteDetail Flow

1. User clicks a map feature in `CityMap.jsx`
2. `onMapClick` queries rendered features in priority order: highlight stars → building dots → district centroids → mode-specific layers
3. For the first match, calls `selectItem(type, properties)` on the context
4. `SiteDetail.jsx` reads `selectedItem` from context and renders the matching layout:
   - `type === 'highlight'` → ScoreRings for energy/waste/nexus, BESS kWh + savings, recommendation
   - `type === 'building'` → solar kWh/yr, annual cost, GHG, BESS kWh + savings
   - `type === 'district'` → solar potential, total/solar-ready buildings, BESS savings, EJ%
   - `null` → placeholder prompt

## Data

### Real site data (`top50_scored_xgboost.json`)
50 NYC municipal sites scored by XGBoost model. Fields used:
- `rank`, `site`, `address`, `borough`, `agency`
- `env_justice` — `"Yes"` / `"No"`
- `energy_score`, `waste_score`, `nexus_score` — 0–100
- `recommended_bess_kwh` — 250 / 500 / 750
- `estimated_annual_savings_usd` — number
- `top_recommendation`, `reasoning` — strings shown in SiteDetail

**lat/lng** are derived deterministically in `sites.js` using golden-ratio scatter within each borough's bounding box. They are **not** in the raw JSON.

### District data (`top10_district_analysis.json`)
Top 10 NYC community districts ranked by `total_solar_potential_kwh_yr`. Derived from `data/gold/district_analysis.json` in the repo root. This is the **single source of truth** for all district and building map layers and the RankingsTable.

Top-level fields per district:
- `district_code`, `borough`, `community_district`
- `centroid_lat`, `centroid_lon` — real geographic centroid used for map markers
- `buildings_summary` — `total`, `solar_ready`, `total_solar_potential_kwh_yr`, `total_bess_savings_usd_yr`, `pct_ej`
- `buildings` — array of up to 10 buildings (top 10 by solar potential)
- `waste`, `waste_to_energy`, `complaints`, `infrastructure`, `ai_analysis`

Building fields:
- `site`, `address`, `agency`, `ej` (bool), `roof`, `sqft`
- `latitude`, `longitude` — real geocoded coordinates for map plotting
- `energy` — `solar_production_kwh_yr`, `est_annual_cost_usd`, `ghg_tons_co2e_yr`, etc.
- `bess_recommendation` — `capacity_kwh`, `power_kw`, `est_annual_savings_usd`

### Highlighted buildings (`highlight_buildings.json`)
8 hand-picked priority buildings shown as ★ star icons on the map in all view modes. Keys are **capitalized**: `Site`, `Address`, `Borough`, `Agency`, `"Environmental Justice Area"` (string `"Yes"`/`"No"`). Other fields: `lat`, `lon`, `bbl`, `energy_score`, `waste_score`, `nexus_score`, `recommended_bess_kwh`, `estimated_annual_savings_usd`, `top_recommendation`, `rank`.

When reading these in `CityMap.jsx`, access with `b.Site`, `b.Address`, `b['Environmental Justice Area']`, etc.

### Score thresholds (real data range 88–98)
| Range | Color | Meaning |
|---|---|---|
| ≥ 95 | `#EF4444` red | Highest priority |
| ≥ 90 | `#F59E0B` amber | High priority |
| < 90 | `#3B82F6` blue | Medium priority |

Helper functions in `sites.js`: `scoreColor(s)`, `scoreBg(s)`, `scoreText(s)`.

## Map Layers

Three modes driven by `viewMode`, plus permanent overlays visible in all modes:

| Mode | Layers |
|---|---|
| **Energy** | (energy-sites layer commented out — not currently shown) |
| **Waste** | Borough fill choropleth (orange intensity by `wasteTons`) + district circles (size by refuse tonnage) |
| **Nexus** | Borough fill + EJ area fill (purple outlined) + heatmap by `nexusScore` |
| **All modes** | Gold ring district centroid markers (rank label 1–10) from `top10_district_analysis.json` |
| **All modes** | Building dots (5px emerald green `#10B981`) from `top10_district_analysis.json` |
| **All modes** | Amber star icons (★) from `highlight_buildings.json` with amber glow ring |

Click priority order in `onMapClick`: highlight stars → building dots → district centroids → mode-specific layers.
Hover popup priority: highlight stars → building dots → district centroids → energy sites.

GeoJSON objects (`TOP10_DISTRICTS_GEOJSON`, `DISTRICT_BUILDINGS_GEOJSON`, `HIGHLIGHT_BUILDINGS_GEOJSON`) are all built at module load time in `CityMap.jsx` from imported JSON — no runtime fetch.

Map tile style defined inline as a MapLibre style object in `CityMap.jsx` — CARTO Light Matter raster tiles.

### Upload page map (`upload/page.tsx`)
Separate MapLibre instance — **not** connected to `DashboardContext`. Uses OpenFreeMap `liberty` vector style (`https://tiles.openfreemap.org/styles/liberty`) which includes building footprints for 3D extrusion. Rendered as a 360×260px inset pinned to the bottom-right of the canvas area. Initial view: Midtown Manhattan, pitch 60°, bearing −20°, zoom 15.5. The 2D/3D toggle (currently commented out in the header) calls `mapRef.current?.easeTo()` to animate pitch/bearing. `VIEW_2D = { pitch: 0, bearing: 0, zoom: 14.5 }`, `VIEW_3D = { pitch: 60, bearing: −20, zoom: 15.5 }`.

## Styling Conventions

Dashboard components use **inline styles** (not Tailwind classes). The dashboard uses a **light theme**.

**Dashboard color palette (light theme):**
```
Background:   #F1F5F9
Card:         #FFFFFF   border: #E2E8F0
Text primary: #0F172A
Text muted:   #64748B
Text dim:     #475569 / #334155
Blue accent:  #3B82F6   text: #60A5FA / #3B82F6
Emerald:      #10B981   text: #059669
Amber:        #F59E0B   text: #CA8A04
Red:          #EF4444
Purple:       #8B5CF6   text: #7C3AED
```

**Card pattern:**
```jsx
style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 12, padding: 16 }}
```

**Section label pattern:**
```jsx
style={{ fontSize: 10, fontWeight: 600, color: '#475569', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12 }}
```

## Key Conventions

- **JSX only for components** — no `.tsx` in `src/components/`. TypeScript only in `src/app/`.
- **No recharts** — all charts are hand-rolled SVG inside components.
- **No API calls** — all data is static/derived from the imported JSON.
- **Do not touch** `City_Planning_Nexus.html` — it is a reference artifact only.
- Map click → `selectItem(type, data)` → updates `SiteDetail` via context.
- Borough filter + minScore slider are wired in Sidebar but not yet applied to map layers.
- KPI numbers in `ENERGY_KPIS` are computed dynamically from `ENERGY_SITES` at module load time in `sites.js` — do not hardcode them.
