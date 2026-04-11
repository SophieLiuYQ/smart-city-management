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
| Map tiles | CARTO Dark Matter raster tiles — **free, no API key** |
| Charts | Custom hand-rolled SVG — no recharts |
| State | React Context (`DashboardContext`) — no Zustand/Redux |

## Routes

| Route | File | Description |
|---|---|---|
| `/` | `src/app/page.tsx` | Landing page — `SmartCityManagmentLanding` component |
| `/map` | `src/app/map/page.tsx` | Full NYC Smart City dashboard (map + all 6 sections) |
| `/upload` | `src/app/upload/page.tsx` | Result viewer — 2D/3D toggle, building preview, comment icon |

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
│   ├── theradyme-landing.tsx   # Landing page component
│   ├── Map/
│   │   ├── CityMap.jsx         # MapLibre GL map — 3 layer modes (Energy/Waste/Nexus)
│   │   ├── MapLegend.jsx       # Dynamic legend per viewMode
│   │   └── MapControls.jsx     # Live geocoder search over ENERGY_SITES array
│   ├── Sidebar.jsx             # View toggle pills, nav, borough/score/budget/EJ filters
│   ├── Header.jsx              # Breadcrumb, status pills
│   ├── KPIRibbon.jsx           # 4 KPI cards, switches content with viewMode
│   ├── RankingsTable.jsx       # 3-schema table (energy/waste/nexus), row click → selectSite
│   ├── SiteDetail.jsx          # 3-col panel: identity / score rings / BESS + AI insight
│   ├── FlowVisualization.jsx   # SVG Sankey flows (Energy + Waste side by side)
│   ├── Simulation.jsx          # Tabbed: BESS Dispatch / Waste Forecast / Scenario Planner
│   └── BoroughComparison.jsx   # Horizontal grouped bar + 5 borough summary cards
├── context/
│   └── DashboardContext.jsx    # All shared state (viewMode, selectedId, borough, filters, scenario)
├── hooks/
│   └── useViewMode.js          # Re-exports useDashboard from context
└── data/
    ├── top50_scored_xgboost.json  # Real XGBoost-scored site data (50 sites)
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
- MapLibre CSS must be imported at the page level: `import 'maplibre-gl/dist/maplibre-gl.css'`

## Shared State (DashboardContext)

All dashboard components read from `useDashboard()`. Never use local state for cross-component concerns.

| Field | Type | Description |
|---|---|---|
| `viewMode` | `'energy' \| 'waste' \| 'nexus'` | Drives KPIs, table schema, map layers |
| `selectedId` | `number` | 0-based index into `ENERGY_SITES` |
| `borough` | `string` | Filter — `'All Boroughs'` or a borough name |
| `minScore` | `number` | Filter — hide sites below this nexus score |
| `ejPriority` | `boolean` | EJ toggle |
| `simTab` | `'bess' \| 'waste' \| 'scenario'` | Active simulation tab |
| `scenario` | `object` | Scenario Planner slider values |

Changing `viewMode` via `changeView()` also auto-switches `simTab`.

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

### Score thresholds (real data range 88–98)
| Range | Color | Meaning |
|---|---|---|
| ≥ 95 | `#EF4444` red | Highest priority |
| ≥ 90 | `#F59E0B` amber | High priority |
| < 90 | `#3B82F6` blue | Medium priority |

Helper functions in `sites.js`: `scoreColor(s)`, `scoreBg(s)`, `scoreText(s)`.

## Map Layers

Three modes driven by `viewMode`:

| Mode | Layers |
|---|---|
| **Energy** | Circle layer — radius + color by `nexusScore`. White ring on `selectedId`. Hover popup. |
| **Waste** | Borough fill choropleth (orange intensity by `wasteTons`) + district circles (size by refuse tonnage) |
| **Nexus** | Borough fill + EJ area fill (purple outlined) + heatmap by `nexusScore` |

Map tile style defined inline as a MapLibre style object in `CityMap.jsx` — CARTO Dark Matter raster, no external style JSON file.

## Styling Conventions

Dashboard components use **inline styles** (not Tailwind classes) to match the original `City_Planning_Nexus.html` reference design. Landing page and layout use Tailwind.

**Dashboard color palette:**
```
Background:   #0B1120
Card:         #131C2E  border: #1E293B
Sidebar:      #0F172A
Blue accent:  #3B82F6   text: #60A5FA / #93C5FD
Emerald:      #10B981   text: #6EE7B7
Amber:        #F59E0B   text: #FCD34D
Red:          #EF4444   text: #FCA5A5
Purple:       #8B5CF6   text: #C4B5FD
Text primary: #F1F5F9
Text muted:   #64748B
Text dim:     #475569 / #334155
```

**Upload page** uses a light theme (`#F3F4F6` background, white card, `#1E293B` active toggle).

**Card pattern:**
```jsx
style={{ background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, padding: 16 }}
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
- Map click → `selectSite(id)` + `flyTo()` → updates SiteDetail and table highlight.
- Borough filter + minScore slider filter both the map circles and the rankings table rows.
- KPI numbers in `ENERGY_KPIS` are computed dynamically from `ENERGY_SITES` at module load time in `sites.js` — do not hardcode them.
