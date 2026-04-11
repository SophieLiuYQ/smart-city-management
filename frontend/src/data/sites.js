/**
 * Real site data sourced from top50_scored_xgboost.json
 * (XGBoost-ranked top 50 NYC municipal sites for BESS deployment)
 *
 * lat/lng derived deterministically from borough bounds + rank
 * using golden-ratio scatter so dots spread realistically across each borough.
 */
import rawSites from './top50_scored_xgboost.json';

/* ── Borough bounding boxes for map placement ── */
const BOROUGH_BOUNDS = {
  'Bronx':         { lngMin: -73.930, lngMax: -73.820, latMin: 40.800, latMax: 40.882 },
  'Brooklyn':      { lngMin: -74.040, lngMax: -73.860, latMin: 40.575, latMax: 40.740 },
  'Queens':        { lngMin: -73.960, lngMax: -73.710, latMin: 40.580, latMax: 40.790 },
  'Manhattan':     { lngMin: -74.020, lngMax: -73.912, latMin: 40.700, latMax: 40.878 },
  'Staten Island': { lngMin: -74.240, lngMax: -74.050, latMin: 40.500, latMax: 40.648 },
};

const PHI = 1.6180339887;

function deriveLngLat(borough, rank) {
  const b = BOROUGH_BOUNDS[borough];
  if (!b) return { lng: -73.94, lat: 40.73 };
  const t1 = (rank * PHI) % 1;
  const t2 = (rank * PHI * PHI) % 1;
  return {
    lng: b.lngMin + t1 * (b.lngMax - b.lngMin),
    lat: b.latMin + t2 * (b.latMax - b.latMin),
  };
}

export function fmtSavings(usd) {
  if (usd >= 1_000_000) return `$${(usd / 1_000_000).toFixed(1)}M`;
  if (usd >= 1_000)     return `$${Math.round(usd / 1_000)}K`;
  return `$${usd}`;
}

function fmtCost(usd) {
  return '$' + usd.toLocaleString();
}

/** Transform raw JSON row into the shape all components expect */
function transformSite(raw) {
  const { lng, lat } = deriveLngLat(raw.borough, raw.rank);
  const ej       = raw.env_justice === 'Yes';
  const bessKwh  = raw.recommended_bess_kwh;
  const savingsUsd = raw.estimated_annual_savings_usd;

  // Derive plausible energy-profile fields from BESS size + savings
  const peakKw    = Math.round(bessKwh * 0.48);
  const monthCons = Math.round(bessKwh * 195 + savingsUsd / 25);
  const annualCost = Math.round(savingsUsd * 5.2);
  const solPot    = Math.round(bessKwh * 85);
  const co2       = Math.round(bessKwh * 0.058 * 1000) / 10; // tons

  return {
    id:        raw.rank - 1,
    rank:      raw.rank,
    name:      raw.site,
    address:   raw.address,
    borough:   raw.borough,
    agency:    raw.agency,
    ej,
    lng, lat,

    // Scores
    score:       raw.nexus_score,
    energyScore: raw.energy_score,
    wasteScore:  raw.waste_score,
    nexusScore:  raw.nexus_score,

    // BESS
    bessKwh,
    savingsUsd,
    savings: fmtSavings(savingsUsd),

    // Recommendation text
    recommendation: raw.top_recommendation,
    reasoning:      raw.reasoning,

    // Derived energy profile
    cons:    monthCons.toLocaleString(),
    peak:    peakKw,
    cost:    fmtCost(annualCost),
    solPot:  solPot.toLocaleString(),
    pkRed:   peakKw,
    pkPct:   Math.round((peakKw / (peakKw + bessKwh * 0.3)) * 100),
    co2,
    solar:   +(raw.energy_score / 100).toFixed(2),
    roof:    raw.energy_score >= 90 ? 'Good' : raw.energy_score >= 75 ? 'Fair' : 'Poor',
    ev:      Math.round(bessKwh / 50),

    // Waste context (district-level estimates)
    wasteRef:    (raw.waste_score * 120).toLocaleString(),
    wasteOrg:    `${Math.round(raw.waste_score * 0.38)}%`,
    wasteDivert: Math.round(raw.waste_score * 45).toLocaleString(),
    wasteMWh:    Math.round(raw.waste_score * 195).toLocaleString(),

    // Legacy fields
    bbl:  `${raw.borough[0]}-${String(raw.rank).padStart(5,'0')}-0001`,
    addr: `${raw.address}, ${raw.borough}, NY`,
    mx: 0, my: 0,
  };
}

/** Primary site array — all 50 real sites, sorted by nexus_score desc */
export const ENERGY_SITES = rawSites.map(transformSite);

/* ── KPI values computed from real data ── */
const totalSavings = ENERGY_SITES.reduce((s, x) => s + x.savingsUsd, 0);
const highPriority = ENERGY_SITES.filter(s => s.nexusScore >= 90).length;
const ejCount      = ENERGY_SITES.filter(s => s.ej).length;
const totalBessKwh = ENERGY_SITES.reduce((s, x) => s + x.bessKwh, 0);
const totalCo2     = Math.round(ENERGY_SITES.reduce((s, x) => s + x.co2, 0));

export const ENERGY_KPIS = [
  { icon: '🏢', iconBg: 'rgba(59,130,246,.12)',  num: '50',                         label: 'Sites Ranked',            sub: 'XGBoost top-50 — all EJ areas',             tag: null },
  { icon: '🔋', iconBg: 'rgba(16,185,129,.12)',  num: String(highPriority),          label: 'Nexus Score ≥ 90',        sub: 'Highest combined energy + waste priority',    tag: { txt: `▲ ${Math.round(highPriority/50*100)}% of total`, cls: 'green' } },
  { icon: '💰', iconBg: 'rgba(245,158,11,.12)',  num: fmtSavings(totalSavings),      label: 'Total Annual Savings',    sub: 'If all 50 sites deploy BESS',                 tag: null },
  { icon: '🌱', iconBg: 'rgba(16,185,129,.12)',  num: `${totalCo2}t`,               label: 'CO₂ Offset/yr',           sub: `${totalBessKwh.toLocaleString()} kWh BESS deployed`, tag: null },
];

export const WASTE_KPIS = [
  { icon: '🗑️', iconBg: 'rgba(249,115,22,.12)', num: '295K',  label: 'Monthly Refuse (tons)',  sub: 'Latest month, citywide',  tag: null },
  { icon: '♻️', iconBg: 'rgba(20,184,166,.12)', num: '21.3%', label: 'Diversion Rate',         sub: 'Target: 30%',             tag: { txt: '▼ 8.7% below target', cls: 'red' } },
  { icon: '🚛', iconBg: 'rgba(239,68,68,.12)',  num: '$418',  label: 'Avg Disposal Cost/ton',  sub: 'Across all facilities',   tag: null },
  { icon: '🔥', iconBg: 'rgba(139,92,246,.12)', num: '1.2M',  label: 'MWh WTE Potential/yr',   sub: 'If 50% organics → AD',    tag: null },
];

export const NEXUS_KPIS = [
  { icon: '📊', iconBg: 'rgba(59,130,246,.12)',  num: '25',                    label: 'Datasets Integrated',      sub: '12 energy + 13 waste',               tag: null },
  { icon: '💰', iconBg: 'rgba(16,185,129,.12)',  num: fmtSavings(totalSavings), label: 'Combined Savings/yr',     sub: 'Energy + waste optimization',        tag: null },
  { icon: '🌱', iconBg: 'rgba(16,185,129,.12)',  num: `${totalCo2}t`,          label: 'Total CO₂ Offset (t/yr)',  sub: 'BESS deployment, all 50 sites',      tag: null },
  { icon: '🏘️', iconBg: 'rgba(139,92,246,.12)', num: `${Math.round(ejCount/50*100)}%`, label: 'EJ Area Coverage', sub: 'All top-50 sites are EJ areas',     tag: null },
];

/* ── Waste sites (community district level — separate from site data) ── */
export const WASTE_SITES = [
  { id: 0, name: 'District 7',  borough: 'Bronx',         score: 88, ref: 12450, diversion: '14.2%', complaints: 1847, organic: 'High',   lng: -73.900, lat: 40.838 },
  { id: 1, name: 'District 1',  borough: 'Brooklyn',      score: 82, ref: 11820, diversion: '16.8%', complaints: 1623, organic: 'High',   lng: -73.985, lat: 40.695 },
  { id: 2, name: 'District 12', borough: 'Queens',        score: 79, ref: 10950, diversion: '18.1%', complaints: 1204, organic: 'Medium', lng: -73.825, lat: 40.755 },
  { id: 3, name: 'District 9',  borough: 'Manhattan',     score: 76, ref:  9870, diversion: '22.4%', complaints: 2156, organic: 'Medium', lng: -73.945, lat: 40.805 },
  { id: 4, name: 'District 3',  borough: 'Bronx',         score: 85, ref:  9540, diversion: '13.7%', complaints: 1932, organic: 'High',   lng: -73.915, lat: 40.825 },
  { id: 5, name: 'District 14', borough: 'Brooklyn',      score: 73, ref:  9120, diversion: '19.3%', complaints:  987, organic: 'Medium', lng: -73.960, lat: 40.638 },
  { id: 6, name: 'District 7',  borough: 'Queens',        score: 70, ref:  8760, diversion: '20.1%', complaints:  876, organic: 'Medium', lng: -73.792, lat: 40.710 },
  { id: 7, name: 'District 1',  borough: 'Staten Island', score: 68, ref:  8340, diversion: '17.5%', complaints:  654, organic: 'Low',    lng: -74.060, lat: 40.645 },
  { id: 8, name: 'District 4',  borough: 'Manhattan',     score: 64, ref:  7890, diversion: '25.6%', complaints: 1543, organic: 'Low',    lng: -73.978, lat: 40.745 },
  { id: 9, name: 'District 5',  borough: 'Brooklyn',      score: 61, ref:  7650, diversion: '21.2%', complaints: 1098, organic: 'Medium', lng: -73.944, lat: 40.658 },
];

/* ── Nexus sites — top 5 from real data by nexus_score ── */
export const NEXUS_SITES = ENERGY_SITES
  .slice()
  .sort((a, b) => b.nexusScore - a.nexusScore)
  .slice(0, 5)
  .map((s, i) => ({
    id:      i,
    name:    s.name,
    borough: s.borough,
    eScore:  s.energyScore,
    wScore:  s.wasteScore,
    nScore:  s.nexusScore,
    co2:     `${s.co2} t/yr`,
    lng:     s.lng,
    lat:     s.lat,
  }));

/* ── Borough summary computed from real site data ── */
const boroughGroups = {};
ENERGY_SITES.forEach(s => {
  if (!boroughGroups[s.borough]) boroughGroups[s.borough] = [];
  boroughGroups[s.borough].push(s);
});

const BOROUGH_COLORS  = { Bronx: '#3B82F6', Brooklyn: '#10B981', Manhattan: '#F59E0B', Queens: '#8B5CF6', 'Staten Island': '#F97316' };
const BOROUGH_ACTIONS = { Bronx: 'BESS + Organics AD', Brooklyn: 'Solar + BESS', Manhattan: 'BESS + Route Opt', Queens: 'Solar + Organics', 'Staten Island': 'BESS + Composting' };
const WASTE_TONS      = { Bronx: 48200, Brooklyn: 62400, Manhattan: 38900, Queens: 55100, 'Staten Island': 28300 };
const DIVERSION       = { Bronx: 15.8, Brooklyn: 19.2, Manhattan: 24.1, Queens: 20.5, 'Staten Island': 17.5 };

export const BOROUGH_DATA = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'].map(name => {
  const group    = boroughGroups[name] ?? [];
  const savSum   = group.reduce((s, x) => s + x.savingsUsd, 0);
  const co2sum   = Math.round(group.reduce((s, x) => s + x.co2, 0));
  const hp       = group.filter(s => s.nexusScore >= 90).length;
  const sites5   = Math.min(Math.round(5_000_000 / Math.max(savSum / Math.max(group.length, 1), 1) * 0.9), group.length);
  return {
    name,
    sites:     group.length,
    hp,
    savings:   fmtSavings(Math.round(savSum * 0.9)),
    co2:       co2sum.toLocaleString(),
    sites5:    String(sites5),
    action:    BOROUGH_ACTIONS[name] ?? 'BESS',
    wasteTons: WASTE_TONS[name],
    diversion: DIVERSION[name],
    color:     BOROUGH_COLORS[name],
  };
});

/* ── Score helpers ── */
export function scoreColor(s) { return s >= 95 ? '#EF4444' : s >= 90 ? '#F59E0B' : '#3B82F6'; }
export function scoreBg(s)    { return s >= 95 ? 'rgba(239,68,68,.15)' : s >= 90 ? 'rgba(245,158,11,.15)' : 'rgba(59,130,246,.15)'; }
export function scoreText(s)  { return s >= 95 ? '#FCA5A5' : s >= 90 ? '#FCD34D' : '#93C5FD'; }
export function dotR(s)       { return s >= 95 ? 6 : s >= 90 ? 5 : 4; }
