import { DashboardProvider } from './context/DashboardContext';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import KPIRibbon from './components/KPIRibbon';
import CityMap from './components/Map/CityMap';
import RankingsTable from './components/RankingsTable';
import SiteDetail from './components/SiteDetail';
import FlowVisualization from './components/FlowVisualization';
import Simulation from './components/Simulation';
import BoroughComparison from './components/BoroughComparison';

function Dashboard() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* ── SIDEBAR ── */}
      <Sidebar />

      {/* ── MAIN CONTENT ── */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        {/* ── HEADER ── */}
        <Header />

        {/* ── SCROLLABLE MAIN ── */}
        <main style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 20, flex: 1 }}>

          {/* SECTION 1: KPI RIBBON */}
          <KPIRibbon />

          {/* SECTION 2: MAP + RANKINGS */}
          <div>
            <div style={{
              fontSize: 10, fontWeight: 600, color: '#475569',
              letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
            }}>
              Site Distribution &amp; Rankings
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 14, alignItems: 'start' }}>
              {/* Map — 3fr */}
              <div style={{ aspectRatio: '4/3', minHeight: 360 }}>
                <CityMap />
              </div>
              {/* Rankings table — 2fr */}
              <RankingsTable />
            </div>
          </div>

          {/* SECTION 3: SITE DETAIL */}
          <SiteDetail />

          {/* SECTION 4: FLOW VISUALIZATION */}
          <FlowVisualization />

          {/* SECTION 5: SIMULATION */}
          <Simulation />

          {/* SECTION 6: BOROUGH COMPARISON */}
          <BoroughComparison />

          {/* FOOTER */}
          <div style={{
            borderTop: '1px solid rgba(255,255,255,.05)',
            padding: '14px 0 20px',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          }}>
            <span style={{ fontSize: 11, color: '#334155' }}>NYC Smart City Nexus v2.0 — Spark Hack NYC 2026</span>
            <span style={{ fontSize: 11, color: '#334155' }}>Data: NYC Open Data (25 datasets) · Model: Claude API · Hardware: NVIDIA GB10</span>
            <span style={{ fontSize: 11, color: '#334155' }}>Built with ⚡ by Team Nexus</span>
          </div>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <DashboardProvider>
      <Dashboard />
    </DashboardProvider>
  );
}
