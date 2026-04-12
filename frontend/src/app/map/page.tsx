'use client';

import 'maplibre-gl/dist/maplibre-gl.css';
import { DashboardProvider } from '@/context/DashboardContext';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import KPIRibbon from '@/components/KPIRibbon';
import CityMap from '@/components/Map/CityMap';
import RankingsTable from '@/components/RankingsTable';
import SiteDetail from '@/components/SiteDetail';
import FlowVisualization from '@/components/FlowVisualization';
import Simulation from '@/components/Simulation';
import BoroughComparison from '@/components/BoroughComparison';

function Dashboard() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#F1F5F9', paddingLeft: '50px', paddingRight: '50px' }}>
      {/* <Sidebar /> */}

      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
        <Header />

        <main style={{ padding: 20, display: 'flex', flexDirection: 'column', gap: 20, flex: 1 }}>
          {/* <KPIRibbon /> */}

          <div>
            <div style={{
              fontSize: 10, fontWeight: 600, color: '#475569',
              letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
            }}>
              Site Distribution &amp; Rankings
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 14, alignItems: 'start' }}>
              <div style={{ height: 480 }}>
                <CityMap />
              </div>
              <div style={{ height: 480 }}>
                <RankingsTable />
              </div>
            </div>
          </div>

          <SiteDetail />
          <FlowVisualization />

          <div style={{ padding: '100px' }}></div>
          {/* <Simulation /> */}
          {/* <BoroughComparison /> */}

          <div style={{
            borderTop: '1px solid rgba(0,0,0,.07)',
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

export default function MapPage() {
  return (
    <DashboardProvider>
      <Dashboard />
    </DashboardProvider>
  );
}
