import { useDashboard } from '../context/DashboardContext';
import { ENERGY_KPIS, WASTE_KPIS, NEXUS_KPIS } from '../data/sites';

function KPICard({ kpi }) {
  const tagColor = kpi.tag?.cls === 'green'
    ? { bg: 'rgba(16,185,129,.15)', text: '#6EE7B7' }
    : { bg: 'rgba(239,68,68,.15)', text: '#FCA5A5' };

  return (
    <div style={{
      background: '#131C2E', border: '1px solid #1E293B',
      borderRadius: 12, padding: 14,
    }}>
      <div style={{
        width: 30, height: 30, borderRadius: 8, background: kpi.iconBg,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 16, marginBottom: 10,
      }}>
        {kpi.icon}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, lineHeight: 1, marginBottom: 3, fontVariantNumeric: 'tabular-nums' }}>
        {kpi.num}
      </div>
      <div style={{ fontSize: 12, color: '#64748B', marginBottom: 2 }}>{kpi.label}</div>
      <div style={{ fontSize: 11, color: '#334155' }}>{kpi.sub}</div>
      {kpi.tag && (
        <div style={{ marginTop: 5 }}>
          <span style={{
            display: 'inline-block', padding: '2px 6px', borderRadius: 20,
            fontSize: 11, fontWeight: 500,
            background: tagColor.bg, color: tagColor.text,
          }}>
            {kpi.tag.txt}
          </span>
        </div>
      )}
    </div>
  );
}

export default function KPIRibbon() {
  const { viewMode } = useDashboard();
  const kpis = viewMode === 'energy' ? ENERGY_KPIS : viewMode === 'waste' ? WASTE_KPIS : NEXUS_KPIS;

  return (
    <div>
      <div style={{
        fontSize: 10, fontWeight: 600, color: '#475569',
        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
      }}>
        System Overview
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 12 }}>
        {kpis.map((kpi, i) => <KPICard key={i} kpi={kpi} />)}
      </div>
    </div>
  );
}
