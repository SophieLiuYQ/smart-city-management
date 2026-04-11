import { useDashboard } from '../../context/DashboardContext';

const LEGENDS = {
  energy: [
    { color: '#EF4444', label: 'Score 80–100 (High Priority)' },
    { color: '#F59E0B', label: 'Score 60–79 (Medium)' },
    { color: '#3B82F6', label: 'Score <60 (Low)' },
  ],
  waste: [
    { color: 'rgba(249,115,22,0.35)', label: '>45K tons/mo (High)', square: true },
    { color: 'rgba(249,115,22,0.20)', label: '35–45K tons/mo',      square: true },
    { color: 'rgba(249,115,22,0.08)', label: '<35K tons/mo',        square: true },
    { color: '#EF4444', label: 'District score ≥80' },
    { color: '#F59E0B', label: 'District score 70–79' },
    { color: '#3B82F6', label: 'District score <70' },
  ],
  nexus: [
    { color: 'rgba(192,132,252,0.9)', label: 'High nexus intensity' },
    { color: 'rgba(139,92,246,0.5)',  label: 'Medium nexus' },
    { color: 'rgba(139,92,246,0.15)', label: 'Low nexus' },
    { color: 'rgba(139,92,246,0.3)', label: 'EJ area overlay', square: true },
  ],
};

export default function MapLegend() {
  const { viewMode } = useDashboard();
  const items = LEGENDS[viewMode] ?? [];

  return (
    <div style={{
      position: 'absolute', bottom: 10, left: 10, zIndex: 10,
      background: 'rgba(11,17,32,0.88)', backdropFilter: 'blur(6px)',
      border: '1px solid #1E293B', borderRadius: 8,
      padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 5,
      pointerEvents: 'none',
    }}>
      <div style={{ fontSize: 9, fontWeight: 600, color: '#475569', letterSpacing: '0.08em', marginBottom: 2 }}>
        {viewMode.toUpperCase()} LEGEND
      </div>
      {items.map((item, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {item.square ? (
            <span style={{
              width: 10, height: 10, borderRadius: 2,
              background: item.color,
              border: '1px solid rgba(255,255,255,0.1)',
              display: 'inline-block', flexShrink: 0,
            }} />
          ) : (
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: item.color,
              display: 'inline-block', flexShrink: 0,
            }} />
          )}
          <span style={{ fontSize: 10, color: '#64748B', whiteSpace: 'nowrap' }}>{item.label}</span>
        </div>
      ))}
    </div>
  );
}
