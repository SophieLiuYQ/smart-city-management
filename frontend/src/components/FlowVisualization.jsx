/** Custom SVG Sankey-style flow diagrams — Energy Flow + Waste Flow */

const CARD = { background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, padding: 16 };

/* ── Energy Flow nodes ── */
const E_SOURCES = [
  { label: 'Grid (ConEd)', value: '44,200 GWh', color: '#64748B', y: 10 },
  { label: 'Solar',        value: '2,800 GWh',  color: '#F59E0B', y: 54 },
  { label: 'Biogas',       value: '680 GWh',    color: '#10B981', y: 92 },
  { label: 'Wind',         value: '4,320 GWh',  color: '#7DD3FC', y: 130 },
];
const E_SECTORS = [
  { label: 'Commercial',   value: '22,400 GWh', y: 8  },
  { label: 'Residential',  value: '18,900 GWh', y: 48 },
  { label: 'Municipal',    value: '6,200 GWh',  y: 88 },
  { label: 'Industrial',   value: '4,500 GWh',  y: 128 },
];
const E_USES = [
  { label: 'Direct Use',   value: '42,000 GWh', color: '#64748B', y: 10 },
  { label: 'EV Charging',  value: '3,200 GWh',  color: '#3B82F6', y: 52 },
  { label: 'BESS Storage', value: '2,800 GWh',  color: '#10B981', y: 92 },
  { label: 'Losses',       value: '4,000 GWh',  color: '#EF4444', y: 130 },
];

/* ── Waste Flow nodes ── */
const W_SOURCES = [
  { label: 'Residential',   value: '3.1M tons', color: '#3B82F6', y: 10  },
  { label: 'Commercial',    value: '5.2M tons', color: '#64748B', y: 50  },
  { label: 'Institutional', value: '1.4M tons', color: '#14B8A6', y: 90  },
  { label: 'Construction',  value: '4.3M tons', color: '#78716C', y: 130 },
];
const W_COLLECTION = [
  { label: 'DSNY Trucks',   value: '3.5M tons', y: 10  },
  { label: 'Private Carters',value: '8.8M tons', y: 48 },
  { label: 'Drop-off',      value: '0.7M tons', y: 90  },
  { label: 'Self-haul',     value: '1.0M tons', y: 128 },
];
const W_DESTS = [
  { label: 'Landfill',      value: '7.7M (55%)', color: '#EF4444', y: 10  },
  { label: 'WTE Plants',    value: '3.1M (22%)', color: '#F97316', y: 52  },
  { label: 'Recycling',     value: '2.5M (18%)', color: '#3B82F6', y: 92  },
  { label: 'Composting',    value: '0.7M (5%)',  color: '#10B981', y: 130 },
];

function FlowNode({ x, y, w = 80, h = 28, label, value, color = '#94A3B8' }) {
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={4} fill="#1E293B" stroke="#334155" strokeWidth={1} />
      <text x={x + w / 2} y={y + 11} fill="#94A3B8" fontSize={8} textAnchor="middle" fontFamily="system-ui">{label}</text>
      <text x={x + w / 2} y={y + 21} fill={color} fontSize={7.5} textAnchor="middle" fontFamily="system-ui">{value}</text>
    </g>
  );
}

function FlowPath({ x1, y1, x2, y2, h = 12, color = '#64748B', opacity = 0.3 }) {
  const mx = (x1 + x2) / 2;
  const d = `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2} L${x2},${y2 + h} C${mx},${y2 + h} ${mx},${y1 + h} ${x1},${y1 + h} Z`;
  return (
    <path d={d} fill={color} opacity={opacity} />
  );
}

function EnergyFlow() {
  return (
    <div style={CARD}>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#94A3B8', marginBottom: 12 }}>Energy Flow — Citywide</div>
      <svg viewBox="0 0 430 175" width="100%" style={{ overflow: 'visible' }}>
        {/* Source → Sector flows */}
        {E_SOURCES.map((src, si) =>
          E_SECTORS.slice(0, 2).map((sec, di) => (
            <FlowPath key={`es-${si}-${di}`}
              x1={80} y1={src.y + 4 + di * 4}
              x2={185} y2={sec.y + 4 + si * 3}
              h={6} color={src.color} opacity={0.2}
            />
          ))
        )}
        {/* Sector → End Use flows */}
        {E_SECTORS.map((sec, si) =>
          E_USES.slice(0, 2).map((use, di) => (
            <FlowPath key={`su-${si}-${di}`}
              x1={275} y1={sec.y + 4 + di * 3}
              x2={360} y2={use.y + 4 + si * 4}
              h={5} color={E_SOURCES[si % 4].color} opacity={0.18}
            />
          ))
        )}

        {/* Source nodes */}
        {E_SOURCES.map((s, i) => <FlowNode key={i} x={0} y={s.y} label={s.label} value={s.value} color={s.color} />)}
        {/* Sector nodes */}
        {E_SECTORS.map((s, i) => <FlowNode key={i} x={185} y={s.y} label={s.label} value={s.value} />)}
        {/* End use nodes */}
        {E_USES.map((s, i) => <FlowNode key={i} x={360} y={s.y} label={s.label} value={s.value} color={s.color} />)}
      </svg>
    </div>
  );
}

function WasteFlow() {
  return (
    <div style={CARD}>
      <div style={{ fontSize: 12, fontWeight: 600, color: '#94A3B8', marginBottom: 12 }}>Waste Flow — Citywide</div>
      <svg viewBox="0 0 430 175" width="100%" style={{ overflow: 'visible' }}>
        {/* Source → Collection */}
        {W_SOURCES.map((src, si) =>
          W_COLLECTION.slice(0, 2).map((col, di) => (
            <FlowPath key={`wc-${si}-${di}`}
              x1={80} y1={src.y + 4 + di * 4}
              x2={185} y2={col.y + 4 + si * 3}
              h={6} color={src.color} opacity={0.2}
            />
          ))
        )}
        {/* Collection → Destination */}
        {W_COLLECTION.map((col, si) =>
          W_DESTS.slice(0, 2).map((dest, di) => (
            <FlowPath key={`cd-${si}-${di}`}
              x1={275} y1={col.y + 4 + di * 3}
              x2={360} y2={dest.y + 4 + si * 4}
              h={5} color={W_DESTS[si % 4].color} opacity={0.2}
            />
          ))
        )}

        {W_SOURCES.map((s, i) => <FlowNode key={i} x={0}   y={s.y} label={s.label} value={s.value} color={s.color} />)}
        {W_COLLECTION.map((s, i) => <FlowNode key={i} x={185} y={s.y} label={s.label} value={s.value} />)}
        {W_DESTS.map((s, i) => <FlowNode key={i} x={360} y={s.y} label={s.label} value={s.value} color={s.color} />)}
      </svg>
    </div>
  );
}

export default function FlowVisualization() {
  return (
    <div>
      <div style={{
        fontSize: 10, fontWeight: 600, color: '#475569',
        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
      }}>
        Citywide Flow Visualization
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <EnergyFlow />
        <WasteFlow />
      </div>
    </div>
  );
}
