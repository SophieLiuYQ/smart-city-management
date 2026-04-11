import { BOROUGH_DATA } from '../data/sites';

const CARD = { background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, padding: 16 };

function BoroughBarChart() {
  const maxSites = Math.max(...BOROUGH_DATA.map(b => b.sites));
  const W = 540, H = 240, pad = { l: 90, r: 16, t: 12, b: 12 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const rowH = iH / BOROUGH_DATA.length;
  const barH = rowH * 0.35;
  const xs = v => (v / maxSites) * iW;
  const gridVals = [0, 250, 500, 750, 1000];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ overflow: 'visible' }}>
      {/* Grid lines */}
      {gridVals.map(v => (
        <g key={v}>
          <line
            x1={pad.l + xs(v)} y1={pad.t}
            x2={pad.l + xs(v)} y2={H - pad.b}
            stroke="rgba(19,29,46,.8)" strokeWidth={1}
          />
          <text x={pad.l + xs(v)} y={H - pad.b + 10} fill="#475569" fontSize={8} textAnchor="middle">{v}</text>
        </g>
      ))}

      {BOROUGH_DATA.map((b, i) => {
        const y = pad.t + i * rowH;
        return (
          <g key={b.name}>
            {/* Label */}
            <text x={pad.l - 8} y={y + rowH / 2 + 3} fill="#94A3B8" fontSize={10} textAnchor="end" fontWeight={500}>
              {b.name === 'Staten Island' ? 'S. Island' : b.name}
            </text>
            {/* Total sites bar */}
            <rect
              x={pad.l} y={y + rowH * 0.08}
              width={xs(b.sites)} height={barH}
              rx={3} fill="#243148"
            />
            {/* High priority bar */}
            <rect
              x={pad.l} y={y + rowH * 0.08 + barH + 3}
              width={xs(b.hp)} height={barH}
              rx={3} fill="#3B82F6"
            />
            {/* Value labels */}
            <text x={pad.l + xs(b.sites) + 4} y={y + rowH * 0.08 + barH - 3} fill="#475569" fontSize={8}>{b.sites}</text>
            <text x={pad.l + xs(b.hp) + 4} y={y + rowH * 0.08 + barH * 2 + 1} fill="#60A5FA" fontSize={8}>{b.hp}</text>
          </g>
        );
      })}
    </svg>
  );
}

function BoroughCard({ b }) {
  return (
    <div style={CARD}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%', background: b.color,
          display: 'inline-block', flexShrink: 0,
        }} />
        <span style={{ fontSize: 12, fontWeight: 600 }}>{b.name}</span>
      </div>
      <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}>If budget = $5M</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: '#60A5FA' }}>{b.sites5} sites</div>
      <div style={{ marginTop: 6, display: 'grid', gap: 3 }}>
        <div style={{ fontSize: 11, color: '#6EE7B7' }}>{b.savings}/yr savings</div>
        <div style={{ fontSize: 11, color: '#93C5FD' }}>{b.co2} t CO₂</div>
        <div style={{ fontSize: 11, color: '#C4B5FD', marginTop: 2 }}>{b.action}</div>
      </div>
    </div>
  );
}

export default function BoroughComparison() {
  return (
    <div>
      <div style={{
        fontSize: 10, fontWeight: 600, color: '#475569',
        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
      }}>
        Borough-Level Overview
      </div>

      <div style={CARD}>
        {/* Legend */}
        <div style={{ display: 'flex', gap: 14, marginBottom: 10 }}>
          {[['#243148', 'Total Sites'], ['#3B82F6', 'High Priority (≥70)']].map(([c, l]) => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 12, color: '#64748B' }}>
              <span style={{ width: 10, height: 10, borderRadius: 2, background: c, display: 'inline-block' }} />
              {l}
            </span>
          ))}
        </div>
        <BoroughBarChart />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: 10, marginTop: 12 }}>
        {BOROUGH_DATA.map(b => <BoroughCard key={b.name} b={b} />)}
      </div>
    </div>
  );
}
