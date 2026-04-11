import { useMemo } from 'react';
import { useDashboard } from '../context/DashboardContext';
import { ENERGY_SITES, scoreColor } from '../data/sites';

function KVItem({ label, value, valueStyle }) {
  return (
    <div style={{ background: '#0B1120', borderRadius: 6, padding: '8px 10px' }}>
      <div style={{ fontSize: 10, color: '#475569', marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 13, fontWeight: 600, ...valueStyle }}>{value}</div>
    </div>
  );
}

function ScoreRing({ score, label }) {
  const r = 28;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - score / 100);
  const color = scoreColor(score);
  return (
    <div style={{ textAlign: 'center' }}>
      <svg viewBox="0 0 64 64" width={64} height={64} style={{ display: 'block', margin: '0 auto' }}>
        <circle cx="32" cy="32" r={r} fill="none" stroke="#1E293B" strokeWidth="6" />
        <circle cx="32" cy="32" r={r} fill="none"
          stroke={color} strokeWidth="6"
          strokeDasharray={circ.toFixed(1)} strokeDashoffset={offset.toFixed(1)}
          strokeLinecap="round" transform="rotate(-90 32 32)"
        />
        <text x="32" y="37" textAnchor="middle" fill="#F1F5F9" fontSize="14" fontWeight="700" fontFamily="system-ui">{score}</text>
      </svg>
      <div style={{ fontSize: 10, color: '#475569', marginTop: 2 }}>{label}</div>
    </div>
  );
}

export default function SiteDetail() {
  const { selectedId } = useDashboard();
  const site = useMemo(() => ENERGY_SITES[selectedId] ?? ENERGY_SITES[0], [selectedId]);

  const roofStyle = site.roof === 'Good'
    ? { bg: 'rgba(16,185,129,.15)', text: '#6EE7B7' }
    : site.roof === 'Fair'
    ? { bg: 'rgba(245,158,11,.15)', text: '#FCD34D' }
    : { bg: 'rgba(239,68,68,.15)', text: '#FCA5A5' };

  return (
    <div>
      <div style={{
        fontSize: 10, fontWeight: 600, color: '#475569',
        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
        display: 'flex', alignItems: 'center', gap: 6,
      }}>
        Site Detail —
        <span style={{ color: '#60A5FA', textTransform: 'none', letterSpacing: 0, fontSize: 12, fontWeight: 500 }}>
          {site.name}
        </span>
        <span style={{ background: 'rgba(139,92,246,.15)', color: '#C4B5FD', padding: '1px 7px', borderRadius: 20, fontSize: 10, fontWeight: 600, marginLeft: 4 }}>
          Rank #{site.rank}
        </span>
      </div>

      <div style={{ background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, padding: 16 }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 20 }}>

          {/* ── Column 1: Identity ── */}
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, marginBottom: 10, lineHeight: 1.3 }}>{site.name}</div>
            <div style={{ display: 'grid', gap: 6 }}>
              <KVItem label="ADDRESS" value={`${site.address}, ${site.borough}, NY`} valueStyle={{ fontSize: 11, color: '#94A3B8' }} />
              <KVItem label="AGENCY"  value={site.agency} />
              <KVItem label="BBL"     value={site.bbl}    valueStyle={{ fontFamily: 'monospace', fontSize: 12, color: '#64748B' }} />
              <div style={{ background: '#0B1120', borderRadius: 6, padding: '8px 10px' }}>
                <div style={{ fontSize: 10, color: '#475569', marginBottom: 2 }}>EJ STATUS</div>
                <span style={{
                  display: 'inline-block', padding: '2px 6px', borderRadius: 20, fontSize: 11, fontWeight: 500,
                  background: site.ej ? 'rgba(16,185,129,.15)' : 'rgba(100,116,139,.15)',
                  color: site.ej ? '#6EE7B7' : '#94A3B8',
                }}>
                  {site.ej ? '✅ EJ Area' : 'Non-EJ'}
                </span>
              </div>
              <div style={{ background: '#0B1120', borderRadius: 6, padding: '8px 10px' }}>
                <div style={{ fontSize: 10, color: '#475569', marginBottom: 4 }}>DERIVED ENERGY PROFILE</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
                  <div style={{ color: '#64748B' }}>Monthly avg <span style={{ color: '#F1F5F9', fontWeight: 600 }}>{site.cons} kWh</span></div>
                  <div style={{ color: '#64748B' }}>Peak demand <span style={{ color: '#F1F5F9', fontWeight: 600 }}>{site.peak} kW</span></div>
                  <div style={{ color: '#64748B' }}>Annual cost <span style={{ color: '#FCA5A5', fontWeight: 600 }}>{site.cost}</span></div>
                  <div style={{ color: '#64748B' }}>Solar pot. <span style={{ color: '#FCD34D', fontWeight: 600 }}>{site.solPot} kWh/yr</span></div>
                  <div style={{ color: '#64748B' }}>Roof <span style={{ display: 'inline-block', padding: '1px 5px', borderRadius: 20, fontSize: 10, background: roofStyle.bg, color: roofStyle.text }}>{site.roof}</span></div>
                  <div style={{ color: '#64748B' }}>EV ports <span style={{ color: '#F1F5F9', fontWeight: 600 }}>{site.ev} nearby</span></div>
                </div>
              </div>
            </div>
          </div>

          {/* ── Column 2: Score breakdown ── */}
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, color: '#475569', letterSpacing: '0.08em', marginBottom: 12 }}>XGBoost SCORE BREAKDOWN</div>
            <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: 14 }}>
              <ScoreRing score={site.energyScore} label="Energy" />
              <ScoreRing score={site.wasteScore}  label="Waste"  />
              <ScoreRing score={site.nexusScore}  label="Nexus"  />
            </div>
            {/* Score bars */}
            {[
              { label: 'Energy Score', val: site.energyScore, color: '#3B82F6' },
              { label: 'Waste Score',  val: site.wasteScore,  color: '#F59E0B' },
              { label: 'Nexus Score',  val: site.nexusScore,  color: '#8B5CF6' },
            ].map(({ label, val, color }) => (
              <div key={label} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#64748B', marginBottom: 3 }}>
                  <span>{label}</span><span style={{ color, fontWeight: 700 }}>{val}/100</span>
                </div>
                <div style={{ height: 4, background: '#1E293B', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${val}%`, background: color, borderRadius: 2, transition: 'width 0.4s ease' }} />
                </div>
              </div>
            ))}
            {/* Waste context */}
            <div style={{ marginTop: 10, background: '#0B1120', borderRadius: 6, padding: '8px 10px', fontSize: 11, color: '#64748B' }}>
              <div style={{ fontSize: 9, color: '#334155', marginBottom: 4, fontWeight: 600, letterSpacing: '0.08em' }}>DISTRICT WASTE CONTEXT</div>
              Refuse: <span style={{ color: '#F59E0B' }}>{site.wasteRef} t/mo</span> · Organic: <span style={{ color: '#10B981' }}>{site.wasteOrg}</span>
              {' '}→ <span style={{ color: '#10B981' }}>{site.wasteDivert} t/mo</span> → <span style={{ color: '#10B981' }}>~{site.wasteMWh} MWh/yr</span> biogas
            </div>
          </div>

          {/* ── Column 3: BESS recommendation ── */}
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, color: '#475569', letterSpacing: '0.08em', marginBottom: 12 }}>BESS RECOMMENDATION</div>
            <div style={{ display: 'grid', gap: 6 }}>
              <KVItem label="Recommended Capacity" value={`${site.bessKwh} kWh BESS`} valueStyle={{ color: '#93C5FD' }} />
              <KVItem label="Est. Annual Savings"   value={`${site.savings}/yr`}        valueStyle={{ color: '#6EE7B7' }} />
              <KVItem label="Peak Reduction"        value={`${site.pkRed} kW (${site.pkPct}%)`} />
              <KVItem label="CO₂ Offset"            value={`${site.co2} tons/yr`}       valueStyle={{ color: '#6EE7B7' }} />
            </div>
            <div style={{
              marginTop: 10, background: 'rgba(59,130,246,.06)',
              border: '1px solid rgba(59,130,246,.2)', borderRadius: 8, padding: '10px 12px',
            }}>
              <div style={{ fontSize: 9, color: '#475569', fontWeight: 600, letterSpacing: '0.08em', marginBottom: 5 }}>TOP RECOMMENDATION</div>
              <div style={{ fontSize: 11, color: '#93C5FD', lineHeight: 1.5 }}>{site.recommendation}</div>
            </div>
          </div>
        </div>

        {/* ── AI / Model Insight bar ── */}
        <div style={{
          marginTop: 14, background: 'rgba(139,92,246,.05)',
          border: '1px solid rgba(139,92,246,.2)', borderRadius: 8, padding: 12,
          display: 'flex', alignItems: 'flex-start', gap: 10,
        }}>
          <span style={{
            display: 'inline-block', padding: '2px 6px', borderRadius: 20, fontSize: 11,
            fontWeight: 500, background: 'rgba(139,92,246,.15)', color: '#C4B5FD',
            whiteSpace: 'nowrap', flexShrink: 0,
          }}>
            🤖 XGBoost Insight
          </span>
          <span style={{ fontSize: 12, color: '#94A3B8', lineHeight: 1.6 }}>
            {site.reasoning}
            {' '}Estimated savings of <strong style={{ color: '#10B981' }}>{site.savings}/yr</strong> with
            a <strong style={{ color: '#93C5FD' }}>{site.bessKwh} kWh BESS</strong> deployment,
            offsetting ~<strong style={{ color: '#6EE7B7' }}>{site.co2} tons CO₂/yr</strong>.
            {site.ej && ' This site serves an Environmental Justice community — deployment is equity-weighted.'}
          </span>
        </div>
      </div>
    </div>
  );
}
