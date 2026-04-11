import { useMemo } from 'react';
import { useDashboard } from '../context/DashboardContext';
import { ENERGY_SITES, WASTE_SITES, NEXUS_SITES, scoreBg, scoreText } from '../data/sites';

function ScoreBadge({ score }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      padding: '2px 7px', borderRadius: 20, fontSize: 11, fontWeight: 700,
      background: scoreBg(score), color: scoreText(score),
      minWidth: 32,
    }}>
      {score}
    </span>
  );
}

function OrganicPill({ organic }) {
  const map = {
    High:   { bg: 'rgba(16,185,129,.15)',  text: '#6EE7B7' },
    Medium: { bg: 'rgba(245,158,11,.15)', text: '#FCD34D' },
    Low:    { bg: 'rgba(100,116,139,.15)', text: '#94A3B8' },
  };
  const s = map[organic] ?? map.Low;
  return (
    <span style={{
      display: 'inline-block', padding: '2px 6px', borderRadius: 20, fontSize: 11,
      background: s.bg, color: s.text, fontWeight: 500,
    }}>
      {organic}
    </span>
  );
}

const TH = { padding: '6px 8px', textAlign: 'left', fontSize: 10, color: '#475569', fontWeight: 600, letterSpacing: '0.05em', borderBottom: '1px solid #1E293B', whiteSpace: 'nowrap' };
const TD = { padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12 };

export default function RankingsTable() {
  const { viewMode, selectedId, selectSite, borough, minScore } = useDashboard();

  const { rows, count } = useMemo(() => {
    if (viewMode === 'energy') {
      const r = ENERGY_SITES.filter(s =>
        (borough === 'All Boroughs' || s.borough === borough) && s.score >= minScore
      );
      return { rows: r, count: r.length };
    }
    if (viewMode === 'waste') {
      const r = WASTE_SITES.filter(s =>
        (borough === 'All Boroughs' || s.borough === borough) && s.score >= minScore
      );
      return { rows: r, count: r.length };
    }
    // nexus
    const r = NEXUS_SITES.filter(s => borough === 'All Boroughs' || s.borough === borough);
    return { rows: r, count: r.length };
  }, [viewMode, borough, minScore]);

  return (
    <div style={{ background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 14px', borderBottom: '1px solid #1E293B',
      }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>Top Sites</span>
        <span style={{
          display: 'inline-block', padding: '2px 6px', borderRadius: 20, fontSize: 11,
          background: 'rgba(59,130,246,.15)', color: '#93C5FD',
        }}>
          {count} results
        </span>
      </div>

      {/* Scrollable table */}
      <div style={{ overflowY: 'auto', maxHeight: 360 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          {viewMode === 'energy' && (
            <>
              <thead>
                <tr>
                  <th style={TH}>#</th>
                  <th style={TH}>Site</th>
                  <th style={TH}>Borough</th>
                  <th style={TH}>Energy</th>
                  <th style={TH}>Waste</th>
                  <th style={TH}>Nexus</th>
                  <th style={TH}>BESS</th>
                  <th style={TH}>Savings/yr</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((s) => {
                  const isSelected = s.id === selectedId;
                  return (
                    <tr
                      key={s.id}
                      onClick={() => selectSite(s.id)}
                      style={{
                        cursor: 'pointer',
                        background: isSelected ? 'rgba(59,130,246,.08)' : 'transparent',
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={e => !isSelected && (e.currentTarget.style.background = 'rgba(59,130,246,.04)')}
                      onMouseLeave={e => !isSelected && (e.currentTarget.style.background = 'transparent')}
                    >
                      <td style={{ ...TD, color: '#475569', borderLeft: isSelected ? '2px solid #3B82F6' : '2px solid transparent' }}>{s.rank}</td>
                      <td style={{ ...TD, fontWeight: 500, maxWidth: 130, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                          title={s.name}>{s.name}</td>
                      <td style={TD}>
                        <span style={{ background: 'rgba(100,116,139,.15)', color: '#94A3B8', padding: '2px 6px', borderRadius: 20, fontSize: 10 }}>
                          {s.borough === 'Staten Island' ? 'S.I.' : s.borough}
                        </span>
                      </td>
                      <td style={TD}><ScoreBadge score={s.energyScore} /></td>
                      <td style={TD}><ScoreBadge score={s.wasteScore} /></td>
                      <td style={TD}><ScoreBadge score={s.nexusScore} /></td>
                      <td style={{ ...TD, color: '#93C5FD', fontVariantNumeric: 'tabular-nums' }}>{s.bessKwh} kWh</td>
                      <td style={{ ...TD, color: '#10B981', fontWeight: 600 }}>{s.savings}</td>
                    </tr>
                  );
                })}
              </tbody>
            </>
          )}

          {viewMode === 'waste' && (
            <>
              <thead>
                <tr>
                  <th style={TH}>#</th>
                  <th style={TH}>District</th>
                  <th style={TH}>Borough</th>
                  <th style={TH}>Score</th>
                  <th style={TH}>Refuse t/mo</th>
                  <th style={TH}>Diversion</th>
                  <th style={TH}>Organic</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((s, i) => (
                  <tr key={s.id} style={{ cursor: 'pointer' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,.04)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ ...TD, color: '#475569' }}>{i + 1}</td>
                    <td style={{ ...TD, fontWeight: 500 }}>{s.name}</td>
                    <td style={TD}><span style={{ background: 'rgba(100,116,139,.15)', color: '#94A3B8', padding: '2px 6px', borderRadius: 20, fontSize: 10 }}>{s.borough}</span></td>
                    <td style={TD}><ScoreBadge score={s.score} /></td>
                    <td style={{ ...TD, fontVariantNumeric: 'tabular-nums' }}>{s.ref.toLocaleString()}</td>
                    <td style={{ ...TD, color: '#F59E0B' }}>{s.diversion}</td>
                    <td style={TD}><OrganicPill organic={s.organic} /></td>
                  </tr>
                ))}
              </tbody>
            </>
          )}

          {viewMode === 'nexus' && (
            <>
              <thead>
                <tr>
                  <th style={TH}>#</th>
                  <th style={TH}>Location</th>
                  <th style={TH}>Borough</th>
                  <th style={TH}>Energy</th>
                  <th style={TH}>Waste</th>
                  <th style={TH}>Nexus</th>
                  <th style={TH}>CO₂</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((s, i) => (
                  <tr key={s.id} style={{ cursor: 'pointer' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,.04)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <td style={{ ...TD, color: '#475569' }}>{i + 1}</td>
                    <td style={{ ...TD, fontWeight: 500 }}>{s.name}</td>
                    <td style={TD}><span style={{ background: 'rgba(100,116,139,.15)', color: '#94A3B8', padding: '2px 6px', borderRadius: 20, fontSize: 10 }}>{s.borough}</span></td>
                    <td style={TD}><ScoreBadge score={s.eScore} /></td>
                    <td style={TD}><span style={{ background: 'rgba(245,158,11,.15)', color: '#FCD34D', padding: '2px 7px', borderRadius: 20, fontSize: 11, fontWeight: 700 }}>{s.wScore}</span></td>
                    <td style={TD}><span style={{ background: 'rgba(139,92,246,.15)', color: '#C4B5FD', padding: '2px 7px', borderRadius: 20, fontSize: 11, fontWeight: 700 }}>{s.nScore}</span></td>
                    <td style={{ ...TD, color: '#10B981' }}>{s.co2}</td>
                  </tr>
                ))}
              </tbody>
            </>
          )}
        </table>
      </div>
    </div>
  );
}
