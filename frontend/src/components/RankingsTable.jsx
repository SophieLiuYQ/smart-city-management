'use client';

import { useState } from 'react';
import { TOP10_DISTRICTS_GEOJSON } from '../data/top10Districts';
import { DISTRICT_BUILDINGS } from '../data/districtBuildings';

const districts = TOP10_DISTRICTS_GEOJSON.features.map(f => f.properties);

/* ── style constants ── */
const TH = {
  padding: '6px 8px', textAlign: 'left', fontSize: 10,
  color: '#475569', fontWeight: 600, letterSpacing: '0.05em',
  borderBottom: '1px solid #1E293B', whiteSpace: 'nowrap',
};
const TD = { padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12 };
const BTH = {
  padding: '5px 8px', textAlign: 'left', fontSize: 10,
  color: '#334155', fontWeight: 600, letterSpacing: '0.05em',
  borderBottom: '1px solid rgba(30,41,59,.6)', whiteSpace: 'nowrap',
};
const BTD = { padding: '5px 8px', borderBottom: '1px solid rgba(30,41,59,.35)', fontSize: 11 };

function fmtKwh(n) {
  if (n == null) return '—';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}K`;
  return String(Math.round(n));
}
function fmtUsd(n) {
  if (n == null || n === 0) return '—';
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n}`;
}

/* ── building sub-table rendered inside an expanded district row ── */
function BuildingsPanel({ code }) {
  const slice = DISTRICT_BUILDINGS[code] ?? [];
  const total = slice.length;

  return (
    <tr>
      <td colSpan={8} style={{ padding: 0, background: '#0D1526' }}>
        <div style={{ padding: '10px 14px 14px' }}>
          {/* sub-header */}
          <div style={{
            fontSize: 10, fontWeight: 600, color: '#475569',
            letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
          }}>
            {total} Buildings · {code}
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 600 }}>
              <thead>
                <tr style={{ background: 'rgba(30,41,59,.5)' }}>
                  <th style={BTH}>Site</th>
                  <th style={BTH}>Address</th>
                  <th style={BTH}>Agency</th>
                  <th style={{ ...BTH, textAlign: 'center' }}>EJ</th>
                  <th style={{ ...BTH, textAlign: 'right' }}>Solar kWh/yr</th>
                  <th style={{ ...BTH, textAlign: 'right' }}>Ann. Cost</th>
                  <th style={{ ...BTH, textAlign: 'right' }}>GHG t CO₂</th>
                  <th style={{ ...BTH, textAlign: 'right' }}>BESS kWh</th>
                  <th style={{ ...BTH, textAlign: 'right' }}>BESS Savings/yr</th>
                </tr>
              </thead>
              <tbody>
                {slice.map((b, i) => (
                  <tr
                    key={i}
                    style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(15,23,42,.4)' }}
                  >
                    <td style={{ ...BTD, fontWeight: 500, color: '#CBD5E1', maxWidth: 120, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                        title={b.site}>{b.site || '—'}</td>
                    <td style={{ ...BTD, color: '#94A3B8', maxWidth: 140, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                        title={b.address}>{b.address || '—'}</td>
                    <td style={BTD}>
                      <span style={{
                        background: 'rgba(100,116,139,.15)', color: '#94A3B8',
                        padding: '1px 5px', borderRadius: 20, fontSize: 10,
                      }}>{b.agency || '—'}</span>
                    </td>
                    <td style={{ ...BTD, textAlign: 'center' }}>
                      {b.ej
                        ? <span style={{ background: 'rgba(139,92,246,.15)', color: '#C4B5FD', padding: '1px 5px', borderRadius: 20, fontSize: 10 }}>EJ</span>
                        : <span style={{ color: '#334155', fontSize: 10 }}>—</span>
                      }
                    </td>
                    <td style={{ ...BTD, textAlign: 'right', color: '#FDE047', fontVariantNumeric: 'tabular-nums' }}>
                      {fmtKwh(b.solar_kwh_yr)}
                    </td>
                    <td style={{ ...BTD, textAlign: 'right', color: '#93C5FD', fontVariantNumeric: 'tabular-nums' }}>
                      {fmtUsd(b.annual_cost_usd)}
                    </td>
                    <td style={{ ...BTD, textAlign: 'right', color: '#6EE7B7', fontVariantNumeric: 'tabular-nums' }}>
                      {b.ghg_co2 != null ? b.ghg_co2 : '—'}
                    </td>
                    <td style={{ ...BTD, textAlign: 'right', color: '#FCD34D', fontVariantNumeric: 'tabular-nums' }}>
                      {b.bess_kwh != null ? `${b.bess_kwh}` : '—'}
                    </td>
                    <td style={{ ...BTD, textAlign: 'right', color: '#10B981', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                      {fmtUsd(b.bess_savings_usd)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

        </div>
      </td>
    </tr>
  );
}

/* ── main component ── */
export default function RankingsTable() {
  const [expanded, setExpanded] = useState(null);

  const toggle = (code) => setExpanded(prev => prev === code ? null : code);

  return (
    <div style={{ background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, overflow: 'hidden' }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 14px', borderBottom: '1px solid #1E293B',
      }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>Top 10 Districts — Solar Potential</span>
        <span style={{
          display: 'inline-block', padding: '2px 6px', borderRadius: 20, fontSize: 11,
          background: 'rgba(234,179,8,.15)', color: '#FDE047',
        }}>
          ☀ kWh/yr ranked
        </span>
      </div>

      {/* Scrollable table */}
      <div style={{ overflowY: 'auto', maxHeight: 520 }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={TH}></th>
              <th style={TH}>#</th>
              <th style={TH}>District</th>
              <th style={TH}>Borough</th>
              <th style={TH}>Solar kWh/yr</th>
              <th style={TH}>BESS Savings/yr</th>
              <th style={TH}>Buildings</th>
              <th style={{ ...TH, textAlign: 'center' }}>Solar Ready</th>
              <th style={{ ...TH, textAlign: 'center' }}>EJ %</th>
            </tr>
          </thead>
          <tbody>
            {districts.map((d) => {
              const isOpen = expanded === d.code;
              return (
                <>
                  {/* District summary row */}
                  <tr
                    key={d.code}
                    onClick={() => toggle(d.code)}
                    style={{
                      cursor: 'pointer',
                      background: isOpen ? 'rgba(234,179,8,.06)' : 'transparent',
                      transition: 'background 0.15s',
                    }}
                    onMouseEnter={e => { if (!isOpen) e.currentTarget.style.background = 'rgba(234,179,8,.03)'; }}
                    onMouseLeave={e => { if (!isOpen) e.currentTarget.style.background = 'transparent'; }}
                  >
                    {/* chevron */}
                    <td style={{ ...TD, width: 28, paddingRight: 0 }}>
                      <span style={{
                        display: 'inline-block', fontSize: 10, color: '#475569',
                        transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
                        transition: 'transform 0.2s',
                        userSelect: 'none',
                      }}>▶</span>
                    </td>

                    {/* rank */}
                    <td style={{
                      ...TD,
                      borderLeft: `2px solid ${d.rank === 1 ? '#EAB308' : 'transparent'}`,
                    }}>
                      <span style={{
                        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                        width: 20, height: 20, borderRadius: '50%', fontSize: 10, fontWeight: 700,
                        background: d.rank <= 3 ? 'rgba(234,179,8,.2)' : 'rgba(100,116,139,.1)',
                        color: d.rank <= 3 ? '#FDE047' : '#64748B',
                      }}>
                        {d.rank}
                      </span>
                    </td>

                    {/* code + CD */}
                    <td style={{ ...TD, fontWeight: 600, color: '#F1F5F9' }}>
                      {d.code}
                      <span style={{ color: '#475569', fontWeight: 400, marginLeft: 4, fontSize: 11 }}>
                        CD{d.district}
                      </span>
                    </td>

                    {/* borough */}
                    <td style={TD}>
                      <span style={{
                        background: 'rgba(100,116,139,.15)', color: '#94A3B8',
                        padding: '2px 6px', borderRadius: 20, fontSize: 10,
                      }}>
                        {d.borough === 'Staten Island' ? 'S.I.' : d.borough}
                      </span>
                    </td>

                    {/* solar */}
                    <td style={{ ...TD, color: '#FDE047', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                      {fmtKwh(d.solar_kwh_yr)}
                    </td>

                    {/* bess savings */}
                    <td style={{ ...TD, color: '#10B981', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                      {fmtUsd(d.bess_savings_usd)}
                    </td>

                    {/* buildings count */}
                    <td style={{ ...TD, color: '#93C5FD', fontVariantNumeric: 'tabular-nums' }}>
                      {d.buildings}
                    </td>

                    {/* solar ready */}
                    <td style={{ ...TD, textAlign: 'center' }}>
                      <span style={{
                        background: 'rgba(234,179,8,.12)', color: '#FCD34D',
                        padding: '2px 7px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                      }}>
                        {d.solar_ready}
                      </span>
                    </td>

                    {/* ej % */}
                    <td style={{ ...TD, textAlign: 'center' }}>
                      {d.pct_ej > 20 ? (
                        <span style={{
                          background: 'rgba(139,92,246,.15)', color: '#C4B5FD',
                          padding: '2px 7px', borderRadius: 20, fontSize: 11, fontWeight: 600,
                        }}>
                          {d.pct_ej}%
                        </span>
                      ) : (
                        <span style={{ color: '#475569', fontSize: 11 }}>{d.pct_ej}%</span>
                      )}
                    </td>
                  </tr>

                  {/* Collapsible buildings panel */}
                  {isOpen && <BuildingsPanel key={`${d.code}-panel`} code={d.code} />}
                </>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
