'use client';

import { useState } from 'react';
import districtData from '../data/top10_district_analysis.json';
import { useDashboard } from '../context/DashboardContext';

/* ── derived data from the single source of truth ── */
const districts = districtData.map((d, i) => ({
  rank: i + 1,
  code: d.district_code,
  district: d.community_district,
  borough: d.borough,
  solar_kwh_yr: d.buildings_summary.total_solar_potential_kwh_yr,
  bess_savings_usd: d.buildings_summary.total_bess_savings_usd_yr,
  buildings: d.buildings_summary.total,
  solar_ready: d.buildings_summary.solar_ready,
  pct_ej: d.buildings_summary.pct_ej,
  centroid_lat: d.centroid_lat,
  centroid_lon: d.centroid_lon,
}));

const buildingsMap = Object.fromEntries(
  districtData.map(d => [
    d.district_code,
    d.buildings.map(b => ({
      site: b.site,
      address: b.address,
      agency: b.agency,
      ej: b.ej,
      solar_kwh_yr: b.energy?.solar_production_kwh_yr,
      annual_cost_usd: b.energy?.est_annual_cost_usd,
      ghg_co2: b.energy?.ghg_tons_co2e_yr,
      bess_kwh: b.bess_recommendation?.capacity_kwh,
      bess_savings_usd: b.bess_recommendation?.est_annual_savings_usd,
      lat: b.latitude,
      lng: b.longitude,
    })),
  ])
);

/* ── style constants ── */
const TH = {
  padding: '6px 8px', textAlign: 'left', fontSize: 10,
  color: '#64748B', fontWeight: 600, letterSpacing: '0.05em',
  borderBottom: '1px solid #1E293B', whiteSpace: 'nowrap',
};
const TD = { padding: '6px 8px', borderBottom: '1px solid rgba(226,232,240,.5)', fontSize: 12 };
const BTH = {
  padding: '5px 8px', textAlign: 'left', fontSize: 10,
  color: '#94A3B8', fontWeight: 600, letterSpacing: '0.05em',
  borderBottom: '1px solid rgba(226,232,240,.6)', whiteSpace: 'nowrap',
};
const BTD = { padding: '5px 8px', borderBottom: '1px solid rgba(226,232,240,.35)', fontSize: 11 };

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

/* ── building sub-table ── */
function BuildingsPanel({ code, onBuildingClick }) {
  const slice = buildingsMap[code] ?? [];

  return (
    <tr>
      <td colSpan={9} style={{ padding: 0, background: '#F8FAFC' }}>
        <div style={{ padding: '10px 14px 14px' }}>
          <div style={{
            fontSize: 10, fontWeight: 600, color: '#64748B',
            letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 8,
          }}>
            {slice.length} Buildings · {code}
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 600 }}>
              <thead>
                <tr style={{ background: 'rgba(226,232,240,.5)' }}>
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
                    onClick={() => b.lat != null && b.lng != null && onBuildingClick(b)}
                    style={{
                      background: i % 2 === 0 ? 'transparent' : 'rgba(241,245,249,.5)',
                      cursor: b.lat != null && b.lng != null ? 'pointer' : 'default',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => { if (b.lat != null) e.currentTarget.style.background = 'rgba(16,185,129,0.06)'; }}
                    onMouseLeave={e => { e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'rgba(241,245,249,.5)'; }}
                  >
                    <td style={{ ...BTD, fontWeight: 500, color: '#334155', maxWidth: 120, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
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
                        : <span style={{ color: '#94A3B8', fontSize: 10 }}>—</span>
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
  const { selectItem, setFlyToCoords } = useDashboard();

  const flyTo = (lat, lng, zoom) =>
    setFlyToCoords({ lat, lng, zoom, _t: Date.now() });

  const handleDistrictClick = (d) => {
    setExpanded(prev => prev === d.code ? null : d.code);
    if (d.centroid_lat != null && d.centroid_lon != null) {
      flyTo(d.centroid_lat, d.centroid_lon, 13);
      selectItem('district', {
        rank: d.rank, code: d.code, district: d.district, borough: d.borough,
        solar_kwh_yr: d.solar_kwh_yr, bess_savings_usd: d.bess_savings_usd,
        buildings: d.buildings, solar_ready: d.solar_ready, pct_ej: d.pct_ej,
      });
    }
  };

  const handleBuildingClick = (b) => {
    flyTo(b.lat, b.lng, 16);
    selectItem('building', {
      site: b.site, address: b.address, agency: b.agency, ej: b.ej,
      solar_kwh_yr: b.solar_kwh_yr, annual_cost_usd: b.annual_cost_usd,
      ghg_co2: b.ghg_co2, bess_kwh: b.bess_kwh, bess_savings_usd: b.bess_savings_usd,
    });
  };

  return (
    <div style={{ background: '#FFFFFF', border: '1px solid #E2E8F0', borderRadius: 12, overflow: 'hidden', display: 'flex', flexDirection: 'column', flex: 1 }}>
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
      <div style={{ overflowY: 'auto', flex: 1 }}>
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
                  <tr
                    key={d.code}
                    onClick={() => handleDistrictClick(d)}
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
                        display: 'inline-block', fontSize: 10, color: '#64748B',
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
                    <td style={{ ...TD, fontWeight: 600, color: '#0F172A' }}>
                      {d.code}
                      <span style={{ color: '#64748B', fontWeight: 400, marginLeft: 4, fontSize: 11 }}>
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
                        <span style={{ color: '#64748B', fontSize: 11 }}>{d.pct_ej}%</span>
                      )}
                    </td>
                  </tr>

                  {/* Collapsible buildings panel */}
                  {isOpen && (
                    <BuildingsPanel
                      key={`${d.code}-panel`}
                      code={d.code}
                      onBuildingClick={handleBuildingClick}
                    />
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
