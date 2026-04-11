import { useMemo } from 'react';
import { useDashboard } from '../context/DashboardContext';
import { HOURLY_DISPATCH, SOC_DATA, WASTE_FORECAST, DIVERSION_GAP, SCENARIO_DEPLOYMENTS } from '../data/chartData';

const CARD = { background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, padding: 16 };
const GRID_COLOR = 'rgba(19,29,46,.8)';
const TICK_COLOR = '#475569';

/* ─── Simple SVG Line/Area chart engine ─── */
function miniChart(data, key, color, maxVal, yOffset = 0) {
  const W = 400, H = 120;
  const pad = { l: 30, r: 10, t: 8, b: 20 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const xScale = i => pad.l + (i / (data.length - 1)) * iW;
  const yScale = v => pad.t + iH - ((v - yOffset) / (maxVal - yOffset)) * iH;
  const pts = data.map((d, i) => `${xScale(i)},${yScale(d[key])}`).join(' ');
  const area = `M${xScale(0)},${yScale(0 > yOffset ? 0 : yOffset)} ${data.map((d, i) => `L${xScale(i)},${yScale(d[key])}`).join(' ')} L${xScale(data.length - 1)},${yScale(yOffset)} Z`;
  return { pts, area, xScale, yScale, W, H, pad };
}

function DispatchChart() {
  const max = 320;
  const W = 420, H = 130, pad = { l: 32, r: 8, t: 8, b: 22 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const xs = i => pad.l + (i / 23) * iW;
  const ys = v => pad.t + iH - (v / max) * iH;

  const series = [
    { key: 'demand', color: '#3B82F6', fill: 'rgba(59,130,246,.18)' },
    { key: 'solar',  color: '#F59E0B', fill: 'rgba(245,158,11,.18)' },
    { key: 'bess',   color: '#10B981', fill: 'rgba(16,185,129,.18)' },
  ];

  const hGrid = [0, 80, 160, 240, 320];
  const xLabels = [{ i: 0, l: '12AM' }, { i: 6, l: '6AM' }, { i: 12, l: '12PM' }, { i: 18, l: '6PM' }, { i: 23, l: '11PM' }];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ overflow: 'visible' }}>
      {/* Grid lines */}
      {hGrid.map(v => (
        <g key={v}>
          <line x1={pad.l} y1={ys(v)} x2={W - pad.r} y2={ys(v)} stroke={GRID_COLOR} strokeWidth={1} />
          <text x={pad.l - 4} y={ys(v) + 3} fill={TICK_COLOR} fontSize={8} textAnchor="end">{v}</text>
        </g>
      ))}
      {/* X labels */}
      {xLabels.map(({ i, l }) => (
        <text key={i} x={xs(i)} y={H - 4} fill={TICK_COLOR} fontSize={8} textAnchor="middle">{l}</text>
      ))}
      {/* Area + lines */}
      {series.map(({ key, color, fill }) => {
        const areaPath = `M${xs(0)},${ys(0)} ${HOURLY_DISPATCH.map((d, i) => `L${xs(i)},${ys(d[key])}`).join(' ')} L${xs(23)},${ys(0)} Z`;
        const linePath = HOURLY_DISPATCH.map((d, i) => `${i === 0 ? 'M' : 'L'}${xs(i)},${ys(d[key])}`).join(' ');
        return (
          <g key={key}>
            <path d={areaPath} fill={fill} />
            <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
          </g>
        );
      })}
    </svg>
  );
}

function SocChart() {
  const W = 420, H = 130, pad = { l: 32, r: 8, t: 8, b: 22 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const xs = i => pad.l + (i / 23) * iW;
  const ys = v => pad.t + iH - (v / 100) * iH;
  const linePath = SOC_DATA.map((d, i) => `${i === 0 ? 'M' : 'L'}${xs(i)},${ys(d.soc)}`).join(' ');
  const areaPath = `M${xs(0)},${ys(0)} ${SOC_DATA.map((d, i) => `L${xs(i)},${ys(d.soc)}`).join(' ')} L${xs(23)},${ys(0)} Z`;
  const hGrid = [0, 25, 50, 75, 100];
  const xLabels = [{ i: 0, l: '12AM' }, { i: 6, l: '6AM' }, { i: 12, l: '12PM' }, { i: 18, l: '6PM' }, { i: 23, l: '11PM' }];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ overflow: 'visible' }}>
      {hGrid.map(v => (
        <g key={v}>
          <line x1={pad.l} y1={ys(v)} x2={W - pad.r} y2={ys(v)} stroke={GRID_COLOR} strokeWidth={1} />
          <text x={pad.l - 4} y={ys(v) + 3} fill={TICK_COLOR} fontSize={8} textAnchor="end">{v}%</text>
        </g>
      ))}
      {xLabels.map(({ i, l }) => (
        <text key={i} x={xs(i)} y={H - 4} fill={TICK_COLOR} fontSize={8} textAnchor="middle">{l}</text>
      ))}
      {/* Ref lines */}
      <line x1={pad.l} y1={ys(10)} x2={W - pad.r} y2={ys(10)} stroke="#EF4444" strokeWidth={1} strokeDasharray="5 5" />
      <text x={W - pad.r + 2} y={ys(10) + 3} fill="#EF4444" fontSize={7}>Min</text>
      <line x1={pad.l} y1={ys(95)} x2={W - pad.r} y2={ys(95)} stroke="#F59E0B" strokeWidth={1} strokeDasharray="5 5" />
      <text x={W - pad.r + 2} y={ys(95) + 3} fill="#F59E0B" fontSize={7}>Max</text>
      {/* SOC */}
      <path d={areaPath} fill="rgba(16,185,129,.08)" />
      <path d={linePath} fill="none" stroke="#10B981" strokeWidth={2} strokeLinejoin="round" />
    </svg>
  );
}

function WasteLineChart() {
  const W = 420, H = 130, pad = { l: 34, r: 8, t: 8, b: 22 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const months = WASTE_FORECAST.map(d => d.month);
  const xs = i => pad.l + (i / (months.length - 1)) * iW;
  const ys = v => pad.t + iH - ((v - 0) / 400) * iH;
  const series = [
    { key: 'refuse',   color: '#EF4444' },
    { key: 'recycling',color: '#3B82F6' },
    { key: 'organics', color: '#10B981' },
  ];
  const hGrid = [0, 100, 200, 300, 400];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ overflow: 'visible' }}>
      {hGrid.map(v => (
        <g key={v}>
          <line x1={pad.l} y1={ys(v)} x2={W - pad.r} y2={ys(v)} stroke={GRID_COLOR} strokeWidth={1} />
          <text x={pad.l - 4} y={ys(v) + 3} fill={TICK_COLOR} fontSize={8} textAnchor="end">{v}</text>
        </g>
      ))}
      {months.map((m, i) => i % 3 === 0 && (
        <text key={i} x={xs(i)} y={H - 4} fill={TICK_COLOR} fontSize={8} textAnchor="middle">{m}</text>
      ))}
      {series.map(({ key, color }) => {
        const path = WASTE_FORECAST.map((d, i) => `${i === 0 ? 'M' : 'L'}${xs(i)},${ys(d[key])}`).join(' ');
        return (
          <g key={key}>
            <path d={path} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" />
            {WASTE_FORECAST.map((d, i) => <circle key={i} cx={xs(i)} cy={ys(d[key])} r={2} fill={color} />)}
          </g>
        );
      })}
    </svg>
  );
}

function DiversionBarChart() {
  const W = 420, H = 130, pad = { l: 50, r: 8, t: 8, b: 22 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;
  const n = DIVERSION_GAP.length;
  const barW = (iW / n) * 0.35;
  const groupW = iW / n;
  const ys = v => pad.t + iH * (1 - v / 100);
  const barH = v => iH * (v / 100);
  const hGrid = [0, 25, 50, 75, 100];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ overflow: 'visible' }}>
      {hGrid.map(v => (
        <g key={v}>
          <line x1={pad.l} y1={ys(v)} x2={W - pad.r} y2={ys(v)} stroke={GRID_COLOR} strokeWidth={1} />
          <text x={pad.l - 4} y={ys(v) + 3} fill={TICK_COLOR} fontSize={8} textAnchor="end">{v}%</text>
        </g>
      ))}
      {DIVERSION_GAP.map((d, i) => {
        const cx = pad.l + i * groupW + groupW / 2;
        return (
          <g key={i}>
            <rect x={cx - barW - 2} y={ys(d.current)} width={barW} height={barH(d.current)} rx={2} fill="#334155" />
            <rect x={cx + 2} y={ys(d.target)} width={barW} height={barH(d.target)} rx={2} fill="#10B981" opacity={0.7} />
            <text x={cx} y={H - 4} fill={TICK_COLOR} fontSize={7.5} textAnchor="middle">{d.material}</text>
          </g>
        );
      })}
    </svg>
  );
}

function BessTab() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
      <div style={CARD}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#94A3B8', marginBottom: 8 }}>24h Energy Profile (kW)</div>
        <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
          {[['#3B82F6','Demand'],['#F59E0B','Solar'],['#10B981','BESS']].map(([c,l]) => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#64748B' }}>
              <span style={{ width: 8, height: 2, background: c, display: 'inline-block', borderRadius: 1 }} />{l}
            </span>
          ))}
        </div>
        <DispatchChart />
      </div>
      <div style={CARD}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#94A3B8', marginBottom: 8 }}>Battery State of Charge (%)</div>
        <SocChart />
      </div>
      {/* Summary */}
      <div style={{ gridColumn: '1/-1', background: 'rgba(30,41,59,.4)', borderRadius: 8, padding: '10px 16px', display: 'flex', justifyContent: 'space-around' }}>
        {[
          ['Daily Savings','$38.50'],['Peak Shaved','87.2 kW (25.4%)'],
          ['Battery Cycles','0.85/day'],['Grid Independence','34% of peak hrs'],
        ].map(([l, v]) => (
          <div key={l} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 11, color: '#64748B' }}>{l}</div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#F1F5F9', marginTop: 2 }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function WasteTab() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
      <div style={CARD}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#94A3B8', marginBottom: 8 }}>12-Month Waste Forecast (K tons)</div>
        <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
          {[['#EF4444','Refuse'],['#3B82F6','Recycling'],['#10B981','Organics']].map(([c,l]) => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#64748B' }}>
              <span style={{ width: 8, height: 2, background: c, display: 'inline-block', borderRadius: 1 }} />{l}
            </span>
          ))}
        </div>
        <WasteLineChart />
      </div>
      <div style={CARD}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#94A3B8', marginBottom: 8 }}>Diversion Gap by Material</div>
        <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
          {[['#334155','Current %'],['#10B981','Target %']].map(([c,l]) => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: '#64748B' }}>
              <span style={{ width: 8, height: 8, borderRadius: 2, background: c, display: 'inline-block' }} />{l}
            </span>
          ))}
        </div>
        <DiversionBarChart />
        <div style={{ marginTop: 10, background: 'rgba(16,185,129,.05)', border: '1px solid rgba(16,185,129,.2)', borderRadius: 6, padding: '6px 10px', fontSize: 11, color: '#94A3B8' }}>
          Closing the organics diversion gap alone = <strong style={{ color: '#10B981' }}>45,000 MWh/yr</strong> of biogas potential — enough to power <strong style={{ color: '#10B981' }}>4,100 homes</strong>.
        </div>
      </div>
    </div>
  );
}

function ScenarioTab() {
  const { scenario, updateScenario } = useDashboard();

  const { budget: b, bess, solar, organics, route, equity } = scenario;
  const sum = bess + solar + organics + route;
  const bessSites   = Math.round((b * bess    / 100) * 1000 / 340);
  const solarSites  = Math.round((b * solar   / 100) * 1000 / 480);
  const orgTons     = Math.round((b * organics / 100) * 1000 / 4.9);
  const routeDist   = Math.round((b * route   / 100) * 1000 / 180);
  const savings     = ((b * bess / 100 * 0.8) + (b * solar / 100 * 0.7) + (b * organics / 100 * 0.75) + (b * route / 100 * 0.79)).toFixed(1);
  const co2         = Math.round(bessSites * 42 + solarSites * 65 + orgTons * 0.07 + routeDist * 28);
  const payback     = (b / parseFloat(savings || 1)).toFixed(1);
  const ejCov       = Math.min(98, Math.round(50 + equity * 60));

  const sliders = [
    ['Total Budget ($M)',       'budget',   1, 50,   1,    `$${b}M`],
    ['BESS Allocation (%)',     'bess',     0, 100,  5,    `${bess}%`],
    ['Solar Allocation (%)',    'solar',    0, 100,  5,    `${solar}%`],
    ['Organics Diversion (%)', 'organics', 0, 100,  5,    `${organics}%`],
    ['Route Optimization (%)','route',     0, 100,  5,    `${route}%`],
    ['Equity Weighting',       'equity',   0, 1,    0.05, equity.toFixed(2)],
  ];

  return (
    <div>
      {/* Sliders */}
      <div style={{ background: 'rgba(30,41,59,.3)', borderRadius: 12, padding: 16, marginBottom: 14 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 32px' }}>
          {sliders.map(([label, key, min, max, step, display]) => (
            <div key={key}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748B', marginBottom: 3 }}>
                <span>{label}</span>
                <span style={{ color: '#60A5FA' }}>{display}</span>
              </div>
              <input
                type="range" min={min} max={max} step={step}
                value={scenario[key]}
                onChange={e => updateScenario(key, parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#3B82F6' }}
              />
            </div>
          ))}
        </div>
        {sum > 100 && (
          <div style={{ marginTop: 8, fontSize: 11, color: '#EF4444' }}>
            ⚠ Allocations sum to {sum}% (exceeds 100%) — adjust to optimize
          </div>
        )}
      </div>

      {/* Impact summary */}
      <div style={{
        background: 'linear-gradient(to right, rgba(59,130,246,.05), rgba(139,92,246,.05), rgba(16,185,129,.05))',
        border: '1px solid rgba(255,255,255,.05)', borderRadius: 12, padding: 16, marginBottom: 14,
      }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: '#475569', letterSpacing: '0.08em', marginBottom: 12 }}>
          PROJECTED IMPACT — ${b}M Budget
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 14 }}>
          {[
            { icon: '🔋', label: 'BESS', detail: `Deploy at ${bessSites} sites → save $${(b*bess/100*.8).toFixed(1)}M/yr` },
            { icon: '☀️', label: 'Solar', detail: `Install on ${solarSites} rooftops → ${(solarSites*0.7).toFixed(1)} GWh/yr` },
            { icon: '🌱', label: 'Organics', detail: `Divert ${(orgTons/1000).toFixed(0)}K tons → ${Math.round(orgTons*.005)}K MWh biogas` },
            { icon: '🚛', label: 'Routes', detail: `Optimize ${routeDist} districts → $${(b*route/100*.79).toFixed(1)}M/yr` },
          ].map(({ icon, label, detail }) => (
            <div key={label} style={{ background: 'rgba(30,41,59,.5)', borderRadius: 8, padding: 10 }}>
              <div style={{ fontSize: 14, marginBottom: 4 }}>{icon} <span style={{ fontWeight: 600, fontSize: 12 }}>{label}</span></div>
              <div style={{ fontSize: 10, color: '#64748B', lineHeight: 1.4 }}>{detail}</div>
            </div>
          ))}
        </div>
        <div style={{ borderTop: '1px solid rgba(255,255,255,.06)', paddingTop: 12, display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
          {[
            ['Annual Savings', `$${savings}M`, '#10B981'],
            ['CO₂ Offset', `${co2.toLocaleString()} t/yr`, '#6EE7B7'],
            ['Payback', `${payback} years`, '#60A5FA'],
            ['EJ Coverage', `${ejCov}%`, '#C4B5FD'],
          ].map(([l, v, c]) => (
            <div key={l} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 11, color: '#475569' }}>{l}</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: c, marginTop: 2 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Deployment table */}
      <div style={{ background: '#131C2E', border: '1px solid #1E293B', borderRadius: 12, overflow: 'hidden' }}>
        <div style={{ padding: '10px 14px', borderBottom: '1px solid #1E293B', fontSize: 12, fontWeight: 600 }}>
          Ranked Deployment Actions
        </div>
        <div style={{ overflowY: 'auto', maxHeight: 240 }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['#','Action','Location','Investment','Return/yr','CO₂','Payback','Equity'].map(h => (
                  <th key={h} style={{ padding: '6px 8px', textAlign: 'left', fontSize: 10, color: '#475569', fontWeight: 600, letterSpacing: '0.05em', borderBottom: '1px solid #1E293B', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {SCENARIO_DEPLOYMENTS.map(d => (
                <tr key={d.priority}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,.04)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12, color: '#475569' }}>{d.priority}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12, color: '#60A5FA', fontWeight: 500 }}>{d.action}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 11, color: '#94A3B8' }}>{d.location}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12 }}>{d.invest}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12, color: '#10B981', fontWeight: 600 }}>{d.ret}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12, color: '#6EE7B7' }}>{d.co2}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 12 }}>{d.payback}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid rgba(30,41,59,.5)', fontSize: 11 }}>
                    {d.ej
                      ? <span style={{ background: 'rgba(16,185,129,.15)', color: '#6EE7B7', padding: '1px 6px', borderRadius: 20, fontSize: 10 }}>✅ EJ</span>
                      : <span style={{ color: '#334155' }}>—</span>
                    }
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

const TABS = [
  { id: 'bess',     label: '⚡ BESS Dispatch'  },
  { id: 'waste',    label: '🗑️ Waste Forecast'  },
  { id: 'scenario', label: '🎯 Scenario Planner' },
];

export default function Simulation() {
  const { simTab, setSimTab } = useDashboard();

  return (
    <div>
      <div style={{
        fontSize: 10, fontWeight: 600, color: '#475569',
        letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 12,
      }}>
        Simulation
      </div>

      {/* Tab Pills */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 14 }}>
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setSimTab(tab.id)}
            style={{
              padding: '5px 12px', borderRadius: 20, cursor: 'pointer',
              fontSize: 12, fontWeight: 500, transition: 'all 0.2s',
              background: simTab === tab.id ? 'rgba(59,130,246,.15)' : 'transparent',
              color: simTab === tab.id ? '#60A5FA' : '#64748B',
              border: simTab === tab.id ? '1px solid rgba(59,130,246,.3)' : '1px solid transparent',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {simTab === 'bess'     && <BessTab />}
      {simTab === 'waste'    && <WasteTab />}
      {simTab === 'scenario' && <ScenarioTab />}
    </div>
  );
}
