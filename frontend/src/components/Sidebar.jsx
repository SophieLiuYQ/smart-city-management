import { useDashboard } from '../context/DashboardContext';

const NAV_ITEMS = [
  { icon: '📊', label: 'Dashboard',  active: true  },
  { icon: '🗺️', label: 'Site Map',   active: false },
  { icon: '📋', label: 'Rankings',   active: false },
  { icon: '📈', label: 'Analytics',  active: false },
  { icon: '⚙️', label: 'Settings',   active: false },
];

const BOROUGHS = ['All Boroughs', 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'];

export default function Sidebar() {
  const {
    viewMode, changeView,
    borough, setBorough,
    minScore, setMinScore,
    budget, setBudget,
    ejPriority, setEjPriority,
  } = useDashboard();

  return (
    <aside style={{
      width: 240, minWidth: 240, background: '#0F172A',
      borderRight: '1px solid rgba(255,255,255,.05)',
      position: 'sticky', top: 0, height: '100vh',
      overflowY: 'auto', display: 'flex', flexDirection: 'column',
    }}>
      {/* Logo */}
      <div style={{ padding: '20px 16px 14px', borderBottom: '1px solid rgba(255,255,255,.05)' }}>
        <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: '-0.3px' }}>⚡ NYC Nexus</div>
        <div style={{ fontSize: 10, color: '#475569', marginTop: 3, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
          Energy · Waste · Intelligence
        </div>
      </div>

      {/* View Mode Toggle */}
      <div style={{ padding: '10px 12px 0' }}>
        <div style={{ display: 'flex', gap: 4, background: '#0B1120', borderRadius: 10, padding: 4 }}>
          {['energy', 'waste', 'nexus'].map(v => (
            <button
              key={v}
              onClick={() => changeView(v)}
              style={{
                flex: 1, padding: '5px 4px', borderRadius: 7, textAlign: 'center',
                cursor: 'pointer', fontSize: 12, fontWeight: 600, border: 'none',
                background: viewMode === v ? '#3B82F6' : '#1E293B',
                color: viewMode === v ? '#fff' : '#64748B',
                transition: 'all 0.2s',
              }}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Navigation */}
      <nav style={{ padding: 8 }}>
        {NAV_ITEMS.map(item => (
          <div
            key={item.label}
            style={{
              padding: '7px 10px', borderRadius: 8, cursor: 'pointer',
              display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2,
              color: item.active ? '#60A5FA' : '#64748B',
              background: item.active ? 'rgba(30,41,59,.7)' : 'transparent',
              fontSize: 13,
            }}
            onMouseEnter={e => !item.active && (e.currentTarget.style.background = 'rgba(30,41,59,.4)')}
            onMouseLeave={e => !item.active && (e.currentTarget.style.background = 'transparent')}
          >
            <span>{item.icon}</span>
            <span>{item.label}</span>
          </div>
        ))}
      </nav>

      {/* Filters */}
      <div style={{
        margin: '0 10px', borderTop: '1px solid rgba(255,255,255,.06)',
        paddingTop: 12, flex: 1,
      }}>
        <div style={{
          fontSize: 10, fontWeight: 600, color: '#334155',
          letterSpacing: '0.1em', marginBottom: 10, padding: '0 4px',
        }}>
          FILTERS
        </div>

        {/* Borough */}
        <div style={{ marginBottom: 10, padding: '0 2px' }}>
          <div style={{ fontSize: 11, color: '#64748B', marginBottom: 4 }}>Borough</div>
          <select
            value={borough}
            onChange={e => setBorough(e.target.value)}
            style={{
              background: '#1E293B', color: '#F1F5F9',
              border: '1px solid #2D3D54', borderRadius: 6,
              padding: '5px 7px', fontSize: 12, width: '100%',
            }}
          >
            {BOROUGHS.map(b => <option key={b}>{b}</option>)}
          </select>
        </div>

        {/* Min Score */}
        <div style={{ marginBottom: 10, padding: '0 2px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748B', marginBottom: 4 }}>
            <span>Min Priority Score</span>
            <span style={{ color: '#60A5FA' }}>{minScore}</span>
          </div>
          <input
            type="range" min="0" max="100" step="5" value={minScore}
            onChange={e => setMinScore(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#3B82F6' }}
          />
        </div>

        {/* Budget */}
        <div style={{ marginBottom: 10, padding: '0 2px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748B', marginBottom: 4 }}>
            <span>Budget ($M)</span>
            <span style={{ color: '#60A5FA' }}>${budget}M</span>
          </div>
          <input
            type="range" min="1" max="50" step="1" value={budget}
            onChange={e => setBudget(Number(e.target.value))}
            style={{ width: '100%', accentColor: '#3B82F6' }}
          />
        </div>

        {/* EJ Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 2px', marginBottom: 4 }}>
          <span style={{ fontSize: 11, color: '#64748B' }}>Prioritize EJ Areas</span>
          <div
            onClick={() => setEjPriority(!ejPriority)}
            style={{
              position: 'relative', width: 34, height: 18,
              background: ejPriority ? '#3B82F6' : '#334155',
              borderRadius: 18, cursor: 'pointer', transition: 'background 0.3s',
            }}
          >
            <div style={{
              position: 'absolute', top: 3, left: ejPriority ? 15 : 3,
              width: 12, height: 12, background: '#fff', borderRadius: '50%',
              transition: 'left 0.3s',
            }} />
          </div>
        </div>

        <button
          style={{
            width: '100%', background: '#3B82F6', color: '#fff', border: 'none',
            borderRadius: 8, padding: 8, fontSize: 12, fontWeight: 600,
            cursor: 'pointer', marginTop: 12, transition: 'background 0.2s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = '#2563EB'}
          onMouseLeave={e => e.currentTarget.style.background = '#3B82F6'}
        >
          Apply Filters
        </button>
      </div>
    </aside>
  );
}
