import { useDashboard } from '../context/DashboardContext';

export default function Header() {
  const { viewMode } = useDashboard();

  return (
    <header style={{
      background: 'rgba(15,23,42,.95)', height: 52,
      display: 'flex', alignItems: 'center',
      padding: '0 20px', justifyContent: 'space-between',
      borderBottom: '1px solid rgba(255,255,255,.05)',
      position: 'sticky', top: 0, zIndex: 10,
    }}>
      <span style={{ fontSize: 12, color: '#94A3B8' }}>
        Dashboard{' '}
        <span style={{ color: '#334155' }}>/</span>{' '}
        <span style={{ color: '#60A5FA' }}>
          {viewMode.charAt(0).toUpperCase() + viewMode.slice(1)}
        </span>
      </span>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{
          display: 'inline-block', padding: '2px 6px', borderRadius: 20,
          fontSize: 11, fontWeight: 500,
          background: 'rgba(59,130,246,.15)', color: '#93C5FD',
        }}>
          ● Claude API
        </span>
        <span style={{
          display: 'inline-block', padding: '2px 6px', borderRadius: 20,
          fontSize: 11, fontWeight: 500,
          background: 'rgba(16,185,129,.15)', color: '#6EE7B7',
        }}>
          ● Pipeline Complete
        </span>
        <span style={{ fontSize: 11, color: '#334155' }}>Apr 10, 2026 2:34 PM</span>
      </div>
    </header>
  );
}
