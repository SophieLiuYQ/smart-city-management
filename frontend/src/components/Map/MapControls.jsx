import { useState, useCallback, useRef } from 'react';
import { ENERGY_SITES } from '../../data/sites';

export default function MapControls({ onSearch }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  const handleChange = useCallback((e) => {
    const q = e.target.value;
    setQuery(q);
    if (q.length >= 2) {
      const matched = ENERGY_SITES.filter(s =>
        s.name.toLowerCase().includes(q.toLowerCase()) ||
        s.borough.toLowerCase().includes(q.toLowerCase()) ||
        s.agency.toLowerCase().includes(q.toLowerCase())
      ).slice(0, 5);
      setResults(matched);
      setOpen(matched.length > 0);
    } else {
      setResults([]);
      setOpen(false);
    }
  }, []);

  const handleSelect = useCallback((site) => {
    setQuery(site.name);
    setOpen(false);
    onSearch(site.name);
    inputRef.current?.blur();
  }, [onSearch]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Escape') { setOpen(false); setQuery(''); }
    if (e.key === 'Enter' && query.length >= 2) {
      onSearch(query);
      setOpen(false);
    }
  }, [query, onSearch]);

  return (
    <div style={{
      position: 'absolute', top: 10, right: 50, zIndex: 20,
      width: 220,
    }}>
      <div style={{ position: 'relative' }}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Search sites or boroughs…"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => results.length && setOpen(true)}
          style={{
            width: '100%',
            background: 'rgba(11,17,32,0.92)',
            backdropFilter: 'blur(6px)',
            border: '1px solid #1E293B',
            borderRadius: 8,
            padding: '6px 10px 6px 30px',
            color: '#F1F5F9',
            fontSize: 11,
            outline: 'none',
          }}
        />
        {/* Search icon */}
        <svg style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)', opacity: 0.4 }}
          width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#94A3B8" strokeWidth="2.5">
          <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
        </svg>
        {/* Clear */}
        {query && (
          <button
            onClick={() => { setQuery(''); setResults([]); setOpen(false); }}
            style={{ position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
              background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: 12 }}>
            ✕
          </button>
        )}
      </div>

      {/* Dropdown results */}
      {open && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
          background: '#131C2E', border: '1px solid #1E293B',
          borderRadius: 8, overflow: 'hidden',
          boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          zIndex: 30,
        }}>
          {results.map(site => (
            <div
              key={site.id}
              onClick={() => handleSelect(site)}
              style={{
                padding: '7px 10px', cursor: 'pointer',
                borderBottom: '1px solid rgba(30,41,59,0.5)',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(59,130,246,0.08)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <div style={{ fontSize: 11, fontWeight: 500, color: '#F1F5F9', marginBottom: 1 }}>{site.name}</div>
              <div style={{ fontSize: 10, color: '#475569' }}>{site.borough} · {site.agency}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
