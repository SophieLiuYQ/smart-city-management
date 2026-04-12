'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import Map, { NavigationControl } from 'react-map-gl/maplibre';
import type { MapRef } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

type ViewMode = '2D' | '3D';

const LLM_OPTIONS = [
  'Qwen 3.5',
  'GPT-4o',
  'Claude Sonnet 4.5',
  'Gemini 1.5 Pro',
  'Llama 3.3 70B',
  'Mistral Large',
];

const MAP_STYLE = 'https://tiles.openfreemap.org/styles/liberty';

const INITIAL_VIEW = {
  longitude: -73.9857,
  latitude: 40.7549,
  zoom: 15.5,
  pitch: 60,
  bearing: -20,
};

const VIEW_2D = { pitch: 0, bearing: 0, zoom: 14.5 };
const VIEW_3D = { pitch: 60, bearing: -20, zoom: 15.5 };

// ── Icons ──────────────────────────────────────────────────────────────────

function CheckCircleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#374151" strokeWidth="1.8">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 12l3 3 5-5" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="1.8">
      <path d="M23 4v6h-6" />
      <path d="M1 20v-6h6" />
      <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
    </svg>
  );
}

function DotsIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="#9CA3AF">
      <circle cx="5" cy="12" r="1.5" />
      <circle cx="12" cy="12" r="1.5" />
      <circle cx="19" cy="12" r="1.5" />
    </svg>
  );
}

function CommentIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="1.8">
      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
    </svg>
  );
}


// ── Page ───────────────────────────────────────────────────────────────────

export default function UploadPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('3D');
  const [llm, setLlm] = useState(LLM_OPTIONS[0]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const t = setTimeout(() => setLoading(false), 4500);
    return () => clearTimeout(t);
  }, []);
  const mapRef = useRef<MapRef>(null);

  const handleModeChange = useCallback((mode: ViewMode) => {
    setViewMode(mode);
    const target = mode === '3D' ? VIEW_3D : VIEW_2D;
    mapRef.current?.easeTo({ ...target, duration: 900 });
  }, []);

  const handleRefresh = useCallback(() => {
    mapRef.current?.easeTo({ ...INITIAL_VIEW, duration: 700 });
    setViewMode('3D');
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      background: '#F3F4F6',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 24,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    }}>
      <div style={{
        width: '100%',
        maxWidth: 900,
        background: '#FFFFFF',
        borderRadius: 14,
        border: '1px solid #E5E7EB',
        boxShadow: '0 1px 6px rgba(0,0,0,0.07)',
        overflow: 'hidden',
      }}>

        {/* ── HEADER ── */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 16px',
          borderBottom: '1px solid #F3F4F6',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <CheckCircleIcon />
            <span style={{ fontSize: 15, fontWeight: 600, color: '#111827' }}>Result</span>
          </div>

          <select
            value={llm}
            onChange={e => setLlm(e.target.value)}
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: '#374151',
              background: '#F9FAFB',
              border: '1px solid #E5E7EB',
              borderRadius: 8,
              padding: '5px 10px',
              cursor: 'pointer',
              outline: 'none',
            }}
          >
            {LLM_OPTIONS.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>

          {/* <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              display: 'flex',
              background: '#F3F4F6',
              borderRadius: 8,
              padding: 3,
              gap: 2,
            }}>
              {(['2D', '3D'] as ViewMode[]).map(mode => (
                <button
                  key={mode}
                  onClick={() => handleModeChange(mode)}
                  style={{
                    padding: '4px 12px',
                    borderRadius: 6,
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: 13,
                    fontWeight: 500,
                    transition: 'all 0.15s',
                    background: viewMode === mode ? '#1E293B' : 'transparent',
                    color: viewMode === mode ? '#FFFFFF' : '#6B7280',
                  }}
                >
                  {mode}
                </button>
              ))}
            </div>

            <button onClick={handleRefresh} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '4px 6px', borderRadius: 6, display: 'flex', alignItems: 'center',
            }}>
              <DotsIcon />
            </button>

            <button onClick={handleRefresh} style={{
              background: 'none', border: 'none', cursor: 'pointer',
              padding: '4px 6px', borderRadius: 6, display: 'flex', alignItems: 'center',
            }}>
              <RefreshIcon />
            </button>
          </div> */}
        </div>

        {/* ── CANVAS — image fills area, map inset bottom-right ── */}
        <div style={{ position: 'relative', height: 520, background: '#FAFAFA' }}>

          {/* Main image area */}
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#FAFAFA' }}>
            {loading ? (
              <>
                <style>{`
                  @keyframes spin { to { transform: rotate(360deg); } }
                `}</style>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
                  <div style={{
                    width: 40, height: 40, borderRadius: '50%',
                    border: '3px solid #E2E8F0',
                    borderTopColor: '#0F172A',
                    animation: 'spin 0.8s linear infinite',
                  }} />
                  <span style={{ fontSize: 12, color: '#94A3B8', letterSpacing: '0.04em' }}>
                    Analyzing image…
                  </span>
                </div>
              </>
            ) : (
              <img
                src="/img2.jpeg"
                alt="Site photo"
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
              />
            )}
          </div>

          {/* Status label */}
          <div style={{
            position: 'absolute',
            top: 12,
            left: 14,
            background: 'rgba(255,255,255,0.88)',
            backdropFilter: 'blur(6px)',
            borderRadius: 8,
            border: '1px solid #E5E7EB',
            padding: '4px 10px',
            fontSize: 11,
            color: '#6B7280',
            letterSpacing: '0.03em',
            pointerEvents: 'none',
            zIndex: 10,
          }}>
            Ready for review
          </div>

          {/* ── TAGS — bottom left ── */}
          {!loading && <div style={{
            position: 'absolute',
            bottom: 14,
            left: 14,
            display: 'flex',
            gap: 6,
            zIndex: 10,
          }}>
            {[
              { label: 'Glass',   bg: '#EFF6FF', color: '#3B82F6', border: '#BFDBFE' },
              { label: 'Paper',   bg: '#F0FDF4', color: '#16A34A', border: '#BBF7D0' },
              { label: 'Organic', bg: '#FFF7ED', color: '#EA580C', border: '#FED7AA' },
            ].map(({ label, bg, color, border }) => (
              <span key={label} style={{
                background: bg,
                color,
                border: `1px solid ${border}`,
                borderRadius: 20,
                fontSize: 11,
                fontWeight: 600,
                padding: '4px 10px',
                letterSpacing: '0.03em',
              }}>
                {label}
              </span>
            ))}
          </div>}

          {/* ── MAP INSET — bottom right ── */}
          {!loading && <div style={{
            position: 'absolute',
            bottom: 14,
            right: 14,
            width: 360,
            height: 260,
            borderRadius: 10,
            overflow: 'hidden',
            border: '1px solid #E5E7EB',
            boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
            zIndex: 10,
          }}>
            <Map
              ref={mapRef}
              initialViewState={INITIAL_VIEW}
              style={{ width: '100%', height: '100%' }}
              mapStyle={MAP_STYLE}
            >
              <NavigationControl position="top-right" showCompass={false} />
            </Map>

            {/* map mode badge */}
            <div style={{
              position: 'absolute',
              bottom: 6,
              left: 8,
              background: 'rgba(0,0,0,0.55)',
              backdropFilter: 'blur(4px)',
              borderRadius: 5,
              padding: '2px 7px',
              fontSize: 10,
              fontWeight: 600,
              color: '#fff',
              letterSpacing: '0.06em',
              pointerEvents: 'none',
            }}>
              {viewMode}
            </div>
          </div>}
        </div>

      </div>
    </div>
  );
}
