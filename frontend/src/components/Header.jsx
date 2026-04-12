'use client';

import { useState } from 'react';
import { useDashboard } from '../context/DashboardContext';
import EntryFormPopup from './EntryFormPopup';

export default function Header() {
  const { viewMode } = useDashboard();
  const [showUpload, setShowUpload] = useState(false);

  return (
    <>
      <EntryFormPopup open={showUpload} onClose={() => setShowUpload(false)} />

      <header style={{
        background: 'rgba(248,250,252,.97)', height: 52,
        display: 'flex', alignItems: 'center',
        padding: '0 20px', justifyContent: 'space-between',
        borderBottom: '1px solid rgba(0,0,0,.07)',
        position: 'sticky', top: 0, zIndex: 10,
      }}>
        <span style={{ fontSize: 12, color: '#64748B' }}>
          Dashboard{' '}
          <span style={{ color: '#CBD5E1' }}>/</span>{' '}
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
          <span style={{ fontSize: 11, color: '#94A3B8' }}>Apr 10, 2026 2:34 PM</span>

          {/* Upload entry button */}
          <button
            onClick={() => setShowUpload(true)}
            style={{
              display: 'flex', alignItems: 'center', gap: 5,
              padding: '5px 12px', borderRadius: 8,
              border: '1px solid #E2E8F0',
              background: '#0F172A', color: '#FFFFFF',
              fontSize: 11, fontWeight: 600,
              cursor: 'pointer', letterSpacing: '0.04em',
            }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload Entry
          </button>
        </div>
      </header>
    </>
  );
}
