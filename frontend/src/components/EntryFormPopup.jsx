'use client';

import { useState } from 'react';

const BOROUGHS = ['Manhattan', 'Brooklyn', 'Queens', 'The Bronx', 'Staten Island'];

const todayStr = () =>
  new Date().toLocaleDateString('en-US', {
    timeZone: 'America/New_York',
    month: '2-digit',
    day: '2-digit',
    year: 'numeric',
  });

const nowStr = () =>
  new Date().toLocaleTimeString('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });

const emptyForm = () => ({
  siteName: '',
  location: '',
  borough: '',
  date: todayStr(),
  time: nowStr(),
  notes: '',
  photo: null,
});

const inputCls =
  'w-full rounded-md bg-white border border-[#E2E8F0] text-[0.82rem] text-[#0F172A] placeholder:text-[#94A3B8] px-3 py-2 outline-none focus:border-[#94A3B8] transition-colors';
const labelCls =
  'block text-[0.62rem] font-semibold uppercase tracking-[0.12em] text-[#64748B] mb-1';

/** Shared entry-form popup used by the landing page and the dashboard header. */
export default function EntryFormPopup({ open, onClose }) {
  const [dragOver, setDragOver] = useState(false);
  const [form, setForm] = useState(emptyForm());
  const [photoPreview, setPhotoPreview] = useState(null);

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const handlePhoto = (file) => {
    const reader = new FileReader();
    reader.onload = (ev) => setPhotoPreview(ev.target?.result ?? null);
    reader.readAsDataURL(file);

    setForm((f) => ({
      ...f,
      photo: file,
      siteName: 'PS 123 Mahalia Jackson School',
      location: '40 W 112th St, New York, NY 10026',
      borough: 'Manhattan',
      date: todayStr(),
      time: nowStr(),
      notes:
        'Mixed recyclables detected: paper, cardboard, bubble wrap. Bin overflow observed on north side.',
    }));
  };

  const handleClose = () => {
    setForm(emptyForm());
    setPhotoPreview(null);
    onClose();
  };

  const saveEntry = () => {
    window.location.href = '/upload';
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(0,0,0,0.75)] p-4"
      onClick={handleClose}
    >
      <div
        className="relative w-full max-w-[680px] rounded-2xl bg-white shadow-[0_40px_80px_rgba(0,0,0,0.25)] overflow-hidden border border-[#E2E8F0]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* close */}
        <button
          onClick={handleClose}
          className="absolute right-4 top-4 z-10 text-[#94A3B8] hover:text-[#475569] transition-colors text-[1.1rem] leading-none"
          aria-label="Close"
        >
          ✕
        </button>

        <div className="grid grid-cols-1 sm:grid-cols-2">
          {/* LEFT — photo upload */}
          <div className="flex flex-col p-5 sm:border-r sm:border-[#E2E8F0] bg-[#F8FAFC]">
            <div className="text-[0.62rem] font-semibold uppercase tracking-[0.12em] text-[#64748B] mb-3">
              Photo
            </div>

            <label
              className={`flex flex-1 min-h-[220px] flex-col items-center justify-center gap-3 rounded-xl cursor-pointer border-2 border-dashed transition-colors ${
                dragOver
                  ? 'border-[#94A3B8] bg-[#E2E8F0]'
                  : 'border-[#CBD5E1] bg-[#F1F5F9]'
              }`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const file = e.dataTransfer.files[0];
                if (file) handlePhoto(file);
              }}
            >
              <input
                type="file"
                accept="image/png,image/jpeg"
                className="sr-only"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handlePhoto(f); }}
              />
              {photoPreview ? (
                <img
                  src={photoPreview}
                  alt="preview"
                  className="w-full h-full object-cover rounded-xl"
                />
              ) : (
                <>
                  <span className="flex items-center justify-center w-9 h-9 rounded-full border border-[#CBD5E1] text-[#94A3B8] text-xl leading-none">
                    +
                  </span>
                  <div className="text-center">
                    <div className="text-[0.8rem] text-[#475569] font-medium">Upload photo</div>
                    <div className="mt-1 text-[0.62rem] text-[#94A3B8] leading-normal max-w-35 mx-auto">
                      Drag &amp; drop or click to browse
                      <br />
                      PNG, JPG up to 10MB
                    </div>
                  </div>
                </>
              )}
            </label>
          </div>

          {/* RIGHT — details form */}
          <div className="flex flex-col p-5 gap-4">
            <div className="text-[0.9rem] font-semibold text-[#0F172A] tracking-[-0.01em]">
              Details
            </div>

            <div>
              <label className={labelCls}>Site name</label>
              <input
                type="text"
                placeholder="e.g. PS 123 School Complex"
                value={form.siteName}
                onChange={set('siteName')}
                className={inputCls}
              />
            </div>

            <div>
              <label className={labelCls}>Location</label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[#94A3B8] text-[0.8rem]">
                  📍
                </span>
                <input
                  type="text"
                  placeholder="Address or borough"
                  value={form.location}
                  onChange={set('location')}
                  className={`${inputCls} pl-8`}
                />
              </div>
            </div>

            <div>
              <label className={labelCls}>Borough</label>
              <select
                value={form.borough}
                onChange={set('borough')}
                className={`${inputCls} appearance-none`}
              >
                <option value="">Select borough</option>
                {BOROUGHS.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelCls}>Date</label>
                <input
                  type="text"
                  value={form.date}
                  onChange={set('date')}
                  className={inputCls}
                />
              </div>
              <div>
                <label className={labelCls}>Time</label>
                <input
                  type="text"
                  value={form.time}
                  onChange={set('time')}
                  className={inputCls}
                />
              </div>
            </div>

            <div>
              <label className={labelCls}>Notes (optional)</label>
              <textarea
                placeholder="Additional observations..."
                value={form.notes}
                onChange={set('notes')}
                rows={3}
                className={`${inputCls} resize-none`}
              />
            </div>

            <button
              type="button"
              onClick={saveEntry}
              className="mt-auto w-full rounded-md bg-[#0F172A] border border-[#0F172A] py-2.5 text-[0.82rem] font-semibold uppercase tracking-[0.08em] text-white hover:bg-[#1E293B] transition-colors"
            >
              Save entry ↗
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
