"use client";

import { useEffect, useState, type FormEvent } from "react";

const NAV_ITEMS = [
  { href: "/map", label: "Research" },
  { href: "/upload", label: "Product" },
];

const HERO_LINES = ["Smart", "City Nexus"];
const DISCOVERY_BARS = [0, 1, 2, 3];

const panelRound =
  "rounded-[1.6rem] border border-[rgba(19,19,19,0.34)] shadow-[0_22px_40px_rgba(0,0,0,0.18)]";
const panelCard =
  "rounded-[1rem] border border-[rgba(19,19,19,0.34)] shadow-[0_12px_32px_rgba(0,0,0,0.14)]";
const microLabel =
  "text-[0.6rem] font-medium uppercase tracking-[0.16em] text-[rgba(19,19,19,0.72)]";
const bodyCopy = "text-[rgba(19,19,19,0.72)]";
const shellPadding = "px-3 sm:px-4 lg:px-6";

const getNewYorkTime = () =>
  new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(new Date());

/* ── Popup form state ── */
type EntryForm = {
  siteName: string;
  location: string;
  borough: string;
  date: string;
  time: string;
  notes: string;
  photo: File | null;
};

const BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "The Bronx", "Staten Island"];

const todayISO = new Date().toLocaleDateString("en-US", {
  timeZone: "America/New_York",
  month: "2-digit",
  day: "2-digit",
  year: "numeric",
});

const nowTime = new Date().toLocaleTimeString("en-US", {
  timeZone: "America/New_York",
  hour: "2-digit",
  minute: "2-digit",
  hour12: true,
});

export function SmartCityManagmentLanding() {
  const [estTime, setEstTime] = useState("--:--");
  const [showPopup, setShowPopup] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [form, setForm] = useState<EntryForm>({
    siteName: "",
    location: "",
    borough: "",
    date: todayISO,
    time: nowTime,
    notes: "",
    photo: null,
  });
  const [photoPreview, setPhotoPreview] = useState<string | null>(null);

  useEffect(() => {
    const updateTime = () => setEstTime(getNewYorkTime());
    updateTime();
    const intervalId = window.setInterval(updateTime, 30_000);
    return () => window.clearInterval(intervalId);
  }, []);

  const set = (k: keyof EntryForm) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) =>
    setForm((f) => ({ ...f, [k]: e.target.value }));

  const handlePhoto = (file: File) => {
    setForm((f) => ({ ...f, photo: file }));
    const reader = new FileReader();
    reader.onload = (ev) => setPhotoPreview(ev.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setShowPopup(true);
  };

  const closePopup = () => setShowPopup(false);

  const saveEntry = () => {
    window.location.href = "/upload";
  };

  /* ── field style helpers ── */
  const inputCls =
    "w-full rounded-md bg-white border border-[#E2E8F0] text-[0.82rem] text-[#0F172A] placeholder:text-[#94A3B8] px-3 py-2 outline-none focus:border-[#94A3B8] transition-colors";
  const labelCls = "block text-[0.62rem] font-semibold uppercase tracking-[0.12em] text-[#64748B] mb-1";

  return (
    <>
      {/* ── POPUP MODAL ── */}
      {showPopup && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(0,0,0,0.75)] p-4"
          onClick={closePopup}
        >
          <div
            className="relative w-full max-w-[680px] rounded-2xl bg-white shadow-[0_40px_80px_rgba(0,0,0,0.25)] overflow-hidden border border-[#E2E8F0]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* close */}
            <button
              onClick={closePopup}
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

                {/* drop zone */}
                <label
                  className={`flex flex-1 min-h-[220px] flex-col items-center justify-center gap-3 rounded-xl cursor-pointer border-2 border-dashed transition-colors ${
                    dragOver ? "border-[#94A3B8] bg-[#E2E8F0]" : "border-[#CBD5E1] bg-[#F1F5F9]"
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
                    <img src={photoPreview} alt="preview" className="w-full h-full object-cover rounded-xl" />
                  ) : (
                    <>
                      <span className="flex items-center justify-center w-9 h-9 rounded-full border border-[#CBD5E1] text-[#94A3B8] text-xl leading-none">
                        +
                      </span>
                      <div className="text-center">
                        <div className="text-[0.8rem] text-[#475569] font-medium">Upload photo</div>
                        <div className="mt-1 text-[0.62rem] text-[#94A3B8] leading-normal max-w-35 mx-auto">
                          Drag &amp; drop or click to browse<br />PNG, JPG up to 10MB
                        </div>
                      </div>
                    </>
                  )}
                </label>
              </div>

              {/* RIGHT — details form */}
              <div className="flex flex-col p-5 gap-4">
                <div className="text-[0.9rem] font-semibold text-[#0F172A] tracking-[-0.01em]">Details</div>

                {/* Site name */}
                <div>
                  <label className={labelCls}>Site name</label>
                  <input
                    type="text"
                    placeholder="e.g. PS 123 School Complex"
                    value={form.siteName}
                    onChange={set("siteName")}
                    className={inputCls}
                  />
                </div>

                {/* Location */}
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
                      onChange={set("location")}
                      className={`${inputCls} pl-8`}
                    />
                  </div>
                </div>

                {/* Borough */}
                <div>
                  <label className={labelCls}>Borough</label>
                  <select value={form.borough} onChange={set("borough")} className={`${inputCls} appearance-none`}>
                    <option value="">Select borough</option>
                    {BOROUGHS.map((b) => <option key={b} value={b}>{b}</option>)}
                  </select>
                </div>

                {/* Date + Time */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={labelCls}>Date</label>
                    <input
                      type="text"
                      value={form.date}
                      onChange={set("date")}
                      className={inputCls}
                    />
                  </div>
                  <div>
                    <label className={labelCls}>Time</label>
                    <input
                      type="text"
                      value={form.time}
                      onChange={set("time")}
                      className={inputCls}
                    />
                  </div>
                </div>

                {/* Notes */}
                <div>
                  <label className={labelCls}>Notes (optional)</label>
                  <textarea
                    placeholder="Additional observations..."
                    value={form.notes}
                    onChange={set("notes")}
                    rows={3}
                    className={`${inputCls} resize-none`}
                  />
                </div>

                {/* Save */}
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
      )}

    <div className="overflow-x-hidden bg-canvas">
      <main className={`mx-auto w-full max-w-[1440px] ${shellPadding} pt-3 sm:pt-4`}>
        {/* ── TOPBAR ── */}
        <header
          className={`${panelRound} flex flex-col items-start gap-4 rounded-[1.4rem] bg-panel px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6 sm:rounded-full sm:px-6 sm:py-3.5`}
        >
          <div
            aria-hidden="true"
            className="shrink-0 h-[15px] w-[313px]"
          />

          <nav
            aria-label="Primary"
            className="flex w-full flex-wrap items-center gap-x-4 gap-y-2 text-[0.62rem] font-semibold uppercase tracking-[0.14em] sm:w-auto sm:justify-end sm:gap-[clamp(18px,2.8vw,48px)] sm:text-[0.66rem]"
          >
            {NAV_ITEMS.map((item) => (
              <a key={item.href} href={item.href}>
                {item.label}
              </a>
            ))}
          </nav>
        </header>

        {/* ── HERO GRID ── */}
        <section className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1.75fr)_minmax(18rem,1fr)]">
          {/* LEFT — hero panel */}
          <article
            className={`${panelRound} flex min-h-[420px] min-w-0 flex-col overflow-hidden bg-hero px-4 pb-4 pt-4 sm:min-h-[460px] sm:px-6 sm:pb-5 sm:pt-5 lg:min-h-[490px] lg:px-7`}
          >
            <div
              className={`flex flex-wrap items-center justify-between gap-3 ${microLabel}`}
            >
              <span>System Status: Active</span>
              <span>EST {estTime}</span>
            </div>

            <div className="mt-3 h-px w-full bg-[rgba(19,19,19,0.5)]" />

            <div className="flex flex-1 flex-col gap-6 pt-6 sm:gap-7 sm:pt-8">
              <h1 className="font-display text-[clamp(2.4rem,12vw,7.2rem)] uppercase leading-[0.88] tracking-[-0.055em]">
                {HERO_LINES.map((line) => (
                  <span key={line} className="block">
                    {line}
                  </span>
                ))}
              </h1>

              <p
                className={`max-w-[34ch] border-l-[2px] border-[rgba(19,19,19,0.6)] pl-4 text-[clamp(0.84rem,2.4vw,1.06rem)] leading-[1.55] ${bodyCopy}`}
              >
                NYC&apos;s energy and waste data into one unified intelligence
                layer. Built to run entirely on-device using NVIDIA.
              </p>
            </div>

            <div className="mt-auto pt-6 sm:pt-8">
              <div className={microLabel}>initialize report</div>

              <form
                className="mt-2 grid grid-cols-1 border border-[rgba(19,19,19,0.6)] sm:grid-cols-[minmax(0,1fr)_auto]"
                onSubmit={handleSubmit}
              >
                <input
                  aria-label="Corporate email"
                  type="email"
                  placeholder="ENTR_LOCATION"
                  className="bg-transparent px-4 py-3.5 text-[0.95rem] uppercase tracking-[0.06em] outline-none placeholder:text-[rgba(19,19,19,0.4)] sm:px-5 sm:py-4"
                />
                <button
                  type="submit"
                  className="w-full border-t border-[rgba(19,19,19,0.6)] bg-[#0a0a0a] px-7 py-3.5 font-display text-[0.95rem] uppercase tracking-[-0.02em] text-hero sm:w-auto sm:border-l sm:border-t-0 sm:py-4"
                >
                  Begin {"\u2192"}
                </button>
              </form>

              <div
                className={`mt-2.5 flex items-center gap-2.5 text-[0.54rem] uppercase tracking-[0.11em] ${bodyCopy}`}
              >
                <span className="signal-bars" aria-hidden="true" />
                <span>captured live</span>
              </div>
            </div>
          </article>

          {/* RIGHT — audit + processing */}
          <div className="grid gap-3 min-[768px]:grid-rows-[1fr_auto]">
            <article
              id="audit-shield"
              className={`${panelRound} flex flex-col bg-panel px-4 pb-4 pt-4 sm:px-5 sm:pb-5 sm:pt-5`}
            >
              <div className={microLabel}>Audit Risk Assessment</div>

              <div className="mt-3 font-display text-[clamp(2.4rem,5vw,3.6rem)] uppercase leading-none tracking-[-0.06em]">
                0.04%
              </div>
              <p
                className={`mt-2 text-[0.82rem] leading-[1.5] ${bodyCopy}`}
              >
                Historical industry avg: 2.4%
              </p>

              <div className="mt-4 grid min-h-[68px] grid-cols-1 border border-[rgba(19,19,19,0.55)] sm:grid-cols-[1fr_auto]">
                <div className="flex items-center px-4">
                  <span className={microLabel}>Risk Lvl</span>
                </div>
                <div className="flex items-center justify-center border-t border-[rgba(19,19,19,0.55)] px-5 py-3 sm:border-l sm:border-t-0 sm:py-0">
                  <span className="font-display text-[clamp(1.4rem,2.8vw,2rem)] uppercase leading-none tracking-[-0.06em]">
                    Low
                  </span>
                </div>
              </div>

              <div className="mt-auto pt-8">
                <div className={microLabel}>Deduction Discovery</div>
                <div className="mt-7 grid grid-cols-4 gap-2">
                  {DISCOVERY_BARS.map((i) => (
                    <span
                      key={i}
                      aria-hidden="true"
                      className="block h-[2px] bg-[rgba(19,19,19,0.6)]"
                    />
                  ))}
                </div>
              </div>
            </article>

            <article
              className={`${panelRound} flex min-h-[72px] items-center justify-center bg-accent px-5 py-4 sm:min-h-[82px] sm:px-6 sm:py-5`}
            >
              <span className="text-[0.62rem] uppercase tracking-[0.18em] text-[rgba(19,19,19,0.65)]">
                Recents
              </span>
            </article>
          </div>
        </section>
      </main>

      {/* ── LOWER BAND ── */}
      <section id="capabilities" className="mt-4 bg-band py-6 sm:mt-5 sm:py-8">
        <div className={`mx-auto w-full max-w-[1440px] ${shellPadding}`}>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {/* Card 1 — TS26 */}
            <article
              className={`${panelCard} flex min-h-[200px] flex-col bg-module px-4 pb-4 pt-4 sm:min-h-[220px] sm:px-5 sm:pt-5`}
            >
              <div className="font-display text-[clamp(2.4rem,3.6vw,3.6rem)] uppercase leading-none tracking-[-0.06em]">
                TS26
              </div>
              <div className="mt-2.5 h-px bg-[rgba(19,19,19,0.5)]" />
              <div className={`${microLabel} mt-3`}>Module A</div>
              <div className="mt-2 max-w-[14ch] font-display text-[clamp(1.05rem,1.4vw,1.4rem)] uppercase leading-[1.2] tracking-[-0.05em]">
                Retroactive Scanning
              </div>
              <p
                className={`mt-3 max-w-[26ch] text-[0.85rem] leading-[1.48] ${bodyCopy}`}
              >
                Deep scan of 3 previous fiscal years for unclaimed R&amp;D
                credits.
              </p>
            </article>

            {/* Card 2 — Compliance Engine */}
            <article
              className={`${panelCard} flex min-h-[200px] flex-col bg-panel px-4 pb-4 pt-4 sm:min-h-[220px] sm:px-5 sm:pt-5`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className={microLabel}>Compliance Engine</div>
                  <div className="mt-2 max-w-[9ch] font-display text-[clamp(1.1rem,1.6vw,1.55rem)] uppercase leading-[1.2] tracking-[-0.05em]">
                    Global Entity Sync
                  </div>
                </div>

                <svg
                  aria-hidden="true"
                  viewBox="0 0 54 54"
                  className="mt-0.5 size-[44px] shrink-0"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect
                    x="27"
                    y="1.7"
                    width="35.8"
                    height="35.8"
                    transform="rotate(45 27 1.7)"
                    stroke="rgba(19,19,19,0.68)"
                    strokeWidth="1.5"
                  />
                  <path
                    d="M21.5 31.4L31.2 21.7M25 21.7H31.2V27.9"
                    stroke="rgba(19,19,19,0.68)"
                    strokeWidth="1.5"
                    strokeLinecap="square"
                  />
                </svg>
              </div>

              <div className="mt-auto grid min-h-[72px] grid-cols-1 border border-[rgba(19,19,19,0.55)] sm:grid-cols-2">
                <div className="flex flex-col justify-center px-4 py-3">
                  <div className={microLabel}>Jurisdictions</div>
                  <div className="mt-1 font-display text-[1.2rem] uppercase leading-none tracking-[-0.05em]">
                    142
                  </div>
                </div>
                <div className="flex flex-col justify-center border-t border-[rgba(19,19,19,0.55)] px-4 py-3 sm:border-l sm:border-t-0">
                  <div className={microLabel}>Speed</div>
                  <div className="mt-1 font-display text-[1.2rem] uppercase leading-none tracking-[-0.05em]">
                    2.47s
                  </div>
                </div>
              </div>
            </article>

            {/* Card 3 — Data Security */}
            <article
              id="pricing"
              className={`${panelCard} flex min-h-[200px] flex-col bg-hero px-4 pb-4 pt-4 md:col-span-2 sm:min-h-[220px] sm:px-5 sm:pt-5 xl:col-span-1`}
            >
              <div className={microLabel}>Data Security</div>
              <div className="mt-4 inline-flex w-fit items-center justify-center border border-[rgba(19,19,19,0.6)] px-4 py-2.5 font-display text-[clamp(1.05rem,1.7vw,1.6rem)] uppercase leading-none tracking-[-0.05em]">
                AES-256
              </div>
              <p
                className={`mt-5 max-w-[26ch] text-[0.85rem] leading-[1.48] ${bodyCopy}`}
              >
                Military-grade encryption for all financial ledgers.
                Zero-knowledge architecture.
              </p>
              <span className="barcode mt-auto" aria-hidden="true" />
            </article>
          </div>

          <footer
            id="login"
            className="mt-8 flex flex-col gap-3 text-[0.58rem] uppercase leading-[1.6] tracking-[0.1em] text-[rgba(226,221,208,0.28)] sm:mt-10 sm:text-[0.62rem] min-[620px]:flex-row min-[620px]:items-end min-[620px]:justify-between"
          >
            <div>
              Antler
              <br />
              New York
            </div>
            <div className="min-[620px]:text-right">
              SYS_ID: #E0A15E
              <br />
              LATENCY: 12MS
            </div>
          </footer>
        </div>
      </section>
    </div>
    </>
  );
}
