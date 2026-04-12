"use client";

import { useEffect, useState, type FormEvent } from "react";
import EntryFormPopup from "./EntryFormPopup";

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

type NetworkInfo = {
  downlink?: number;
  effectiveType?: string;
  addEventListener?: (type: "change", listener: () => void) => void;
  removeEventListener?: (type: "change", listener: () => void) => void;
};

const getNewYorkTime = () =>
  new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(new Date());

const getNetworkInfo = (): NetworkInfo | undefined => {
  const nav = navigator as Navigator & {
    connection?: NetworkInfo;
    mozConnection?: NetworkInfo;
    webkitConnection?: NetworkInfo;
  };

  return nav.connection ?? nav.mozConnection ?? nav.webkitConnection;
};

const getWifiStrength = () => {
  const connection = getNetworkInfo();

  if (!connection) {
    return navigator.onLine ? "Online" : "Offline";
  }

  const downlink = connection.downlink ?? 0;
  const effectiveType = connection.effectiveType?.toUpperCase();

  let strength = "Weak";
  if (downlink >= 25) {
    strength = "Excellent";
  } else if (downlink >= 10) {
    strength = "Strong";
  } else if (downlink >= 3) {
    strength = "Moderate";
  }

  return effectiveType ? `${strength} (${effectiveType})` : strength;
};

export function SmartCityManagmentLanding() {
  const [estTime, setEstTime] = useState("--:--");
  const [wifiStrength, setWifiStrength] = useState("--");
  const [showPopup, setShowPopup] = useState(false);

  useEffect(() => {
    const updateTime = () => setEstTime(getNewYorkTime());
    updateTime();
    const intervalId = window.setInterval(updateTime, 30_000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    const updateWifiStrength = () => {
      setWifiStrength(getWifiStrength());
    };

    updateWifiStrength();
    const connection = getNetworkInfo();
    connection?.addEventListener?.("change", updateWifiStrength);
    const intervalId = window.setInterval(updateWifiStrength, 15_000);

    return () => {
      connection?.removeEventListener?.("change", updateWifiStrength);
      window.clearInterval(intervalId);
    };
  }, []);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setShowPopup(true);
  };

  return (
    <>
      <EntryFormPopup open={showPopup} onClose={() => setShowPopup(false)} />

    <div className="overflow-x-hidden bg-canvas">
      <main className={`mx-auto w-full max-w-[1440px] ${shellPadding} pt-3 sm:pt-4`}>
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
              <div className={microLabel}>Simulation</div>

              <div className="mt-3 font-display text-[clamp(2.4rem,5vw,3.6rem)] uppercase leading-none tracking-[-0.06em]">
                Energy
              </div>
              <p
                className={`mt-2 text-[0.82rem] leading-[1.5] ${bodyCopy}`}
              >
                total power saved per day: 87.2 kW
              </p>

              <div className="mt-4 grid min-h-[68px] grid-cols-1 border border-[rgba(19,19,19,0.55)] sm:grid-cols-[1fr_auto]">
                <div className="flex items-center px-4">
                  <span className={microLabel}>Daily Savings</span>
                </div>
                <div className="flex items-center justify-center border-t border-[rgba(19,19,19,0.55)] px-5 py-3 sm:border-l sm:border-t-0 sm:py-0">
                  <span className="font-display text-[clamp(1.4rem,2.8vw,2rem)] uppercase leading-none tracking-[-0.06em]">
                    $38.50
                  </span>
                </div>
              </div>

              <div className="mt-auto pt-8">
                <div className={microLabel}>NYC Open Data</div>
                <div className="flow-strip-wrap mt-7 grid grid-cols-4 gap-2">
                  {DISCOVERY_BARS.map((i) => (
                    <span
                      key={i}
                      aria-hidden="true"
                      className="flow-strip block h-[2px]"
                    />
                  ))}
                </div>
              </div>
            </article>

            <article
              className={`${panelRound} flex min-h-[72px] flex-col items-center justify-center gap-2 bg-accent px-5 py-4 sm:min-h-[82px] sm:px-6 sm:py-5`}
            >
              <div className="grid w-full grid-cols-2 items-center gap-2 sm:gap-3">
                {NAV_ITEMS.map((item, index) => (
                  <a
                    key={item.href}
                    href={item.href}
                    className={`inline-flex min-w-[104px] justify-center rounded-full border border-transparent bg-[rgba(19,19,19,0.16)] px-4 py-1.5 text-[0.76rem] font-semibold uppercase tracking-[0.14em] text-[rgba(19,19,19,0.8)] transition-[background-color,border-color,color,transform] hover:bg-[rgba(19,19,19,0.24)] active:scale-[0.98] active:border-white active:bg-transparent active:text-white focus-visible:border-white focus-visible:outline-none ${
                      index === 0 ? "justify-self-start" : "justify-self-end"
                    }`}
                  >
                    {item.label}
                  </a>
                ))}
              </div>
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
                Waste
              </div>
              <div className="mt-2.5 h-px bg-[rgba(19,19,19,0.5)]" />
              <div className={`${microLabel} mt-3`}>NYC Open Data</div>
              <div className="mt-2 max-w-[14ch] font-display text-[clamp(1.05rem,1.4vw,1.4rem)] uppercase leading-[1.2] tracking-[-0.05em]">
                Clean
              </div>
              <p
                className={`mt-3 max-w-[26ch] text-[0.85rem] leading-[1.48] ${bodyCopy}`}
              >
                Optimize trash clean for 2.4 million units from NYC Open Data.
              </p>
            </article>

            {/* Card 2 — Compliance Engine */}
            <article
              className={`${panelCard} flex min-h-[200px] flex-col bg-panel px-4 pb-4 pt-4 sm:min-h-[220px] sm:px-5 sm:pt-5`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className={microLabel}>Fine Tuning</div>
                  <div className="mt-2 max-w-[14ch] font-display text-[clamp(1.1rem,1.6vw,1.55rem)] uppercase leading-[1.2] tracking-[-0.05em]">
                    Nemotron-Nano 12B
                    <br />
                    Qwen VL-8B
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
                  <div className={microLabel}>samples</div>
                  <div className="mt-1 font-display text-[1.2rem] uppercase leading-none tracking-[-0.05em]">
                    5000
                  </div>
                </div>
                <div className="flex flex-col justify-center border-t border-[rgba(19,19,19,0.55)] px-4 py-3 sm:border-l sm:border-t-0">
                  <div className={microLabel}>batch</div>
                  <div className="mt-1 font-display text-[1.2rem] uppercase leading-none tracking-[-0.05em]">
                    4
                  </div>
                </div>
              </div>
            </article>

            {/* Card 3 — Data Security */}
            <article
              id="pricing"
              className={`${panelCard} flex min-h-[200px] flex-col bg-hero px-4 pb-4 pt-4 md:col-span-2 sm:min-h-[220px] sm:px-5 sm:pt-5 xl:col-span-1`}
            >
              <div className={microLabel}>Local Processing &amp; Privacy</div>
              <div className="mt-4 inline-flex w-fit items-center justify-center border border-[rgba(19,19,19,0.6)] px-4 py-2.5 font-display text-[clamp(1.05rem,1.7vw,1.6rem)] uppercase leading-none tracking-[-0.05em]">
                NVIDIA GB10
              </div>
              <p
                className={`mt-5 max-w-[26ch] text-[0.85rem] leading-[1.48] ${bodyCopy}`}
              >
                128 GB unified memory, 1 PFLOP FP4 on Acer Veriton GN100.
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
              SYS_ID: 6
              <br />
              WIFI: {wifiStrength}
            </div>
          </footer>
        </div>
      </section>
    </div>
    </>
  );
}
