/**
 * Symbolsatz der Oberfläche.
 *
 * Bewusst als eingebettete SVG statt einer Symbolschrift oder Emoji:
 *
 *  - Emoji sehen auf jedem Betriebssystem anders aus und wirken beiläufig.
 *    Für eine Anwendung, der jemand Gesundheitsunterlagen anvertraut, ist das
 *    der falsche Ton.
 *  - Eine Symbolschrift aus dem Netz nachzuladen scheidet aus: Es darf nichts
 *    an fremde Server gehen, und ohne Internet müsste alles trotzdem laufen.
 *
 * Alle Symbole zeichnen auf einem 24er-Raster mit gleicher Strichstärke und
 * erben die Textfarbe, damit sie sich überall einfügen.
 */
import type { ReactElement } from "react";

export type SymbolName =
  | "uebersicht"
  | "unterlagen"
  | "assistent"
  | "quellen"
  | "brief"
  | "einstellungen"
  | "hochladen"
  | "schild"
  | "hilfe"
  | "haken"
  | "warnung"
  | "fehler"
  | "info"
  | "kopieren"
  | "papierkorb"
  | "herunterladen"
  | "plus"
  | "pfeil"
  | "senden"
  | "kalender"
  | "uhr"
  | "abmelden"
  | "schluessel"
  | "datei"
  | "suche";

/** Pfaddaten je Symbol. Alle auf 24x24 gezeichnet. */
const PFADE: Record<SymbolName, ReactElement> = {
  uebersicht: (
    <>
      <rect x="3" y="3" width="7.5" height="7.5" rx="1.5" />
      <rect x="13.5" y="3" width="7.5" height="7.5" rx="1.5" />
      <rect x="3" y="13.5" width="7.5" height="7.5" rx="1.5" />
      <rect x="13.5" y="13.5" width="7.5" height="7.5" rx="1.5" />
    </>
  ),
  unterlagen: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </>
  ),
  assistent: (
    <>
      <path d="M21 12a8 8 0 0 1-8 8H7l-4 3 1.2-4.2A8 8 0 1 1 21 12z" />
      <path d="M8.5 11.5h7M8.5 14.5h4" />
    </>
  ),
  quellen: (
    <>
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H19v15H6.5A2.5 2.5 0 0 0 4 20.5z" />
      <path d="M19 18v3H6.5A2.5 2.5 0 0 1 4 18.5" />
      <path d="M8 7.5h7M8 11h5" />
    </>
  ),
  brief: (
    <>
      <rect x="3.5" y="5" width="17" height="14" rx="2" />
      <path d="M3.5 7.5 12 13l8.5-5.5" />
    </>
  ),
  einstellungen: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 14.5a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5v.2a2 2 0 1 1-4 0v-.1a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0 .3-1.8 1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H9a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 1 1.5 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V9a1.6 1.6 0 0 0 1.5 1h.2a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z" />
    </>
  ),
  hochladen: (
    <>
      <path d="M20 16.5A4.5 4.5 0 0 0 17.8 8a6 6 0 0 0-11.6 1.5A4 4 0 0 0 6.5 17" />
      <path d="M12 12v8M8.5 15.5 12 12l3.5 3.5" />
    </>
  ),
  schild: (
    <>
      <path d="M12 3l7.5 3v5.5c0 4.6-3.1 8.4-7.5 9.5-4.4-1.1-7.5-4.9-7.5-9.5V6z" />
      <path d="M9 12l2.2 2.2L15.5 10" />
    </>
  ),
  hilfe: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.6 9.5a2.5 2.5 0 1 1 3.3 2.4c-.6.2-.9.8-.9 1.4v.4" />
      <path d="M12 17.2h.01" />
    </>
  ),
  haken: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M8 12.2l2.8 2.8L16 9.5" />
    </>
  ),
  warnung: (
    <>
      <path d="M10.3 4.3 2.8 17.2A2 2 0 0 0 4.5 20.2h15a2 2 0 0 0 1.7-3L13.7 4.3a2 2 0 0 0-3.4 0z" />
      <path d="M12 9.5v4M12 17h.01" />
    </>
  ),
  fehler: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M15 9l-6 6M9 9l6 6" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" />
    </>
  ),
  kopieren: (
    <>
      <rect x="8.5" y="8.5" width="12" height="12" rx="2" />
      <path d="M15.5 8.5v-2a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </>
  ),
  papierkorb: (
    <>
      <path d="M4 6.5h16M9.5 6.5V4.8a1.3 1.3 0 0 1 1.3-1.3h2.4a1.3 1.3 0 0 1 1.3 1.3v1.7" />
      <path d="M6.5 6.5l.9 12.2a2 2 0 0 0 2 1.8h5.2a2 2 0 0 0 2-1.8l.9-12.2" />
      <path d="M10.5 10.5v6M13.5 10.5v6" />
    </>
  ),
  herunterladen: (
    <>
      <path d="M12 3.5v11M8 11l4 3.5 4-3.5" />
      <path d="M4.5 16v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V16" />
    </>
  ),
  plus: <path d="M12 5.5v13M5.5 12h13" />,
  pfeil: <path d="M4.5 12h14M13 6.5l5.5 5.5-5.5 5.5" />,
  senden: <path d="M4 12l16-7.5-4 16-4.2-5.6z M11.8 14.9 20 4.5" />,
  kalender: (
    <>
      <rect x="3.5" y="5.5" width="17" height="15" rx="2" />
      <path d="M3.5 10h17M8.5 3.5v4M15.5 3.5v4" />
    </>
  ),
  uhr: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5.2l3.2 1.9" />
    </>
  ),
  abmelden: (
    <>
      <path d="M14.5 3.5h3a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2h-3" />
      <path d="M9.5 16.5 4.5 12l5-4.5M4.5 12h10" />
    </>
  ),
  schluessel: (
    <>
      <circle cx="8" cy="12" r="4" />
      <path d="M12 12h9M18 12v3.5M15 12v2.5" />
    </>
  ),
  datei: (
    <>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5M8.5 13h7M8.5 16.5h4.5" />
    </>
  ),
  suche: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="M16 16l4.5 4.5" />
    </>
  ),
};

export function Symbol({
  name,
  groesse = 20,
  strich = 1.7,
}: {
  name: SymbolName;
  groesse?: number;
  /** Strichstärke. Kleine Symbole vertragen etwas mehr. */
  strich?: number;
}) {
  return (
    <svg
      className="symbol"
      width={groesse}
      height={groesse}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={strich}
      strokeLinecap="round"
      strokeLinejoin="round"
      // Symbole begleiten immer eine Beschriftung; Vorleseprogramme sollen
      // sie deshalb überspringen statt sie doppelt anzusagen.
      aria-hidden="true"
      focusable="false"
    >
      {PFADE[name]}
    </svg>
  );
}

/** Wortmarke: stilisierte schützende Hand über einem Herzen. */
export function Wortmarke({ groesse = 34 }: { groesse?: number }) {
  return (
    <svg
      width={groesse}
      height={groesse}
      viewBox="0 0 40 40"
      fill="none"
      aria-hidden="true"
      focusable="false"
    >
      <rect width="40" height="40" rx="11" fill="url(#marke-verlauf)" />
      <path
        d="M20 29.5c-5.6-3.3-9-6.9-9-10.8a4.6 4.6 0 0 1 9-1.6 4.6 4.6 0 0 1 9 1.6c0 3.9-3.4 7.5-9 10.8z"
        fill="#ffffff"
        fillOpacity="0.95"
      />
      <path
        d="M12.5 13.5c2.2-2.6 5-4 7.5-4s5.3 1.4 7.5 4"
        stroke="#ffffff"
        strokeOpacity="0.55"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <defs>
        <linearGradient id="marke-verlauf" x1="0" y1="0" x2="40" y2="40">
          <stop stopColor="#1d7ea8" />
          <stop offset="1" stopColor="#0d4f6e" />
        </linearGradient>
      </defs>
    </svg>
  );
}
