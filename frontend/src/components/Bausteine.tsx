/**
 * Wiederverwendete Bausteine.
 *
 * Alles, was auf mehreren Seiten vorkommt, steht hier genau einmal. Dadurch
 * sehen Karten, Hinweise und Formularfelder überall gleich aus - das ist der
 * Unterschied zwischen "wirkt gebaut" und "wirkt zusammengesteckt".
 */
import type { ReactNode } from "react";
import type { Quelle } from "../api";
import { Symbol, type SymbolName } from "./Symbole";

// ---------------------------------------------------------------------------
// HINWEISE
// ---------------------------------------------------------------------------
type HinweisArt = "info" | "erfolg" | "warnung" | "fehler";

const HINWEIS_SYMBOL: Record<HinweisArt, SymbolName> = {
  info: "info",
  erfolg: "haken",
  warnung: "warnung",
  fehler: "fehler",
};

export function Hinweis({ art = "info", children }: { art?: HinweisArt; children: ReactNode }) {
  // role="alert" unterbricht Vorleseprogramme sofort - das ist bei Fehlern
  // gewollt, bei allem anderen unhöflich.
  return (
    <div className={`hinweis ${art}`} role={art === "fehler" ? "alert" : "status"}>
      <Symbol name={HINWEIS_SYMBOL[art]} groesse={19} />
      <div>{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// KARTE
// ---------------------------------------------------------------------------
export function Karte({
  titel,
  symbol,
  werkzeuge,
  fuss,
  flach,
  children,
}: {
  titel?: string;
  symbol?: SymbolName;
  /** Schaltflächen rechts in der Kopfzeile. */
  werkzeuge?: ReactNode;
  fuss?: ReactNode;
  /** Ohne Innenabstand - für Karten, die selbst scrollende Bereiche enthalten. */
  flach?: boolean;
  children: ReactNode;
}) {
  return (
    <section className="karte">
      {titel && (
        <header className="karte-kopf">
          {symbol && <Symbol name={symbol} />}
          <h2>{titel}</h2>
          {werkzeuge && <div className="karte-kopf-rechts">{werkzeuge}</div>}
        </header>
      )}
      {flach ? children : <div className="karte-koerper">{children}</div>}
      {fuss && <footer className="karte-fuss">{fuss}</footer>}
    </section>
  );
}

// ---------------------------------------------------------------------------
// KENNZAHL
// ---------------------------------------------------------------------------
export type KachelFarbe = "blau" | "tuerkis" | "gruen" | "gelb" | "rot";

export function Kachel({
  titel,
  wert,
  zusatz,
  symbol,
  farbe = "blau",
  klein,
}: {
  titel: string;
  wert: string;
  zusatz?: string;
  symbol: SymbolName;
  farbe?: KachelFarbe;
  /** Für Werte, die keine Zahl sind (Datum, Text). */
  klein?: boolean;
}) {
  return (
    <article className="kachel">
      <div className={`kachel-symbol ${farbe}`}>
        <Symbol name={symbol} groesse={22} />
      </div>
      <div style={{ minWidth: 0 }}>
        <div className="kachel-titel">{titel}</div>
        <div className={`kachel-wert${klein ? " klein" : ""}`}>{wert}</div>
        {zusatz && <div className="kachel-zusatz">{zusatz}</div>}
      </div>
    </article>
  );
}

// ---------------------------------------------------------------------------
// PLAKETTE
// ---------------------------------------------------------------------------
export function Plakette({
  art,
  punkt,
  children,
}: {
  art?: "gruen" | "gelb" | "rot" | "blau";
  punkt?: boolean;
  children: ReactNode;
}) {
  return (
    <span className={`plakette${art ? ` ${art}` : ""}`}>
      {punkt && <span className="punkt" aria-hidden="true" />}
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// EINGABEFELD
// ---------------------------------------------------------------------------
export function Feld({
  label,
  wert,
  aendern,
  hilfe,
  platzhalter,
  pflicht,
  art = "text",
}: {
  label: string;
  wert: string;
  aendern: (wert: string) => void;
  hilfe?: string;
  platzhalter?: string;
  pflicht?: boolean;
  art?: "text" | "date";
}) {
  const kennung = `feld-${label.replace(/\W+/g, "-").toLowerCase()}`;
  return (
    <div className="feld">
      <label htmlFor={kennung}>
        {label}
        {pflicht && <span aria-hidden="true"> *</span>}
        {pflicht && <span className="nur-vorlesen"> (Pflichtangabe)</span>}
      </label>
      <input
        id={kennung}
        type={art}
        value={wert}
        placeholder={platzhalter}
        aria-describedby={hilfe ? `${kennung}-hilfe` : undefined}
        onChange={(e) => aendern(e.target.value)}
      />
      {hilfe && (
        <div className="hilfe" id={`${kennung}-hilfe`}>
          {hilfe}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// AUSWAHLFELD MIT BESCHREIBUNG
// ---------------------------------------------------------------------------
export function Wahlfeld({
  name,
  gewaehlt,
  waehlen,
  titel,
  text,
}: {
  name: string;
  gewaehlt: boolean;
  waehlen: () => void;
  titel: string;
  text?: string;
}) {
  return (
    <label className="wahlfeld">
      <input type="radio" name={name} checked={gewaehlt} onChange={waehlen} />
      <span>
        <span className="wahlfeld-titel">{titel}</span>
        {text && <span className="wahlfeld-text">{text}</span>}
      </span>
    </label>
  );
}

// ---------------------------------------------------------------------------
// FORTSCHRITT ÜBER DIE DREI ARBEITSSCHRITTE
// ---------------------------------------------------------------------------
export interface SchrittZustand {
  titel: string;
  fertig: boolean;
  status: string;
}

export function Fortschritt({ schritte }: { schritte: SchrittZustand[] }) {
  let laufendGesetzt = false;
  return (
    <ol className="schritte" aria-label="Arbeitsschritte">
      {schritte.map((schritt, index) => {
        let klasse = "schritt";
        if (schritt.fertig) {
          klasse += " fertig";
        } else if (!laufendGesetzt) {
          klasse += " laufend";
          laufendGesetzt = true;
        }
        return (
          <li key={schritt.titel} className={klasse}>
            <span className="schritt-nummer" aria-hidden="true">
              {schritt.fertig ? <Symbol name="haken" groesse={16} strich={2.4} /> : index + 1}
            </span>
            <span style={{ minWidth: 0 }}>
              <span className="schritt-text">{schritt.titel}</span>
              <span className="nur-vorlesen">{schritt.fertig ? " (erledigt): " : ": "}</span>
              <span className="schritt-status" style={{ display: "block" }}>
                {schritt.status}
              </span>
            </span>
          </li>
        );
      })}
    </ol>
  );
}

// ---------------------------------------------------------------------------
// QUELLENANZEIGE
// ---------------------------------------------------------------------------
/** Kürzt eine Herkunft auf etwas, das in eine schmale Spalte passt. */
function kurzerName(quelle: string): string {
  if (/^https?:\/\//i.test(quelle)) {
    try {
      return new URL(quelle).hostname.replace(/^www\./, "");
    } catch {
      return quelle;
    }
  }
  return quelle.split(/[\\/]/).pop() ?? quelle;
}

export function Quellenliste({ quellen }: { quellen: Quelle[] }) {
  if (quellen.length === 0) return null;
  return (
    <div className="quellenliste">
      {quellen.map((q) => (
        <details className="quelle" key={`${q.herkunft}-${q.nummer}`}>
          <summary>
            <span className={`ziffer${q.herkunft === "nutzer" ? " nutzer" : ""}`} aria-hidden="true">
              {q.nummer}
            </span>
            <span style={{ minWidth: 0 }}>
              <span className="nur-vorlesen">Beleg {q.nummer}: </span>
              <span className="quelle-name">{kurzerName(q.quelle)}</span>
              <span className="quelle-art">
                {q.art}
                {q.ueberschrift ? ` · ${q.ueberschrift}` : ""}
              </span>
            </span>
          </summary>
          <div className="quelle-text">{q.ausschnitt}</div>
        </details>
      ))}
    </div>
  );
}

/** Leerzustand: erklärt, was hier stehen wird, statt nur leer zu sein. */
export function Leerbereich({
  symbol,
  titel,
  text,
}: {
  symbol: SymbolName;
  titel: string;
  text: string;
}) {
  return (
    <div style={{ textAlign: "center", padding: "var(--a5) var(--a3)", color: "var(--gedaempft)" }}>
      <div style={{ display: "grid", placeItems: "center", marginBottom: "var(--a2)", color: "var(--rand-kraeftig)" }}>
        <Symbol name={symbol} groesse={30} />
      </div>
      <div style={{ fontWeight: 620, color: "var(--text)" }}>{titel}</div>
      <div style={{ fontSize: "0.85rem", marginTop: "var(--a1)", lineHeight: 1.55 }}>{text}</div>
    </div>
  );
}
