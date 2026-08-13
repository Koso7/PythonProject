/** Wiederverwendete Bausteine der Oberfläche. */
import type { ReactNode } from "react";
import type { Quelle } from "../api";

// ---------------------------------------------------------------------------
// HINWEISE
// ---------------------------------------------------------------------------
type HinweisArt = "info" | "erfolg" | "warnung" | "fehler";

const SYMBOLE: Record<HinweisArt, string> = {
  info: "💡",
  erfolg: "✅",
  warnung: "⚠️",
  fehler: "⛔",
};

export function Hinweis({
  art = "info",
  children,
}: {
  art?: HinweisArt;
  children: ReactNode;
}) {
  // role="status" sorgt dafür, dass Vorleseprogramme die Meldung ansagen.
  return (
    <div className={`hinweis ${art}`} role={art === "fehler" ? "alert" : "status"}>
      <span aria-hidden="true">{SYMBOLE[art]} </span>
      {children}
    </div>
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
}: {
  label: string;
  wert: string;
  aendern: (wert: string) => void;
  hilfe?: string;
  platzhalter?: string;
  pflicht?: boolean;
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
        type="text"
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
        let zeichen = "";
        if (schritt.fertig) {
          klasse += " fertig";
          zeichen = "✓ ";
        } else if (!laufendGesetzt) {
          klasse += " laufend";
          zeichen = "→ ";
          laufendGesetzt = true;
        }
        return (
          <li key={schritt.titel} className={klasse}>
            <div className="schritt-kopf">Schritt {index + 1}</div>
            <div className="schritt-text">{schritt.titel}</div>
            <div className="schritt-status">
              {zeichen}
              {schritt.status}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

// ---------------------------------------------------------------------------
// QUELLENANZEIGE
// ---------------------------------------------------------------------------
export function Quellen({ quellen }: { quellen: Quelle[] }) {
  if (quellen.length === 0) return null;
  const eigene = quellen.filter((q) => q.herkunft === "nutzer");
  const fachwissen = quellen.filter((q) => q.herkunft === "fachwissen");

  const gruppe = (titel: string, liste: Quelle[]) =>
    liste.length > 0 && (
      <div key={titel}>
        <h3>{titel}</h3>
        {liste.map((q) => (
          <div className="quelle" key={`${q.herkunft}-${q.nummer}`}>
            <div className="quelle-art">
              {q.art}
              {q.ueberschrift ? ` · ${q.ueberschrift}` : ""}
            </div>
            <div className="quelle-kopf">
              <span className="ziffer" aria-hidden="true">
                {q.nummer}
              </span>
              <span className="nur-vorlesen">Beleg {q.nummer}: </span>
              {q.quelle}
            </div>
            <div className="quelle-text">{q.ausschnitt}</div>
          </div>
        ))}
      </div>
    );

  return (
    <details>
      <summary>📚 Verwendete Quellen ({quellen.length}) – anzeigen</summary>
      <p>
        Die hochgestellten Ziffern in der Antwort verweisen auf diese Abschnitte. Angezeigt
        wird jeweils die Textstelle, die tatsächlich verwendet wurde.
      </p>
      {gruppe("Aus Ihren eigenen Unterlagen", eigene)}
      {gruppe("Aus dem geprüften Fachwissen", fachwissen)}
    </details>
  );
}
