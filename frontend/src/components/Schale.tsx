/**
 * Die Schale um alle Arbeitsbereiche: Seitenleiste, Kopfleiste, Fußzeile.
 *
 * Nach Don Norman:
 *  Sichtbarkeit  - die Seitenleiste zeigt jederzeit alle Bereiche und wo man
 *                  gerade ist; die Kopfleiste den Zustand der Sitzung.
 *  Mapping       - die Reihenfolge der Einträge ist die Reihenfolge der Arbeit.
 *  Signifier     - Zähler an den Einträgen sagen, was dort schon vorliegt.
 *  Rückmeldung   - der aktive Eintrag ist farblich und über aria-current gesetzt.
 */
import type { ReactNode } from "react";
import { Plakette } from "./Bausteine";
import { Symbol, Wortmarke, type SymbolName } from "./Symbole";

export type Bereich =
  | "uebersicht"
  | "unterlagen"
  | "assistent"
  | "quellen"
  | "brief"
  | "einstellungen";

interface Eintrag {
  schluessel: Bereich;
  beschriftung: string;
  symbol: SymbolName;
  gruppe: string;
}

const EINTRAEGE: Eintrag[] = [
  { schluessel: "uebersicht", beschriftung: "Übersicht", symbol: "uebersicht", gruppe: "Start" },
  { schluessel: "unterlagen", beschriftung: "Unterlagen", symbol: "unterlagen", gruppe: "Ihr Fall" },
  { schluessel: "assistent", beschriftung: "KI-Assistent", symbol: "assistent", gruppe: "Ihr Fall" },
  { schluessel: "quellen", beschriftung: "Belege & Quellen", symbol: "quellen", gruppe: "Ihr Fall" },
  { schluessel: "brief", beschriftung: "Widerspruch", symbol: "brief", gruppe: "Ergebnis" },
  { schluessel: "einstellungen", beschriftung: "Einstellungen", symbol: "einstellungen", gruppe: "Konto" },
];

export const BEREICH_TITEL: Record<Bereich, { titel: string; unterzeile: string }> = {
  uebersicht: {
    titel: "Übersicht",
    unterzeile: "Der Stand Ihres Falls auf einen Blick",
  },
  unterlagen: {
    titel: "Unterlagen",
    unterzeile: "Bescheid, Gutachten und weitere Nachweise einlesen",
  },
  assistent: {
    titel: "KI-Assistent",
    unterzeile: "Ihre Unterlagen gegen das Fachwissen prüfen lassen",
  },
  quellen: {
    titel: "Belege & Quellen",
    unterzeile: "Woher jede Aussage der Auswertung stammt",
  },
  brief: {
    titel: "Widerspruch",
    unterzeile: "Das fertige Schreiben zum Ausdrucken und Unterschreiben",
  },
  einstellungen: {
    titel: "Einstellungen",
    unterzeile: "Zugangscode, Gültigkeit, Darstellung und Löschung",
  },
};

export function Schale({
  bereich,
  wechseln,
  anzahlUnterlagen,
  anzahlQuellen,
  tageGueltig,
  kopfWerkzeuge,
  children,
}: {
  bereich: Bereich;
  wechseln: (b: Bereich) => void;
  anzahlUnterlagen: number;
  anzahlQuellen: number;
  tageGueltig: number | null;
  kopfWerkzeuge?: ReactNode;
  children: ReactNode;
}) {
  const zaehler: Partial<Record<Bereich, number>> = {
    unterlagen: anzahlUnterlagen,
    quellen: anzahlQuellen,
  };

  let letzteGruppe = "";
  const kopf = BEREICH_TITEL[bereich];

  return (
    <div className="schale">
      <a className="sprungmarke" href="#hauptbereich">
        Direkt zum Inhalt springen
      </a>

      <nav className="seitenleiste" aria-label="Bereiche">
        <div className="marke">
          <Wortmarke />
          <div>
            <div className="marke-name">Pflegehilfe Online</div>
            <div className="marke-zusatz">Unterstützung beim Pflegegrad-Widerspruch</div>
          </div>
        </div>

        <ul className="navigation">
          {EINTRAEGE.map((eintrag) => {
            const neueGruppe = eintrag.gruppe !== letzteGruppe;
            letzteGruppe = eintrag.gruppe;
            const zahl = zaehler[eintrag.schluessel];
            return (
              <li key={eintrag.schluessel}>
                {neueGruppe && <div className="nav-gruppe">{eintrag.gruppe}</div>}
                <button
                  type="button"
                  aria-current={bereich === eintrag.schluessel ? "page" : undefined}
                  onClick={() => wechseln(eintrag.schluessel)}
                >
                  <Symbol name={eintrag.symbol} />
                  {eintrag.beschriftung}
                  {zahl !== undefined && zahl > 0 && (
                    <span className="nav-zahl">
                      {zahl}
                      <span className="nur-vorlesen"> Einträge</span>
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ul>

        <div className="leiste-fuss">
          <div className="schutzhinweis">
            <div className="schutzhinweis-kopf">
              <Symbol name="schild" groesse={18} />
              Ihre Daten bleiben hier
            </div>
            <p>
              Unterlagen, Auswertung und Sprachmodell laufen ausschließlich auf diesem Rechner.
              Es geht nichts an Anbieter im Internet.
            </p>
          </div>
        </div>
      </nav>

      <div className="arbeitsflaeche">
        <header className="kopfleiste">
          <div>
            <div className="kopfleiste-titel">{kopf.titel}</div>
            <div className="kopfleiste-unterzeile">{kopf.unterzeile}</div>
          </div>
          <div className="kopfleiste-rechts">
            <Plakette art="gruen" punkt>
              Sitzung aktiv
            </Plakette>
            {tageGueltig !== null && (
              <Plakette art={tageGueltig <= 3 ? "gelb" : undefined}>
                Noch {tageGueltig} Tage gültig
              </Plakette>
            )}
            {kopfWerkzeuge}
          </div>
        </header>

        <main className="inhalt" id="hauptbereich">
          {children}
        </main>

        <footer className="fusszeile">
          <strong>Rechtlicher Hinweis:</strong> Dieser Assistent ersetzt keine Rechtsberatung.
          Alle erstellten Texte sind Entwürfe und müssen vor dem Absenden selbst geprüft werden.
          Die Verarbeitung findet ausschließlich örtlich statt.
        </footer>
      </div>
    </div>
  );
}
