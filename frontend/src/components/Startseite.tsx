/**
 * Einstieg: neue Sitzung beginnen oder mit Zugangscode fortsetzen.
 *
 * Zweigeteilt: links die Handlung, rechts die Erklärung. Wer das erste Mal
 * hier landet, soll in wenigen Sekunden erfassen, worum es geht, was mit den
 * Unterlagen geschieht und was als Erstes zu tun ist.
 */
import { useState } from "react";
import { ApiFehler, sitzungAnlegen, sitzungLaden } from "../api";
import { Hinweis } from "./Bausteine";
import { Symbol, Wortmarke } from "./Symbole";

const ABLAUF = [
  [
    "Unterlagen hochladen",
    "Bescheid, Gutachten des Medizinischen Dienstes, Pflegetagebuch und Arztberichte als PDF.",
  ],
  [
    "Prüfen lassen",
    "Der Assistent vergleicht das Gutachten mit Ihren übrigen Nachweisen und sucht Ansatzpunkte – jede Aussage mit Belegstelle.",
  ],
  [
    "Widerspruch erstellen",
    "Ein fertiges Schreiben im Geschäftsbriefformat, zum Ausdrucken und Unterschreiben.",
  ],
];

const MERKMALE = [
  "Keine Anmeldung – wir fragen weder Namen noch E-Mail-Adresse",
  "Verarbeitung ausschließlich auf diesem Rechner",
  "Auch das Sprachmodell läuft örtlich – nichts geht ins Internet",
  "Gespeichertes ist verschlüsselt und wird nach vier Wochen vollständig gelöscht",
  "Sofortige Löschung jederzeit auf Knopfdruck",
];

export function Startseite({
  anmelden,
}: {
  anmelden: (token: string, ablauf: string) => void;
}) {
  const [code, setCode] = useState("");
  const [fehler, setFehler] = useState("");
  const [laeuft, setLaeuft] = useState<"neu" | "weiter" | null>(null);

  async function neuAnfangen() {
    setFehler("");
    setLaeuft("neu");
    try {
      const sitzung = await sitzungAnlegen();
      anmelden(sitzung.token, sitzung.expires_at);
    } catch (e) {
      setFehler(e instanceof ApiFehler ? e.message : "Unbekannter Fehler.");
    } finally {
      setLaeuft(null);
    }
  }

  async function fortsetzen(ereignis: React.FormEvent) {
    ereignis.preventDefault();
    setFehler("");
    if (!code.trim()) {
      setFehler("Bitte geben Sie zuerst Ihren Zugangscode ein.");
      return;
    }
    setLaeuft("weiter");
    try {
      const sitzung = await sitzungLaden(code.trim());
      anmelden(sitzung.token, sitzung.expires_at);
    } catch (e) {
      if (e instanceof ApiFehler && e.status === 410) {
        setFehler(
          "Dieser Zugangscode ist abgelaufen. Aus Datenschutzgründen wurden alle zugehörigen " +
            "Daten bereits vollständig gelöscht.",
        );
      } else if (e instanceof ApiFehler && e.status === 404) {
        setFehler("Dieser Zugangscode ist unbekannt. Bitte prüfen Sie ihn auf Tippfehler.");
      } else {
        setFehler(e instanceof ApiFehler ? e.message : "Unbekannter Fehler.");
      }
    } finally {
      setLaeuft(null);
    }
  }

  return (
    <div className="start">
      <main className="start-inhalt">
        <div className="start-marke">
          <Wortmarke groesse={44} />
          <div>
            <div style={{ fontSize: "1.1rem", fontWeight: 680, letterSpacing: "-0.01em" }}>
              Pflegehilfe Online
            </div>
            <div style={{ fontSize: "0.82rem", color: "var(--gedaempft)" }}>
              Unterstützung beim Pflegegrad-Widerspruch
            </div>
          </div>
        </div>

        <h1>Wurde Ihr Pflegegrad zu niedrig eingestuft?</h1>
        <p className="start-vorspann">
          Dieser Assistent prüft Ihre Pflegeunterlagen gegen die amtlichen Begutachtungs-Richtlinien
          und hilft Ihnen, einen begründeten Widerspruch zu verfassen. Ohne Anmeldung, ohne dass
          Ihre Unterlagen diesen Rechner verlassen.
        </p>

        {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

        <button
          className="knopf haupt breit"
          onClick={neuAnfangen}
          disabled={laeuft !== null}
          style={{ minHeight: "3.2rem", fontSize: "1rem" }}
        >
          {laeuft === "neu" ? "Sitzung wird angelegt …" : "Kostenlos beginnen"}
          {laeuft !== "neu" && <Symbol name="pfeil" groesse={19} />}
        </button>
        <p className="hilfe" style={{ textAlign: "center", marginTop: "var(--a2)" }}>
          Sie erhalten einen Zugangscode, mit dem Sie jederzeit weiterarbeiten können.
        </p>

        <div className="trenner">oder</div>

        <form onSubmit={fortsetzen}>
          <div className="feld">
            <label htmlFor="zugangscode">Sie haben schon einen Zugangscode?</label>
            <input
              id="zugangscode"
              type="text"
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Code hier einfügen"
              autoComplete="off"
              spellCheck={false}
            />
            <div className="hilfe">
              Damit geht es genau dort weiter, wo Sie aufgehört haben – mit Unterlagen,
              Gesprächsverlauf und Schreiben.
            </div>
          </div>
          <button className="knopf breit" type="submit" disabled={laeuft !== null}>
            {laeuft === "weiter" ? "Arbeitsstand wird geladen …" : "Weiterarbeiten"}
          </button>
        </form>

        <div style={{ marginTop: "var(--a6)" }}>
          <Hinweis art="warnung">
            <strong>Dieser Assistent ersetzt keine Rechtsberatung.</strong> Alle erstellten Texte
            sind Entwürfe. Prüfen Sie sie vor dem Absenden sorgfältig – im Zweifel mit einem
            Sozialverband wie VdK oder SoVD.
          </Hinweis>
        </div>
      </main>

      <aside className="start-bild">
        <div>
          <h2 style={{ marginBottom: "var(--a5)" }}>So läuft es ab</h2>
          <ol className="ablaufliste">
            {ABLAUF.map(([titel, text], index) => (
              <li className="ablaufschritt" key={titel}>
                <span className="ablaufschritt-nummer" aria-hidden="true">
                  {index + 1}
                </span>
                <div>
                  <h3>{titel}</h3>
                  <p>{text}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>

        <div
          style={{
            borderTop: "1px solid rgba(255,255,255,0.16)",
            paddingTop: "var(--a5)",
          }}
        >
          <h2 style={{ marginBottom: "var(--a3)", display: "flex", alignItems: "center", gap: "var(--a2)" }}>
            <Symbol name="schild" groesse={22} />
            Ihre Daten bleiben hier
          </h2>
          <ul className="merkmale">
            {MERKMALE.map((merkmal) => (
              <li className="merkmal" key={merkmal}>
                <Symbol name="haken" groesse={17} />
                {merkmal}
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  );
}
