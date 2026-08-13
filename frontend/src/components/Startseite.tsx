/** Einstieg: neue Sitzung beginnen oder mit Zugangscode fortsetzen. */
import { useState } from "react";
import { ApiFehler, sitzungAnlegen, sitzungLaden } from "../api";
import { Hinweis } from "./Bausteine";

const SCHRITTE = [
  ["1", "Unterlagen hochladen", "Bescheid, Gutachten, Pflegetagebuch und Arztberichte als PDF."],
  ["2", "Prüfen lassen", "Der Assistent vergleicht das Gutachten mit Ihren Unterlagen."],
  ["3", "Widerspruch erstellen", "Fertiges Schreiben zum Ausdrucken und Unterschreiben."],
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
          "Dieser Zugangscode ist abgelaufen. Aus Datenschutzgründen wurden alle " +
            "zugehörigen Daten bereits vollständig gelöscht.",
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
    <main className="huelle">
      <h1>⚖️ Pflegehilfe Online</h1>
      <p style={{ fontSize: "1.1rem" }}>
        <strong>Unterstützung beim Widerspruch gegen einen Pflegegradbescheid.</strong> Dieser
        Assistent prüft Ihre Pflegeunterlagen und hilft Ihnen, einen begründeten Widerspruch zu
        verfassen. Eine Anmeldung ist nicht nötig – wir fragen weder Ihren Namen noch Ihre
        E-Mail-Adresse ab.
      </p>

      <h2>So läuft es ab</h2>
      <div className="raster">
        {SCHRITTE.map(([nummer, titel, text]) => (
          <div className="karte" key={nummer}>
            <div className="quelle-art">Schritt {nummer}</div>
            <h3>{titel}</h3>
            <p>{text}</p>
          </div>
        ))}
      </div>

      {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

      <div className="raster" style={{ marginTop: "1.5rem" }}>
        <section>
          <h2>🆕 Neu anfangen</h2>
          <p>
            Sie starten mit einer leeren Sitzung. Danach erhalten Sie einen persönlichen
            Zugangscode, mit dem Sie später weiterarbeiten können.
          </p>
          <button
            className="knopf haupt"
            onClick={neuAnfangen}
            disabled={laeuft !== null}
            style={{ width: "100%" }}
          >
            {laeuft === "neu" ? "Sitzung wird angelegt …" : "Neue Sitzung starten"}
          </button>
        </section>

        <section>
          <h2>🔑 Mit Zugangscode fortsetzen</h2>
          <p>
            Sie haben schon einen Zugangscode? Dann geht es genau dort weiter, wo Sie aufgehört
            haben – mit Unterlagen, Gesprächsverlauf und Schreiben.
          </p>
          <form onSubmit={fortsetzen}>
            <div className="feld">
              <label htmlFor="zugangscode">Ihr Zugangscode</label>
              <input
                id="zugangscode"
                type="text"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                autoComplete="off"
                spellCheck={false}
              />
            </div>
            <button className="knopf" type="submit" disabled={laeuft !== null} style={{ width: "100%" }}>
              {laeuft === "weiter" ? "Arbeitsstand wird geladen …" : "Weiterarbeiten"}
            </button>
          </form>
        </section>
      </div>

      <h2 style={{ marginTop: "2rem" }}>🔒 Ihre Daten bleiben auf diesem Rechner</h2>
      <div className="raster">
        <ul>
          <li>Verarbeitung <strong>ausschließlich auf diesem Rechner</strong></li>
          <li><strong>Keine Übertragung</strong> an Unternehmen im Internet</li>
          <li>Gespeicherte Daten sind <strong>verschlüsselt</strong></li>
        </ul>
        <ul>
          <li>Nach <strong>4 Wochen</strong> wird alles <strong>vollständig gelöscht</strong></li>
          <li>Verlängerung um 3 Tage jederzeit möglich</li>
          <li>Sofortige Löschung auf Knopfdruck</li>
        </ul>
      </div>

      <Hinweis art="warnung">
        <strong>Wichtiger Hinweis:</strong> Dieser Assistent ersetzt keine Rechtsberatung. Prüfen
        Sie alle erstellten Texte vor dem Absenden sorgfältig selbst.
      </Hinweis>
    </main>
  );
}
