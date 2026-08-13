/** Reiter 4: Zugangscode, Gültigkeit, Darstellung und Löschung. */
import { useEffect, useState } from "react";
import { ApiFehler, sitzungLoeschen, sitzungVerlaengern, zustandLaden, type Zustand } from "../api";
import { Hinweis } from "./Bausteine";

const SITZUNGSTAGE = 28;

export function EinstellungenTab({
  token,
  ablauf,
  setAblauf,
  schriftgroesse,
  setSchriftgroesse,
  hoherKontrast,
  setHoherKontrast,
  abmelden,
  geloescht,
}: {
  token: string;
  ablauf: string;
  setAblauf: (a: string) => void;
  schriftgroesse: number;
  setSchriftgroesse: (g: number) => void;
  hoherKontrast: boolean;
  setHoherKontrast: (k: boolean) => void;
  abmelden: () => void;
  geloescht: () => void;
}) {
  const [zustand, setZustand] = useState<Zustand | null>(null);
  const [bestaetigt, setBestaetigt] = useState(false);
  const [meldung, setMeldung] = useState("");
  const [fehler, setFehler] = useState("");

  useEffect(() => {
    zustandLaden().then(setZustand).catch(() => setZustand(null));
  }, []);

  const ablaufDatum = ablauf ? new Date(ablauf) : null;
  const verbleibendMs = ablaufDatum ? ablaufDatum.getTime() - Date.now() : 0;
  const tage = Math.max(Math.floor(verbleibendMs / 86_400_000), 0);
  const stunden = Math.max(Math.floor((verbleibendMs % 86_400_000) / 3_600_000), 0);

  async function verlaengern() {
    setFehler("");
    try {
      const antwort = await sitzungVerlaengern(token);
      setAblauf(antwort.expires_at);
      setMeldung("Ihre Sitzung wurde um 3 Tage verlängert.");
    } catch (e) {
      setFehler(e instanceof ApiFehler ? e.message : "Die Verlängerung hat nicht geklappt.");
    }
  }

  async function allesLoeschen() {
    setFehler("");
    try {
      await sitzungLoeschen(token);
      geloescht();
    } catch (e) {
      setFehler(e instanceof ApiFehler ? e.message : "Das Löschen ist fehlgeschlagen.");
    }
  }

  return (
    <section aria-labelledby="einstellungen-titel">
      <h2 id="einstellungen-titel">Einstellungen</h2>

      <h3>🔑 Ihr Zugangscode</h3>
      <p>
        Mit diesem Code arbeiten Sie später weiter. <strong>Schreiben Sie ihn sich auf.</strong>{" "}
        Er lässt sich nicht wiederherstellen.
      </p>
      <p className="zugangscode">{token}</p>
      <button
        className="knopf"
        onClick={() => {
          navigator.clipboard?.writeText(token);
          setMeldung("Der Zugangscode wurde in die Zwischenablage kopiert.");
        }}
      >
        📋 Code kopieren
      </button>

      <h3 style={{ marginTop: "1.5rem" }}>
        ⏳ Noch {tage} Tage und {stunden} Stunden gültig
      </h3>
      <div className="fortschrittsbalken" role="img"
           aria-label={`Noch ${tage} von ${SITZUNGSTAGE} Tagen gültig`}>
        <div style={{ width: `${Math.min((tage / SITZUNGSTAGE) * 100, 100)}%` }} />
      </div>
      {ablaufDatum && (
        <p className="hilfe">
          Automatische, vollständige Löschung am{" "}
          {ablaufDatum.toLocaleDateString("de-DE")} um{" "}
          {ablaufDatum.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })} Uhr.
        </p>
      )}
      <button className="knopf haupt" onClick={verlaengern}>
        ➕ Um 3 Tage verlängern
      </button>

      {meldung && <Hinweis art="erfolg">{meldung}</Hinweis>}
      {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

      <h3 style={{ marginTop: "1.5rem" }}>👁️ Darstellung</h3>
      <div className="raster">
        <div className="feld">
          <label htmlFor="schriftgroesse">Schriftgröße</label>
          <select
            id="schriftgroesse"
            value={schriftgroesse}
            onChange={(e) => setSchriftgroesse(Number(e.target.value))}
          >
            <option value={18}>Normal</option>
            <option value={21}>Groß</option>
            <option value={24}>Sehr groß</option>
          </select>
        </div>
        <div className="feld">
          <label htmlFor="kontrast" style={{ display: "inline" }}>
            <input
              id="kontrast"
              type="checkbox"
              checked={hoherKontrast}
              onChange={(e) => setHoherKontrast(e.target.checked)}
            />{" "}
            Hoher Kontrast
          </label>
          <div className="hilfe">Verstärkt Schwarz-Weiß-Kontraste und Umrandungen.</div>
        </div>
      </div>

      <h3 style={{ marginTop: "1.5rem" }}>⚙️ Technische Angaben</h3>
      {zustand ? (
        <ul>
          <li>Wissensdatenbank: <strong>{zustand.wissensbasis_abschnitte} Abschnitte</strong></li>
          <li>Vektordatenbank: {zustand.vektordatenbank}</li>
          <li>Neubewertung der Treffer: {zustand.neubewertung}</li>
          <li>Sprachmodell: {zustand.sprachmodell}</li>
        </ul>
      ) : (
        <p>Der Betriebszustand ist gerade nicht abrufbar.</p>
      )}
      <p className="hilfe">Alle Verarbeitung findet ausschließlich auf diesem Rechner statt.</p>

      <h3 style={{ marginTop: "1.5rem" }}>🚪 Sitzung verlassen</h3>
      <p>
        Der Bildschirm wird geleert, Ihre Daten bleiben gespeichert. Mit Ihrem Zugangscode kommen
        Sie jederzeit zurück.
      </p>
      <button className="knopf" onClick={abmelden}>
        Abmelden
      </button>

      <h3 style={{ marginTop: "1.5rem" }}>🗑️ Alles löschen und beenden</h3>
      <p>
        Löscht <strong>sofort und unwiderruflich</strong>: Zugangscode, Inhalte Ihrer Unterlagen,
        Gesprächsverlauf und Schreiben. Ein Wiedereinstieg ist danach nicht mehr möglich.
      </p>
      <label style={{ fontWeight: 400 }}>
        <input
          type="checkbox"
          checked={bestaetigt}
          onChange={(e) => setBestaetigt(e.target.checked)}
        />{" "}
        Ja, ich möchte wirklich alles endgültig löschen.
      </label>
      <button
        className="knopf gefahr"
        style={{ marginTop: "0.6rem" }}
        disabled={!bestaetigt}
        onClick={allesLoeschen}
      >
        Jetzt alles löschen und beenden
      </button>
    </section>
  );
}
