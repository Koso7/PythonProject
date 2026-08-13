/** Zugangscode, Gültigkeit, Darstellung und Löschung. */
import { useEffect, useState } from "react";
import { ApiFehler, sitzungLoeschen, sitzungVerlaengern, zustandLaden, type Zustand } from "../api";
import { Hinweis, Karte } from "./Bausteine";
import { Symbol } from "./Symbole";

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
    zustandLaden()
      .then(setZustand)
      .catch(() => setZustand(null));
  }, []);

  const ablaufDatum = ablauf ? new Date(ablauf) : null;
  const verbleibendMs = ablaufDatum ? ablaufDatum.getTime() - Date.now() : 0;
  const tage = Math.max(Math.floor(verbleibendMs / 86_400_000), 0);
  const stunden = Math.max(Math.floor((verbleibendMs % 86_400_000) / 3_600_000), 0);
  const anteil = Math.min((tage / SITZUNGSTAGE) * 100, 100);

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
    <div className="spalten zwei">
      <div className="stapel">
        <Karte
          titel="Ihr Zugangscode"
          symbol="schluessel"
          fuss="Ohne diesen Code kommt niemand an Ihre Daten – auch wir nicht. Er lässt sich nicht wiederherstellen."
        >
          <p style={{ fontSize: "0.92rem", color: "var(--gedaempft)" }}>
            Mit diesem Code arbeiten Sie später weiter. <strong>Schreiben Sie ihn sich auf</strong>{" "}
            oder legen Sie ihn in Ihrem Passwortspeicher ab.
          </p>
          <p className="zugangscode">{token}</p>
          <button
            className="knopf"
            onClick={() => {
              navigator.clipboard?.writeText(token);
              setMeldung("Der Zugangscode wurde in die Zwischenablage kopiert.");
            }}
          >
            <Symbol name="kopieren" groesse={17} />
            Code kopieren
          </button>
        </Karte>

        <Karte titel="Gültigkeit der Sitzung" symbol="uhr">
          <div style={{ display: "flex", alignItems: "baseline", gap: "var(--a2)", marginBottom: "var(--a3)" }}>
            <span style={{ fontSize: "1.6rem", fontWeight: 690, letterSpacing: "-0.02em" }}>
              {tage} Tage
            </span>
            <span style={{ color: "var(--gedaempft)", fontSize: "0.9rem" }}>
              und {stunden} Stunden verbleibend
            </span>
          </div>
          <div
            className={`fortschrittsbalken${tage <= 3 ? " rot" : tage <= 7 ? " gelb" : ""}`}
            role="img"
            aria-label={`Noch ${tage} von ${SITZUNGSTAGE} Tagen gültig`}
          >
            <div style={{ width: `${anteil}%` }} />
          </div>
          {ablaufDatum && (
            <p className="hilfe">
              Automatische, vollständige Löschung am {ablaufDatum.toLocaleDateString("de-DE")} um{" "}
              {ablaufDatum.toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })} Uhr.
            </p>
          )}
          <button className="knopf haupt" style={{ marginTop: "var(--a3)" }} onClick={verlaengern}>
            <Symbol name="plus" groesse={17} />
            Um 3 Tage verlängern
          </button>

          {meldung && (
            <div style={{ marginTop: "var(--a4)" }}>
              <Hinweis art="erfolg">{meldung}</Hinweis>
            </div>
          )}
          {fehler && (
            <div style={{ marginTop: "var(--a4)" }}>
              <Hinweis art="fehler">{fehler}</Hinweis>
            </div>
          )}
        </Karte>

        <Karte titel="Darstellung" symbol="einstellungen">
          <div className="feld">
            <label htmlFor="schriftgroesse">Schriftgröße</label>
            <select
              id="schriftgroesse"
              value={schriftgroesse}
              onChange={(e) => setSchriftgroesse(Number(e.target.value))}
            >
              <option value={18}>Normal (18 px)</option>
              <option value={21}>Groß (21 px)</option>
              <option value={24}>Sehr groß (24 px)</option>
            </select>
            <div className="hilfe">Vergrößert die gesamte Oberfläche, nicht nur den Fließtext.</div>
          </div>

          <label className="wahlfeld">
            <input
              id="kontrast"
              type="checkbox"
              checked={hoherKontrast}
              onChange={(e) => setHoherKontrast(e.target.checked)}
            />
            <span>
              <span className="wahlfeld-titel">Hoher Kontrast</span>
              <span className="wahlfeld-text">
                Reines Schwarz auf Weiß, kräftigere Umrandungen, keine Farbflächen.
              </span>
            </span>
          </label>
        </Karte>
      </div>

      <div className="stapel">
        <Karte titel="Technische Angaben" symbol="info"
               fuss="Alle Verarbeitung findet ausschließlich auf diesem Rechner statt.">
          {zustand ? (
            <ul className="dateiliste">
              {[
                ["Wissensdatenbank", `${zustand.wissensbasis_abschnitte} Abschnitte`],
                ["Vektordatenbank", zustand.vektordatenbank],
                ["Neubewertung der Treffer", zustand.neubewertung],
                ["Sprachmodell", zustand.sprachmodell],
              ].map(([bezeichnung, wert]) => (
                <li className="dateizeile" key={bezeichnung}>
                  <span style={{ minWidth: 0, flex: 1 }}>
                    <span className="dateiname">{bezeichnung}</span>
                  </span>
                  <span style={{ fontSize: "0.85rem", color: "var(--gedaempft)", textAlign: "right" }}>
                    {wert}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="hilfe">Der Betriebszustand ist gerade nicht abrufbar.</p>
          )}
        </Karte>

        <Karte titel="Sitzung verlassen" symbol="abmelden">
          <p style={{ fontSize: "0.92rem", color: "var(--gedaempft)" }}>
            Der Bildschirm wird geleert, <strong>Ihre Daten bleiben gespeichert</strong>. Mit Ihrem
            Zugangscode kommen Sie jederzeit zurück – auch von einem anderen Rechner mit dieser
            Anwendung.
          </p>
          <button className="knopf" onClick={abmelden}>
            <Symbol name="abmelden" groesse={17} />
            Abmelden
          </button>
        </Karte>

        <Karte titel="Alles löschen und beenden" symbol="papierkorb">
          <Hinweis art="warnung">
            Gelöscht werden <strong>sofort und unwiderruflich</strong>: Zugangscode, die Inhalte
            Ihrer Unterlagen, der Gesprächsverlauf und das Schreiben. Ein Wiedereinstieg ist danach
            nicht mehr möglich.
          </Hinweis>

          <label className="wahlfeld">
            <input
              type="checkbox"
              checked={bestaetigt}
              onChange={(e) => setBestaetigt(e.target.checked)}
            />
            <span>
              <span className="wahlfeld-titel">Ja, ich möchte wirklich alles endgültig löschen.</span>
            </span>
          </label>

          <button
            className="knopf gefahr breit"
            style={{ marginTop: "var(--a2)" }}
            disabled={!bestaetigt}
            onClick={allesLoeschen}
          >
            <Symbol name="papierkorb" groesse={17} />
            Jetzt alles löschen und beenden
          </button>
        </Karte>
      </div>
    </div>
  );
}
