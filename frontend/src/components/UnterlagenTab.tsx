/** Unterlagen hochladen und verwalten. */
import { useRef, useState } from "react";
import { ApiFehler, unterlageEntfernen, unterlagenHochladen, type UploadErgebnis } from "../api";
import { Hinweis, Karte, Leerbereich } from "./Bausteine";
import { Symbol } from "./Symbole";

const MAX_MB = 30;

const EMPFEHLUNGEN = [
  ["Pflegegradbescheid", "Die Entscheidung der Pflegekasse, gegen die sich der Widerspruch richtet."],
  ["Gutachten des Medizinischen Dienstes", "Das wichtigste Dokument – daraus stammen die Einzelpunkte je Modul."],
  ["Pflegetagebuch", "Belegt den tatsächlichen Hilfebedarf über mehrere Tage."],
  ["Arzt- und Krankenhausberichte", "Diagnosen und Befunde, die im Gutachten fehlen könnten."],
];

export function UnterlagenTab({
  token,
  dokumente,
  setDokumente,
}: {
  token: string;
  dokumente: string[];
  setDokumente: (namen: string[]) => void;
}) {
  const [auswahl, setAuswahl] = useState<File[]>([]);
  const [laeuft, setLaeuft] = useState(false);
  const [ergebnisse, setErgebnisse] = useState<UploadErgebnis[]>([]);
  const [fehler, setFehler] = useState("");
  const [ueberAblage, setUeberAblage] = useState(false);
  const dateifeld = useRef<HTMLInputElement>(null);

  function auswaehlen(dateien: FileList | null) {
    setAuswahl(Array.from(dateien ?? []).filter((d) => d.type === "application/pdf"));
  }

  async function einlesen() {
    if (auswahl.length === 0) return;
    setLaeuft(true);
    setFehler("");
    setErgebnisse([]);
    try {
      const antwort = await unterlagenHochladen(token, auswahl);
      setErgebnisse(antwort.ergebnisse);
      setDokumente(antwort.dokumente);
      setAuswahl([]);
      if (dateifeld.current) dateifeld.current.value = "";
    } catch (e) {
      setFehler(e instanceof ApiFehler ? e.message : "Das Einlesen ist fehlgeschlagen.");
    } finally {
      setLaeuft(false);
    }
  }

  async function entfernen(name: string) {
    try {
      const antwort = await unterlageEntfernen(token, name);
      setDokumente(antwort.dokumente);
    } catch (e) {
      setFehler(e instanceof ApiFehler ? e.message : "Das Entfernen ist fehlgeschlagen.");
    }
  }

  const erfolgreiche = ergebnisse.filter((e) => e.erfolgreich);
  const misslungene = ergebnisse.filter((e) => !e.erfolgreich);

  return (
    <div className="spalten zwei-schmal">
      <div className="stapel">
        <Karte
          titel="Unterlagen hinzufügen"
          symbol="hochladen"
          fuss={`Zulässig sind PDF-Dateien bis ${MAX_MB} MB je Datei. Mehrere Dateien lassen sich gleichzeitig auswählen.`}
        >
          {/* Die Fläche ist zugleich Ablage und Auslöser für die Dateiauswahl.
              Das Eingabefeld bleibt bedienbar - nur unsichtbar -, damit
              Tastatur und Vorleseprogramme unverändert funktionieren. */}
          <div
            className={`ablage${ueberAblage || auswahl.length > 0 ? " bereit" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setUeberAblage(true);
            }}
            onDragLeave={() => setUeberAblage(false)}
            onDrop={(e) => {
              e.preventDefault();
              setUeberAblage(false);
              auswaehlen(e.dataTransfer.files);
            }}
          >
            <Symbol name="hochladen" groesse={34} />
            <div className="ablage-titel">Dateien hierher ziehen</div>
            <div className="ablage-text">oder unten auswählen – nur PDF</div>

            <div className="feld" style={{ marginTop: "var(--a4)", marginBottom: 0 }}>
              <label htmlFor="dateiwahl" className="nur-vorlesen">
                Dateien auswählen
              </label>
              <input
                id="dateiwahl"
                ref={dateifeld}
                type="file"
                accept="application/pdf"
                multiple
                onChange={(e) => auswaehlen(e.target.files)}
              />
            </div>
          </div>

          {auswahl.length > 0 && (
            <div style={{ marginTop: "var(--a4)" }}>
              <Hinweis art="erfolg">
                <strong>{auswahl.length} Datei(en) ausgewählt.</strong> Klicken Sie jetzt auf
                „Unterlagen einlesen“.
              </Hinweis>
            </div>
          )}

          <button
            className="knopf haupt breit"
            style={{ marginTop: "var(--a4)" }}
            onClick={einlesen}
            disabled={auswahl.length === 0 || laeuft}
          >
            <Symbol name="hochladen" groesse={18} />
            {laeuft ? "Die Unterlagen werden gelesen …" : "Unterlagen einlesen"}
          </button>

          {laeuft && (
            <p className="hilfe" role="status" style={{ marginTop: "var(--a3)" }}>
              Eingescannte Unterlagen brauchen Texterkennung. Das kann eine Minute je Datei dauern –
              bitte lassen Sie das Fenster offen.
            </p>
          )}

          {fehler && (
            <div style={{ marginTop: "var(--a4)" }}>
              <Hinweis art="fehler">{fehler}</Hinweis>
            </div>
          )}
          {erfolgreiche.length > 0 && (
            <div style={{ marginTop: "var(--a4)" }}>
              <Hinweis art="erfolg">
                <strong>{erfolgreiche.length} Dokument(e) eingelesen</strong> – daraus{" "}
                {erfolgreiche.reduce((summe, e) => summe + e.abschnitte, 0)} durchsuchbare
                Textabschnitte. Weiter geht es beim <strong>KI-Assistenten</strong>.
              </Hinweis>
            </div>
          )}
          {misslungene.map((e) => (
            <div style={{ marginTop: "var(--a3)" }} key={e.dateiname}>
              <Hinweis art="warnung">
                <strong>„{e.dateiname}“:</strong> {e.hinweis}
              </Hinweis>
            </div>
          ))}
        </Karte>

        <Karte titel={`Eingelesene Unterlagen (${dokumente.length})`} symbol="unterlagen">
          {dokumente.length === 0 ? (
            <Leerbereich
              symbol="unterlagen"
              titel="Noch keine Unterlagen"
              text="Ohne Unterlagen kann der Assistent Ihren Fall nicht prüfen. Am wichtigsten sind der Bescheid und das Gutachten des Medizinischen Dienstes."
            />
          ) : (
            <ul className="dateiliste">
              {dokumente.map((name) => (
                <li className="dateizeile" key={name}>
                  <Symbol name="datei" />
                  <span className="dateiname">{name}</span>
                  <button
                    className="knopf klein gefahr"
                    onClick={() => entfernen(name)}
                    aria-label={`${name} entfernen`}
                  >
                    <Symbol name="papierkorb" groesse={16} />
                    Entfernen
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Karte>
      </div>

      <div className="stapel">
        <Karte titel="Was hilft dem Assistenten?" symbol="info">
          <ul className="dateiliste">
            {EMPFEHLUNGEN.map(([titel, text]) => (
              <li className="dateizeile" key={titel} style={{ alignItems: "flex-start" }}>
                <span style={{ color: "var(--erfolg-700)", marginTop: "0.1rem" }}>
                  <Symbol name="haken" groesse={18} />
                </span>
                <span style={{ minWidth: 0 }}>
                  <span className="dateiname">{titel}</span>
                  <span className="hilfe" style={{ marginTop: 0 }}>
                    {text}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </Karte>

        <Karte titel="Was mit Ihren Unterlagen geschieht" symbol="schild">
          <p style={{ fontSize: "0.92rem", color: "var(--gedaempft)" }}>
            <strong style={{ color: "var(--text)" }}>Verarbeitung:</strong> Ihre Dateien werden
            ausschließlich auf diesem Rechner gelesen und ausgewertet. Sie werden nicht an ein
            Unternehmen im Internet übertragen und nicht zum Trainieren künstlicher Intelligenz
            verwendet. Auch das Sprachmodell läuft örtlich.
          </p>
          <p style={{ fontSize: "0.92rem", color: "var(--gedaempft)" }}>
            <strong style={{ color: "var(--text)" }}>Speicherung:</strong> Der Arbeitsstand bleibt
            nur erhalten, solange Sie Ihren Zugangscode haben. Alles Gespeicherte ist verschlüsselt.
            Die Suchdaten Ihrer Unterlagen liegen ausschließlich im Arbeitsspeicher.
          </p>
          <p style={{ fontSize: "0.92rem", color: "var(--gedaempft)" }}>
            <strong style={{ color: "var(--text)" }}>Löschung:</strong> Spätestens vier Wochen nach
            Beginn wird die Sitzung vollständig gelöscht. Unter <strong>Einstellungen</strong>{" "}
            können Sie jederzeit sofort selbst löschen.
          </p>
        </Karte>
      </div>
    </div>
  );
}
