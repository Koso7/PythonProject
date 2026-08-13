/** Reiter 1: Unterlagen hochladen und verwalten. */
import { useRef, useState } from "react";
import { ApiFehler, unterlageEntfernen, unterlagenHochladen, type UploadErgebnis } from "../api";
import { Hinweis } from "./Bausteine";

const MAX_MB = 30;

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
  const dateifeld = useRef<HTMLInputElement>(null);

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
    <section aria-labelledby="unterlagen-titel">
      <h2 id="unterlagen-titel">Schritt 1 – Ihre Unterlagen hochladen</h2>
      <p>
        Laden Sie alle Unterlagen hoch, die für den Widerspruch wichtig sind. Sie können{" "}
        <strong>mehrere Dateien gleichzeitig</strong> auswählen.
      </p>

      <Hinweis>
        <strong>Besonders hilfreich:</strong> der Pflegegradbescheid, das Gutachten des
        Medizinischen Dienstes, Ihr Pflegetagebuch sowie Arzt- und Krankenhausberichte.
      </Hinweis>

      <details>
        <summary>🔒 Datenschutzhinweis – bitte einmal lesen</summary>
        <p>
          <strong>Was mit Ihren Unterlagen passiert:</strong> Ihre Dateien werden ausschließlich
          auf diesem Rechner gelesen und ausgewertet. Sie werden <strong>nicht</strong> an ein
          Unternehmen im Internet übertragen und <strong>nicht</strong> zum Trainieren von
          künstlicher Intelligenz verwendet. Auch das Sprachmodell läuft örtlich.
        </p>
        <p>
          <strong>Wie gespeichert wird:</strong> Ihr Arbeitsstand bleibt nur erhalten, solange Sie
          Ihren Zugangscode haben. Alles Gespeicherte ist verschlüsselt; ohne den Code kann
          niemand darauf zugreifen.
        </p>
        <p>
          <strong>Wie gelöscht wird:</strong> Spätestens 4 Wochen nach Beginn wird die Sitzung
          vollständig gelöscht. Im Reiter <strong>Einstellungen</strong> können Sie jederzeit
          sofort selbst löschen.
        </p>
      </details>

      <p>
        <strong>Zulässig:</strong> PDF-Dateien, höchstens {MAX_MB} MB je Datei.
      </p>

      <div className="feld">
        <label htmlFor="dateiwahl">Dateien auswählen</label>
        <input
          id="dateiwahl"
          ref={dateifeld}
          type="file"
          accept="application/pdf"
          multiple
          onChange={(e) => setAuswahl(Array.from(e.target.files ?? []))}
        />
      </div>

      {auswahl.length > 0 && (
        <Hinweis art="erfolg">
          <strong>{auswahl.length} Datei(en) ausgewählt.</strong> Klicken Sie jetzt auf
          „Unterlagen einlesen“.
        </Hinweis>
      )}

      <button className="knopf haupt" onClick={einlesen} disabled={auswahl.length === 0 || laeuft}>
        {laeuft ? "Die Unterlagen werden gelesen …" : "📥 Unterlagen einlesen"}
      </button>

      {laeuft && (
        <p role="status" style={{ marginTop: "0.8rem" }}>
          Eingescannte Unterlagen brauchen Texterkennung, das kann eine Minute je Datei dauern.
        </p>
      )}

      {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

      {erfolgreiche.length > 0 && (
        <Hinweis art="erfolg">
          <strong>{erfolgreiche.length} Dokument(e) eingelesen</strong>, daraus{" "}
          {erfolgreiche.reduce((summe, e) => summe + e.abschnitte, 0)} durchsuchbare
          Textabschnitte. Weiter geht es im Reiter „KI-Assistent“.
        </Hinweis>
      )}
      {misslungene.map((e) => (
        <Hinweis art="warnung" key={e.dateiname}>
          „{e.dateiname}“: {e.hinweis}
        </Hinweis>
      ))}

      <h3 style={{ marginTop: "1.5rem" }}>Eingelesene Unterlagen</h3>
      {dokumente.length === 0 ? (
        <Hinweis>
          Noch keine Unterlagen vorhanden. Ohne Unterlagen kann der Assistent Ihren Fall nicht
          prüfen.
        </Hinweis>
      ) : (
        <ul style={{ listStyle: "none", padding: 0 }}>
          {dokumente.map((name) => (
            <li
              key={name}
              className="karte"
              style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem" }}
            >
              <span>📄 {name}</span>
              <button className="knopf gefahr" onClick={() => entfernen(name)}>
                Entfernen
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
