/** Reiter 3: Angaben ausfüllen und das Widerspruchsschreiben erzeugen. */
import { useEffect, useState } from "react";
import {
  ApiFehler,
  briefErzeugen,
  briefPruefen,
  type Briefangaben,
  type Perspektive,
} from "../api";
import { Feld, Hinweis } from "./Bausteine";

export function BriefTab({
  token,
  angaben,
  setAngaben,
  entwurf,
  pdfErzeugt,
}: {
  token: string;
  angaben: Briefangaben;
  setAngaben: (a: Briefangaben) => void;
  entwurf: string;
  pdfErzeugt: () => void;
}) {
  const [fehlend, setFehlend] = useState<string[]>([]);
  const [luecken, setLuecken] = useState<string[]>([]);
  const [laeuft, setLaeuft] = useState(false);
  const [fehler, setFehler] = useState("");
  const [fertig, setFertig] = useState(false);

  const aendern = (feld: keyof Briefangaben, wert: string | boolean) =>
    setAngaben({ ...angaben, [feld]: wert });

  // Die Prüfung läuft im Dienst, damit Oberfläche und Brief dieselben Regeln
  // verwenden. Verzögert, damit nicht jeder Tastendruck eine Anfrage auslöst.
  useEffect(() => {
    const zeitgeber = setTimeout(() => {
      briefPruefen(token, angaben)
        .then((p) => {
          setFehlend(p.fehlende_angaben);
          setLuecken(p.offene_platzhalter);
        })
        .catch(() => undefined);
    }, 400);
    return () => clearTimeout(zeitgeber);
  }, [token, angaben]);

  async function erzeugen() {
    setLaeuft(true);
    setFehler("");
    setFertig(false);
    try {
      const blob = await briefErzeugen(token, angaben);
      const url = URL.createObjectURL(blob);
      const verweis = document.createElement("a");
      verweis.href = url;
      verweis.download = "Widerspruch_Pflegegrad.pdf";
      verweis.click();
      URL.revokeObjectURL(url);
      setFertig(true);
      pdfErzeugt();
    } catch (e) {
      setFehler(e instanceof ApiFehler ? e.message : "Das PDF konnte nicht erstellt werden.");
    } finally {
      setLaeuft(false);
    }
  }

  const fuerAndere = angaben.perspektive === "angehoerige";

  return (
    <section aria-labelledby="brief-titel">
      <h2 id="brief-titel">Schritt 3 – Widerspruch als PDF erstellen</h2>
      <p>
        Hier entsteht Ihr fertiges Widerspruchsschreiben zum Ausdrucken und Unterschreiben. Der
        Aufbau folgt dem Musterbrief der Verbraucherzentrale.
      </p>

      <h3>1. Wer legt den Widerspruch ein?</h3>
      <fieldset style={{ border: "1px solid var(--rand)", borderRadius: 8, padding: "1rem", marginBottom: "1rem" }}>
        <legend>Perspektive des Schreibens</legend>
        <label style={{ fontWeight: 400 }}>
          <input
            type="radio"
            name="perspektive"
            checked={!fuerAndere}
            onChange={() => aendern("perspektive", "selbst" satisfies Perspektive)}
          />{" "}
          Ich bin selbst betroffen – das Schreiben wird in der Ich-Form verfasst.
        </label>
        <label style={{ fontWeight: 400, marginTop: "0.5rem" }}>
          <input
            type="radio"
            name="perspektive"
            checked={fuerAndere}
            onChange={() => aendern("perspektive", "angehoerige" satisfies Perspektive)}
          />{" "}
          Ich schreibe für eine andere Person – über sie wird namentlich geschrieben.
        </label>
      </fieldset>

      {fuerAndere && (
        <div className="raster">
          <Feld
            label="Name der pflegebedürftigen Person"
            wert={angaben.versichert_name}
            aendern={(w) => aendern("versichert_name", w)}
            platzhalter="Elfriede Müller"
            pflicht
          />
          <Feld
            label="Verhältnis zu Ihnen"
            wert={angaben.verhaeltnis}
            aendern={(w) => aendern("verhaeltnis", w)}
            hilfe="Zum Beispiel Mutter, Vater, Ehefrau. Erscheint im Brief als „meine Mutter, …“."
            platzhalter="Mutter"
          />
        </div>
      )}

      <h3>2. Ihre Angaben</h3>
      <div className="raster">
        <div>
          <p><strong>Absender – Ihre Anschrift</strong></p>
          <Feld label="Ihr Vor- und Nachname" wert={angaben.absender_name}
                aendern={(w) => aendern("absender_name", w)} platzhalter="Sabine Müller" pflicht />
          <Feld label="Ihre Straße und Hausnummer" wert={angaben.absender_strasse}
                aendern={(w) => aendern("absender_strasse", w)} platzhalter="Lindenweg 12" pflicht />
          <Feld label="Ihre Postleitzahl und Ihr Ort" wert={angaben.absender_plz_ort}
                aendern={(w) => aendern("absender_plz_ort", w)} platzhalter="30159 Hannover" pflicht />
          <Feld label="Ort für die Datumszeile" wert={angaben.ort}
                aendern={(w) => aendern("ort", w)}
                hilfe="Erscheint oben rechts vor dem Datum. Kann leer bleiben." platzhalter="Hannover" />
        </div>
        <div>
          <p><strong>Empfänger – die Pflegekasse</strong></p>
          <Feld label="Name der Pflegekasse" wert={angaben.kasse_name}
                aendern={(w) => aendern("kasse_name", w)} platzhalter="Muster-Pflegekasse" pflicht />
          <Feld label="Straße und Hausnummer der Pflegekasse" wert={angaben.kasse_strasse}
                aendern={(w) => aendern("kasse_strasse", w)} />
          <Feld label="Postleitzahl und Ort der Pflegekasse" wert={angaben.kasse_plz_ort}
                aendern={(w) => aendern("kasse_plz_ort", w)} />
        </div>
      </div>

      <p><strong>Angaben zum Bescheid</strong></p>
      <div className="raster">
        <Feld label="Datum des Bescheids" wert={angaben.bescheid_datum}
              aendern={(w) => aendern("bescheid_datum", w)}
              hilfe="Das Datum oben auf Ihrem Bescheid." platzhalter="12.03.2025" pflicht />
        <Feld label="Aktenzeichen" wert={angaben.aktenzeichen}
              aendern={(w) => aendern("aktenzeichen", w)}
              hilfe="Steht meist oben auf dem Bescheid. Kann leer bleiben." />
        <Feld label="Versichertennummer" wert={angaben.versichert_nr}
              aendern={(w) => aendern("versichert_nr", w)} />
      </div>

      <h3>3. Begründung des Widerspruchs</h3>
      {entwurf ? (
        <button className="knopf" onClick={() => aendern("begruendung", entwurf)}>
          ⬇️ Entwurf aus dem Gespräch übernehmen
        </button>
      ) : (
        <Hinweis>
          Noch kein Entwurf vorhanden. Nutzen Sie im Reiter <strong>„KI-Assistent“</strong> die
          Aufgabe <strong>„Widerspruch schreiben“</strong>.
        </Hinweis>
      )}

      <div className="feld" style={{ marginTop: "1rem" }}>
        <label htmlFor="begruendung">Begründung (Sie können den Text frei bearbeiten)</label>
        <textarea
          id="begruendung"
          rows={14}
          value={angaben.begruendung}
          onChange={(e) => aendern("begruendung", e.target.value)}
          aria-describedby="begruendung-hilfe"
        />
        <div className="hilfe" id="begruendung-hilfe">
          Anrede, Betreff und Grußformel ergänzt die Vorlage automatisch. Schreiben Sie hier nur
          die Begründung.
        </div>
      </div>

      {luecken.length > 0 && (
        <Hinweis art="warnung">
          <strong>Im Text stehen noch Lücken.</strong> Diese Angaben konnte der Assistent nicht
          aus Ihren Unterlagen entnehmen:
          <ul>
            {luecken.slice(0, 8).map((l) => (
              <li key={l}><code>{l}</code></li>
            ))}
          </ul>
          Bitte ersetzen Sie sie, bevor Sie das Schreiben abschicken.
        </Hinweis>
      )}

      <label style={{ fontWeight: 400 }}>
        <input
          type="checkbox"
          checked={angaben.begruendung_folgt}
          onChange={(e) => aendern("begruendung_folgt", e.target.checked)}
        />{" "}
        Begründung später nachreichen (Widerspruch zunächst nur fristwahrend einlegen)
      </label>

      <div className="feld" style={{ marginTop: "1.5rem" }}>
        <label htmlFor="anlagen">Anlagen (freiwillig)</label>
        <textarea
          id="anlagen"
          rows={3}
          value={angaben.anlagen ?? ""}
          onChange={(e) => aendern("anlagen", e.target.value)}
          placeholder={"Kopie des Bescheids vom 12.03.2025\nPflegetagebuch für zwei Wochen\nÄrztliches Attest"}
          aria-describedby="anlagen-hilfe"
        />
        <div className="hilfe" id="anlagen-hilfe">
          Eine Unterlage je Zeile. Diese Liste erscheint unter Ihrer Unterschrift. Bleibt das
          Feld leer, entfällt der Abschnitt.
        </div>
      </div>

      <h3 style={{ marginTop: "1.5rem" }}>4. PDF erstellen</h3>
      {fehlend.length > 0 ? (
        <Hinweis art="warnung">
          <strong>Es fehlen noch Angaben.</strong> Sobald alles ausgefüllt ist, wird der Knopf frei:
          <ul>
            {fehlend.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </Hinweis>
      ) : (
        <Hinweis art="erfolg">Alle Pflichtangaben liegen vor.</Hinweis>
      )}

      {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

      <button className="knopf haupt" onClick={erzeugen} disabled={fehlend.length > 0 || laeuft}>
        {laeuft ? "Ihr Schreiben wird erstellt …" : "📄 PDF erstellen und herunterladen"}
      </button>

      {fertig && (
        <Hinweis art="erfolg">
          <strong>Ihr Widerspruchsschreiben wurde heruntergeladen.</strong> Drucken Sie es aus und{" "}
          <strong>unterschreiben Sie es von Hand</strong>. Versenden Sie es per Post – am besten
          als Einwurfeinschreiben – oder per Fax. Eine E-Mail wahrt die Frist nicht.
        </Hinweis>
      )}
    </section>
  );
}
