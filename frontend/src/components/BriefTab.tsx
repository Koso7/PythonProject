/** Angaben ausfüllen und das Widerspruchsschreiben erzeugen. */
import { useEffect, useState } from "react";
import {
  ApiFehler,
  briefErzeugen,
  briefPruefen,
  type Briefangaben,
  type Perspektive,
} from "../api";
import { Feld, Hinweis, Karte, Wahlfeld } from "./Bausteine";
import { Symbol } from "./Symbole";

export function BriefTab({
  token,
  angaben,
  setAngaben,
  entwurf,
  pdfErzeugt,
  zumAssistenten,
}: {
  token: string;
  angaben: Briefangaben;
  setAngaben: (a: Briefangaben) => void;
  entwurf: string;
  pdfErzeugt: () => void;
  zumAssistenten: () => void;
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
  const bereit = fehlend.length === 0;

  return (
    <div className="spalten zwei-schmal">
      <div className="stapel">
        <Karte titel="1 · Wer legt den Widerspruch ein?" symbol="info">
          <fieldset>
            <legend className="nur-vorlesen">Perspektive des Schreibens</legend>
            <Wahlfeld
              name="perspektive"
              gewaehlt={!fuerAndere}
              waehlen={() => aendern("perspektive", "selbst" satisfies Perspektive)}
              titel="Ich bin selbst betroffen"
              text="Das Schreiben wird in der Ich-Form verfasst."
            />
            <Wahlfeld
              name="perspektive"
              gewaehlt={fuerAndere}
              waehlen={() => aendern("perspektive", "angehoerige" satisfies Perspektive)}
              titel="Ich schreibe für eine andere Person"
              text="Über die pflegebedürftige Person wird namentlich geschrieben, unterschrieben wird von Ihnen."
            />
          </fieldset>

          {fuerAndere && (
            <div className="raster" style={{ marginTop: "var(--a4)" }}>
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
        </Karte>

        <Karte titel="2 · Absender und Empfänger" symbol="brief">
          <div className="raster">
            <div>
              <h3 style={{ marginBottom: "var(--a3)" }}>Ihre Anschrift</h3>
              <Feld label="Ihr Vor- und Nachname" wert={angaben.absender_name}
                    aendern={(w) => aendern("absender_name", w)} platzhalter="Sabine Müller" pflicht />
              <Feld label="Straße und Hausnummer" wert={angaben.absender_strasse}
                    aendern={(w) => aendern("absender_strasse", w)} platzhalter="Lindenweg 12" pflicht />
              <Feld label="Postleitzahl und Ort" wert={angaben.absender_plz_ort}
                    aendern={(w) => aendern("absender_plz_ort", w)} platzhalter="30159 Hannover" pflicht />
              <Feld label="Ort für die Datumszeile" wert={angaben.ort}
                    aendern={(w) => aendern("ort", w)}
                    hilfe="Erscheint oben rechts vor dem Datum. Kann leer bleiben." platzhalter="Hannover" />
            </div>
            <div>
              <h3 style={{ marginBottom: "var(--a3)" }}>Die Pflegekasse</h3>
              <Feld label="Name der Pflegekasse" wert={angaben.kasse_name}
                    aendern={(w) => aendern("kasse_name", w)} platzhalter="Muster-Pflegekasse" pflicht />
              <Feld label="Straße und Hausnummer" wert={angaben.kasse_strasse}
                    aendern={(w) => aendern("kasse_strasse", w)} platzhalter="Kassenallee 4" />
              <Feld label="Postleitzahl und Ort" wert={angaben.kasse_plz_ort}
                    aendern={(w) => aendern("kasse_plz_ort", w)} platzhalter="30159 Hannover" />
            </div>
          </div>

          <h3 style={{ margin: "var(--a3) 0 var(--a3)" }}>Angaben zum Bescheid</h3>
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
        </Karte>

        <Karte
          titel="3 · Begründung des Widerspruchs"
          symbol="assistent"
          werkzeuge={
            entwurf && (
              <button className="knopf klein" onClick={() => aendern("begruendung", entwurf)}>
                <Symbol name="herunterladen" groesse={16} />
                Entwurf übernehmen
              </button>
            )
          }
          fuss="Anrede, Betreff und Grußformel ergänzt die Vorlage automatisch. Hier steht nur die Begründung."
        >
          {!entwurf && (
            <Hinweis>
              <strong>Noch kein Entwurf vorhanden.</strong> Lassen Sie den Assistenten zuerst einen
              Vorschlag schreiben – Sie können ihn danach frei bearbeiten.{" "}
              <button className="knopf klein" style={{ marginLeft: "var(--a2)" }} onClick={zumAssistenten}>
                Zum Assistenten
              </button>
            </Hinweis>
          )}

          <div className="feld">
            <label htmlFor="begruendung">Begründung – Sie können den Text frei bearbeiten</label>
            <textarea
              id="begruendung"
              rows={16}
              value={angaben.begruendung}
              onChange={(e) => aendern("begruendung", e.target.value)}
            />
          </div>

          {luecken.length > 0 && (
            <Hinweis art="warnung">
              <strong>Im Text stehen noch Lücken.</strong> Diese Angaben konnte der Assistent nicht
              aus Ihren Unterlagen entnehmen:
              <ul>
                {luecken.slice(0, 8).map((l) => (
                  <li key={l}>
                    <code>{l}</code>
                  </li>
                ))}
              </ul>
              Bitte ersetzen Sie sie, bevor Sie das Schreiben abschicken.
            </Hinweis>
          )}

          <label className="wahlfeld" style={{ marginTop: "var(--a3)" }}>
            <input
              type="checkbox"
              checked={angaben.begruendung_folgt}
              onChange={(e) => aendern("begruendung_folgt", e.target.checked)}
            />
            <span>
              <span className="wahlfeld-titel">Begründung später nachreichen</span>
              <span className="wahlfeld-text">
                Der Widerspruch wird zunächst nur fristwahrend eingelegt. Im Brief steht dann, dass
                die Begründung folgt.
              </span>
            </span>
          </label>

          <div className="feld" style={{ marginTop: "var(--a4)" }}>
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
              Eine Unterlage je Zeile. Die Liste erscheint unter Ihrer Unterschrift. Bleibt das Feld
              leer, entfällt der Abschnitt.
            </div>
          </div>
        </Karte>
      </div>

      {/* --- rechte Spalte: Prüfung und Erzeugung ---------------------------- */}
      <div className="stapel">
        <Karte titel="4 · PDF erstellen" symbol="herunterladen">
          {bereit ? (
            <Hinweis art="erfolg">Alle Pflichtangaben liegen vor.</Hinweis>
          ) : (
            <Hinweis art="warnung">
              <strong>Es fehlen noch Angaben.</strong> Sobald alles ausgefüllt ist, wird der Knopf
              frei:
              <ul>
                {fehlend.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </Hinweis>
          )}

          {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

          <button className="knopf haupt breit" onClick={erzeugen} disabled={!bereit || laeuft}>
            <Symbol name="herunterladen" groesse={18} />
            {laeuft ? "Ihr Schreiben wird erstellt …" : "PDF erstellen und herunterladen"}
          </button>

          {fertig && (
            <div style={{ marginTop: "var(--a4)" }}>
              <Hinweis art="erfolg">
                <strong>Ihr Widerspruchsschreiben wurde heruntergeladen.</strong>
              </Hinweis>
            </div>
          )}
        </Karte>

        <Karte titel="So geht es weiter" symbol="info">
          <ol style={{ margin: 0, paddingLeft: "1.2rem", fontSize: "0.92rem", color: "var(--gedaempft)", lineHeight: 1.7 }}>
            <li>
              <strong style={{ color: "var(--text)" }}>Ausdrucken</strong> und den Text noch einmal
              in Ruhe durchlesen.
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Von Hand unterschreiben</strong> – ohne
              Unterschrift ist der Widerspruch unwirksam.
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Per Post senden</strong>, am besten als
              Einwurfeinschreiben. Auch ein Fax wahrt die Frist.
            </li>
            <li>
              Eine <strong style={{ color: "var(--text)" }}>Kopie behalten</strong> und den
              Sendebeleg aufbewahren.
            </li>
          </ol>
          <Hinweis art="warnung">
            Eine einfache E-Mail wahrt die Widerspruchsfrist <strong>nicht</strong>.
          </Hinweis>
        </Karte>

        <Karte titel="Aufbau des Schreibens" symbol="brief">
          <p style={{ fontSize: "0.9rem", color: "var(--gedaempft)" }}>
            Das PDF folgt der Form eines Geschäftsbriefs nach DIN 5008: Anschriftfeld, Betreffzeile,
            Anrede, Begründung, Grußformel und Unterschriftfeld – mit Falz- und Lochmarken, sodass
            es in einen Fensterumschlag passt.
          </p>
          <p style={{ fontSize: "0.9rem", color: "var(--gedaempft)" }}>
            Inhaltlich orientiert es sich am Musterbrief der Verbraucherzentrale.
          </p>
        </Karte>
      </div>
    </div>
  );
}
