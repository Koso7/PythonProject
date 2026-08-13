/**
 * Übersicht: der Stand des Falls auf einen Blick.
 *
 * Die Seite beantwortet drei Fragen, ohne dass jemand suchen muss:
 * Wo stehe ich? Wie viel Zeit bleibt? Was ist als Nächstes zu tun?
 */
import { fristBerechnen, datumFormatieren } from "../fristen";
import type { Briefangaben, Nachricht, Quelle } from "../api";
import { Fortschritt, Hinweis, Kachel, Karte, Leerbereich, type SchrittZustand } from "./Bausteine";
import { Symbol } from "./Symbole";
import type { Bereich } from "./Schale";

/** Kürzt einen langen Text an einer Wortgrenze. */
function kuerzen(text: string, zeichen: number): string {
  if (text.length <= zeichen) return text;
  const schnitt = text.lastIndexOf(" ", zeichen);
  return `${text.slice(0, schnitt > 0 ? schnitt : zeichen)} …`;
}

export function Uebersicht({
  dokumente,
  verlauf,
  quellen,
  angaben,
  fristZugang,
  setFristZugang,
  schritte,
  wechseln,
}: {
  dokumente: string[];
  verlauf: Nachricht[];
  quellen: Quelle[];
  angaben: Briefangaben;
  fristZugang: string;
  setFristZugang: (wert: string) => void;
  schritte: SchrittZustand[];
  wechseln: (b: Bereich) => void;
}) {
  const auswertungen = verlauf.filter((n) => n.role === "assistant");
  const letzte = auswertungen[auswertungen.length - 1];
  const eigene = quellen.filter((q) => q.herkunft === "nutzer");
  const fachwissen = quellen.filter((q) => q.herkunft === "fachwissen");

  // Zugangsdatum: was die Person angibt, sonst ersatzweise das Bescheiddatum.
  const grundlage = fristZugang.trim() || angaben.bescheid_datum;
  const frist = fristBerechnen(grundlage);
  const ausBescheid = !fristZugang.trim() && Boolean(angaben.bescheid_datum);

  /** Der jeweils nächste sinnvolle Schritt - genau einer, nicht fünf Angebote. */
  const naechster = (() => {
    if (dokumente.length === 0)
      return {
        titel: "Unterlagen hochladen",
        text: "Der Assistent braucht zuerst Ihren Bescheid und möglichst das Gutachten des Medizinischen Dienstes.",
        knopf: "Zu den Unterlagen",
        ziel: "unterlagen" as Bereich,
      };
    if (auswertungen.length === 0)
      return {
        titel: "Unterlagen prüfen lassen",
        text: "Der Assistent vergleicht das Gutachten mit Ihren übrigen Nachweisen und sucht Ansatzpunkte für den Widerspruch.",
        knopf: "Zum Assistenten",
        ziel: "assistent" as Bereich,
      };
    return {
      titel: "Widerspruch erstellen",
      text: "Die Auswertung liegt vor. Jetzt fehlen nur noch Ihre Anschrift und die Angaben zum Bescheid.",
      knopf: "Zum Widerspruch",
      ziel: "brief" as Bereich,
    };
  })();

  return (
    <>
      <div className="kachelreihe">
        <Kachel
          symbol="unterlagen"
          farbe="blau"
          titel="Eingelesene Unterlagen"
          wert={String(dokumente.length)}
          zusatz={dokumente.length === 0 ? "noch keine vorhanden" : "durchsuchbar aufbereitet"}
        />
        <Kachel
          symbol="assistent"
          farbe="tuerkis"
          titel="Auswertungen"
          wert={String(auswertungen.length)}
          zusatz={auswertungen.length === 0 ? "noch keine Anfrage" : "Antworten des Assistenten"}
        />
        <Kachel
          symbol="quellen"
          farbe="gruen"
          titel="Belegte Fundstellen"
          wert={String(quellen.length)}
          zusatz={
            quellen.length === 0
              ? "erscheinen nach der Auswertung"
              : `${eigene.length} aus Ihren Unterlagen, ${fachwissen.length} aus dem Fachwissen`
          }
        />
        <Kachel
          symbol="kalender"
          farbe={
            !frist ? "blau" : frist.stufe === "gruen" ? "gruen" : frist.stufe === "gelb" ? "gelb" : "rot"
          }
          titel="Widerspruchsfrist"
          wert={frist ? datumFormatieren(frist.ende) : "offen"}
          klein
          zusatz={frist ? frist.text : "Zugangsdatum noch nicht angegeben"}
        />
      </div>

      <Fortschritt schritte={schritte} />

      <div className="spalten drei">
        {/* --- links: Frist und nächster Schritt ---------------------------- */}
        <div className="stapel">
          <Karte titel="Fristenrechner" symbol="kalender">
            <div className="feld">
              <label htmlFor="frist-zugang">Wann ist der Bescheid bei Ihnen angekommen?</label>
              <input
                id="frist-zugang"
                type="text"
                inputMode="numeric"
                value={fristZugang}
                placeholder={angaben.bescheid_datum || "TT.MM.JJJJ"}
                onChange={(e) => setFristZugang(e.target.value)}
                aria-describedby="frist-hilfe"
              />
              <div className="hilfe" id="frist-hilfe">
                {ausBescheid
                  ? "Gerechnet wird gerade mit dem Bescheiddatum aus Ihren Angaben. Maßgeblich ist aber der Tag, an dem der Brief bei Ihnen ankam."
                  : "Nicht das Datum auf dem Bescheid, sondern der Tag, an dem der Brief bei Ihnen im Briefkasten lag."}
              </div>
            </div>

            {frist ? (
              <>
                <div className={`frist ${frist.stufe === "abgelaufen" ? "rot" : frist.stufe}`}>
                  <div className="kachel-titel">Widerspruch muss eingegangen sein bis</div>
                  <div className="frist-datum">{datumFormatieren(frist.ende)}</div>
                  <div className="frist-rest">{frist.text}</div>
                </div>
                {frist.stufe === "abgelaufen" ? (
                  <Hinweis art="fehler">
                    <strong>Die Monatsfrist ist rechnerisch abgelaufen.</strong> Ein Widerspruch
                    kann trotzdem noch möglich sein – etwa wenn der Bescheid keine korrekte
                    Rechtsbehelfsbelehrung enthielt oder Sie die Frist unverschuldet versäumt haben.
                    Lassen Sie das bitte prüfen, zum Beispiel bei einem Sozialverband.
                  </Hinweis>
                ) : (
                  <p className="hilfe" style={{ marginTop: "var(--a3)" }}>
                    Ein Monat ab Bekanntgabe (§ 84 SGG). Der Widerspruch muss <strong>eingegangen</strong>{" "}
                    sein, nicht abgeschickt – rechnen Sie den Postweg ein. Diese Berechnung ist eine
                    Orientierung und keine Rechtsauskunft.
                  </p>
                )}
              </>
            ) : (
              <Leerbereich
                symbol="kalender"
                titel="Noch keine Frist berechnet"
                text="Tragen Sie oben das Datum ein, an dem der Bescheid bei Ihnen ankam. Format: TT.MM.JJJJ"
              />
            )}
          </Karte>

          <Karte titel="Das ist jetzt dran" symbol="pfeil">
            <h3 style={{ marginBottom: "var(--a2)" }}>{naechster.titel}</h3>
            <p style={{ color: "var(--gedaempft)", fontSize: "0.92rem" }}>{naechster.text}</p>
            <button
              className="knopf haupt breit"
              style={{ marginTop: "var(--a3)" }}
              onClick={() => wechseln(naechster.ziel)}
            >
              {naechster.knopf}
              <Symbol name="pfeil" groesse={18} />
            </button>
          </Karte>
        </div>

        {/* --- Mitte: die letzte Auswertung --------------------------------- */}
        <Karte
          titel="Letzte Auswertung"
          symbol="assistent"
          werkzeuge={
            auswertungen.length > 0 && (
              <button className="knopf klein" onClick={() => wechseln("assistent")}>
                Gespräch öffnen
              </button>
            )
          }
          fuss={
            auswertungen.length > 0
              ? "Die Auswertung ist ein Entwurf. Prüfen Sie jede Aussage anhand der Belegstellen."
              : undefined
          }
        >
          {letzte ? (
            <div style={{ whiteSpace: "pre-wrap", fontSize: "0.94rem", lineHeight: 1.65 }}>
              {kuerzen(letzte.content, 1400)}
            </div>
          ) : (
            <Leerbereich
              symbol="assistent"
              titel="Noch keine Auswertung"
              text={
                dokumente.length === 0
                  ? "Laden Sie zuerst Ihre Unterlagen hoch. Danach kann der Assistent Ihren Fall prüfen."
                  : "Wechseln Sie zum Assistenten und wählen Sie dort eine der vorbereiteten Aufgaben."
              }
            />
          )}
        </Karte>

        {/* --- rechts: Belege ----------------------------------------------- */}
        <div className="stapel">
          <Karte titel="Aus Ihren Unterlagen" symbol="datei">
            {eigene.length > 0 ? (
              <ul className="dateiliste">
                {eigene.slice(0, 4).map((q) => (
                  <li className="dateizeile" key={q.nummer}>
                    <span className="ziffer nutzer" aria-hidden="true">
                      {q.nummer}
                    </span>
                    <span style={{ minWidth: 0 }}>
                      <span className="dateiname">{q.quelle}</span>
                      <span className="quelle-art">{q.art}</span>
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <Leerbereich
                symbol="datei"
                titel="Noch keine Fundstellen"
                text="Hier erscheinen die Stellen aus Ihren eigenen Unterlagen, auf die sich die Auswertung stützt."
              />
            )}
            {eigene.length > 4 && (
              <button
                className="knopf stumm klein"
                style={{ marginTop: "var(--a2)" }}
                onClick={() => wechseln("quellen")}
              >
                Alle {eigene.length} Stellen anzeigen
              </button>
            )}
          </Karte>

          <Karte titel="Verwendete Fachquellen" symbol="quellen">
            {fachwissen.length > 0 ? (
              <ul className="dateiliste">
                {fachwissen.slice(0, 4).map((q) => (
                  <li className="dateizeile" key={q.nummer}>
                    <span className="ziffer" aria-hidden="true">
                      {q.nummer}
                    </span>
                    <span style={{ minWidth: 0 }}>
                      <span className="dateiname">{q.quelle}</span>
                      <span className="quelle-art">{q.art}</span>
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <Leerbereich
                symbol="quellen"
                titel="Noch keine Fachquellen"
                text="Der Assistent belegt seine Aussagen mit Begutachtungs-Richtlinien, SGB XI und geprüften Ratgebern."
              />
            )}
            {fachwissen.length > 4 && (
              <button
                className="knopf stumm klein"
                style={{ marginTop: "var(--a2)" }}
                onClick={() => wechseln("quellen")}
              >
                Alle {fachwissen.length} Quellen anzeigen
              </button>
            )}
          </Karte>
        </div>
      </div>
    </>
  );
}
