/** Gespräch mit dem Assistenten. */
import { useEffect, useRef, useState } from "react";
import {
  ApiFehler,
  aktionenLaden,
  chatStellen,
  sitzungLaden,
  type Aktion,
  type Nachricht,
  type Perspektive,
  type Quelle,
} from "../api";
import { Hinweis, Karte, Quellenliste } from "./Bausteine";
import { Symbol } from "./Symbole";

export function ChatTab({
  token,
  hatUnterlagen,
  verlauf,
  setVerlauf,
  quellen,
  setQuellen,
  perspektive,
  versicherteName,
  verhaeltnis,
  entwurfGesetzt,
  zuUnterlagen,
}: {
  token: string;
  hatUnterlagen: boolean;
  verlauf: Nachricht[];
  setVerlauf: (n: Nachricht[]) => void;
  quellen: Quelle[];
  setQuellen: (q: Quelle[]) => void;
  perspektive: Perspektive;
  versicherteName: string;
  verhaeltnis: string;
  entwurfGesetzt: (text: string) => void;
  zuUnterlagen: () => void;
}) {
  const [aktionen, setAktionen] = useState<Aktion[]>([]);
  const [eingabe, setEingabe] = useState("");
  const [laeuft, setLaeuft] = useState(false);
  const [status, setStatus] = useState("");
  const [suchfrage, setSuchfrage] = useState("");
  const [teilantwort, setTeilantwort] = useState("");
  const [fehler, setFehler] = useState("");
  const verlaufFeld = useRef<HTMLDivElement>(null);

  useEffect(() => {
    aktionenLaden()
      .then(setAktionen)
      .catch(() => setAktionen([]));
  }, []);

  // Immer die neueste Nachricht zeigen.
  useEffect(() => {
    if (verlaufFeld.current) {
      verlaufFeld.current.scrollTop = verlaufFeld.current.scrollHeight;
    }
  }, [verlauf, teilantwort, status]);

  async function fragen(aktion?: Aktion, freieFrage?: string) {
    const anzeige = aktion ? aktion.nutzertext : (freieFrage ?? "").trim();
    if (!anzeige || laeuft) return;

    setLaeuft(true);
    setFehler("");
    setSuchfrage("");
    setTeilantwort("");
    setStatus("Die Anfrage wird vorbereitet …");
    const bisher = [...verlauf, { role: "user" as const, content: anzeige }];
    setVerlauf(bisher);
    setEingabe("");

    let gesammelt = "";
    let ergebnisErhalten = false;
    try {
      await chatStellen(
        token,
        {
          aktion: aktion?.schluessel,
          frage: aktion ? undefined : anzeige,
          perspektive,
          versicherte_name: versicherteName,
          verhaeltnis,
        },
        (meldung) => {
          if (meldung.art === "status") setStatus(meldung.text ?? "");
          else if (meldung.art === "fehler") {
            // Der Dienst meldet einen Abbruch. Ohne diesen Zweig bliebe die
            // Anzeige im Wartezustand stehen.
            ergebnisErhalten = true;
            setFehler(meldung.text ?? "Die Antwort konnte nicht erzeugt werden.");
            setVerlauf(verlauf);
            setTeilantwort("");
          } else if (meldung.art === "suchfrage") setSuchfrage(meldung.text ?? "");
          else if (meldung.art === "text") {
            gesammelt += meldung.text ?? "";
            setTeilantwort(gesammelt);
          } else if (meldung.art === "ergebnis") {
            ergebnisErhalten = true;
            setVerlauf([...bisher, { role: "assistant", content: meldung.antwort ?? "" }]);
            setQuellen(meldung.quellen ?? []);
            if (aktion?.schluessel === "schreiben") {
              // Die briefreife Fassung liegt im Dienst; sie enthält keine
              // Anrede und keine Belegziffern mehr.
              sitzungLaden(token)
                .then((s) => entwurfGesetzt((s.data?.letter_draft as string) ?? meldung.antwort ?? ""))
                .catch(() => entwurfGesetzt(meldung.antwort ?? ""));
            }
            setTeilantwort("");
          }
        },
      );

      // Bricht der Strom ab, ohne ein Ergebnis zu liefern, bliebe die Anzeige
      // sonst stumm im Wartezustand stehen.
      if (!ergebnisErhalten) {
        setFehler(
          "Die Antwort wurde unterwegs abgebrochen. Bitte versuchen Sie es noch einmal – " +
            "falls es erneut auftritt, prüfen Sie das Fenster des Hintergrunddienstes.",
        );
        setVerlauf(verlauf);
        setTeilantwort("");
      }
    } catch (e) {
      setFehler(
        e instanceof ApiFehler && e.status === 0
          ? "Der Hintergrunddienst ist nicht erreichbar."
          : "Der Assistent ist gerade nicht erreichbar. Läuft LM Studio mit geladenem Modell?",
      );
      setVerlauf(verlauf);
      setTeilantwort("");
    } finally {
      setLaeuft(false);
      setStatus("");
    }
  }

  return (
    <div className="spalten zwei-schmal">
      <div className="stapel">
        {!hatUnterlagen && (
          <Hinweis art="warnung">
            <strong>Der Assistent braucht zuerst Ihre Unterlagen.</strong> Ohne Bescheid und
            Gutachten kann er nichts prüfen.{" "}
            <button
              className="knopf klein"
              style={{ marginLeft: "var(--a2)" }}
              onClick={zuUnterlagen}
            >
              Unterlagen hochladen
            </button>
          </Hinweis>
        )}

        <Karte titel="Was möchten Sie tun?" symbol="assistent">
          <div className="aufgaben">
            {aktionen.map((aktion) => (
              <button
                key={aktion.schluessel}
                className="aufgabe"
                disabled={!hatUnterlagen || laeuft}
                onClick={() => fragen(aktion)}
              >
                <span className="aufgabe-titel">{aktion.titel}</span>
                <span className="aufgabe-text">{aktion.beschreibung}</span>
              </button>
            ))}
          </div>
        </Karte>

        <Karte titel="Gespräch" symbol="assistent" flach
          werkzeuge={
            verlauf.length > 0 && (
              <button
                className="knopf stumm klein"
                onClick={() => {
                  setVerlauf([]);
                  setQuellen([]);
                }}
              >
                <Symbol name="papierkorb" groesse={16} />
                Verlauf löschen
              </button>
            )
          }
        >
          <div className="verlauf" ref={verlaufFeld} aria-live="polite" aria-label="Gesprächsverlauf">
            {verlauf.length === 0 && !teilantwort && !laeuft && (
              <div className="leerer-verlauf">
                <Symbol name="assistent" groesse={40} />
                <div style={{ fontWeight: 620, color: "var(--text)" }}>
                  Hier erscheint Ihr Gespräch
                </div>
                <p style={{ fontSize: "0.9rem", marginTop: "var(--a2)" }}>
                  Wählen Sie oben eine der vorbereiteten Aufgaben – oder stellen Sie unten Ihre
                  eigene Frage in ganz normalen Worten.
                </p>
              </div>
            )}

            {verlauf.map((nachricht, index) => (
              <div
                key={index}
                className={`blase ${nachricht.role === "user" ? "nutzer" : "assistent"}`}
              >
                <div className="blase-kopf">
                  {nachricht.role === "user" ? "Ihre Frage" : "Assistent"}
                </div>
                <div className="blase-koerper">{nachricht.content}</div>
              </div>
            ))}

            {teilantwort && (
              <div className="blase assistent">
                <div className="blase-kopf">Assistent</div>
                <div className="blase-koerper">{teilantwort}</div>
              </div>
            )}

            {laeuft && !teilantwort && (
              <div className="blase assistent">
                <div className="blase-kopf">Assistent</div>
                <div className="blase-koerper" style={{ display: "flex", alignItems: "center", gap: "var(--a3)" }}>
                  <span className="tippen" aria-hidden="true">
                    <span />
                    <span />
                    <span />
                  </span>
                  <span style={{ color: "var(--gedaempft)", fontSize: "0.9rem" }}>
                    {status || "Einen Moment …"}
                  </span>
                </div>
              </div>
            )}
          </div>

          <form
            className="eingabezeile"
            onSubmit={(e) => {
              e.preventDefault();
              fragen(undefined, eingabe);
            }}
          >
            <label htmlFor="chateingabe" className="nur-vorlesen">
              Ihre Frage
            </label>
            <textarea
              id="chateingabe"
              rows={1}
              value={eingabe}
              disabled={!hatUnterlagen || laeuft}
              placeholder={hatUnterlagen ? "Stellen Sie eine Frage …" : "Zuerst Unterlagen hochladen"}
              onChange={(e) => setEingabe(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  fragen(undefined, eingabe);
                }
              }}
            />
            <button
              className="knopf haupt"
              type="submit"
              disabled={!eingabe.trim() || laeuft}
              aria-label="Frage senden"
            >
              <Symbol name="senden" groesse={19} />
            </button>
          </form>
        </Karte>

        {suchfrage && (
          <p className="hilfe" role="status">
            <Symbol name="suche" groesse={15} /> Gesucht wurde nach: <em>„{suchfrage}“</em>
          </p>
        )}
        {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}
      </div>

      <div className="stapel">
        <Karte
          titel={`Belege der letzten Antwort (${quellen.length})`}
          symbol="quellen"
          fuss="Die hochgestellten Ziffern in der Antwort verweisen auf diese Abschnitte."
        >
          {quellen.length > 0 ? (
            <Quellenliste quellen={quellen} />
          ) : (
            <p className="hilfe">
              Sobald der Assistent geantwortet hat, steht hier zu jeder Ziffer die Textstelle, auf
              die sie sich stützt.
            </p>
          )}
        </Karte>

        <Karte titel="So lesen Sie die Antwort" symbol="info">
          <p style={{ fontSize: "0.9rem", color: "var(--gedaempft)" }}>
            Der Assistent belegt jede Aussage mit einer hochgestellten Ziffer. Eine Antwort{" "}
            <strong>ohne</strong> solche Ziffern bekommt eine Warnung vorangestellt – dann stützt
            sie sich auf nichts Nachprüfbares und kann sachlich falsch sein.
          </p>
          <p style={{ fontSize: "0.9rem", color: "var(--gedaempft)" }}>
            Besonders bei Punktzahlen, Fristen und Zuständigkeiten lohnt der Blick in die
            Belegstelle. Die Texte sind Entwürfe, keine Rechtsberatung.
          </p>
        </Karte>
      </div>
    </div>
  );
}
