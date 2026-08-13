/** Reiter 2: Gespräch mit dem Assistenten. */
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
import { Hinweis, Quellen } from "./Bausteine";

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
    aktionenLaden().then(setAktionen).catch(() => setAktionen([]));
  }, []);

  // Immer die neueste Nachricht zeigen.
  useEffect(() => {
    if (verlaufFeld.current) {
      verlaufFeld.current.scrollTop = verlaufFeld.current.scrollHeight;
    }
  }, [verlauf, teilantwort]);

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
    <section aria-labelledby="chat-titel">
      <h2 id="chat-titel">Schritt 2 – Mit dem Assistenten prüfen</h2>

      {!hatUnterlagen && (
        <Hinweis art="warnung">
          <strong>Die Aufgaben sind noch gesperrt.</strong> Der Assistent braucht zuerst Ihre
          Unterlagen. Wechseln Sie dafür in den Reiter <strong>„Unterlagen“</strong>.
        </Hinweis>
      )}

      <p>
        <strong>Was möchten Sie tun?</strong>
      </p>
      <div className="aufgaben">
        {aktionen.map((aktion) => (
          <button
            key={aktion.schluessel}
            className="knopf"
            disabled={!hatUnterlagen || laeuft}
            title={hatUnterlagen ? aktion.beschreibung : "Zuerst Unterlagen hochladen."}
            onClick={() => fragen(aktion)}
          >
            {aktion.titel}
          </button>
        ))}
      </div>

      <div className="verlauf" ref={verlaufFeld} aria-live="polite" aria-label="Gesprächsverlauf">
        {verlauf.length === 0 && !teilantwort && (
          <p>
            <strong>Hier erscheint Ihr Gespräch.</strong> Wählen Sie oben eine Aufgabe oder
            stellen Sie unten Ihre eigene Frage.
          </p>
        )}
        {verlauf.map((nachricht, index) => (
          <div key={index} className={`blase ${nachricht.role === "user" ? "nutzer" : ""}`}>
            <div className="blase-kopf">
              {nachricht.role === "user" ? "🧑 Ihre Frage" : "⚖️ Assistent"}
            </div>
            <div style={{ whiteSpace: "pre-wrap" }}>{nachricht.content}</div>
          </div>
        ))}
        {teilantwort && (
          <div className="blase">
            <div className="blase-kopf">⚖️ Assistent</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{teilantwort}</div>
          </div>
        )}
      </div>

      {suchfrage && (
        <p role="status">
          🔎 Gesucht wurde nach: <em>„{suchfrage}“</em>
        </p>
      )}
      {laeuft && status && <p role="status">⏳ {status}</p>}
      {fehler && <Hinweis art="fehler">{fehler}</Hinweis>}

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
          rows={2}
          value={eingabe}
          disabled={!hatUnterlagen || laeuft}
          placeholder={hatUnterlagen ? "Ihre Frage eingeben …" : "Zuerst Unterlagen hochladen"}
          onChange={(e) => setEingabe(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              fragen(undefined, eingabe);
            }
          }}
        />
        <button className="knopf haupt" type="submit" disabled={!eingabe.trim() || laeuft}>
          Senden
        </button>
      </form>

      <Quellen quellen={quellen} />

      {verlauf.length > 0 && (
        <button
          className="knopf"
          style={{ marginTop: "1rem" }}
          onClick={() => {
            setVerlauf([]);
            setQuellen([]);
          }}
        >
          🗑️ Gespräch löschen
        </button>
      )}
    </section>
  );
}
