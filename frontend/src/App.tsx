/**
 * Pflegehilfe Online – Oberfläche.
 *
 * Gestaltungsleitlinien (nach Don Norman):
 *  Sichtbarkeit  – Kopfzeile und Fortschrittsanzeige zeigen jederzeit den Stand.
 *  Rückmeldung   – jede Aktion meldet sich sofort zurück.
 *  Constraints   – was noch nicht möglich ist, bleibt gesperrt und erklärt sich.
 *  Mapping       – die Reihenfolge der Reiter entspricht der Reihenfolge der Arbeit.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  sitzungLaden,
  sitzungSpeichern,
  type Briefangaben,
  type Nachricht,
  type Perspektive,
  type Quelle,
} from "./api";
import { Fortschritt, Hinweis } from "./components/Bausteine";
import { BriefTab } from "./components/BriefTab";
import { ChatTab } from "./components/ChatTab";
import { EinstellungenTab } from "./components/EinstellungenTab";
import { Startseite } from "./components/Startseite";
import { UnterlagenTab } from "./components/UnterlagenTab";

const SPEICHER_TOKEN = "pflegehilfe.token";

const LEERE_ANGABEN: Briefangaben = {
  absender_name: "", absender_strasse: "", absender_plz_ort: "", ort: "",
  kasse_name: "", kasse_strasse: "", kasse_plz_ort: "",
  versichert_name: "", versichert_nr: "", aktenzeichen: "", bescheid_datum: "",
  begruendung: "", begruendung_folgt: false, perspektive: "selbst", verhaeltnis: "",
  anlagen: "",
};

type Reiter = "unterlagen" | "chat" | "brief" | "einstellungen";

const REITER: { schluessel: Reiter; beschriftung: string }[] = [
  { schluessel: "unterlagen", beschriftung: "📁 Unterlagen" },
  { schluessel: "chat", beschriftung: "💬 KI-Assistent" },
  { schluessel: "brief", beschriftung: "📄 PDF erstellen" },
  { schluessel: "einstellungen", beschriftung: "⚙️ Einstellungen" },
];

export default function App() {
  const [token, setToken] = useState<string | null>(null);
  const [ablauf, setAblauf] = useState("");
  const [reiter, setReiter] = useState<Reiter>("unterlagen");

  const [dokumente, setDokumente] = useState<string[]>([]);
  const [verlauf, setVerlauf] = useState<Nachricht[]>([]);
  const [quellen, setQuellen] = useState<Quelle[]>([]);
  const [entwurf, setEntwurf] = useState("");
  const [angaben, setAngaben] = useState<Briefangaben>(LEERE_ANGABEN);
  const [pdfFertig, setPdfFertig] = useState(false);

  const [schriftgroesse, setSchriftgroesse] = useState(18);
  const [hoherKontrast, setHoherKontrast] = useState(false);
  const [geloescht, setGeloescht] = useState(false);
  const [ladefehler, setLadefehler] = useState("");

  // Darstellungswünsche auf das Dokument anwenden.
  useEffect(() => {
    document.documentElement.style.setProperty("--grund", `${schriftgroesse}px`);
    document.documentElement.dataset.kontrast = hoherKontrast ? "hoch" : "normal";
  }, [schriftgroesse, hoherKontrast]);

  const standUebernehmen = useCallback((daten: Record<string, unknown>) => {
    setDokumente((daten.document_names as string[]) ?? []);
    setVerlauf((daten.messages as Nachricht[]) ?? []);
    setQuellen((daten.last_sources as Quelle[]) ?? []);
    // Der Dienst legt eine briefreife Fassung ab; sie ist bereits von
    // Anrede, Grußformel und Belegziffern befreit.
    setEntwurf(
      (daten.letter_draft as string) ?? (daten.last_generated_letter as string) ?? "",
    );
    const brief = daten.brief as Partial<Briefangaben> | undefined;
    if (brief) setAngaben({ ...LEERE_ANGABEN, ...brief });
    const darstellung = daten.darstellung as { schrift?: number; kontrast?: boolean } | undefined;
    if (darstellung?.schrift) setSchriftgroesse(darstellung.schrift);
    if (darstellung?.kontrast !== undefined) setHoherKontrast(darstellung.kontrast);
  }, []);

  // Bei laufender Sitzung nach dem Neuladen der Seite fortsetzen.
  useEffect(() => {
    const gespeichert = sessionStorage.getItem(SPEICHER_TOKEN);
    if (!gespeichert) return;
    sitzungLaden(gespeichert)
      .then((sitzung) => {
        setToken(sitzung.token);
        setAblauf(sitzung.expires_at);
        standUebernehmen(sitzung.data ?? {});
      })
      .catch(() => sessionStorage.removeItem(SPEICHER_TOKEN));
  }, [standUebernehmen]);

  const anmelden = useCallback(
    async (neuerToken: string, neuerAblauf: string) => {
      setToken(neuerToken);
      setAblauf(neuerAblauf);
      setGeloescht(false);
      sessionStorage.setItem(SPEICHER_TOKEN, neuerToken);
      try {
        const sitzung = await sitzungLaden(neuerToken);
        standUebernehmen(sitzung.data ?? {});
      } catch {
        setLadefehler("Der Arbeitsstand konnte nicht vollständig geladen werden.");
      }
    },
    [standUebernehmen],
  );

  // Änderungen sichern. Die Unterlagen selbst verwaltet der Dienst, hier gehen
  // nur die Angaben des Formulars und die Darstellungswünsche mit.
  useEffect(() => {
    if (!token) return;
    const zeitgeber = setTimeout(() => {
      sitzungLaden(token)
        .then((sitzung) =>
          sitzungSpeichern(token, {
            ...(sitzung.data ?? {}),
            brief: angaben,
            darstellung: { schrift: schriftgroesse, kontrast: hoherKontrast },
          }),
        )
        .catch(() => undefined);
    }, 800);
    return () => clearTimeout(zeitgeber);
  }, [token, angaben, schriftgroesse, hoherKontrast]);

  const abmelden = useCallback(() => {
    sessionStorage.removeItem(SPEICHER_TOKEN);
    setToken(null);
    setDokumente([]);
    setVerlauf([]);
    setQuellen([]);
    setEntwurf("");
    setAngaben(LEERE_ANGABEN);
    setPdfFertig(false);
    setReiter("unterlagen");
  }, []);

  const schritte = useMemo(
    () => [
      {
        titel: "Unterlagen hochladen",
        fertig: dokumente.length > 0,
        status: dokumente.length > 0 ? `${dokumente.length} eingelesen` : "noch offen",
      },
      {
        titel: "Mit dem Assistenten prüfen",
        fertig: verlauf.some((n) => n.role === "assistant"),
        status: verlauf.some((n) => n.role === "assistant") ? "Auswertung liegt vor" : "noch offen",
      },
      {
        titel: "Widerspruch als PDF",
        fertig: pdfFertig,
        status: pdfFertig ? "PDF erstellt" : "noch offen",
      },
    ],
    [dokumente, verlauf, pdfFertig],
  );

  if (!token) {
    return (
      <>
        {geloescht && (
          <div className="huelle" style={{ paddingBottom: 0 }}>
            <Hinweis art="erfolg">
              <strong>Alle Ihre Daten wurden vollständig gelöscht.</strong> Vielen Dank für Ihr
              Vertrauen.
            </Hinweis>
          </div>
        )}
        <Startseite anmelden={anmelden} />
      </>
    );
  }

  const tage = ablauf
    ? Math.max(Math.floor((new Date(ablauf).getTime() - Date.now()) / 86_400_000), 0)
    : null;

  return (
    <div className="huelle">
      <header className="kopf">
        <h1 className="kopf-titel">⚖️ Pflegehilfe Online</h1>
        <div className="plaketten">
          <span className="plakette aktiv">Sitzung aktiv</span>
          {tage !== null && <span className="plakette">noch {tage} Tage gültig</span>}
          <span className={`plakette ${dokumente.length > 0 ? "aktiv" : "offen"}`}>
            {dokumente.length > 0 ? `${dokumente.length} Unterlage(n)` : "keine Unterlagen"}
          </span>
        </div>
      </header>

      <Fortschritt schritte={schritte} />

      {ladefehler && <Hinweis art="warnung">{ladefehler}</Hinweis>}

      <div className="reiter" role="tablist" aria-label="Arbeitsschritte">
        {REITER.map((eintrag) => (
          <button
            key={eintrag.schluessel}
            role="tab"
            aria-selected={reiter === eintrag.schluessel}
            aria-controls={`bereich-${eintrag.schluessel}`}
            id={`reiter-${eintrag.schluessel}`}
            onClick={() => setReiter(eintrag.schluessel)}
          >
            {eintrag.beschriftung}
          </button>
        ))}
      </div>

      <div id={`bereich-${reiter}`} role="tabpanel" aria-labelledby={`reiter-${reiter}`}>
        {reiter === "unterlagen" && (
          <UnterlagenTab token={token} dokumente={dokumente} setDokumente={setDokumente} />
        )}
        {reiter === "chat" && (
          <ChatTab
            token={token}
            hatUnterlagen={dokumente.length > 0}
            verlauf={verlauf}
            setVerlauf={setVerlauf}
            quellen={quellen}
            setQuellen={setQuellen}
            perspektive={angaben.perspektive as Perspektive}
            versicherteName={angaben.versichert_name}
            verhaeltnis={angaben.verhaeltnis}
            entwurfGesetzt={setEntwurf}
          />
        )}
        {reiter === "brief" && (
          <BriefTab
            token={token}
            angaben={angaben}
            setAngaben={setAngaben}
            entwurf={entwurf}
            pdfErzeugt={() => setPdfFertig(true)}
          />
        )}
        {reiter === "einstellungen" && (
          <EinstellungenTab
            token={token}
            ablauf={ablauf}
            setAblauf={setAblauf}
            schriftgroesse={schriftgroesse}
            setSchriftgroesse={setSchriftgroesse}
            hoherKontrast={hoherKontrast}
            setHoherKontrast={setHoherKontrast}
            abmelden={abmelden}
            geloescht={() => {
              setGeloescht(true);
              abmelden();
            }}
          />
        )}
      </div>

      <footer className="fusszeile">
        Dieser Assistent ersetzt keine Rechtsberatung. Alle erstellten Texte müssen vor dem
        Absenden geprüft werden. Die Verarbeitung findet ausschließlich örtlich statt.
      </footer>
    </div>
  );
}
