/**
 * Pflegehilfe Online – Oberfläche.
 *
 * Gestaltungsleitlinien (nach Don Norman):
 *  Sichtbarkeit  – Seitenleiste, Kopfleiste und Fortschritt zeigen jederzeit
 *                  den Stand: wo man ist, was vorliegt, wie lange noch.
 *  Rückmeldung   – jede Aktion meldet sich sofort zurück, auch die langen.
 *  Constraints   – was noch nicht möglich ist, bleibt gesperrt und erklärt sich,
 *                  statt kommentarlos ins Leere zu führen.
 *  Mapping       – die Reihenfolge der Bereiche ist die Reihenfolge der Arbeit.
 *  Affordances   – Karten, Ablageflächen und Schaltflächen sehen aus wie das,
 *                  was man mit ihnen tun kann.
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
import { Hinweis } from "./components/Bausteine";
import { BriefTab } from "./components/BriefTab";
import { ChatTab } from "./components/ChatTab";
import { EinstellungenTab } from "./components/EinstellungenTab";
import { QuellenTab } from "./components/QuellenTab";
import { Schale, type Bereich } from "./components/Schale";
import { Startseite } from "./components/Startseite";
import { Symbol } from "./components/Symbole";
import { Uebersicht } from "./components/Uebersicht";
import { UnterlagenTab } from "./components/UnterlagenTab";

const SPEICHER_TOKEN = "pflegehilfe.token";

const LEERE_ANGABEN: Briefangaben = {
  absender_name: "", absender_strasse: "", absender_plz_ort: "", ort: "",
  kasse_name: "", kasse_strasse: "", kasse_plz_ort: "",
  versichert_name: "", versichert_nr: "", aktenzeichen: "", bescheid_datum: "",
  begruendung: "", begruendung_folgt: false, perspektive: "selbst", verhaeltnis: "",
  anlagen: "",
};

export default function App() {
  const [token, setToken] = useState<string | null>(null);
  const [ablauf, setAblauf] = useState("");
  const [bereich, setBereich] = useState<Bereich>("uebersicht");

  const [dokumente, setDokumente] = useState<string[]>([]);
  const [verlauf, setVerlauf] = useState<Nachricht[]>([]);
  const [quellen, setQuellen] = useState<Quelle[]>([]);
  const [entwurf, setEntwurf] = useState("");
  const [angaben, setAngaben] = useState<Briefangaben>(LEERE_ANGABEN);
  const [fristZugang, setFristZugang] = useState("");
  const [pdfFertig, setPdfFertig] = useState(false);

  const [schriftgroesse, setSchriftgroesse] = useState(18);
  const [hoherKontrast, setHoherKontrast] = useState(false);
  const [geloescht, setGeloescht] = useState(false);
  const [ladefehler, setLadefehler] = useState("");
  const [kopiert, setKopiert] = useState(false);

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
    const frist = daten.frist as { zugang?: string } | undefined;
    if (frist?.zugang) setFristZugang(frist.zugang);
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
      setBereich("uebersicht");
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
            frist: { zugang: fristZugang },
            darstellung: { schrift: schriftgroesse, kontrast: hoherKontrast },
          }),
        )
        .catch(() => undefined);
    }, 800);
    return () => clearTimeout(zeitgeber);
  }, [token, angaben, fristZugang, schriftgroesse, hoherKontrast]);

  const abmelden = useCallback(() => {
    sessionStorage.removeItem(SPEICHER_TOKEN);
    setToken(null);
    setDokumente([]);
    setVerlauf([]);
    setQuellen([]);
    setEntwurf("");
    setAngaben(LEERE_ANGABEN);
    setFristZugang("");
    setPdfFertig(false);
    setBereich("uebersicht");
  }, []);

  const schritte = useMemo(() => {
    const hatAuswertung = verlauf.some((n) => n.role === "assistant");
    return [
      {
        titel: "Unterlagen hochladen",
        fertig: dokumente.length > 0,
        status: dokumente.length > 0 ? `${dokumente.length} eingelesen` : "noch offen",
      },
      {
        titel: "Mit dem Assistenten prüfen",
        fertig: hatAuswertung,
        status: hatAuswertung ? "Auswertung liegt vor" : "noch offen",
      },
      {
        titel: "Widerspruch als PDF",
        fertig: pdfFertig,
        status: pdfFertig ? "PDF erstellt" : "noch offen",
      },
    ];
  }, [dokumente, verlauf, pdfFertig]);

  if (!token) {
    return (
      <>
        {geloescht && (
          <div style={{ padding: "var(--a4) var(--a4) 0", maxWidth: "60rem", margin: "0 auto" }}>
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
    <Schale
      bereich={bereich}
      wechseln={setBereich}
      anzahlUnterlagen={dokumente.length}
      anzahlQuellen={quellen.length}
      tageGueltig={tage}
      kopfWerkzeuge={
        <button
          className="knopf klein"
          onClick={() => {
            navigator.clipboard?.writeText(token);
            setKopiert(true);
            setTimeout(() => setKopiert(false), 2500);
          }}
          title="Ihren Zugangscode in die Zwischenablage kopieren"
        >
          <Symbol name={kopiert ? "haken" : "schluessel"} groesse={16} />
          {kopiert ? "Kopiert" : "Zugangscode"}
        </button>
      }
    >
      {ladefehler && <Hinweis art="warnung">{ladefehler}</Hinweis>}

      {bereich === "uebersicht" && (
        <Uebersicht
          dokumente={dokumente}
          verlauf={verlauf}
          quellen={quellen}
          angaben={angaben}
          fristZugang={fristZugang}
          setFristZugang={setFristZugang}
          schritte={schritte}
          wechseln={setBereich}
        />
      )}

      {bereich === "unterlagen" && (
        <UnterlagenTab token={token} dokumente={dokumente} setDokumente={setDokumente} />
      )}

      {bereich === "assistent" && (
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
          zuUnterlagen={() => setBereich("unterlagen")}
        />
      )}

      {bereich === "quellen" && <QuellenTab quellen={quellen} />}

      {bereich === "brief" && (
        <BriefTab
          token={token}
          angaben={angaben}
          setAngaben={setAngaben}
          entwurf={entwurf}
          pdfErzeugt={() => setPdfFertig(true)}
          zumAssistenten={() => setBereich("assistent")}
        />
      )}

      {bereich === "einstellungen" && (
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
    </Schale>
  );
}
