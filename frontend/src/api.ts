/**
 * Anbindung an den örtlichen Dienst.
 *
 * Alle Aufrufe gehen an denselben Rechner; im Entwicklungsbetrieb leitet Vite
 * sie weiter (siehe vite.config.ts). Es verlässt kein Datum den Rechner.
 */

const BASIS = import.meta.env.VITE_API_URL ?? "";

export type Perspektive = "selbst" | "angehoerige";

export interface Sitzung {
  token: string;
  expires_at: string;
}

export interface Nachricht {
  role: "user" | "assistant";
  content: string;
}

export interface Quelle {
  nummer: number;
  quelle: string;
  art: string;
  ausschnitt: string;
  bewertung: number;
  herkunft: "nutzer" | "fachwissen";
  ueberschrift: string;
}

export interface Aktion {
  schluessel: string;
  titel: string;
  beschreibung: string;
  nutzertext: string;
  braucht_perspektive: boolean;
}

export interface Zustand {
  wissensbasis_abschnitte: number;
  vektordatenbank: string;
  neubewertung: string;
  sprachmodell: string;
}

export interface UploadErgebnis {
  dateiname: string;
  erfolgreich: boolean;
  abschnitte: number;
  hinweis: string;
}

export interface Briefangaben {
  absender_name: string;
  absender_strasse: string;
  absender_plz_ort: string;
  ort: string;
  kasse_name: string;
  kasse_strasse: string;
  kasse_plz_ort: string;
  versichert_name: string;
  versichert_nr: string;
  aktenzeichen: string;
  bescheid_datum: string;
  begruendung: string;
  begruendung_folgt: boolean;
  anlagen?: string;
  perspektive: Perspektive;
  verhaeltnis: string;
}

export interface Briefpruefung {
  fehlende_angaben: string[];
  offene_platzhalter: string[];
}

/** Fehler mit einer für Menschen lesbaren Meldung. */
export class ApiFehler extends Error {
  readonly status: number;

  constructor(status: number, meldung: string) {
    super(meldung);
    this.name = "ApiFehler";
    this.status = status;
  }
}

async function anfrage<T>(pfad: string, optionen: RequestInit = {}): Promise<T> {
  let antwort: Response;
  try {
    antwort = await fetch(`${BASIS}${pfad}`, optionen);
  } catch {
    throw new ApiFehler(0, "Der Hintergrunddienst ist nicht erreichbar. Läuft er bereits?");
  }
  if (!antwort.ok) {
    const text = await antwort.text().catch(() => "");
    let meldung = text;
    try {
      meldung = JSON.parse(text).detail ?? text;
    } catch {
      /* Klartext beibehalten */
    }
    throw new ApiFehler(antwort.status, meldung || `Fehler ${antwort.status}`);
  }
  return antwort.status === 204 ? (undefined as T) : ((await antwort.json()) as T);
}

// ---------------------------------------------------------------------------
// SITZUNG
// ---------------------------------------------------------------------------
export const sitzungAnlegen = () => anfrage<Sitzung>("/session", { method: "POST" });

export const sitzungLaden = (token: string) =>
  anfrage<Sitzung & { data: Record<string, unknown> }>(`/session/${token}`);

export const sitzungSpeichern = (token: string, data: Record<string, unknown>) =>
  anfrage<Sitzung>(`/session/${token}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });

export const sitzungVerlaengern = (token: string) =>
  anfrage<{ expires_at: string }>(`/session/${token}/extend`, { method: "POST" });

export const sitzungLoeschen = (token: string) =>
  anfrage<{ message: string }>(`/session/${token}`, { method: "DELETE" });

// ---------------------------------------------------------------------------
// UNTERLAGEN
// ---------------------------------------------------------------------------
export async function unterlagenHochladen(token: string, dateien: File[]) {
  const formular = new FormData();
  dateien.forEach((datei) => formular.append("files", datei));
  return anfrage<{ ergebnisse: UploadErgebnis[]; dokumente: string[] }>(
    `/session/${token}/documents`,
    { method: "POST", body: formular },
  );
}

export const unterlageEntfernen = (token: string, dateiname: string) =>
  anfrage<{ dokumente: string[] }>(
    `/session/${token}/documents/${encodeURIComponent(dateiname)}`,
    { method: "DELETE" },
  );

// ---------------------------------------------------------------------------
// ASSISTENT
// ---------------------------------------------------------------------------
export const aktionenLaden = () => anfrage<Aktion[]>("/actions");
export const zustandLaden = () => anfrage<Zustand>("/status");

export interface ChatAnfrage {
  aktion?: string;
  frage?: string;
  perspektive: Perspektive;
  versicherte_name: string;
  verhaeltnis: string;
}

export interface ChatMeldung {
  art: "status" | "suchfrage" | "text" | "ergebnis" | "fehler";
  text?: string;
  antwort?: string;
  quellen?: Quelle[];
  suchfrage?: string;
  umformuliert?: boolean;
  ohne_beleg?: boolean;
}

/**
 * Stellt eine Frage und meldet den Fortschritt, sobald er eintrifft.
 *
 * Der Dienst sendet einen Ereignisstrom; dadurch erscheint die Antwort beim
 * Entstehen, statt dass minutenlang nichts passiert.
 */
export async function chatStellen(
  token: string,
  anfrageDaten: ChatAnfrage,
  beiMeldung: (meldung: ChatMeldung) => void,
  abbruch?: AbortSignal,
): Promise<void> {
  const antwort = await fetch(`${BASIS}/session/${token}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(anfrageDaten),
    signal: abbruch,
  });
  if (!antwort.ok || !antwort.body) {
    const text = await antwort.text().catch(() => "");
    throw new ApiFehler(antwort.status, text || "Die Anfrage ist fehlgeschlagen.");
  }

  const leser = antwort.body.getReader();
  const dekoder = new TextDecoder();
  let puffer = "";

  for (;;) {
    const { done, value } = await leser.read();
    if (done) break;
    puffer += dekoder.decode(value, { stream: true });

    // Ereignisse sind durch eine Leerzeile getrennt.
    let grenze: number;
    while ((grenze = puffer.indexOf("\n\n")) !== -1) {
      const block = puffer.slice(0, grenze);
      puffer = puffer.slice(grenze + 2);
      const zeile = block.split("\n").find((z) => z.startsWith("data: "));
      if (!zeile) continue;
      try {
        beiMeldung(JSON.parse(zeile.slice(6)) as ChatMeldung);
      } catch {
        /* unvollständige Blöcke überspringen */
      }
    }
  }
}

// ---------------------------------------------------------------------------
// WIDERSPRUCHSSCHREIBEN
// ---------------------------------------------------------------------------
export const briefPruefen = (token: string, angaben: Briefangaben) =>
  anfrage<Briefpruefung>(`/session/${token}/letter/validate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(angaben),
  });

export async function briefErzeugen(token: string, angaben: Briefangaben): Promise<Blob> {
  const antwort = await fetch(`${BASIS}/session/${token}/letter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(angaben),
  });
  if (!antwort.ok) {
    const text = await antwort.text().catch(() => "");
    let meldung = text;
    try {
      meldung = JSON.parse(text).detail ?? text;
    } catch {
      /* Klartext beibehalten */
    }
    throw new ApiFehler(antwort.status, meldung);
  }
  return antwort.blob();
}
