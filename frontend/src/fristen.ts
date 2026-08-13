/**
 * Berechnung der Widerspruchsfrist.
 *
 * Der Widerspruch muss binnen eines Monats nach Bekanntgabe des Bescheids bei
 * der Pflegekasse eingehen (§ 84 Abs. 1 SGG). Maßgeblich ist nicht das Datum
 * auf dem Bescheid, sondern wann er zugegangen ist; bei Zusendung mit der Post
 * gilt er am dritten Tag nach der Aufgabe als bekannt gegeben (§ 37 Abs. 2
 * SGB X). Diese Anwendung rechnet mit dem Tag, den die Person selbst angibt,
 * und sagt deutlich dazu, dass die Frist im Zweifel zu prüfen ist.
 *
 * "Ein Monat" heißt: derselbe Tag im Folgemonat (§ 26 Abs. 1 SGB X i. V. m.
 * § 188 Abs. 2 BGB). Gibt es diesen Tag dort nicht - etwa der 31. Januar, dem
 * kein 31. Februar folgt -, endet die Frist am letzten Tag des Folgemonats.
 * Genau diesen Fall bekommt eine naive Rechnung mit "Monat + 1" falsch: aus
 * dem 31. Januar würde der 3. März.
 */

/** Liest "12.03.2025" oder "2025-03-12". Gibt bei Unsinn null zurück. */
export function datumLesen(text: string): Date | null {
  const wert = (text || "").trim();
  if (!wert) return null;

  let jahr: number, monat: number, tag: number;
  const deutsch = wert.match(/^(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})$/);
  const iso = wert.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (deutsch) {
    [, tag, monat, jahr] = deutsch.map(Number) as unknown as [never, number, number, number];
  } else if (iso) {
    [, jahr, monat, tag] = iso.map(Number) as unknown as [never, number, number, number];
  } else {
    return null;
  }

  const datum = new Date(jahr, monat - 1, tag);
  // Fängt den 31.02. ab: Date rollt still in den März weiter.
  if (
    datum.getFullYear() !== jahr ||
    datum.getMonth() !== monat - 1 ||
    datum.getDate() !== tag
  ) {
    return null;
  }
  return datum;
}

/** Der letzte Tag, an dem der Widerspruch eingehen darf. */
export function fristEnde(zugang: Date): Date {
  const jahr = zugang.getFullYear();
  const monat = zugang.getMonth();
  const tag = zugang.getDate();

  // Tag 0 des übernächsten Monats = letzter Tag des Folgemonats.
  const letzterImFolgemonat = new Date(jahr, monat + 2, 0).getDate();
  return new Date(jahr, monat + 1, Math.min(tag, letzterImFolgemonat));
}

/** Volle Tage von heute bis zum Stichtag. Negativ, wenn er vorbei ist. */
export function tageBis(stichtag: Date): number {
  const heute = new Date();
  const a = Date.UTC(heute.getFullYear(), heute.getMonth(), heute.getDate());
  const b = Date.UTC(stichtag.getFullYear(), stichtag.getMonth(), stichtag.getDate());
  return Math.round((b - a) / 86_400_000);
}

export type Dringlichkeit = "gruen" | "gelb" | "rot" | "abgelaufen";

/** Ab wann die Anzeige die Farbe wechselt. */
export function dringlichkeit(tage: number): Dringlichkeit {
  if (tage < 0) return "abgelaufen";
  if (tage <= 3) return "rot";
  if (tage <= 10) return "gelb";
  return "gruen";
}

export function datumFormatieren(datum: Date): string {
  return datum.toLocaleDateString("de-DE", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });
}

export interface Fristangabe {
  ende: Date;
  tage: number;
  stufe: Dringlichkeit;
  text: string;
}

/** Fasst alles zusammen, was die Oberfläche über die Frist anzeigen will. */
export function fristBerechnen(zugangstext: string): Fristangabe | null {
  const zugang = datumLesen(zugangstext);
  if (!zugang) return null;
  const ende = fristEnde(zugang);
  const tage = tageBis(ende);
  const stufe = dringlichkeit(tage);
  let text: string;
  if (tage < 0) text = `seit ${Math.abs(tage)} Tag(en) abgelaufen`;
  else if (tage === 0) text = "Heute ist der letzte Tag";
  else if (tage === 1) text = "Nur noch 1 Tag";
  else text = `Noch ${tage} Tage`;
  return { ende, tage, stufe, text };
}
