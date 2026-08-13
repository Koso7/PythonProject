/**
 * Belege und Quellen.
 *
 * Der Bereich beantwortet die Frage, die über die Verlässlichkeit entscheidet:
 * Woher weiß der Assistent das? Jede hochgestellte Ziffer in einer Antwort
 * findet sich hier mit der Textstelle wieder, die tatsächlich verwendet wurde.
 */
import type { Quelle } from "../api";
import { Hinweis, Karte, Leerbereich, Quellenliste } from "./Bausteine";

export function QuellenTab({ quellen }: { quellen: Quelle[] }) {
  const eigene = quellen.filter((q) => q.herkunft === "nutzer");
  const fachwissen = quellen.filter((q) => q.herkunft === "fachwissen");

  if (quellen.length === 0) {
    return (
      <Karte titel="Noch keine Belege" symbol="quellen">
        <Leerbereich
          symbol="quellen"
          titel="Hier erscheinen die Belegstellen"
          text="Sobald der Assistent Ihren Fall ausgewertet hat, steht hier zu jeder hochgestellten Ziffer der Abschnitt, auf den sie verweist – aus Ihren eigenen Unterlagen und aus dem geprüften Fachwissen."
        />
      </Karte>
    );
  }

  return (
    <>
      <Hinweis>
        Die hochgestellten Ziffern in den Antworten des Assistenten verweisen auf diese Abschnitte.
        Angezeigt wird jeweils <strong>die Textstelle, die tatsächlich verwendet wurde</strong> –
        nicht das ganze Dokument. Klicken Sie einen Eintrag an, um sie zu lesen.
      </Hinweis>

      <div className="spalten zwei">
        <Karte
          titel={`Aus Ihren Unterlagen (${eigene.length})`}
          symbol="datei"
          fuss="Diese Stellen stammen aus den Dateien, die Sie selbst hochgeladen haben."
        >
          {eigene.length > 0 ? (
            <Quellenliste quellen={eigene} />
          ) : (
            <Leerbereich
              symbol="datei"
              titel="Keine eigene Fundstelle"
              text="Die letzte Antwort stützte sich allein auf das Fachwissen. Ergänzende Unterlagen – etwa ein Pflegetagebuch – geben dem Assistenten mehr Anhaltspunkte."
            />
          )}
        </Karte>

        <Karte
          titel={`Geprüftes Fachwissen (${fachwissen.length})`}
          symbol="quellen"
          fuss="Begutachtungs-Richtlinien, SGB XI und Ratgeber amtlicher Stellen und Verbraucherverbände."
        >
          {fachwissen.length > 0 ? (
            <Quellenliste quellen={fachwissen} />
          ) : (
            <Leerbereich
              symbol="quellen"
              titel="Keine Fachquelle"
              text="Die letzte Antwort bezog sich ausschließlich auf Ihre eigenen Unterlagen."
            />
          )}
        </Karte>
      </div>
    </>
  );
}
