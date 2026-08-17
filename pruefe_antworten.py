"""Prüfbatterie für die Antwortqualität.

Stellt dem Assistenten freie Fragen - nicht die vorbereiteten Aufgaben - und
prüft für jede, ob sie überhaupt beantwortbar ist und ob die Antwort die
erwarteten Angaben enthält. Damit lässt sich belegen, was der Assistent kann
und wo die Wissensbasis Lücken hat.

Aufruf (LM Studio und die Vektordatenbank müssen laufen):
    python pruefe_antworten.py            – alle Fragen
    python pruefe_antworten.py fristen    – nur eine Gruppe
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass, field
from typing import List, Sequence

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
)

import pflege_rag


@dataclass
class Pruefung:
    """Eine Frage und woran sich eine brauchbare Antwort erkennen lässt."""

    gruppe: str
    frage: str
    # Mindestens eines dieser Wörter muss vorkommen, sonst ging die Antwort
    # am Thema vorbei. Kleinschreibung, es wird ohne Rücksicht darauf gesucht.
    erwartet: Sequence[str] = ()
    # Diese Wörter dürfen nicht vorkommen.
    verboten: Sequence[str] = ()
    # Wird eine Ablehnung erwartet?
    soll_ablehnen: bool = False


PRUEFUNGEN: List[Pruefung] = [
    # --- Begutachtung: Module, Punkte, Bewertungsstufen ---------------------
    Pruefung("module", "Welche sechs Module werden bei der Pflegebegutachtung bewertet?",
             ["mobilität", "selbstversorgung", "kognitiv"]),
    Pruefung("module", "Wie stark wird das Modul Selbstversorgung gewichtet?",
             ["40", "prozent"]),
    Pruefung("module", "Was bedeutet die Bewertungsstufe überwiegend unselbständig?",
             ["unselbständig", "unselbstständig"]),
    Pruefung("module", "Wie werden die Module 2 und 3 zusammen gewertet?",
             ["höher", "nur", "einer"]),
    Pruefung("module", "Ab wie vielen Gesamtpunkten bekommt man Pflegegrad 3?",
             ["47,5", "47.5", "47"]),
    Pruefung("module", "Ab wie vielen Punkten beginnt Pflegegrad 2?",
             ["27", "27,5", "27.5"]),

    # --- Widerspruch: Frist, Form, Ablauf ------------------------------------
    Pruefung("widerspruch", "Wie lange habe ich Zeit, um Widerspruch einzulegen?",
             ["monat", "vier wochen", "frist"]),
    Pruefung("widerspruch", "Muss der Widerspruch schriftlich sein?",
             ["schriftlich", "unterschrift", "schriftform"]),
    Pruefung("widerspruch", "Kostet ein Widerspruch gegen den Pflegegrad Geld?",
             ["kostenlos", "kosten", "gebühren", "keine"]),
    Pruefung("widerspruch", "Was passiert, nachdem ich Widerspruch eingelegt habe?",
             ["prüf", "erneut", "begutachtung", "widerspruchsbescheid"]),
    Pruefung("widerspruch", "Kann ich den Widerspruch auch ohne Begründung einlegen?",
             ["frist", "nachreichen", "begründung"]),
    Pruefung("widerspruch", "Was kann ich tun, wenn der Widerspruch abgelehnt wird?",
             ["klage", "sozialgericht", "gericht"]),

    # --- Antrag und Verfahren ------------------------------------------------
    Pruefung("antrag", "Wie stelle ich einen Antrag auf einen Pflegegrad?",
             ["pflegekasse", "antrag"]),
    Pruefung("antrag", "Wer führt die Begutachtung durch?",
             ["medizinische", "dienst", "medicproof"]),
    Pruefung("antrag", "Wie läuft ein Begutachtungstermin ab?",
             ["gutachter", "besuch", "termin", "häuslich"]),
    Pruefung("antrag", "Wofür brauche ich ein Pflegetagebuch?",
             ["tagebuch", "dokument", "nachweis", "hilfebedarf"]),

    # --- Leistungen ----------------------------------------------------------
    Pruefung("leistungen", "Was ist der Unterschied zwischen Pflegegeld und Pflegesachleistung?",
             ["pflegegeld", "sachleistung"]),
    Pruefung("leistungen", "Was ist der Entlastungsbetrag?",
             ["entlastungsbetrag", "125"]),
    Pruefung("leistungen", "Welche Leistungen gibt es bei Pflegegrad 2?",
             ["pflegegeld", "euro", "leistung"]),

    # --- Rechtsgrundlagen ----------------------------------------------------
    Pruefung("recht", "Was regelt § 14 SGB XI?",
             ["pflegebedürftig", "begriff", "selbständig"]),
    Pruefung("recht", "Was steht in § 15 SGB XI?",
             ["pflegegrad", "punkt", "ermittlung"]),
    Pruefung("recht", "Was besagt § 18 SGB XI?",
             ["begutachtung", "medizinisch", "dienst"]),

    # --- Themenfremd: muss abgelehnt werden ----------------------------------
    Pruefung("themenfremd", "Wie ist das Wetter aktuell in Berlin?", soll_ablehnen=True),
    Pruefung("themenfremd", "Wer hat die Fußball-WM 2014 gewonnen?", soll_ablehnen=True),
    Pruefung("themenfremd", "Schreibe mir ein Gedicht über den Herbst.", soll_ablehnen=True),
    Pruefung("themenfremd", "Wie backe ich einen Käsekuchen?", soll_ablehnen=True),
    Pruefung("themenfremd", "Was ist die Hauptstadt von Frankreich?", soll_ablehnen=True),
    Pruefung("themenfremd", "Gib mir Anlagetipps für meine Ersparnisse.", soll_ablehnen=True),
]


@dataclass
class Ergebnis:
    pruefung: Pruefung
    antwort: str
    bewertung: float
    quellen: int
    dauer: float
    fehler: List[str] = field(default_factory=list)

    @property
    def bestanden(self) -> bool:
        return not self.fehler


# Die schweren Bestandteile einmal laden und wiederverwenden. Beim ersten
# Aufruf dauert das Laden des Neubewertungsmodells rund 20 Sekunden.
_geladen: dict = {}


def _llm():
    if "llm" not in _geladen:
        _geladen["llm"] = pflege_rag.create_llm()
    return _geladen["llm"]


def _fachwissen() -> pflege_rag.HybridIndex:
    """Öffnet die Wissensdatenbank und baut den Stichwortindex dazu auf."""
    if "fach" not in _geladen:
        speicher = pflege_rag.open_expert_database(pflege_rag.create_embeddings())
        abschnitte = pflege_rag.load_all_expert_chunks(speicher)
        _geladen["fach"] = pflege_rag.HybridIndex(speicher, abschnitte)
    return _geladen["fach"]


def pruefe(pruefung: Pruefung, reranker, fach) -> Ergebnis:
    start = time.time()
    ergebnis = pflege_rag.prepare_context(fach, None, pruefung.frage, reranker=reranker)

    if ergebnis.themenfremd:
        antwort = pflege_rag.ABLEHNUNG_THEMENFREMD
    else:
        nachrichten = pflege_rag.build_messages(
            ergebnis.system_prompt, [{"role": "user", "content": pruefung.frage}]
        )
        antwort = "".join(pflege_rag.stream_answer(_llm(), nachrichten))
        antwort = pflege_rag.render_citations(
            pflege_rag.strip_context_headers(antwort), ergebnis.nummern
        )

    fehler = []
    klein = antwort.lower()
    abgelehnt = pflege_rag.ABLEHNUNG_THEMENFREMD[:40] in antwort

    if pruefung.soll_ablehnen:
        if not abgelehnt:
            fehler.append("hätte ablehnen müssen, hat aber geantwortet")
    else:
        if abgelehnt:
            fehler.append("wurde fälschlich als themenfremd abgelehnt")
        elif pruefung.erwartet and not any(w.lower() in klein for w in pruefung.erwartet):
            fehler.append(f"keines der erwarteten Wörter: {list(pruefung.erwartet)}")
        for wort in pruefung.verboten:
            if wort.lower() in klein:
                fehler.append(f"verbotenes Wort: {wort}")

    return Ergebnis(pruefung, antwort, ergebnis.beste_bewertung,
                    len(ergebnis.quellen), time.time() - start, fehler)


def main() -> int:
    nur = sys.argv[1] if len(sys.argv) > 1 else ""
    aufgaben = [p for p in PRUEFUNGEN if not nur or p.gruppe == nur]
    if not aufgaben:
        gruppen = sorted({p.gruppe for p in PRUEFUNGEN})
        print(f"Unbekannte Gruppe. Verfügbar: {', '.join(gruppen)}")
        return 2

    print(f"{len(aufgaben)} Fragen werden geprüft. Das dauert einige Minuten.\n")
    reranker = pflege_rag.create_reranker()
    fach = _fachwissen()

    ergebnisse = []
    letzte_gruppe = ""
    for pruefung in aufgaben:
        if pruefung.gruppe != letzte_gruppe:
            letzte_gruppe = pruefung.gruppe
            print(f"\n--- {pruefung.gruppe.upper()} ---")
        e = pruefe(pruefung, reranker, fach)
        ergebnisse.append(e)
        zeichen = "OK    " if e.bestanden else "FEHLER"
        print(f"{zeichen} [{e.bewertung:.2f}] {pruefung.frage}", flush=True)
        if not e.bestanden:
            for f in e.fehler:
                print(f"         -> {f}")
            print(f"         Antwort: {e.antwort[:180]}")

    bestanden = sum(1 for e in ergebnisse if e.bestanden)
    print(f"\n{'='*70}")
    print(f"{bestanden} von {len(ergebnisse)} bestanden "
          f"({bestanden / len(ergebnisse) * 100:.0f} %)")
    gesamt = sum(e.dauer for e in ergebnisse)
    print(f"Gesamtdauer {gesamt:.0f}s, im Mittel {gesamt/len(ergebnisse):.1f}s je Frage")

    durchgefallen = [e for e in ergebnisse if not e.bestanden]
    if durchgefallen:
        print(f"\nNicht bestanden ({len(durchgefallen)}):")
        for e in durchgefallen:
            print(f"  [{e.pruefung.gruppe}] {e.pruefung.frage}")
    return 0 if not durchgefallen else 1


if __name__ == "__main__":
    raise SystemExit(main())
