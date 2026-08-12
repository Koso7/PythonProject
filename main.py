"""Diagnosewerkzeug für die Kommandozeile.

Führt eine Frage durch dieselbe Such- und Antwortkette wie die Weboberfläche
und zeigt dabei offen, welche Abschnitte gefunden, wie sie bewertet und welche
davon tatsächlich zitiert wurden. Damit lässt sich die Antwortqualität prüfen,
ohne die Oberfläche zu starten.

Aufruf:
    python main.py                      – Dialogbetrieb
    python main.py "Meine Frage"        – eine einzelne Frage

Hinweis: Die Vektordatenbank kann nur von einem Prozess gleichzeitig geöffnet
werden. Die Weboberfläche muss dafür beendet sein.
"""

from __future__ import annotations

import sys
import textwrap
import time

from dotenv import load_dotenv

import pflege_rag

load_dotenv()

BREITE = 100


def trennlinie(zeichen: str = "-") -> None:
    print(zeichen * BREITE)


def zeige_quellen(ergebnis: pflege_rag.RetrievalResult, zitiert: set[int]) -> None:
    trennlinie()
    print("GEFUNDENE ABSCHNITTE")
    trennlinie()
    for quelle in ergebnis.quellen:
        markierung = "zitiert " if quelle.nummer in zitiert else "ungenutzt"
        herkunft = "Nutzerunterlage" if quelle.herkunft == "nutzer" else "Fachwissen"
        print(f"[{quelle.nummer}] {markierung} | Bewertung {quelle.bewertung:5.3f} | {herkunft}")
        print(f"     Quelle: {quelle.quelle}")
        if quelle.ueberschrift:
            print(f"     Abschnitt: {quelle.ueberschrift}")
        for zeile in textwrap.wrap(quelle.ausschnitt, BREITE - 7):
            print(f"     {zeile}")
        print()


def beantworte(frage: str, index, reranker, llm) -> None:
    start = time.time()
    ergebnis = pflege_rag.prepare_context(index, None, frage, reranker=reranker)
    suchdauer = time.time() - start

    print("\nAntwort:")
    trennlinie()
    start = time.time()
    antwort = "".join(
        pflege_rag.stream_answer(llm, pflege_rag.build_messages(ergebnis.system_prompt, [
            {"role": "user", "content": frage}
        ]))
    )
    antwortdauer = time.time() - start

    zitiert = set(pflege_rag.cited_numbers(antwort)) & set(ergebnis.nummern)
    print(pflege_rag.render_citations(antwort, ergebnis.nummern))
    print()
    zeige_quellen(ergebnis, zitiert)

    ohne_beleg = "keine" if zitiert else "KEINE – die Antwort nennt keine Belegstelle!"
    print(f"Suche {suchdauer:.1f}s | Antwort {antwortdauer:.1f}s | "
          f"{len(ergebnis.quellen)} Abschnitte gefunden | zitiert: {sorted(zitiert) or ohne_beleg}")


def main() -> int:
    print("=" * BREITE)
    print("Pflege-Assistent – Diagnose der Antwortqualität")
    print("=" * BREITE)

    print("Wissensdatenbank wird geöffnet …")
    index = pflege_rag.HybridIndex(None, [])
    try:
        speicher = pflege_rag.open_expert_database(pflege_rag.create_embeddings())
        abschnitte = pflege_rag.load_all_expert_chunks(speicher)
        index = pflege_rag.HybridIndex(speicher, abschnitte)
        print(f"  {len(abschnitte)} Abschnitte geladen.")
    except Exception as fehler:
        print(f"  Fehler: {fehler}")
        print("  Läuft die Weboberfläche noch? Sie sperrt die Vektordatenbank.")
        return 1

    print("Neubewertungsmodell wird geladen (einmalig, etwa 20 Sekunden) …")
    reranker = pflege_rag.create_reranker()
    llm = pflege_rag.create_llm()
    print("Bereit.\n")

    if len(sys.argv) > 1:
        beantworte(" ".join(sys.argv[1:]), index, reranker, llm)
        return 0

    print("Frage eingeben, 'exit' zum Beenden.")
    while True:
        try:
            frage = input("\nFrage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if frage.lower() in {"exit", "quit", "ende"}:
            return 0
        if frage:
            beantworte(frage, index, reranker, llm)


if __name__ == "__main__":
    sys.exit(main())
