"""Prüfwerkzeug für die Wissensdatenbank.

Zeigt, welche Quellen enthalten sind, wie viele Abschnitte je Quelle entstanden
sind und wie lang diese im Mittel sind. Damit lässt sich belegen, dass jedes
Dokument tatsächlich verarbeitet wurde.

Aufruf:
    python check_db.py              – Übersicht
    python check_db.py --dump       – zusätzlich alle Abschnitte in eine Textdatei

Hinweis: Die Weboberfläche muss beendet sein, sie sperrt die Datenbank.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv

import pflege_rag

load_dotenv()

DUMP_DATEI = "datenbank_dump.txt"


def main() -> int:
    try:
        speicher = pflege_rag.open_expert_database(pflege_rag.create_embeddings())
        abschnitte = pflege_rag.load_all_expert_chunks(speicher)
    except Exception as fehler:
        print(f"Die Wissensdatenbank ist nicht lesbar: {fehler}")
        print("Läuft die Weboberfläche noch? Sie sperrt die Datenbank.")
        return 1

    if not abschnitte:
        print("Die Wissensdatenbank ist leer. Bitte zuerst 'python ingest.py' ausführen.")
        return 1

    je_quelle: dict[str, list[int]] = defaultdict(list)
    arten: Counter[str] = Counter()
    module: Counter[int] = Counter()
    for abschnitt in abschnitte:
        quelle = abschnitt.metadata.get("source", "unbekannt")
        je_quelle[quelle].append(len(abschnitt.page_content))
        arten[abschnitt.metadata.get("doc_kind", "Dokument")] += 1
        for nummer in abschnitt.metadata.get("modules", []) or []:
            module[nummer] += 1

    print("=" * 100)
    print(f"WISSENSDATENBANK – {len(abschnitte)} Abschnitte aus {len(je_quelle)} Quellen")
    print("=" * 100)
    print(f"{'Quelle':<66}{'Abschnitte':>12}{'Ø Zeichen':>12}")
    print("-" * 100)
    for quelle, laengen in sorted(je_quelle.items(), key=lambda p: len(p[1]), reverse=True):
        name = quelle if len(quelle) <= 64 else quelle[:61] + "..."
        print(f"{name:<66}{len(laengen):>12}{sum(laengen) // len(laengen):>12}")

    print("-" * 100)
    print("Dokumentarten:", ", ".join(f"{art} ({anzahl})" for art, anzahl in arten.most_common()))
    if module:
        zuordnung = ", ".join(
            f"Modul {nummer}: {module[nummer]}" for nummer in sorted(module)
        )
        print("Modulzuordnung:", zuordnung)

    if "--dump" in sys.argv:
        with open(DUMP_DATEI, "w", encoding="utf-8") as datei:
            for abschnitt in abschnitte:
                datei.write(f"=== QUELLE: {abschnitt.metadata.get('source', 'unbekannt')} ===\n")
                datei.write(abschnitt.page_content + "\n\n" + "=" * 60 + "\n\n")
        print(f"\nAlle Abschnitte wurden nach '{DUMP_DATEI}' geschrieben.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
