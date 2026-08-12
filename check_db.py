import os
from qdrant_client import QdrantClient

# Verbinde zur lokalen Qdrant Datenbank
QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
client = QdrantClient(path=QDRANT_DIR)

print("Lade alle Chunks aus der Vektordatenbank...")

# Hole alle Einträge (bis zu 10.000)
records = client.scroll(
    collection_name="pflege_fachwissen",
    limit=10000,
    with_payload=True  # Das ist wichtig, hier steckt der Text drin
)[0]

# Schreibe alles in eine übersichtliche Textdatei
with open("datenbank_dump.txt", "w", encoding="utf-8") as f:
    for record in records:
        payload = record.payload
        source = payload.get("metadata", {}).get("source", "Unbekannt")
        text = payload.get("page_content", "Kein Text")

        f.write(f"=== QUELLE: {source} ===\n")
        f.write(f"{text}\n")
        f.write("\n" + "=" * 50 + "\n\n")

print("✅ Fertig! Öffne die Datei 'datenbank_dump.txt' in deinem Projektordner.")