# Pflegehilfe Online

Ein KI-gestützter Assistent, der Betroffene und Angehörige beim **Widerspruch gegen einen
Pflegegradbescheid** unterstützt. Hochgeladene Unterlagen werden mit dem Gutachten des
Medizinischen Dienstes abgeglichen; daraus entsteht ein fertiges Widerspruchsschreiben als PDF.

Universitäres Studienprojekt. **Alle Verarbeitung findet örtlich statt** – es gehen keine Daten an
Anbieter im Internet.

---

## 1. Voraussetzungen

| Was | Warum |
|---|---|
| **Python 3.11 oder neuer** | Hintergrunddienst und Suche |
| **Node.js 20 oder neuer** ([nodejs.org](https://nodejs.org)) | Weboberfläche (React) |
| **Docker Desktop** ([docker.com](https://www.docker.com/products/docker-desktop/)) | Vektordatenbank als Dienst |
| **LM Studio** ([lmstudio.ai](https://lmstudio.ai)) | führt Sprachmodell und Einbettungen örtlich aus |
| ca. **12 GB freier Speicher** | Modelle, Wissensdatenbank, ONNX-Fassung des Rerankers |
| 16 GB Arbeitsspeicher empfohlen | |

### Modelle in LM Studio laden

In LM Studio unter *Discover* herunterladen und im Reiter *Developer* **beide** laden, dann den
Server auf Port `1234` starten:

| Modell | Rolle |
|---|---|
| `mistralai/mistral-nemo-instruct-2407` | erzeugt die Antworten |
| `text-embedding-bge-m3` | wandelt Texte in Vektoren für die Suche |

> Ohne laufenden LM-Studio-Server meldet die Anwendung „Der Assistent ist gerade nicht erreichbar“.

---

## 2. Einrichtung

```bash
git clone https://github.com/Koso7/PythonProject.git
cd PythonProject

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt

npm install --prefix frontend
```

> Docker, Node und LM Studio laufen alle auf diesem Rechner. **Docker ist keine Cloud** – der
> Behälter mit der Vektordatenbank läuft örtlich und ist an `127.0.0.1` gebunden, also nicht
> einmal aus dem eigenen Netzwerk erreichbar.

### Grafikkarte (optional, aber deutlich schneller)

Die Neubewertung der Suchtreffer läuft auf der Grafikkarte rund achtmal schneller. Dafür muss
`onnxruntime-directml` **nach** allen anderen Paketen installiert werden, weil es das gewöhnliche
`onnxruntime` ersetzt:

```bash
pip install --force-reinstall --no-deps onnxruntime-directml
```

Funktioniert mit jeder DirectX-12-Grafikkarte unter Windows (AMD, NVIDIA, Intel). **Ohne passende
Karte läuft alles unverändert auf dem Prozessor** – es ist keine Einstellung nötig, die Anwendung
erkennt das selbst. Unter *Einstellungen* wird angezeigt, was gerade verwendet wird.

### Zugangsdaten anlegen

```bash
copy .env.example .env            # Windows
# cp .env.example .env            # Linux/macOS
```

Dann in `.env` einen Schlüssel für die Verschlüsselung der Sitzungsdaten eintragen:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

> Der `ENCRYPTION_KEY` darf sich später **nicht mehr ändern**, sonst sind gespeicherte Sitzungen
> nicht mehr lesbar. Ohne Eintrag erzeugt der Dienst bei jedem Start einen Wegwerf-Schlüssel.

---

## 3. Wissensdatenbank aufbauen

Zuerst die Vektordatenbank starten. Sie läuft als eigener Dienst, damit `ingest.py` und die
Anwendung gleichzeitig darauf zugreifen können:

```bash
docker compose up -d
```

Dann die Wissensdatenbank füllen. Einmalig, dauert je nach Rechner 10 bis 20 Minuten.
**LM Studio muss dabei laufen.**

```bash
python ingest.py
```

Das Skript liest alle PDF-Dateien aus `daten/`, die Paragrafen des SGB XI sowie geprüfte Webseiten
ein und legt sie in der Vektordatenbank ab. Am Ende steht ein Prüfbericht, der für jedes Dokument zeigt, wie
viel Text gewonnen wurde – so lässt sich belegen, dass jede Quelle vollständig verarbeitet wurde.

Zum Nachschauen, was in der Datenbank gelandet ist:

```bash
python check_db.py
```

---

## 4. Starten

Drei Dinge müssen laufen: die Vektordatenbank, der Hintergrunddienst und die Oberfläche.

```bash
docker compose up -d
```

```bash
.venv\Scripts\python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

```bash
npm run dev --prefix frontend
```

Danach im Browser: <http://localhost:5173>

> Alle drei lauschen bewusst **nur örtlich** (`127.0.0.1`). Die Pflegeunterlagen sind dadurch nicht
> aus dem Netzwerk erreichbar. Der erste Start des Dienstes dauert etwa eine Minute, weil das
> Modell für die Neubewertung geladen wird.

Zum Beenden:

```bash
docker compose down
```

---

## 5. Aufbau des Projekts

```
frontend/         Weboberfläche (React, TypeScript, Vite): vier Reiter, Bedienführung
  src/api.ts        Anbindung an den Dienst, einschließlich Ereignisstrom für den Chat
  src/components/   Startseite, Unterlagen, Chat, PDF, Einstellungen
backend.py        Dienst (FastAPI): Zugangscodes, verschlüsselte Ablage, Schnittstellen
pflege_service.py Fachlogik zwischen Dienst und Suche: Dokumentenaufbereitung, Antwortstrom
pflege_rag.py     Suche und Antworterzeugung: hybride Suche, Neubewertung, Belegstellen
pflege_pdf.py     Erzeugung des Widerspruchsschreibens als Geschäftsbrief nach DIN 5008
ingest.py         Aufbau der Wissensdatenbank aus daten/, SGB XI und geprüften Webseiten
main.py           Diagnose auf der Kommandozeile: zeigt Treffer, Bewertungen und Belege
check_db.py       Übersicht über den Inhalt der Wissensdatenbank
tests/            Testreihe (pytest): Suche, Textaufbereitung, Briefaufbau
docker-compose.yml  Vektordatenbank als örtlicher Dienst
daten/            Fachdokumente für die Wissensdatenbank (amtliche Quellen)
daten/privat/     Unterlagen einzelner Personen – wird nie versioniert
beispielfall/     erfundener Fall zum Ausprobieren (Bescheid, Gutachten, Tagebuch …)
```

### Zum Ausprobieren

Der Ordner `beispielfall/` enthält einen vollständig erfundenen Fall: Bescheid einer
Muster-Pflegekasse, Gutachten des Medizinischen Dienstes, Pflegetagebuch, ärztlicher
Befundbericht und ein Auszug aus der Patientenakte. Die fünf Dateien lassen sich im Reiter
*Unterlagen* hochladen, danach sind alle Aufgaben des Assistenten benutzbar – ohne dass echte
Gesundheitsdaten nötig sind. Die Dateien gehören **nicht** in `daten/`: Dort steht das
Fachwissen, das der Assistent zitiert.

Die Testreihe läuft ohne LM Studio und ohne Docker:

```bash
python -m pytest tests/ -q
```

### Wie eine Antwort entsteht

```
Frage
  ├─ Vektorsuche (bge-m3)        findet sinnverwandte Stellen
  └─ Stichwortsuche (BM25)       findet exakte Begriffe wie "Modul 4" oder "§ 18"
        └─ Rangfusion            verschmilzt beide Trefferlisten
              └─ Neubewertung    Cross-Encoder bewertet jeden Treffer gegen die Frage;
                                 inhaltsleere Stellen erhalten 0 und fallen weg
                    └─ Antwort   nummerierte Abschnitte an das Sprachmodell,
                                 das seine Aussagen mit [1], [2] belegt
```

Bei Aufgaben über alle sechs Begutachtungsmodule wird je Modul einzeln gesucht und bewertet.
Ohne das gewinnen zwei allgemeine Abschnitte die Rangfolge, und das Sprachmodell füllt die
verbleibenden Module mit Erfundenem.

---

## 6. Datenschutz

* Keine Registrierung. Eine Sitzung wird nur über einen zufälligen Zugangscode angesprochen.
* Gespeicherte Sitzungsdaten sind mit Fernet verschlüsselt.
* Hochgeladene Dateien werden für die Umwandlung kurz in eine temporäre Datei geschrieben und
  danach **überschrieben**, bevor sie gelöscht werden.
* Gelöschte Sitzungen werden über `secure_delete` und ein Neuschreiben der Datenbankdatei
  tatsächlich entfernt, nicht nur als frei markiert.
* Sitzungen laufen nach 4 Wochen ab und werden selbsttätig gelöscht; eine Verlängerung um 3 Tage
  ist jederzeit möglich.
* Die Vektordatenbank ist an `127.0.0.1` gebunden und ihre Telemetrie ist abgeschaltet
  (`docker-compose.yml`). Qdrant verlangt von sich aus **kein** Passwort – ohne diese Bindung wäre
  sie aus dem gesamten Netzwerk offen.
* Die Vektoren der hochgeladenen Unterlagen liegen ausschließlich im Arbeitsspeicher und werden
  nie auf die Festplatte geschrieben.

---

## 7. Bekannte Einschränkungen

* Läuft die Vektordatenbank als Docker-Dienst, können alle Werkzeuge gleichzeitig zugreifen.
  Ohne Docker weicht die Anwendung auf den eingebetteten Betrieb aus; dann darf immer nur ein
  Programm gleichzeitig laufen (`ingest.py`, `main.py`, `check_db.py` oder der Dienst).
* Ein Wechsel der Qdrant-Fassung kann den gespeicherten Bestand unlesbar machen. Dann hilft
  `docker compose down -v`, gefolgt von einem erneuten `python ingest.py`.
* Die erzeugten Texte sind **Entwürfe**. Sie ersetzen keine Rechtsberatung und müssen vor dem
  Absenden geprüft werden. Der Assistent weist auf Textstellen hin, die er nicht belegen konnte.
* Antworten ohne Belegstelle werden mit einer Warnung versehen, weil das Sprachmodell in solchen
  Fällen dazu neigt, Angaben aus eigenem Wissen zu ergänzen.

---

## 8. Lizenz und Quellen

Die Dateien in `daten/` stammen aus öffentlich zugänglichen Veröffentlichungen des Medizinischen
Dienstes Bund, des Bundesgesundheitsministeriums, des GKV-Spitzenverbandes, der Verbraucherzentrale
und des VdK. Sie werden hier ausschließlich zu Studienzwecken verwendet; die Rechte liegen bei den
jeweiligen Herausgebern.
