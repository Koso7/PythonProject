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
| **Python 3.11 oder neuer** | Anwendung und Dienst |
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
```

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

Einmalig, dauert je nach Rechner 10 bis 20 Minuten. **LM Studio muss dabei laufen.**

```bash
python ingest.py
```

Das Skript liest alle PDF-Dateien aus `daten/` sowie geprüfte Webseiten ein und legt sie in der
Vektordatenbank `qdrant_db/` ab. Am Ende steht ein Prüfbericht, der für jedes Dokument zeigt, wie
viel Text gewonnen wurde – so lässt sich belegen, dass jede Quelle vollständig verarbeitet wurde.

Zum Nachschauen, was in der Datenbank gelandet ist:

```bash
python check_db.py
```

---

## 4. Starten

Zwei Programme, zwei Fenster:

```bash
python backend.py
```

```bash
streamlit run app.py
```

Danach im Browser: <http://localhost:8501>

> Beide Dienste lauschen bewusst **nur örtlich**. Die Pflegeunterlagen sind dadurch nicht aus dem
> Netzwerk erreichbar.

---

## 5. Aufbau des Projekts

```
app.py            Weboberfläche (Streamlit): vier Reiter, Bedienführung, Fortschrittsanzeige
backend.py        Dienst für die Sitzungsverwaltung (FastAPI): Zugangscodes, verschlüsselte Ablage
pflege_rag.py     Suche und Antworterzeugung: hybride Suche, Neubewertung, Belegstellen
pflege_pdf.py     Erzeugung des Widerspruchsschreibens nach dem Musterbrief der Verbraucherzentrale
ingest.py         Aufbau der Wissensdatenbank aus daten/ und geprüften Webseiten
main.py           Diagnose auf der Kommandozeile: zeigt Treffer, Bewertungen und Belege
check_db.py       Übersicht über den Inhalt der Wissensdatenbank
daten/            Fachdokumente für die Wissensdatenbank (amtliche Quellen)
daten/privat/     Unterlagen einzelner Personen – wird nie versioniert
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
* Die Nutzungsstatistik von Streamlit ist abgeschaltet (`.streamlit/config.toml`).

---

## 7. Bekannte Einschränkungen

* Die Vektordatenbank kann **nur von einem Prozess gleichzeitig** geöffnet werden. `ingest.py`,
  `main.py` und `check_db.py` verlangen deshalb, dass die Weboberfläche beendet ist.
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
