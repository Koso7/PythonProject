"""Aufbau der Wissensdatenbank aus Fachdokumenten und geprüften Webseiten.

Aufruf:  python ingest.py

Das Skript liest alle PDF-Dateien aus ``daten/`` sowie die unten aufgeführten
Webseiten ein, zerlegt sie in Abschnitte und legt sie in der Vektordatenbank ab.
Am Ende steht ein Prüfbericht, der für jedes Dokument zeigt, wie viel Text
gewonnen wurde und wie viele Abschnitte daraus entstanden sind. So lässt sich
belegen, dass wirklich jedes Dokument vollständig gelesen wurde.

Hinweis: Solange die Weboberfläche läuft, ist die Vektordatenbank gesperrt.
Bitte vor dem Aufruf beenden.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("USER_AGENT", "PflegeAssistent/2.0 (Universitaeres Studienprojekt)")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

import pflege_rag

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./daten")
QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
BATCH_SIZE = 32

# Unterlagen einzelner Personen gehören NICHT in die gemeinsame Wissensdatenbank.
# Sie würden sonst als allgemeines "Fachwissen" in den Antworten anderer
# Ratsuchender auftauchen und deren Fall mit fremden Angaben vermischen.
# Solche Dateien eignen sich weiterhin zum Testen des Uploads in der Oberfläche.
# Geprüft am 2026-08-13 durch Texterkennung auf Stichproben:
# MD_Sachsen-Anhalt_Gesundheitsdaten_REAL.pdf ist NICHT anonymisiert. Auf
# Seite 11 stehen im Klartext zwei vollständige Namen, zwei Geburtsdaten und
# eine Wohnadresse, dazu Pflegegrad und Pflegeaufwand. Diese Datei darf die
# Wissensbasis unter keinen Umständen erreichen.
EXCLUDED_FILES = {
    "Gutachten3.pdf",
    "MD_Sachsen-Anhalt_Gesundheitsdaten_REAL.pdf",
}

# Gesetzestexte im Wortlaut, vom amtlichen Dienst des Bundesjustizministeriums.
# Damit kann der Assistent Paragrafen zitieren, statt sie nur aus Ratgebertexten
# zu umschreiben.
GESETZ_URL = "https://www.gesetze-im-internet.de/{buch}/__{nummer}.html"

# Die maßgeblichen Paragrafen im Wortlaut, je Gesetzbuch.
#
# Bis 2026-08-13 waren nur §§ 14, 15 und 18 SGB XI enthalten - also allein die
# Begutachtung. Fragen zum Widerspruchsverfahren selbst (Frist, Form, Kosten,
# Klage) ließen sich damit nicht belegen, obwohl die Anwendung genau dafür da
# ist und die Frist sogar ausrechnet. Deshalb kommen das Sozialgerichtsgesetz
# und das Verwaltungsverfahrensrecht des SGB X hinzu.
GESETZE: dict[str, tuple[str, list[str]]] = {
    # Pflegeversicherung: Begriff, Begutachtung, Leistungen
    "SGB XI": ("sgb_11", [
        "7a",   # Pflegeberatung
        "14",   # Begriff der Pflegebedürftigkeit
        "15",   # Ermittlung des Grades der Pflegebedürftigkeit
        "17",   # Richtlinien der Pflegekassen
        "18",   # Begutachtung und Feststellung
        "33",   # Leistungsvoraussetzungen (Antrag)
        "36",   # Pflegesachleistung
        "37",   # Pflegegeld
        "38",   # Kombinationsleistung
        "39",   # Verhinderungspflege
        "40",   # Pflegehilfsmittel
        "41",   # Tagespflege und Nachtpflege
        "42",   # Kurzzeitpflege
        "43",   # Vollstationäre Pflege
        "45b",  # Entlastungsbetrag
    ]),
    # Sozialgerichtsgesetz: das Widerspruchsverfahren
    "SGG": ("sgg", [
        "78",   # Vorverfahren - Nachprüfung des Verwaltungsakts
        "83",   # Beginn des Vorverfahrens
        "84",   # Widerspruchsfrist
        "85",   # Widerspruchsbescheid
        "87",   # Klagefrist
        "88",   # Untätigkeitsklage
    ]),
    # SGB X: Verwaltungsverfahren, u. a. Bekanntgabe und Kosten
    "SGB X": ("sgb_10", [
        "37",   # Bekanntgabe - maßgeblich für den Fristbeginn
        "41",   # Heilung von Verfahrensfehlern
        "44",   # Rücknahme eines rechtswidrigen Verwaltungsakts
        "63",   # Erstattung der Kosten im Vorverfahren
    ]),
}

URLS_TO_LEARN = [
    # --- Amtliche Stellen ---
    "https://www.bundesgesundheitsministerium.de/pflegegrade.html",
    "https://www.bundesgesundheitsministerium.de/themen/pflege/online-ratgeber-pflege/pflegebeduerftig-was-nun",
    "https://md-bund.de/themen/pflegebeduerftigkeit-und-pflegebegutachtung/das-begutachtungsinstrument.html",
    "https://md-bund.de/themen/pflegebeduerftigkeit-und-pflegebegutachtung/begutachtungs-richtlinien.html",
    "https://md-bund.de/themen/pflegebeduerftigkeit-und-pflegebegutachtung/fragen-und-antworten.html",
    # --- Verbraucherschutz und Sozialverbände ---
    "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/pflegegrad-abgelehnt-so-wehren-sie-sich-mit-widerspruch-und-klage-11547",
    "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/pflegegrad-beantragen-so-gehts-13413",
    "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/begutachtung-durch-medizinischen-dienst-so-koennen-sie-sich-vorbereiten-13414",
    "https://www.vdk.de/aktuelles/aktuelle-meldungen/artikel/widerspruch-gegen-pflegegrad-lohnt-sich-oft/",
    "https://www.betanet.de/pflegetagebuch.html",
    # --- Fachportale ---
    "https://www.pflege.de/pflegekasse-pflegerecht/pflegegrade/widerspruch/",
    "https://www.pflege.de/pflegekasse-pflegerecht/pflegegrade/beantragen/",
    "https://www.pflege.de/pflegende-angehoerige/pflegefall/pflegetagebuch/",
    "https://www.pflegeberatung.de/pflegeanspruch/begutachtung/das-begutachtungsinstrument",
    "https://www.pflege-betreuer.de/de/pflegewissen/pflegerecht-und-ansprueche/widerspruch-gegen-die-pflegegrad-einstufung-einlegen",

    # --- Ergänzt am 2026-08-13 ---------------------------------------------
    # Eine Prüfbatterie freier Fragen (pruefe_antworten.py) zeigte Lücken bei
    # den Themen Ablauf des Widerspruchsverfahrens, Kosten, Klage, Leistungen
    # und Höherstufung. Diese Seiten decken genau das ab.
    "https://www.vdk.de/deutschland/pages/themen/pflege/85737/pflegegrad_abgelehnt_-_keine_angst_vor_widerspruch",
    "https://www.betanet.de/pflegeantrag-und-pflegebegutachtung.html",
    "https://www.betanet.de/widerspruch-im-sozialrecht.html",
    "https://www.betanet.de/pflege-leistungen.html",
    "https://www.bundesgesundheitsministerium.de/themen/pflege/online-ratgeber-pflege/pflegebeduerftig-was-nun/pflegebeduerftigkeit",
    "https://www.pflege.de/pflegekasse-pflegerecht/pflegeleistungen/",
    "https://www.bundesgesundheitsministerium.de/themen/pflege/online-ratgeber-pflege/leistungen-der-pflegeversicherung/leistungen-im-ueberblick",
]


@dataclass
class IngestReport:
    """Prüfergebnis für ein einzelnes Dokument."""

    name: str
    seiten: int = 0
    zeichen: int = 0
    abschnitte: int = 0
    ocr: bool = False
    sekunden: float = 0.0
    fehler: str = ""

    @property
    def ok(self) -> bool:
        return not self.fehler and self.abschnitte > 0


def build_converters() -> tuple[DocumentConverter, DocumentConverter]:
    """Erzeugt zwei Umwandler: einen schnellen und einen mit Texterkennung.

    Digitale PDFs brauchen keine Texterkennung und werden dadurch um ein
    Vielfaches schneller eingelesen. Eingescannte Unterlagen brauchen sie
    zwingend, sonst bleibt ihr Inhalt unsichtbar.
    """
    schnell = PdfPipelineOptions()
    schnell.do_ocr = False
    schnell.do_table_structure = True

    mit_ocr = PdfPipelineOptions()
    mit_ocr.do_ocr = True
    mit_ocr.do_table_structure = True

    return (
        DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=schnell)}),
        DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=mit_ocr)}),
    )


def zaehle_seiten(pfad: str) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(pfad).pages)
    except Exception:
        return 0


def lade_pdf_dokumente() -> tuple[List[Document], List[IngestReport]]:
    """Liest alle PDF-Dateien aus dem Datenordner ein."""
    if not os.path.isdir(DATA_DIR):
        print(f"Ordner {DATA_DIR} nicht gefunden.")
        return [], []

    alle = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))
    dateien = [f for f in alle if f not in EXCLUDED_FILES]
    for uebersprungen in sorted(set(alle) & EXCLUDED_FILES):
        print(f"      übersprungen (Unterlage einer einzelnen Person): {uebersprungen}")
    if not dateien:
        return [], []

    schnell, mit_ocr = build_converters()
    dokumente: List[Document] = []
    berichte: List[IngestReport] = []

    for nummer, dateiname in enumerate(dateien, start=1):
        pfad = os.path.join(DATA_DIR, dateiname).replace("\\", "/")
        bericht = IngestReport(name=dateiname, seiten=zaehle_seiten(pfad))
        bericht.ocr = not pflege_rag.pdf_has_text_layer(pfad)

        art = "Texterkennung" if bericht.ocr else "digital"
        print(f"[{nummer}/{len(dateien)}] {dateiname} ({bericht.seiten} Seiten, {art}) …", flush=True)

        start = time.time()
        try:
            umwandler = mit_ocr if bericht.ocr else schnell
            ergebnis = umwandler.convert(pfad)
            markdown = pflege_rag.clean_text(ergebnis.document.export_to_markdown())
            bericht.zeichen = len(markdown)

            if bericht.zeichen < 200:
                bericht.fehler = "Kaum Text gewonnen – vermutlich Bildqualität zu schlecht."
            else:
                dokumente.append(
                    Document(
                        page_content=markdown,
                        metadata={
                            "source": dateiname,
                            "document_type": "fachdokument",
                            "doc_kind": pflege_rag.classify_document(dateiname),
                        },
                    )
                )
        except Exception as fehler:
            bericht.fehler = f"{type(fehler).__name__}: {str(fehler)[:90]}"

        bericht.sekunden = time.time() - start
        print(f"      {bericht.zeichen} Zeichen in {bericht.sekunden:.0f}s"
              f"{' – ' + bericht.fehler if bericht.fehler else ''}", flush=True)
        berichte.append(bericht)

    return dokumente, berichte


def lade_gesetzestexte() -> tuple[List[Document], List[IngestReport]]:
    """Holt die maßgeblichen Paragrafen im Wortlaut.

    Als Quelle wird der amtliche Dienst des Bundesjustizministeriums verwendet.
    Jeder Paragraf wird ein eigenes Dokument, damit die Quellenangabe später
    "§ 15 SGB XI" lautet und nicht bloß einen Dateinamen nennt.
    """
    import requests
    from bs4 import BeautifulSoup

    dokumente: List[Document] = []
    berichte: List[IngestReport] = []

    auftraege = [
        (gesetz, buch, nummer)
        for gesetz, (buch, paragrafen) in GESETZE.items()
        for nummer in paragrafen
    ]

    for gesetz, buch, nummer in auftraege:
        bezeichnung = f"§ {nummer} {gesetz}"
        bericht = IngestReport(name=bezeichnung)
        start = time.time()
        try:
            antwort = requests.get(GESETZ_URL.format(buch=buch, nummer=nummer), timeout=30)
            antwort.encoding = "utf-8"
            suppe = BeautifulSoup(antwort.text, "html.parser")
            inhalt = suppe.find("div", class_="jnhtml") or suppe.find("div", id="paddingLR12")
            text = pflege_rag.clean_text(inhalt.get_text("\n", strip=True)) if inhalt else ""
            bericht.zeichen = len(text)

            # Nicht über die Länge prüfen, sondern über die Kennzeichnung:
            # § 83 SGG besteht aus einem einzigen Satz mit 59 Zeichen und wäre
            # jeder vernünftigen Untergrenze zum Opfer gefallen. Der Dienst
            # nennt die Paragrafennummer in einem eigenen Element - stimmt sie
            # mit der angeforderten überein, ist die richtige Seite geladen.
            kennung = suppe.find("span", class_="jnenbez")
            gefunden = kennung.get_text(strip=True) if kennung else ""
            if not text or gefunden != f"§ {nummer}":
                bericht.fehler = (
                    f"Erwartet „§ {nummer}“, gefunden „{gefunden or 'nichts'}“ – "
                    "Aufbau der Seite geändert?"
                )
            else:
                # Die Paragrafenüberschrift ("Widerspruchsfrist") steht in
                # einem eigenen Element; fehlt sie, genügt die Bezeichnung.
                titelmarke = suppe.find("span", class_="jnentitel")
                titel = (titelmarke.get_text(strip=True) if titelmarke else "") or bezeichnung
                dokumente.append(
                    Document(
                        page_content=f"# {bezeichnung} – {titel}\n\n{text}",
                        metadata={
                            "source": bezeichnung,
                            "document_type": "gesetz",
                            "doc_kind": "Gesetzestext",
                        },
                    )
                )
        except Exception as fehler:
            bericht.fehler = f"{type(fehler).__name__}: {str(fehler)[:90]}"

        bericht.sekunden = time.time() - start
        print(f"      {bezeichnung}: {bericht.fehler or str(bericht.zeichen) + ' Zeichen'}", flush=True)
        berichte.append(bericht)

    return dokumente, berichte


def lade_webseiten() -> tuple[List[Document], List[IngestReport]]:
    """Liest die geprüften Webseiten ein."""
    dokumente: List[Document] = []
    berichte: List[IngestReport] = []

    for nummer, url in enumerate(URLS_TO_LEARN, start=1):
        kurz = url.split("//")[-1][:64]
        bericht = IngestReport(name=url)
        start = time.time()
        try:
            # Erst den Statuscode prüfen, dann den Inhalt holen.
            #
            # Ohne diese Prüfung landen Fehlerseiten als Fachwissen in der
            # Datenbank: Beim Ergänzen neuer Adressen am 2026-08-13 lieferten
            # vier von neun eine 404 - mit 846 bis 8136 Zeichen Text. Die
            # längeren davon hätten die bisherige Mindestlänge mühelos
            # überschritten und wären als Quelle zitierbar gewesen.
            import requests
            kopf = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            antwort = requests.head(url, timeout=25, headers=kopf, allow_redirects=True)
            if antwort.status_code >= 400:
                # Manche Server beantworten HEAD nicht richtig; dann noch einmal
                # richtig anfragen, bevor die Seite verworfen wird.
                antwort = requests.get(url, timeout=25, headers=kopf)
            if antwort.status_code >= 400:
                raise RuntimeError(f"HTTP {antwort.status_code}")

            geladen = WebBaseLoader(url).load()
            text = pflege_rag.clean_text("\n".join(d.page_content for d in geladen))
            bericht.zeichen = len(text)
            if bericht.zeichen < 400:
                bericht.fehler = "Zu wenig Inhalt – Seite vermutlich nicht abrufbar."
            else:
                dokumente.append(
                    Document(
                        page_content=text,
                        metadata={"source": url, "document_type": "webseite", "doc_kind": "Webseite"},
                    )
                )
        except Exception as fehler:
            bericht.fehler = f"{type(fehler).__name__}: {str(fehler)[:90]}"

        bericht.sekunden = time.time() - start
        status = bericht.fehler or f"{bericht.zeichen} Zeichen"
        print(f"[{nummer}/{len(URLS_TO_LEARN)}] {kurz} – {status}", flush=True)
        berichte.append(bericht)

    return dokumente, berichte


def drucke_pruefbericht(berichte: List[IngestReport]) -> None:
    """Zeigt für jedes Dokument, ob es vollständig verarbeitet wurde."""
    print("\n" + "=" * 96)
    print("PRÜFBERICHT – wurde jedes Dokument vollständig gelesen?")
    print("=" * 96)
    print(f"{'Dokument':<52}{'Seiten':>7}{'Zeichen':>10}{'Abschn.':>9}{'OCR':>5}{'Zeit':>7}  Status")
    print("-" * 96)
    for bericht in berichte:
        name = bericht.name if len(bericht.name) <= 50 else bericht.name[:47] + "..."
        status = "OK" if bericht.ok else (bericht.fehler or "keine Abschnitte")
        seiten = str(bericht.seiten) if bericht.seiten else "-"
        print(f"{name:<52}{seiten:>7}{bericht.zeichen:>10}{bericht.abschnitte:>9}"
              f"{'ja' if bericht.ocr else '-':>5}{bericht.sekunden:>6.0f}s  {status}")
    print("-" * 96)
    fehlerhaft = [b for b in berichte if not b.ok]
    print(f"{len(berichte) - len(fehlerhaft)} von {len(berichte)} Quellen erfolgreich verarbeitet.")
    if fehlerhaft:
        print("\nNicht verwertbar:")
        for bericht in fehlerhaft:
            print(f"  - {bericht.name}: {bericht.fehler or 'keine Abschnitte erzeugt'}")


def build_expert_database() -> int:
    print("=" * 96)
    print("AUFBAU DER WISSENSDATENBANK")
    print("=" * 96)

    print("\n--- PDF-Dokumente ---")
    pdf_docs, pdf_berichte = lade_pdf_dokumente()

    print("\n--- Gesetzestexte ---")
    gesetz_docs, gesetz_berichte = lade_gesetzestexte()

    print("\n--- Webseiten ---")
    web_docs, web_berichte = lade_webseiten()

    alle_docs = pdf_docs + gesetz_docs + web_docs
    if not alle_docs:
        print("\nKeine verwertbaren Quellen gefunden. Abbruch.")
        return 1

    print("\n--- Abschnitte bilden ---")
    berichte = pdf_berichte + gesetz_berichte + web_berichte
    nach_quelle: dict[str, int] = {}
    alle_chunks: List[Document] = []
    for doc in alle_docs:
        chunks = pflege_rag.split_documents([doc])
        nach_quelle[doc.metadata["source"]] = len(chunks)
        alle_chunks.extend(chunks)
    for bericht in berichte:
        bericht.abschnitte = nach_quelle.get(bericht.name, 0)

    print(f"{len(alle_chunks)} verwertbare Abschnitte aus {len(alle_docs)} Quellen.")

    print("\n--- Einbettungen berechnen und speichern ---")
    print(f"Vektordatenbank: {pflege_rag.qdrant_betriebsart()}")
    embeddings = pflege_rag.create_embeddings()
    vektorgroesse = len(embeddings.embed_query("Test"))

    if not pflege_rag.QDRANT_URL and os.path.exists(QDRANT_DIR):
        # Eingebetteter Betrieb: das Verzeichnis wird vollständig neu angelegt.
        shutil.rmtree(QDRANT_DIR)

    client = pflege_rag.create_qdrant_client()
    if pflege_rag.QDRANT_URL:
        # Im Dienstbetrieb bleibt der Speicher bestehen; nur die Sammlung wird
        # ersetzt, damit kein Altbestand mit neuen Abschnitten vermischt wird.
        client.delete_collection(collection_name=pflege_rag.COLLECTION_NAME)
    client.create_collection(
        collection_name=pflege_rag.COLLECTION_NAME,
        vectors_config=VectorParams(size=vektorgroesse, distance=Distance.COSINE),
    )
    speicher = QdrantVectorStore(
        client=client, collection_name=pflege_rag.COLLECTION_NAME, embedding=embeddings
    )

    gesamt = (len(alle_chunks) - 1) // BATCH_SIZE + 1
    for index in range(0, len(alle_chunks), BATCH_SIZE):
        speicher.add_documents(alle_chunks[index : index + BATCH_SIZE])
        print(f"  Block {index // BATCH_SIZE + 1} von {gesamt} gespeichert.", flush=True)

    client.close()
    drucke_pruefbericht(berichte)
    print("\nWissensdatenbank fertig.")
    return 0


if __name__ == "__main__":
    sys.exit(build_expert_database())
