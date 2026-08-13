"""Dienstschicht für Suche, Antworten und Dokumentenverarbeitung.

Bündelt alles, was früher im Prozess der Oberfläche lag, an einer Stelle. Dadurch

* hält nur noch ein Prozess die Vektordatenbank (die Ein-Prozess-Sperre stört
  im Betrieb nicht mehr),
* lassen sich Suche und Antworterzeugung ohne Oberfläche testen,
* spricht die Oberfläche den Dienst ausschließlich über HTTP an und kann
  ausgetauscht werden, ohne die Fachlogik anzufassen.

Die schweren Bestandteile (Wissensdatenbank, Neubewertungsmodell,
PDF-Umwandler) werden einmalig geladen und wiederverwendet.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

from langchain_core.documents import Document

import pflege_rag


# ---------------------------------------------------------------------------
# EINMALIG GELADENE RESSOURCEN
# ---------------------------------------------------------------------------
class _Ressourcen:
    """Lädt Modelle und Wissensdatenbank beim ersten Bedarf, danach nie wieder.

    Die Sperre verhindert, dass zwei gleichzeitige Anfragen dasselbe Modell
    doppelt laden - das würde beim Reranker mehrere Gigabyte kosten.
    """

    def __init__(self) -> None:
        self._sperre = threading.Lock()
        self._embeddings = None
        self._llm = None
        self._reranker = None
        self._expert_index: Optional[pflege_rag.HybridIndex] = None
        self._converters = None

    def embeddings(self):
        with self._sperre:
            if self._embeddings is None:
                self._embeddings = pflege_rag.create_embeddings()
            return self._embeddings

    def llm(self):
        with self._sperre:
            if self._llm is None:
                self._llm = pflege_rag.create_llm()
            return self._llm

    def reranker(self):
        with self._sperre:
            if self._reranker is None:
                try:
                    self._reranker = pflege_rag.create_reranker()
                except Exception as fehler:
                    print(f"Neubewertung nicht verfügbar: {type(fehler).__name__}")
                    self._reranker = False  # merken, damit nicht bei jeder Anfrage neu versucht wird
            return self._reranker or None

    def expert_index(self) -> Optional[pflege_rag.HybridIndex]:
        with self._sperre:
            if self._expert_index is None:
                try:
                    speicher = pflege_rag.open_expert_database(self.embeddings_ohne_sperre())
                    abschnitte = pflege_rag.load_all_expert_chunks(speicher)
                    self._expert_index = pflege_rag.HybridIndex(speicher, abschnitte)
                except Exception as fehler:
                    print(f"Wissensdatenbank nicht lesbar: {fehler}")
                    return None
            return self._expert_index

    def embeddings_ohne_sperre(self):
        """Nur innerhalb eines bereits gehaltenen Sperrbereichs aufrufen."""
        if self._embeddings is None:
            self._embeddings = pflege_rag.create_embeddings()
        return self._embeddings

    def converters(self):
        """PDF-Umwandler: schnell für digitale Dateien, mit Texterkennung für Scans."""
        with self._sperre:
            if self._converters is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                schnell = PdfPipelineOptions()
                schnell.do_ocr = False
                schnell.do_table_structure = True
                mit_ocr = PdfPipelineOptions()
                mit_ocr.do_ocr = True
                mit_ocr.do_table_structure = True
                self._converters = (
                    DocumentConverter(
                        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=schnell)}),
                    DocumentConverter(
                        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=mit_ocr)}),
                )
            return self._converters

    def wissensbasis_umfang(self) -> int:
        index = self.expert_index()
        return len(index.documents) if index else 0


ressourcen = _Ressourcen()


# ---------------------------------------------------------------------------
# SUCHINDEX DER NUTZERDOKUMENTE
# ---------------------------------------------------------------------------
@dataclass
class UserIndexEintrag:
    """Suchindex einer Sitzung, ausschließlich im Arbeitsspeicher."""

    index: Optional[pflege_rag.HybridIndex]
    fingerabdruck: str


class UserIndexCache:
    """Hält die Suchindizes der laufenden Sitzungen vor.

    Der Index wird nur neu gebaut, wenn sich die Dokumente tatsächlich geändert
    haben. Ohne diese Prüfung würde jede Anfrage sämtliche Abschnitte erneut
    einbetten - bei über hundert Abschnitten dauert das mehrere Sekunden.
    """

    def __init__(self) -> None:
        self._sperre = threading.Lock()
        self._eintraege: dict[str, UserIndexEintrag] = {}

    @staticmethod
    def _fingerabdruck(dokumente: Sequence[Document]) -> str:
        return f"{len(dokumente)}:{hash(tuple(d.page_content[:80] for d in dokumente))}"

    def hole(self, token: str, dokumente: Sequence[Document]) -> Optional[pflege_rag.HybridIndex]:
        if not dokumente:
            self.entferne(token)
            return None

        abdruck = self._fingerabdruck(dokumente)
        with self._sperre:
            vorhanden = self._eintraege.get(token)
            if vorhanden is not None and vorhanden.fingerabdruck == abdruck:
                return vorhanden.index

        # Der Aufbau geschieht außerhalb der Sperre; er dauert und würde sonst
        # alle anderen Anfragen blockieren.
        speicher = pflege_rag.build_user_vector_store(list(dokumente), ressourcen.embeddings())
        index = pflege_rag.HybridIndex(speicher, list(dokumente)) if speicher else None

        with self._sperre:
            self._eintraege[token] = UserIndexEintrag(index=index, fingerabdruck=abdruck)
        return index

    def entferne(self, token: str) -> None:
        """Gibt den Arbeitsspeicher einer beendeten Sitzung frei."""
        with self._sperre:
            self._eintraege.pop(token, None)

    def anzahl(self) -> int:
        with self._sperre:
            return len(self._eintraege)


user_indices = UserIndexCache()


# ---------------------------------------------------------------------------
# DOKUMENTENVERARBEITUNG
# ---------------------------------------------------------------------------
@dataclass
class UploadErgebnis:
    """Was beim Einlesen einer Datei herauskam."""

    dateiname: str
    erfolgreich: bool
    abschnitte: int = 0
    hinweis: str = ""


def verarbeite_pdf(inhalt: bytes, dateiname: str) -> tuple[List[Document], UploadErgebnis]:
    """Wandelt eine hochgeladene PDF-Datei in durchsuchbare Abschnitte um."""
    schnell, mit_ocr = ressourcen.converters()
    try:
        dokument = pflege_rag.extract_document_from_pdf(
            inhalt, dateiname, schnell, ocr_converter=mit_ocr
        )
    except Exception:
        # Kein technischer Fehlertext nach außen: er könnte Inhalte preisgeben.
        return [], UploadErgebnis(dateiname, False,
                                  hinweis="Die Datei konnte nicht gelesen werden. Ist sie beschädigt?")

    if dokument is None:
        return [], UploadErgebnis(
            dateiname, False,
            hinweis="Kein lesbarer Text gefunden. Bei eingescannten Unterlagen hilft eine "
                    "bessere Scanqualität.",
        )

    abschnitte = pflege_rag.split_documents([dokument])
    if not abschnitte:
        return [], UploadErgebnis(dateiname, False, hinweis="Die Datei enthält keinen verwertbaren Text.")
    return abschnitte, UploadErgebnis(dateiname, True, abschnitte=len(abschnitte))


# ---------------------------------------------------------------------------
# ANTWORTERZEUGUNG
# ---------------------------------------------------------------------------
@dataclass
class AntwortErgebnis:
    """Ergebnis einer Anfrage an den Assistenten."""

    antwort: str = ""
    quellen: List[dict] = field(default_factory=list)
    suchfrage: str = ""
    umformuliert: bool = False
    ohne_beleg: bool = False


OHNE_BELEG_HINWEIS = (
    "> ⚠️ **Diese Antwort stützt sich auf keine Textstelle** aus Ihren Unterlagen oder dem "
    "geprüften Fachwissen. Sie kann daher sachlich falsch sein – besonders bei Zuständigkeiten, "
    "Fristen und Punktzahlen. Bitte prüfen Sie sie unbedingt nach oder laden Sie passendere "
    "Unterlagen hoch.\n\n"
)

# Eine übernommene Kopfzeile kann sich über zwei Zeilen erstrecken
# ("----- [1] -----" und darunter "Herkunft: ..."). So viele Zeilen hält der
# Strom deshalb zurück, bevor er Text weitergibt.
ZURUECKGEHALTENE_ZEILEN = 2


def _zeilenschnitt(text: str, zurueckhalten: int) -> int:
    """Gibt die Stelle zurück, bis zu der gefahrlos gesendet werden darf.

    Das ist das Ende der letzten Zeile, die noch ``zurueckhalten`` vollständige
    Zeilen hinter sich hat. Gibt es so viele Zeilen noch nicht, wird nichts
    gesendet.
    """
    schnitt = len(text)
    for _ in range(zurueckhalten):
        schnitt = text.rfind("\n", 0, schnitt)
        if schnitt == -1:
            return 0
    return schnitt + 1


def beantworte(
    token: str,
    frage: str,
    anweisung: str,
    verlauf: Sequence[dict],
    dokumente: Sequence[Document],
    zusatzfragen: Sequence[str] = (),
) -> Iterator[dict]:
    """Beantwortet eine Frage und liefert den Fortschritt stückweise.

    Erzeugt eine Folge von Meldungen, die eine Oberfläche unmittelbar anzeigen
    kann: erst der Bearbeitungsstand, dann die Textstücke der Antwort, zuletzt
    das Ergebnis mit den Quellen.
    """
    yield {"art": "status", "text": "Der Assistent wird vorbereitet …"}
    reranker = ressourcen.reranker()

    # Anschlussfragen ohne eigenen Inhalt zuerst auflösen.
    eigenstaendig = pflege_rag.condense_question(ressourcen.llm(), frage, verlauf)
    if eigenstaendig != frage:
        yield {"art": "suchfrage", "text": eigenstaendig}

    yield {"art": "status", "text": "Ihre Unterlagen und das Fachwissen werden durchsucht …"}
    ergebnis = pflege_rag.prepare_context(
        ressourcen.expert_index(),
        user_indices.hole(token, dokumente),
        eigenstaendig,
        reranker=reranker,
        extra_queries=zusatzfragen,
    )

    yield {"art": "status", "text": "Die Antwort wird geschrieben …"}
    nachrichten = pflege_rag.build_messages(
        ergebnis.system_prompt, list(verlauf) + [{"role": "user", "content": anweisung}]
    )

    gesammelt = ""
    gesendet = 0
    for teil in pflege_rag.stream_answer(ressourcen.llm(), nachrichten):
        gesammelt += teil
        # Die letzten Zeilen werden zurückgehalten: eine Kopfzeile lässt sich
        # erst erkennen, wenn sie vollständig da ist. Ohne das blitzt sie im
        # Gesprächsfenster auf, bevor sie wieder verschwindet.
        sauber = pflege_rag.strip_context_headers(
            gesammelt[: _zeilenschnitt(gesammelt, ZURUECKGEHALTENE_ZEILEN)]
        )
        if len(sauber) > gesendet:
            yield {"art": "text", "text": sauber[gesendet:]}
            gesendet = len(sauber)

    gesammelt = pflege_rag.strip_context_headers(gesammelt)
    if len(gesammelt) > gesendet:
        yield {"art": "text", "text": gesammelt[gesendet:]}

    verwendet = set(pflege_rag.cited_numbers(gesammelt)) & set(ergebnis.nummern)
    angezeigt = pflege_rag.render_citations(gesammelt, ergebnis.nummern)
    if not verwendet:
        angezeigt = OHNE_BELEG_HINWEIS + angezeigt

    quellen = [
        q.als_dict() for q in ergebnis.quellen if not verwendet or q.nummer in verwendet
    ]
    yield {
        "art": "ergebnis",
        "antwort": angezeigt,
        "quellen": quellen,
        "suchfrage": eigenstaendig,
        "umformuliert": eigenstaendig != frage,
        "ohne_beleg": not verwendet,
    }
