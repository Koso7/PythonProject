"""Retrieval-Augmented-Generation-Kern des Pflege-Assistenten.

Ablauf einer Anfrage:

    1. Hybride Suche   - Vektorsuche (bge-m3) und Stichwortsuche (BM25) laufen
                         parallel. Die Vektorsuche findet sinnverwandte Stellen,
                         BM25 findet exakte Begriffe wie "Modul 4" oder "§ 18".
    2. Rangfusion      - Beide Trefferlisten werden über Reciprocal Rank Fusion
                         zu einer Kandidatenliste verschmolzen.
    3. Neubewertung    - Ein Cross-Encoder bewertet jeden Kandidaten gegen die
                         Frage. Er erkennt inhaltsleere Stellen wie
                         Tabellenrahmen oder Seitenzahlen und bewertet sie mit 0.
    4. Antwort         - Nur die besten Abschnitte gehen nummeriert an das
                         Sprachmodell, das seine Aussagen mit [1], [2] belegt.

Alle Modelle laufen örtlich: das Sprachmodell und die Einbettungen über
LM Studio, der Cross-Encoder über sentence-transformers auf der CPU.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# KONFIGURATION
# ---------------------------------------------------------------------------
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-nemo-instruct-2407")
QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
COLLECTION_NAME = "pflege_fachwissen"

# Mehrsprachiger Cross-Encoder, passend zur Einbettungsfamilie bge-m3.
# Läuft auf der CPU in rund 1,4 Sekunden für 25 Kandidaten.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
# Einmalig erzeugte ONNX-Fassung für den Betrieb auf der Grafikkarte (rund 2,3 GB).
RERANKER_ONNX_DIR = os.getenv("RERANKER_ONNX_DIR", "./modelle/bge-reranker-v2-m3-onnx")
# Die Abschnitte sind rund 900 Zeichen lang; 384 Token decken sie ab.
RERANKER_MAX_LENGTH = 384

CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 120

# Wie viele Kandidaten jede Suchart liefert und wie viele am Ende übrig bleiben.
DENSE_CANDIDATES = 20
KEYWORD_CANDIDATES = 20
FINAL_EXPERT_CHUNKS = 5
FINAL_USER_CHUNKS = 5
# Bei Aufgaben über alle sechs Module braucht das Sprachmodell mehr Belege,
# sonst bleibt zu einzelnen Modulen nichts übrig und es erfindet Inhalte.
FINAL_USER_CHUNKS_BREIT = 10
FINAL_EXPERT_CHUNKS_BREIT = 6

# Obergrenze für die Neubewertung. Bei Aufgaben mit Zusatzfragen (etwa der
# Differenzanalyse mit sechs Modulen) entstehen sonst mehrere hundert
# Kandidaten, deren Bewertung auf der CPU spürbar Zeit kostet. Die Rangfusion
# hat die aussichtsreichsten Treffer da bereits nach oben sortiert.
MAX_RERANK_CANDIDATES = 40

# Kandidaten unterhalb dieser Bewertung sind für die Frage ohne Aussagekraft.
# Tabellenrahmen und Seitenzahlen erreichen im Test genau 0,000.
RERANK_MIN_SCORE = 0.05

# Begrenzt den mitgesendeten Gesprächsverlauf, damit der Kontext des örtlichen
# Modells nicht überläuft und die gefundenen Belege nicht verdrängt werden.
MAX_HISTORY_MESSAGES = 8

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]

# Häufige Wörter ohne Unterscheidungskraft für die Stichwortsuche.
GERMAN_STOPWORDS = {
    "aber", "alle", "als", "also", "am", "an", "auch", "auf", "aus", "bei", "bin",
    "bis", "das", "dass", "dem", "den", "der", "des", "die", "dies", "diese",
    "diesem", "diesen", "dieser", "durch", "ein", "eine", "einem", "einen",
    "einer", "eines", "er", "es", "für", "hat", "haben", "ich", "im", "in", "ist",
    "kann", "man", "mit", "nach", "nicht", "noch", "nur", "oder", "sich", "sie",
    "sind", "so", "über", "um", "und", "von", "vor", "war", "was", "werden",
    "wenn", "wie", "wird", "zu", "zum", "zur",
}

_UMLAUT_MAP_INIT = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_NORMALIZED_STOPWORDS = {w.translate(_UMLAUT_MAP_INIT) for w in GERMAN_STOPWORDS}

MODULE_NAMES = {
    1: "Mobilität",
    2: "Kognitive und kommunikative Fähigkeiten",
    3: "Verhaltensweisen und psychische Problemlagen",
    4: "Selbstversorgung",
    5: "Bewältigung von krankheits- und therapiebedingten Anforderungen",
    6: "Gestaltung des Alltagslebens und sozialer Kontakte",
}

_WORD_RE = re.compile(r"[a-zäöüß0-9§]+", re.IGNORECASE)
_MODULE_RE = re.compile(r"\bmodul\s*([1-6])\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+")


# ---------------------------------------------------------------------------
# TEXTAUFBEREITUNG UND QUALITÄTSPRÜFUNG
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Entfernt Steuerzeichen, mehrfache Leerzeichen und Leerzeilen."""
    text = text.replace("\x00", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def is_informative(text: str) -> bool:
    """Prüft, ob ein Textabschnitt überhaupt eine Aussage enthält.

    Aussortiert werden reine Tabellenrahmen, Seitenzahlen, Inhaltsverzeichnis-
    Punktlinien und Fragmente ohne zusammenhängende Wörter. Solche Abschnitte
    verstopfen sonst den Kontext und tauchen als sinnlose Quellenangaben auf.
    """
    if not text:
        return False
    kompakt = text.strip()
    if len(kompakt) < MIN_CHUNK_CHARS:
        return False

    buchstaben = sum(1 for zeichen in kompakt if zeichen.isalpha())
    if buchstaben / len(kompakt) < 0.55:
        # Überwiegend Zahlen, Striche oder Trennzeichen.
        return False

    woerter = [w for w in _WORD_RE.findall(kompakt) if len(w) >= 4]
    if len(woerter) < 12:
        return False

    # Inhaltsverzeichnisse bestehen aus Punktketten und Seitenzahlen.
    if kompakt.count("....") > 2 or kompakt.count("· ·") > 2:
        return False

    return True


_UMLAUT_MAP = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def normalize(text: str) -> str:
    """Vereinheitlicht Umlaute für den Textvergleich.

    Texterkennung liefert je nach Scanqualität "Mobilität" oder "Mobilitaet".
    Ohne diese Angleichung würde die Modulerkennung an solchen Schreibweisen
    scheitern.
    """
    return text.lower().translate(_UMLAUT_MAP)


def detect_modules(text: str) -> List[int]:
    """Erkennt, auf welche Begutachtungsmodule sich ein Abschnitt bezieht."""
    gefunden = {int(treffer) for treffer in _MODULE_RE.findall(text)}
    normalisiert = normalize(text)
    for nummer, name in MODULE_NAMES.items():
        # Der Modulname allein ist ebenfalls ein verlässlicher Hinweis.
        hauptbegriff = name.split(" und ")[0].split(" von ")[0].strip()
        if len(hauptbegriff) > 6 and normalize(hauptbegriff) in normalisiert:
            gefunden.add(nummer)
    return sorted(gefunden)


def classify_document(file_name: str) -> str:
    """Ordnet ein Dokument grob ein, damit Quellen verständlich benannt sind."""
    name = file_name.lower()
    if "musterbrief" in name or "widerspruch" in name:
        return "Musterbrief"
    if "richtlinie" in name or "bri" in name or "begutachtungs" in name:
        return "Amtliche Richtlinie"
    if "ratgeber" in name or "bmg" in name or "gkv" in name:
        return "Amtlicher Ratgeber"
    if "gutachten" in name:
        return "Gutachten"
    if "tagebuch" in name:
        return "Pflegetagebuch"
    if "bogen" in name or "fragebogen" in name or "berechnung" in name:
        return "Arbeitshilfe"
    if name.startswith("http"):
        return "Webseite"
    return "Dokument"


def split_documents(documents: Iterable[Document], drop_short: bool = True) -> List[Document]:
    """Teilt Dokumente an Überschriften und danach in überlappende Abschnitte.

    Jeder Abschnitt erhält Metadaten (Dokumentart, betroffene Module,
    Überschrift), die später für Filterung und Quellenanzeige gebraucht werden.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    final_chunks: List[Document] = []
    gesehen = set()
    for doc in documents:
        header_splits = md_splitter.split_text(doc.page_content) or [doc]
        for split in header_splits:
            ueberschrift = split.metadata.get("H3") or split.metadata.get("H2") or split.metadata.get("H1")
            split.metadata.update(doc.metadata)
            if ueberschrift:
                split.metadata["heading"] = ueberschrift

        for chunk in recursive_splitter.split_documents(header_splits):
            inhalt = clean_text(chunk.page_content)
            if drop_short and not is_informative(inhalt):
                continue
            schluessel = (chunk.metadata.get("source", ""), inhalt[:300])
            if schluessel in gesehen:
                continue
            gesehen.add(schluessel)

            chunk.page_content = inhalt
            quelle = chunk.metadata.get("source", "")
            chunk.metadata.setdefault("doc_kind", classify_document(quelle))
            module = detect_modules(inhalt)
            if module:
                chunk.metadata["modules"] = module
            final_chunks.append(chunk)

    return final_chunks


# ---------------------------------------------------------------------------
# PDF-VERARBEITUNG
# ---------------------------------------------------------------------------
def _shred_file(path: str) -> None:
    """Überschreibt eine Datei vor dem Löschen.

    Hochgeladene Unterlagen müssen für die Umwandlung kurz auf die Festplatte.
    Einfaches Löschen gibt nur den Speicherplatz frei, der Inhalt bliebe
    rekonstruierbar.
    """
    try:
        groesse = os.path.getsize(path)
        with open(path, "r+b", buffering=0) as datei:
            datei.write(os.urandom(groesse))
            datei.flush()
            os.fsync(datei.fileno())
    except OSError:
        pass
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def pdf_has_text_layer(
    path: str, stichprobe: int = 12, min_zeichen: int = 100, min_anteil: float = 0.6
) -> bool:
    """Prüft, ob ein PDF durchsuchbaren Text enthält.

    Digitale PDFs lassen sich ohne Texterkennung um ein Vielfaches schneller
    einlesen; eingescannte brauchen sie zwingend, sonst bleibt ihr Inhalt
    unsichtbar.

    Die Stichprobe verteilt sich bewusst über das ganze Dokument. Würde nur der
    Anfang geprüft, gälte ein eingescanntes Gutachten mit maschinell erzeugtem
    Deckblatt fälschlich als digital - und der eigentliche Inhalt ginge
    unbemerkt verloren.
    """
    try:
        from pypdf import PdfReader

        seiten = PdfReader(path).pages
        anzahl = len(seiten)
        if anzahl == 0:
            return False
        schritt = max(1, anzahl // stichprobe)
        auswahl = [seiten[i] for i in range(0, anzahl, schritt)][:stichprobe]
        mit_text = sum(
            1 for seite in auswahl if len((seite.extract_text() or "").strip()) > min_zeichen
        )
        return mit_text >= len(auswahl) * min_anteil
    except Exception:
        # Im Zweifel mit Texterkennung arbeiten - lieber langsam als unvollständig.
        return False


def extract_document_from_pdf(
    file_bytes: bytes, file_name: str, converter, ocr_converter=None
) -> Optional[Document]:
    """Wandelt eine PDF-Datei in ein Textdokument um.

    Ist ``ocr_converter`` angegeben, wird für eingescannte Dateien automatisch
    auf Texterkennung umgeschaltet. Die temporäre Datei wird anschließend
    überschrieben und gelöscht.
    """
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        gewaehlt = converter
        if ocr_converter is not None and not pdf_has_text_layer(tmp_path):
            gewaehlt = ocr_converter

        ergebnis = gewaehlt.convert(tmp_path)
        markdown = clean_text(ergebnis.document.export_to_markdown())
    finally:
        if tmp_path:
            _shred_file(tmp_path)

    if len(markdown) < 30:
        return None
    return Document(
        page_content=markdown,
        metadata={
            "source": file_name,
            "document_type": "nutzerdokument",
            "doc_kind": classify_document(file_name),
        },
    )


# ---------------------------------------------------------------------------
# MODELLE (alles örtlich)
# ---------------------------------------------------------------------------
def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_base=LM_STUDIO_URL,
        openai_api_key="lm-studio",
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )


def create_llm(temperature: float = 0.2, max_tokens: int = 2600) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _dml_verfuegbar() -> bool:
    """Prüft, ob eine DirectX-12-Grafikkarte für Berechnungen bereitsteht."""
    try:
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def create_reranker():
    """Lädt den Cross-Encoder für die Neubewertung der Treffer.

    Bevorzugt wird die Grafikkarte über DirectML. Auf dem Entwicklungsrechner
    (AMD Radeon RX 7800 XT) bewertet sie 180 Textpaare in rund 4 Sekunden statt
    34 Sekunden auf dem Prozessor - bei bitgenau identischen Bewertungen.

    Für AMD-Karten gibt es unter Windows kein PyTorch mit GPU-Unterstützung;
    der Umweg über ONNX Runtime mit DirectML ist deshalb der einzige Weg. Steht
    keine geeignete Karte bereit oder schlägt das Laden fehl, wird ohne
    Zutun auf den Prozessor zurückgefallen.
    """
    from sentence_transformers import CrossEncoder

    if _dml_verfuegbar():
        try:
            # Der einmalige Export nach ONNX dauert rund 40 Sekunden. Liegt das
            # Ergebnis bereits lokal vor, ist das Laden in etwa 5 Sekunden erledigt.
            if not os.path.isdir(RERANKER_ONNX_DIR):
                modell = CrossEncoder(
                    RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH, backend="onnx",
                    model_kwargs={"provider": "DmlExecutionProvider"},
                )
                modell.save_pretrained(RERANKER_ONNX_DIR)
                return modell
            return CrossEncoder(
                RERANKER_ONNX_DIR, max_length=RERANKER_MAX_LENGTH, backend="onnx",
                model_kwargs={"provider": "DmlExecutionProvider"},
            )
        except Exception as fehler:
            print(f"Grafikkarte nicht nutzbar ({type(fehler).__name__}), weiter auf dem Prozessor.")

    return CrossEncoder(RERANKER_MODEL, max_length=RERANKER_MAX_LENGTH)


def reranker_backend() -> str:
    """Gibt zurück, worauf die Neubewertung voraussichtlich läuft."""
    return "Grafikkarte (DirectML)" if _dml_verfuegbar() else "Prozessor"


def open_expert_database(embeddings) -> QdrantVectorStore:
    client = QdrantClient(path=QDRANT_DIR)
    return QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )


def load_all_expert_chunks(vector_store: QdrantVectorStore, limit: int = 20000) -> List[Document]:
    """Liest alle Abschnitte der Wissensdatenbank für die Stichwortsuche aus."""
    treffer, _ = vector_store.client.scroll(
        collection_name=COLLECTION_NAME, limit=limit, with_payload=True, with_vectors=False
    )
    dokumente: List[Document] = []
    for eintrag in treffer:
        nutzdaten = eintrag.payload or {}
        inhalt = nutzdaten.get("page_content", "")
        if inhalt:
            dokumente.append(
                Document(page_content=inhalt, metadata=nutzdaten.get("metadata", {}) or {})
            )
    return dokumente


# ---------------------------------------------------------------------------
# HYBRIDE SUCHE
# ---------------------------------------------------------------------------
_PARAGRAPH_RE = re.compile(r"§+\s*(\d+[a-z]?)", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Zerlegt Text für die Stichwortsuche in bedeutungstragende Wörter.

    Zwei Besonderheiten:

    * Umlaute werden vereinheitlicht, damit "Mobilität" aus einem digitalen PDF
      und "Mobilitaet" aus einer Texterkennung dasselbe Wort ergeben.
    * Paragrafenangaben werden zu einem Wort zusammengezogen ("§ 18" -> "§18").
      Getrennt betrachtet fielen beide Teile durch die Mindestlänge und eine
      Suche nach "§ 18 SGB XI" hätte den Paragrafen gar nicht berücksichtigt.
    """
    normalisiert = _PARAGRAPH_RE.sub(r"§\1", normalize(text))
    return [
        wort
        for wort in _WORD_RE.findall(normalisiert)
        if len(wort) >= 2 and wort not in _NORMALIZED_STOPWORDS
    ]


def reciprocal_rank_fusion(
    ranglisten: Sequence[Sequence[Document]], k: int = 60
) -> List[Document]:
    """Verschmilzt mehrere Trefferlisten zu einer gemeinsamen Rangfolge.

    Ein Abschnitt, den beide Suchverfahren finden, steigt dadurch nach oben,
    ohne dass die Bewertungen der Verfahren vergleichbar sein müssen.
    """
    punkte: dict[str, float] = {}
    dokumente: dict[str, Document] = {}
    for rangliste in ranglisten:
        for position, doc in enumerate(rangliste):
            schluessel = f"{doc.metadata.get('source', '')}|{doc.page_content[:160]}"
            punkte[schluessel] = punkte.get(schluessel, 0.0) + 1.0 / (k + position + 1)
            dokumente.setdefault(schluessel, doc)
    sortiert = sorted(punkte.items(), key=lambda paar: paar[1], reverse=True)
    return [dokumente[schluessel] for schluessel, _ in sortiert]


class HybridIndex:
    """Bündelt Vektorsuche und Stichwortsuche über denselben Bestand."""

    def __init__(self, vector_store, documents: Sequence[Document]):
        self.vector_store = vector_store
        self.documents = list(documents)
        self._bm25 = None
        if self.documents:
            try:
                from rank_bm25 import BM25Okapi

                self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.documents])
            except Exception:
                # Ohne Stichwortsuche bleibt die Vektorsuche voll funktionsfähig.
                self._bm25 = None

    def _dense(self, query: str, k: int) -> List[Document]:
        if self.vector_store is None:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception:
            return []

    def _keyword(self, query: str, k: int) -> List[Document]:
        if self._bm25 is None:
            return []
        begriffe = tokenize(query)
        if not begriffe:
            return []
        werte = self._bm25.get_scores(begriffe)
        beste = sorted(range(len(werte)), key=lambda i: werte[i], reverse=True)[:k]
        return [self.documents[i] for i in beste if werte[i] > 0]

    def search(
        self, queries: Sequence[str], limit: int = MAX_RERANK_CANDIDATES, dense: bool = True
    ) -> List[Document]:
        """Sucht mit einer oder mehreren Formulierungen und fusioniert alles.

        Mit ``dense=False`` bleibt es bei der Stichwortsuche. Das lohnt sich für
        Teilfragen wie "Modul 4 Selbstversorgung": Der Begriff steht wörtlich in
        den Unterlagen, und jede Vektorsuche kostet einen Aufruf des
        Einbettungsmodells - bei sieben Teilfragen summiert sich das spürbar.
        """
        ranglisten: List[List[Document]] = []
        for frage in queries:
            if dense:
                ranglisten.append(self._dense(frage, DENSE_CANDIDATES))
            ranglisten.append(self._keyword(frage, KEYWORD_CANDIDATES))
        return reciprocal_rank_fusion([r for r in ranglisten if r])[:limit]


# ---------------------------------------------------------------------------
# NEUBEWERTUNG
# ---------------------------------------------------------------------------
def rerank(
    reranker, query: str, documents: Sequence[Document], top_n: int,
    min_score: float = RERANK_MIN_SCORE,
) -> List[Tuple[Document, float]]:
    """Bewertet Kandidaten gegen die Frage und behält die besten.

    Abschnitte ohne Aussagekraft (Tabellenrahmen, Seitenzahlen) erhalten vom
    Cross-Encoder eine Bewertung nahe null und fallen über ``min_score`` weg.
    """
    if not documents:
        return []
    if reranker is None:
        return [(doc, 0.0) for doc in documents[:top_n]]

    try:
        werte = reranker.predict([(query, doc.page_content) for doc in documents])
    except Exception:
        return [(doc, 0.0) for doc in documents[:top_n]]

    bewertet = sorted(zip(documents, (float(w) for w in werte)), key=lambda p: p[1], reverse=True)
    behalten = [paar for paar in bewertet if paar[1] >= min_score][:top_n]
    # Lieber etwas Schwaches zeigen als gar nichts, wenn alles unter der
    # Schwelle liegt - die Antwort weist dann selbst auf die dünne Lage hin.
    return behalten or bewertet[:1]


# ---------------------------------------------------------------------------
# QUELLENAUFBEREITUNG
# ---------------------------------------------------------------------------
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def to_superscript(nummer: int) -> str:
    return str(nummer).translate(SUPERSCRIPT)


def strip_junk_lines(text: str) -> str:
    """Entfernt Zeilen ohne Aussage, etwa Tabellenrahmen oder Trennlinien.

    Solche Zeilen hängen sonst am Anfang des ersten Satzes und machen die
    Quellenanzeige unlesbar.
    """
    behalten = []
    for zeile in text.splitlines():
        gestutzt = zeile.strip()
        if not gestutzt:
            continue
        # Tabellenzeilen bestehen aus mehreren durch Striche getrennten Zellen.
        # Für die Anzeige sind sie unbrauchbar; dem Sprachmodell steht der
        # vollständige Abschnitt weiterhin zur Verfügung.
        if gestutzt.count("|") >= 2:
            continue
        if set(gestutzt) <= set("-=_ .·|"):
            continue
        buchstaben = sum(1 for zeichen in gestutzt if zeichen.isalpha())
        if buchstaben < 8 or buchstaben / len(gestutzt) < 0.45:
            continue
        behalten.append(gestutzt)
    return " ".join(behalten)


def best_excerpt(text: str, query: str, max_chars: int = 320) -> str:
    """Wählt die aussagekräftigsten vollständigen Sätze eines Abschnitts.

    Statt stumpf die ersten Zeichen zu zeigen (die oft mitten in einer Tabelle
    beginnen), werden ganze Sätze mit dem größten Bezug zur Frage ausgewählt.
    """
    bereinigt = strip_junk_lines(text)
    saetze = [s.strip() for s in _SENTENCE_RE.split(bereinigt or text) if s.strip()]
    brauchbar = [
        satz for satz in saetze
        if len(satz) > 30 and sum(c.isalpha() for c in satz) / max(len(satz), 1) > 0.6
    ]
    if not brauchbar:
        gekuerzt = " ".join((bereinigt or text).split())
        return gekuerzt[:max_chars] + ("…" if len(gekuerzt) > max_chars else "")

    begriffe = set(tokenize(query))
    bewertet = sorted(
        enumerate(brauchbar),
        key=lambda paar: len(begriffe & set(tokenize(paar[1]))),
        reverse=True,
    )
    startindex = bewertet[0][0] if bewertet else 0

    ausschnitt: List[str] = []
    laenge = 0
    for satz in brauchbar[startindex:]:
        if laenge + len(satz) > max_chars and ausschnitt:
            break
        ausschnitt.append(satz)
        laenge += len(satz) + 1

    ergebnis = " ".join(ausschnitt)
    if startindex > 0:
        ergebnis = "… " + ergebnis
    if startindex + len(ausschnitt) < len(brauchbar):
        ergebnis += " …"
    return ergebnis


@dataclass
class SourceRef:
    """Eine nummerierte Quelle, wie sie unter der Antwort erscheint."""

    nummer: int
    quelle: str
    art: str
    ausschnitt: str
    bewertung: float
    herkunft: str  # "nutzer" oder "fachwissen"
    ueberschrift: str = ""

    def als_dict(self) -> dict:
        return {
            "nummer": self.nummer, "quelle": self.quelle, "art": self.art,
            "ausschnitt": self.ausschnitt, "bewertung": round(self.bewertung, 3),
            "herkunft": self.herkunft, "ueberschrift": self.ueberschrift,
        }


def build_source_refs(
    bewertete: Sequence[Tuple[Document, float]], query: str, herkunft: str, start: int = 1
) -> List[SourceRef]:
    quellen: List[SourceRef] = []
    for versatz, (doc, wert) in enumerate(bewertete):
        quellen.append(
            SourceRef(
                nummer=start + versatz,
                quelle=doc.metadata.get("source", "Unbekannte Quelle"),
                art=doc.metadata.get("doc_kind", "Dokument"),
                ausschnitt=best_excerpt(doc.page_content, query),
                bewertung=wert,
                herkunft=herkunft,
                ueberschrift=str(doc.metadata.get("heading", "") or ""),
            )
        )
    return quellen


_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def render_citations(text: str, gueltige_nummern: Sequence[int]) -> str:
    """Wandelt Belegstellen [1] in echte Hochziffern um.

    Nummern, die es gar nicht gibt, entfernt die Funktion - sonst verweist die
    Antwort auf eine Quelle, die in der Liste unten fehlt.
    """
    erlaubt = set(gueltige_nummern)

    def ersetzen(treffer: re.Match) -> str:
        nummer = int(treffer.group(1))
        return to_superscript(nummer) if nummer in erlaubt else ""

    bereinigt = _CITATION_RE.sub(ersetzen, text)
    # Mehrfache Hochziffern hintereinander lesbar trennen.
    return re.sub(r"[ ]{2,}", " ", bereinigt)


def cited_numbers(text: str) -> List[int]:
    return sorted({int(n) for n in _CITATION_RE.findall(text)})


# ---------------------------------------------------------------------------
# KONTEXT UND PROMPTS
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Du bist ein Assistenzdienst, der Menschen dabei unterstützt, einen Widerspruch gegen einen Pflegegradbescheid vorzubereiten.

So schreibst du:
- Immer auf Deutsch, in einfacher, gut verständlicher Sprache. Kurze Sätze. Fachbegriffe erklärst du kurz.
- Gliedere längere Antworten mit Zwischenüberschriften und Aufzählungen.

Belegpflicht - das ist die wichtigste Regel:
- Jede inhaltliche Aussage belegst du mit der Nummer des Abschnitts, aus dem sie stammt, in eckigen Klammern: [1], [2].
- Verwende ausschließlich die unten aufgeführten Nummern. Erfinde niemals eine Nummer und niemals einen Dateinamen.
- Was in keinem der Abschnitte steht, schreibst du nicht. Fehlt eine Angabe, sage klar:
  "Dazu finde ich in den vorliegenden Unterlagen keine Angabe."
- Rate nichts und ergänze nichts aus eigenem Wissen. Das gilt besonders für Zuständigkeiten,
  Verfahrenswege, Fristen, Punktzahlen und Geldbeträge: Solche Angaben nennst du nur, wenn sie
  wörtlich in einem Abschnitt stehen.
- Schreibe die Belegstelle IMMER als Zahl in eckigen Klammern direkt hinter der Aussage, zum Beispiel:
  "Das Modul Selbstversorgung wird mit 40 Prozent gewichtet [3]."
  Schreibe NIEMALS "(Abschnitt 3)", "(Quelle: …)" oder einen Dateinamen als Beleg.
- Jeder Abschnitt beginnt mit einer Trennzeile der Form ----- [3] ----- Herkunft: … Diese Trennzeilen
  sind reine Technik und gehören NICHT in deine Antwort.
- Eine Belegnummer setzt du nur, wenn der Abschnitt die Aussage wirklich enthält. Setze niemals
  eine Nummer hinter einen Satz, den du selbst ergänzt oder vermutet hast.
- Enthält ein Abschnitt bloß eine Aufzählung von Dokumenttiteln oder Überschriften, ist das KEIN
  Beleg für inhaltliche Feststellungen. Aus einem Dateinamen oder Berichtsdatum darfst du nicht
  ableiten, was in dem Bericht steht.

Weitere Regeln:
- Du gibst keine Rechtsberatung und weist bei rechtlichen Fragen darauf hin, dass eine Beratung dadurch nicht ersetzt wird.
- Unterscheide sauber zwischen dem, was in den Unterlagen der ratsuchenden Person steht, und dem allgemeinen Fachwissen.

Fachlicher Hintergrund: Der Pflegegrad ergibt sich aus sechs Modulen mit unterschiedlicher Gewichtung:
1. Mobilität (10 %), 2. Kognitive und kommunikative Fähigkeiten und 3. Verhaltensweisen und psychische Problemlagen
(zusammen 15 %, es zählt der höhere Wert), 4. Selbstversorgung (40 %),
5. Bewältigung von krankheits- und therapiebedingten Anforderungen (20 %),
6. Gestaltung des Alltagslebens und sozialer Kontakte (15 %).

=== ABSCHNITTE AUS DEN UNTERLAGEN DER RATSUCHENDEN PERSON ===
{user_context}

=== ABSCHNITTE AUS DEM GEPRÜFTEN FACHWISSEN ===
{expert_context}
"""


def format_numbered_context(quellen: Sequence[SourceRef], dokumente: Sequence[Document], titel: str) -> str:
    """Stellt die Abschnitte nummeriert für das Sprachmodell zusammen.

    Die Kopfzeilen sind bewusst als technische Markierung erkennbar, damit das
    Sprachmodell sie nicht versehentlich in die Antwort übernimmt.
    """
    if not quellen:
        return f"({titel}: keine passenden Abschnitte gefunden.)"
    teile = []
    for quelle, doc in zip(quellen, dokumente):
        kopf = f"----- [{quelle.nummer}] ----- Herkunft: {quelle.quelle}"
        if quelle.ueberschrift:
            kopf += f" | Kapitel: {quelle.ueberschrift}"
        teile.append(f"{kopf}\n{doc.page_content}")
    return "\n\n".join(teile)


# Zusatzanweisung, wenn die gefundenen Abschnitte nur schwach zur Frage passen.
# Ohne sie füllt das Sprachmodell die Lücke aus eigenem Wissen - im Test führte
# das zu einer falschen Auskunft über die zuständige Behörde.
WEAK_EVIDENCE_NOTE = """
ACHTUNG – DÜNNE BELEGLAGE: Die unten stehenden Abschnitte passen nur schwach zur Frage.
Sage in diesem Fall ausdrücklich, dass sich die Frage mit den vorliegenden Unterlagen nicht
sicher beantworten lässt, und nenne nur das, was tatsächlich belegt ist. Ergänze auf keinen Fall
Verfahrenswege, Zuständigkeiten, Fristen oder Zahlen aus eigenem Wissen.
"""

# Ab dieser Bewertung gilt ein Treffer als tragfähig.
STRONG_EVIDENCE_SCORE = 0.30


@dataclass
class RetrievalResult:
    """Ergebnis einer Suche: Systemprompt und die belegten Quellen."""

    system_prompt: str
    quellen: List[SourceRef] = field(default_factory=list)
    beste_bewertung: float = 0.0

    @property
    def nummern(self) -> List[int]:
        return [q.nummer for q in self.quellen]

    @property
    def belege_tragfaehig(self) -> bool:
        """Ob mindestens ein Treffer die Frage wirklich abdeckt."""
        return self.beste_bewertung >= STRONG_EVIDENCE_SCORE


# Zusatzfragen, die bei einer Differenzanalyse mitgesucht werden. So kommen
# Belege zu allen sechs Modulen in den Kontext, nicht nur zum erstbesten.
MODULE_QUERIES = [f"Modul {nummer} {name}" for nummer, name in MODULE_NAMES.items()]


def select_per_query(
    index: HybridIndex, reranker, queries: Sequence[str], je_frage: int = 2,
    kandidaten_je_frage: int = 8, min_score: float = 0.0,
) -> List[Tuple[Document, float]]:
    """Wählt für jede Teilfrage eigene Belege aus.

    Eine einzige Neubewertung gegen die Sammelfrage genügt nicht: Bei einer
    Analyse über sechs Module gewinnen sonst zwei allgemeine Abschnitte, und zu
    den übrigen Modulen liegt dem Sprachmodell gar kein Beleg vor - es füllt die
    Lücke dann mit Erfundenem. Deshalb bekommt jede Teilfrage ihre eigene
    Auswahl.
    """
    # Erst alle Teilfragen durchsuchen, dann in EINEM Durchgang neu bewerten.
    # Einzelne Aufrufe des Cross-Encoders haben spürbaren Grundaufwand; gebündelt
    # ist derselbe Umfang um ein Vielfaches schneller.
    paare: List[Tuple[str, Document]] = []
    grenzen: List[Tuple[str, int, int]] = []
    for position, frage in enumerate(queries):
        # Nur die Hauptfrage rechtfertigt eine Vektorsuche; die Modulfragen
        # bestehen aus wörtlich vorkommenden Fachbegriffen.
        kandidaten = index.search([frage], limit=kandidaten_je_frage, dense=(position == 0))
        start = len(paare)
        paare.extend((frage, doc) for doc in kandidaten)
        grenzen.append((frage, start, len(paare)))

    if not paare:
        return []

    if reranker is None:
        werte = [0.0] * len(paare)
    else:
        try:
            werte = [float(w) for w in reranker.predict([(f, d.page_content) for f, d in paare])]
        except Exception:
            werte = [0.0] * len(paare)

    gesammelt: dict[str, Tuple[Document, float]] = {}
    for _, start, ende in grenzen:
        abschnitt = sorted(
            ((paare[i][1], werte[i]) for i in range(start, ende)),
            key=lambda paar: paar[1], reverse=True,
        )
        for doc, wert in [p for p in abschnitt if p[1] >= min_score][:je_frage]:
            schluessel = f"{doc.metadata.get('source', '')}|{doc.page_content[:160]}"
            # Bei Mehrfachtreffern zählt die beste erreichte Bewertung.
            if schluessel not in gesammelt or gesammelt[schluessel][1] < wert:
                gesammelt[schluessel] = (doc, wert)
    return sorted(gesammelt.values(), key=lambda paar: paar[1], reverse=True)


def prepare_context(
    expert_index: Optional[HybridIndex],
    user_index: Optional[HybridIndex],
    question: str,
    reranker=None,
    extra_queries: Sequence[str] = (),
) -> RetrievalResult:
    """Sucht Belege und baut daraus den nummerierten Systemprompt."""
    fragen = [question, *extra_queries]

    if user_index is None:
        user_bewertet: List[Tuple[Document, float]] = []
    elif extra_queries:
        # Aufgaben über alle sechs Module brauchen zu jedem Modul einen Beleg.
        user_bewertet = select_per_query(user_index, reranker, fragen, je_frage=2)[
            :FINAL_USER_CHUNKS_BREIT
        ]
    else:
        # Die eigenen Unterlagen betreffen immer den eigenen Fall. Hier zählt die
        # Reihenfolge, nicht das Aussortieren - deshalb keine Mindestbewertung.
        user_bewertet = rerank(
            reranker, question, user_index.search(fragen), FINAL_USER_CHUNKS, min_score=0.0
        )

    # Die Suche im Fachwissen wird um die zentralen Begriffe des Themas ergänzt.
    fach_fragen = [*fragen, f"{question} Pflegegrad Begutachtung Widerspruch Module Punkte"]
    if expert_index is None:
        fach_bewertet: List[Tuple[Document, float]] = []
    elif extra_queries:
        fach_bewertet = select_per_query(
            expert_index, reranker, fach_fragen, je_frage=1, min_score=RERANK_MIN_SCORE
        )[:FINAL_EXPERT_CHUNKS_BREIT]
    else:
        fach_bewertet = rerank(
            reranker, question, expert_index.search(fach_fragen), FINAL_EXPERT_CHUNKS
        )

    user_quellen = build_source_refs(user_bewertet, question, "nutzer", start=1)
    fach_quellen = build_source_refs(
        fach_bewertet, question, "fachwissen", start=len(user_quellen) + 1
    )

    system_prompt = SYSTEM_PROMPT.format(
        user_context=format_numbered_context(
            user_quellen, [d for d, _ in user_bewertet], "Unterlagen der ratsuchenden Person"
        ),
        expert_context=format_numbered_context(
            fach_quellen, [d for d, _ in fach_bewertet], "Fachwissen"
        ),
    )

    # Passt kein einziger Treffer wirklich gut, wird das Modell ausdrücklich zur
    # Zurückhaltung angewiesen, statt die Lücke aus eigenem Wissen zu füllen.
    beste_bewertung = max(
        (wert for _, wert in list(user_bewertet) + list(fach_bewertet)), default=0.0
    )
    if beste_bewertung < STRONG_EVIDENCE_SCORE:
        system_prompt = WEAK_EVIDENCE_NOTE + "\n" + system_prompt

    return RetrievalResult(
        system_prompt=system_prompt,
        quellen=user_quellen + fach_quellen,
        beste_bewertung=beste_bewertung,
    )


def build_messages(system_prompt: str, history: Sequence[dict]) -> List[dict]:
    """Baut die Nachrichtenliste und begrenzt den Gesprächsverlauf."""
    verlauf = list(history)[-MAX_HISTORY_MESSAGES:]
    return [{"role": "system", "content": system_prompt}] + [
        {"role": n["role"], "content": n["content"]} for n in verlauf
    ]


def stream_answer(llm, messages: Sequence[dict]) -> Iterator[str]:
    """Gibt die Antwort des Sprachmodells stückweise zur Live-Anzeige aus."""
    for teil in llm.stream(list(messages)):
        inhalt = getattr(teil, "content", "")
        if inhalt:
            yield inhalt


# ---------------------------------------------------------------------------
# SCHNELLAKTIONEN IM CHAT
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuickAction:
    """Eine vorbereitete Aufgabe im Chat.

    ``nutzertext`` erscheint im Gesprächsverlauf, ``prompt`` geht an das
    Sprachmodell. Die ausführlichen Formatvorgaben würden im Verlauf sonst als
    scheinbare Nachricht der ratsuchenden Person stehen und ihn unlesbar machen.
    """

    schluessel: str
    titel: str
    beschreibung: str
    nutzertext: str
    prompt: str
    zusatzfragen: Tuple[str, ...] = ()


QUICK_ACTIONS: Tuple[QuickAction, ...] = (
    QuickAction(
        schluessel="einlesen",
        titel="Unterlagen sichten",
        beschreibung="Verschafft einen Überblick: Welche Dokumente liegen vor, welcher Pflegegrad wurde festgestellt, welche Einschränkungen sind belegt.",
        nutzertext="Bitte sichte meine Unterlagen und gib mir einen Überblick.",
        prompt=(
            "Sichte meine hochgeladenen Unterlagen und gib mir einen strukturierten Überblick.\n\n"
            "Gliedere deine Antwort so:\n"
            "1. **Vorliegende Unterlagen** – welches Dokument ist was, und von wann?\n"
            "2. **Ergebnis der Begutachtung** – welcher Pflegegrad wurde festgestellt, mit wie vielen Punkten, "
            "und wie verteilen sich die Punkte auf die sechs Module?\n"
            "3. **Dokumentierte Einschränkungen und Diagnosen** – geordnet nach den sechs Modulen.\n"
            "4. **Erste Auffälligkeiten** – wo wirkt die Bewertung auf den ersten Blick zu niedrig?\n\n"
            "Belege jede Angabe mit der Abschnittsnummer. Wenn eine der vier Angaben in meinen Unterlagen "
            "fehlt, schreibe das ausdrücklich hin, statt sie zu ergänzen."
        ),
    ),
    QuickAction(
        schluessel="differenz",
        titel="Differenzanalyse",
        beschreibung="Vergleicht das Gutachten des Medizinischen Dienstes Modul für Modul mit Ihren übrigen Unterlagen.",
        nutzertext="Bitte vergleiche das Gutachten des Medizinischen Dienstes mit meinen übrigen Unterlagen.",
        prompt=(
            "Führe eine Differenzanalyse durch: Vergleiche die Feststellungen des Medizinischen Dienstes "
            "mit dem, was meine übrigen hochgeladenen Unterlagen belegen.\n\n"
            "Arbeite die sechs Module der Reihe nach ab. Schreibe zu jedem Modul:\n"
            "- **Feststellung des Medizinischen Dienstes:** welche Punkte, welche Begründung? [Nummer]\n"
            "- **Was meine Unterlagen belegen:** [Nummer]\n"
            "- **Bewertung der Abweichung:** keine / geringfügig / erheblich – und warum.\n\n"
            "Strenge Regeln:\n"
            "- Beziehe dich ausschließlich auf Dokumente, die in den Abschnitten tatsächlich als Herkunft "
            "genannt sind. Gehe NICHT davon aus, dass ein Pflegetagebuch, ein Arztbericht oder sonst ein "
            "Dokument vorliegt, wenn es dort nicht auftaucht.\n"
            "- Fehlt zu einem Modul eine Angabe, schreibe „Dazu finde ich in den Unterlagen keine Angabe“ "
            "und gehe weiter. Erfinde weder Feststellungen noch Punktzahlen.\n"
            "- Schließe mit einer Einschätzung, bei welchen Modulen ein Widerspruch am aussichtsreichsten "
            "ist, und nenne, welche Nachweise dafür noch fehlen."
        ),
        zusatzfragen=tuple(MODULE_QUERIES),
    ),
    QuickAction(
        schluessel="argumente",
        titel="Argumente sammeln",
        beschreibung="Stellt die belegbaren Argumente für den Widerspruch zusammen, sortiert nach Überzeugungskraft.",
        nutzertext="Bitte sammle die begründeten Argumente für meinen Widerspruch.",
        prompt=(
            "Stelle die belegbaren Argumente für meinen Widerspruch zusammen.\n\n"
            "Schreibe für jedes Argument:\n"
            "- **Betroffenes Modul**\n"
            "- **Feststellung des Medizinischen Dienstes** (mit Abschnittsnummer)\n"
            "- **Gegenbeleg aus meinen Unterlagen** (mit Abschnittsnummer)\n"
            "- **Warum die Einschätzung damit nicht haltbar ist**\n"
            "- **Einschätzung der Erfolgsaussicht:** hoch / mittel / gering\n\n"
            "Sortiere nach Erfolgsaussicht, das stärkste Argument zuerst. Nimm nur Argumente auf, die du "
            "tatsächlich mit einem Abschnitt belegen kannst. Nenne am Ende, welche Nachweise mir noch fehlen "
            "und welche ich nachreichen sollte."
        ),
        zusatzfragen=tuple(MODULE_QUERIES),
    ),
    QuickAction(
        schluessel="schreiben",
        titel="Widerspruch schreiben",
        beschreibung="Verfasst die Begründung für das Widerspruchsschreiben, fertig zur Übernahme in das PDF.",
        nutzertext="Bitte verfasse die Begründung für mein Widerspruchsschreiben.",
        prompt=(
            "Verfasse die Begründung für mein Widerspruchsschreiben an die Pflegekasse.\n\n"
            "Aufbau:\n"
            "- Ein einleitender Satz, gegen welche Feststellung sich der Widerspruch richtet.\n"
            "- Danach je ein Absatz pro strittigem Modul: Was hat der Medizinische Dienst festgestellt, "
            "was belegen meine Unterlagen, welche höhere Bewertung ist daher angezeigt.\n"
            "- Ein Schlussabsatz mit der Bitte um erneute Begutachtung.\n\n"
            "Zwingende Vorgaben:\n"
            "- Schreibe NUR den Begründungstext. Keine Anrede, keine Grußformel, kein Betreff, "
            "keine Absenderangaben, keine Unterschrift – die ergänzt die Briefvorlage automatisch.\n"
            "- Schreibe in ganzen Sätzen, sachlich und höflich, in der Ich-Form.\n"
            "- Verwende KEINE Belegnummern und KEINE eckigen Klammern im Brieftext. Der Brief geht an eine "
            "Behörde und muss ohne Fußnoten lesbar sein.\n"
            "- Setze KEINE Platzhalter ein. Wenn dir eine Zahl fehlt, formuliere den Satz ohne sie.\n"
            "- Behandle nur Module, zu denen du echte Belege hast. Erwähne fehlende Module gar nicht.\n"
            "- Schreibe nichts über Rechtsberatung, über künstliche Intelligenz oder Empfehlungen an mich."
        ),
        zusatzfragen=tuple(MODULE_QUERIES),
    ),
)

QUICK_ACTION_BY_KEY = {aktion.schluessel: aktion for aktion in QUICK_ACTIONS}
