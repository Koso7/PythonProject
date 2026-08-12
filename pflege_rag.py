"""Gemeinsame RAG-Logik des Pflege-Assistenten.

Enthält Dokumentenaufbereitung, Suche (Retrieval) und die Erzeugung der
Antworten. Bewusst frei von Streamlit-Abhängigkeiten, damit sowohl die
Weboberfläche (app.py) als auch der Aufbau der Wissensdatenbank (ingest.py)
dieselben Funktionen benutzen und sich beide nicht auseinanderentwickeln.

Datenschutz: Alle hier verwendeten Dienste laufen ausschließlich lokal
(LM Studio auf 127.0.0.1, Qdrant als lokales Verzeichnis). Es gehen keine
Nutzerdaten an externe Anbieter.
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, Iterator, List, Optional, Tuple

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

CHUNK_SIZE = 750
CHUNK_OVERLAP = 160
MIN_CHUNK_CHARS = 80

# Anzahl der Nachrichten aus dem Gesprächsverlauf, die an das Sprachmodell
# gehen. Begrenzt, damit der Kontext des lokalen Modells nicht überläuft und
# ältere Nachrichten nicht die abgerufenen Dokumente verdrängen.
MAX_HISTORY_MESSAGES = 8

HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]


# ---------------------------------------------------------------------------
# TEXTAUFBEREITUNG
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Entfernt Steuerzeichen, doppelte Leerzeichen und Leerzeilen."""
    text = text.replace("\x00", " ").replace("\t", " ").replace("  ", " ")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def remove_duplicate_docs(docs: Iterable[Document]) -> List[Document]:
    """Entfernt inhaltlich doppelte Treffer aus einer Ergebnisliste."""
    unique_docs: List[Document] = []
    seen = set()
    for doc in docs:
        key = (doc.metadata.get("source", ""), clean_text(doc.page_content)[:350])
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs


def split_documents(documents: Iterable[Document], drop_short: bool = True) -> List[Document]:
    """Teilt Dokumente zuerst an Überschriften, dann in überlappende Abschnitte."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    final_chunks: List[Document] = []
    seen = set()
    for doc in documents:
        header_splits = md_splitter.split_text(doc.page_content) or [doc]
        for split in header_splits:
            # Die Herkunftsangabe des Ursprungsdokuments muss erhalten bleiben,
            # damit Quellen später korrekt angezeigt werden.
            split.metadata.update(doc.metadata)

        for chunk in recursive_splitter.split_documents(header_splits):
            content = clean_text(chunk.page_content)
            if drop_short and len(content) < MIN_CHUNK_CHARS:
                continue
            key = (chunk.metadata.get("source", ""), content[:300])
            if key in seen:
                continue
            seen.add(key)
            chunk.page_content = content
            final_chunks.append(chunk)

    return final_chunks


# ---------------------------------------------------------------------------
# PDF-VERARBEITUNG
# ---------------------------------------------------------------------------
def _shred_file(path: str) -> None:
    """Überschreibt eine Datei vor dem Löschen.

    Die hochgeladenen Unterlagen müssen für die Umwandlung kurzzeitig auf die
    Festplatte geschrieben werden. Einfaches Löschen gibt den Speicherplatz nur
    frei, der Inhalt bliebe rekonstruierbar. Deshalb wird der Inhalt vorher
    überschrieben.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "r+b", buffering=0) as handle:
            handle.write(os.urandom(size))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        pass
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def extract_document_from_pdf(file_bytes: bytes, file_name: str, converter) -> Optional[Document]:
    """Wandelt eine hochgeladene PDF-Datei in ein Textdokument um.

    Die temporäre Datei wird anschließend überschrieben und gelöscht.
    """
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        result = converter.convert(tmp_path)
        markdown = clean_text(result.document.export_to_markdown())
    finally:
        if tmp_path:
            _shred_file(tmp_path)

    if len(markdown) < 30:
        return None
    return Document(
        page_content=markdown,
        metadata={"source": file_name, "document_type": "nutzerdokument"},
    )


# ---------------------------------------------------------------------------
# MODELLE UND DATENBANK (alles lokal)
# ---------------------------------------------------------------------------
def create_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        openai_api_base=LM_STUDIO_URL,
        openai_api_key="lm-studio",
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )


def create_llm(temperature: float = 0.2, max_tokens: int = 2400) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def open_expert_database(embeddings) -> QdrantVectorStore:
    client = QdrantClient(path=QDRANT_DIR)
    return QdrantVectorStore(
        client=client, collection_name=COLLECTION_NAME, embedding=embeddings
    )


def build_user_vector_store(documents: List[Document], embeddings) -> Optional[QdrantVectorStore]:
    """Legt einen Vektorspeicher für die Nutzerdokumente an.

    Ausschließlich im Arbeitsspeicher (`:memory:`): Die Inhalte der
    hochgeladenen Unterlagen werden dadurch nie unverschlüsselt auf die
    Festplatte geschrieben.
    """
    if not documents:
        return None
    return QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        location=":memory:",
        collection_name="nutzerdokumente",
    )


# ---------------------------------------------------------------------------
# SUCHE
# ---------------------------------------------------------------------------
def search_user_documents(store, question: str, k: int = 6) -> List[Document]:
    if store is None:
        return []
    return remove_duplicate_docs(store.as_retriever(search_kwargs={"k": k}).invoke(question))


def build_expert_search_query(question: str, user_docs: List[Document]) -> str:
    excerpt = "\n\n".join(doc.page_content[:900] for doc in user_docs[:6])
    return (
        "Pflegegrad Widerspruch Musterbrief Pflegekasse Medizinischer Dienst MD "
        "Gutachten Begutachtung Module Punkte\n"
        f"Nutzerfrage: {question}\n"
        f"Relevante Auszüge aus Nutzerdokumenten: {excerpt}"
    )


def search_expert_documents(expert_db, query: str, k: int = 5) -> List[Document]:
    if expert_db is None:
        return []
    return remove_duplicate_docs(expert_db.as_retriever(search_kwargs={"k": k}).invoke(query))


def format_docs_for_prompt(title: str, docs: List[Document]) -> str:
    if not docs:
        return f"{title}: Keine passenden Textstellen gefunden."
    parts = [title]
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        parts.append(f"[{index}] Quelle: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_source_list(docs: List[Document]) -> List[dict]:
    """Erzeugt die Quellenangaben für die Anzeige unter der Antwort."""
    sources: List[dict] = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unbekannte Quelle")
        content = clean_text(doc.page_content)
        preview = content[:250] + "..." if len(content) > 250 else content
        key = (source, preview[:100])
        if key not in seen:
            seen.add(key)
            sources.append({"nr": len(sources) + 1, "source": source, "preview": preview})
    return sources


# ---------------------------------------------------------------------------
# ANTWORTERZEUGUNG
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Du bist ein Assistenzdienst, der Menschen dabei unterstützt, einen Widerspruch gegen einen Pflegegradbescheid vorzubereiten.

So antwortest du:
- Immer auf Deutsch, in einfacher, gut verständlicher Sprache. Kurze Sätze. Fachbegriffe erklärst du in einem Nebensatz.
- Gliedere längere Antworten mit Zwischenüberschriften und Aufzählungen.
- Stütze dich ausschließlich auf die unten stehenden Nutzerdokumente und das Fachwissen. Erfinde nichts.
- Wenn eine Angabe in den Unterlagen fehlt, sage klar: "Dazu finde ich in Ihren Unterlagen keine Angabe." Rate nicht.
- Nenne bei wichtigen Aussagen die Quelle in Klammern. Verwende dabei ausschließlich Dateinamen, die unten
  tatsächlich als "Quelle:" aufgeführt sind. Erfinde niemals einen Dateinamen. Wenn du unsicher bist, lasse
  die Quellenangabe weg.
- Du gibst keine Rechtsberatung. Weise bei rechtlichen Fragen darauf hin, dass eine Rechtsberatung dadurch nicht ersetzt wird.

Fachlicher Hintergrund: Die Einstufung erfolgt über sechs Module, die unterschiedlich gewichtet werden:
1. Mobilität, 2. Kognitive und kommunikative Fähigkeiten, 3. Verhaltensweisen und psychische Problemlagen,
4. Selbstversorgung, 5. Bewältigung von krankheits- und therapiebedingten Anforderungen,
6. Gestaltung des Alltagslebens und sozialer Kontakte.

NUTZERDOKUMENTE (die hochgeladenen Unterlagen):
{user_context}

FACHWISSEN (geprüfte Quellen und Musterbriefe):
{expert_context}
"""

# Die vier Schnellaktionen im Chat.
QUICK_ACTIONS = {
    "einlesen": (
        "Lies alle hochgeladenen Unterlagen sorgfältig durch und gib mir eine strukturierte Übersicht:\n"
        "1. Welche Dokumente liegen vor (Art des Dokuments und Datum)?\n"
        "2. Welcher Pflegegrad wurde festgestellt und mit wie vielen Punkten?\n"
        "3. Welche Diagnosen und pflegerelevanten Einschränkungen sind dokumentiert?\n"
        "4. Was fällt dir auf den ersten Blick auf?\n"
        "Fasse dich klar und verständlich."
    ),
    "differenz": (
        "Führe eine Differenzanalyse durch: Vergleiche das Gutachten des Medizinischen Dienstes mit meinen "
        "übrigen Unterlagen (Pflegetagebuch, Arztberichte, Patientenakte).\n"
        "Gehe dafür die sechs Module der Begutachtung einzeln durch. Zeige für jedes Modul:\n"
        "- Was hat der Medizinische Dienst festgestellt?\n"
        "- Was belegen meine Unterlagen?\n"
        "- Gibt es eine Abweichung, und wie schwer wiegt sie?\n"
        "Hebe deutlich hervor, wo etwas nicht oder zu gering berücksichtigt wurde und deshalb ein Widerspruch "
        "aussichtsreich sein könnte. Wenn du zu einem Modul keine Angaben findest, sage das."
    ),
    "argumente": (
        "Sammle alle begründeten Argumente für meinen Widerspruch. Nenne für jedes Argument:\n"
        "1. Die konkrete Feststellung des Medizinischen Dienstes,\n"
        "2. den Beleg aus meinen Unterlagen mit Quellenangabe,\n"
        "3. warum die Einschätzung des Medizinischen Dienstes damit nicht haltbar ist,\n"
        "4. welches Modul betroffen ist.\n"
        "Sortiere die Argumente danach, wie überzeugend sie sind - das stärkste zuerst. "
        "Stütze dich ausschließlich auf belegbare Angaben aus meinen Unterlagen."
    ),
    "schreiben": (
        "Verfasse jetzt die Begründung für mein Widerspruchsschreiben an die Pflegekasse.\n\n"
        "Wichtige Vorgaben:\n"
        "- Schreibe AUSSCHLIESSLICH den Begründungstext.\n"
        "- KEINE Anrede ('Sehr geehrte Damen und Herren'), KEINE Grußformel, KEINE Absenderangaben, "
        "KEIN Betreff und KEINE Unterschrift. Diese Teile werden automatisch ergänzt.\n"
        "- Schreibe sachlich, höflich und behördentauglich in ganzen Sätzen.\n"
        "- Begründe konkret anhand der sechs Module und belege jeden Punkt mit meinen Unterlagen.\n"
        "- Erfinde keine Sachverhalte und keine Dateinamen. Verwende nur, was in meinen Unterlagen steht.\n"
        "- Führe nur Module auf, zu denen du tatsächlich Angaben in meinen Unterlagen findest. "
        "Lasse Module weg, zu denen dir Angaben fehlen, statt das Fehlen im Brief zu erwähnen.\n"
        "- Der Brief geht an die Pflegekasse. Schreibe deshalb KEINE Hinweise über Rechtsberatung, "
        "über künstliche Intelligenz oder Empfehlungen an mich in den Brieftext.\n"
        "- Schließe mit dem Hinweis, dass um eine erneute Begutachtung gebeten wird."
    ),
}


def build_messages(system_prompt: str, history: List[dict]) -> List[dict]:
    """Baut die Nachrichtenliste für das Sprachmodell.

    Der Gesprächsverlauf wird begrenzt, damit der Kontext des lokalen Modells
    nicht überläuft und die abgerufenen Dokumente nicht verdrängt werden.
    """
    trimmed = history[-MAX_HISTORY_MESSAGES:] if len(history) > MAX_HISTORY_MESSAGES else history
    return [{"role": "system", "content": system_prompt}] + [
        {"role": message["role"], "content": message["content"]} for message in trimmed
    ]


def prepare_context(
    expert_db, user_store, question: str, user_k: int = 6, expert_k: int = 5
) -> Tuple[str, List[Document], List[Document]]:
    """Sucht passende Textstellen und baut daraus den Systemprompt."""
    user_docs = search_user_documents(user_store, question, k=user_k)
    expert_docs = search_expert_documents(
        expert_db, build_expert_search_query(question, user_docs), k=expert_k
    )
    system_prompt = SYSTEM_PROMPT.format(
        user_context=format_docs_for_prompt("Auszüge aus den hochgeladenen Unterlagen", user_docs),
        expert_context=format_docs_for_prompt("Geprüftes Fachwissen", expert_docs),
    )
    return system_prompt, user_docs, expert_docs


def stream_answer(llm, messages: List[dict]) -> Iterator[str]:
    """Liefert die Antwort des Sprachmodells stückweise für die Live-Anzeige."""
    for chunk in llm.stream(messages):
        content = getattr(chunk, "content", "")
        if content:
            yield content
