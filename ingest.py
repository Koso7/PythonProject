import os
import shutil
from typing import List

from dotenv import load_dotenv

os.environ["USER_AGENT"] = "PflegeAssistentStudienprojekt/1.0"

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./daten")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")

URLS_TO_LEARN = [
    "https://www.bundesgesundheitsministerium.de/themen/pflege/pflegebeduerftigkeit/pflegegrade.html",
    "https://www.pflege.de/pflegekasse-pflegerecht/pflegegrade/widerspruch/",
    "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/pflegegrad-abgelehnt-so-wehren-sie-sich-mit-widerspruch-und-klage-11547",
    "https://www.vdk.de/aktuelles/aktuelle-meldungen/artikel/widerspruch-gegen-pflegegrad-lohnt-sich-oft/",
    "https://www.pflege-betreuer.de/de/pflegewissen/pflegerecht-und-ansprueche/widerspruch-gegen-die-pflegegrad-einstufung-einlegen",
    "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/pflegegrad-beantragen-so-gehts-13413",
    "https://www.pflege.de/pflegekasse-pflegerecht/pflegegrade/beantragen/",
    "https://www.bundesgesundheitsministerium.de/themen/pflege/online-ratgeber-pflege/pflegebeduerftig-was-nun",
]


def load_pdf_documents() -> List[Document]:
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ Ordner '{DATA_DIR}' existiert nicht.")
        return []

    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    cleaned_docs = []

    for doc in docs:
        content = clean_text(doc.page_content)

        if len(content.strip()) < 100:
            continue

        doc.page_content = content
        doc.metadata["document_type"] = "pdf"
        doc.metadata["knowledge_type"] = "fachwissen"

        source = doc.metadata.get("source", "Unbekannte PDF")
        doc.metadata["source"] = source.replace("\\", "/")

        cleaned_docs.append(doc)

    print(f"📄 {len(cleaned_docs)} verwertbare PDF-Seiten geladen.")
    return cleaned_docs


def load_web_documents() -> List[Document]:
    all_web_docs = []

    print("🌐 Lade Webseiten...")

    for url in URLS_TO_LEARN:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            for doc in docs:
                content = clean_text(doc.page_content)

                if len(content.strip()) < 100:
                    continue

                doc.page_content = content
                doc.metadata["source"] = url
                doc.metadata["document_type"] = "webseite"
                doc.metadata["knowledge_type"] = "fachwissen"

                all_web_docs.append(doc)

            print(f"✅ Webseite geladen: {url}")

        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {url}: {e}")

    print(f"🌐 {len(all_web_docs)} verwertbare Webseiten-Dokumente geladen.")
    return all_web_docs


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")
    text = text.replace("  ", " ")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def split_documents(documents: List[Document]) -> List[Document]:
    print("✂️ Teile Dokumente in bessere Chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=180,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    cleaned_chunks = []
    seen = set()

    for index, chunk in enumerate(chunks):
        content = clean_text(chunk.page_content)

        if len(content) < 120:
            continue

        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        duplicate_key = (source, page, content[:300])

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)

        chunk.page_content = content
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_size"] = len(content)

        cleaned_chunks.append(chunk)

    print(f"✅ {len(cleaned_chunks)} eindeutige Textabschnitte erstellt.")
    return cleaned_chunks


def build_expert_database():
    print("=" * 70)
    print("📚 Starte Aufbau der Pflegegrad-Wissensdatenbank")
    print("=" * 70)

    documents = []

    pdf_docs = load_pdf_documents()
    web_docs = load_web_documents()

    documents.extend(pdf_docs)
    documents.extend(web_docs)

    if not documents:
        print("❌ Keine Dokumente gefunden. Abbruch.")
        return

    chunks = split_documents(documents)

    if os.path.exists(CHROMA_DIR):
        print(f"🗑️ Lösche alte ChromaDB unter '{CHROMA_DIR}', um Duplikate zu vermeiden.")
        shutil.rmtree(CHROMA_DIR)

    print(f"🧠 Erstelle Embeddings mit Modell: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="pflege_fachwissen",
    )

    print("✅ Wissensdatenbank erfolgreich erstellt.")
    print(f"📁 Speicherort: {CHROMA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    build_expert_database()