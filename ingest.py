import os
from typing import List

from dotenv import load_dotenv

# Verhindert teilweise, dass Webseiten einfache Bot-Anfragen blockieren.
os.environ["USER_AGENT"] = "PflegeAssistentStudienprojekt/1.0"

from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./daten")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


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
        print(f"⚠️ Ordner '{DATA_DIR}' existiert nicht. Bitte anlegen.")
        return []

    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    for doc in docs:
        doc.metadata["document_type"] = "pdf"
        doc.metadata["knowledge_type"] = "fachwissen"

    print(f"📄 {len(docs)} PDF-Seiten geladen.")
    return docs


def load_web_documents() -> List[Document]:
    all_web_docs = []

    print("🌐 Lade Webseiten...")

    for url in URLS_TO_LEARN:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = url
                doc.metadata["document_type"] = "webseite"
                doc.metadata["knowledge_type"] = "fachwissen"

            all_web_docs.extend(docs)
            print(f"✅ Webseite geladen: {url}")

        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {url}: {e}")

    print(f"🌐 {len(all_web_docs)} Webseiten-Dokumente geladen.")
    return all_web_docs


def split_documents(documents: List[Document]) -> List[Document]:
    print("✂️ Teile Dokumente in Chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    print(f"✅ {len(chunks)} Textabschnitte erstellt.")
    return chunks


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

    print("🧠 Erstelle Embeddings und speichere in ChromaDB...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Hinweis:
    # Bei erneutem Ausführen wird die bestehende ChromaDB überschrieben,
    # damit keine alten/veralteten Duplikate bleiben.
    if os.path.exists(CHROMA_DIR):
        print(f"⚠️ Bestehende ChromaDB unter '{CHROMA_DIR}' wird aktualisiert/überschrieben.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("✅ Wissensdatenbank erfolgreich erstellt.")
    print(f"📁 Speicherort: {CHROMA_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    build_expert_database()