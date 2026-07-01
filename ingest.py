import os
import shutil
import warnings
import requests
from typing import List

from dotenv import load_dotenv
from docling.document_converter import DocumentConverter  # NEU: IBM Docling Engine

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["USER_AGENT"] = "PflegeAssistent"

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter  # Harmonisiert
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./daten")
QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
EMBEDDING_MODEL = "text-embedding-bge-m3"
COLLECTION_NAME = "pflege_fachwissen"

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


class LMStudioEmbeddings(Embeddings):
    def __init__(self, base_url="http://localhost:1234/v1", model=EMBEDDING_MODEL):
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.base_url}/embeddings",
            json={"input": texts, "model": self.model}
        )
        response.raise_for_status()
        return [data["embedding"] for data in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\t", " ").replace("  ", " ")
    return "\n".join([line.rstrip() for line in text.splitlines() if line.strip()])


def load_pdf_documents_as_markdown() -> List[Document]:
    if not os.path.exists(DATA_DIR): return []
    docs = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    # Docling Instanziierung
    converter = DocumentConverter()

    for file_name in pdf_files:
        try:
            print(f"📄 Docling konvertiert: {file_name}...")
            file_path = os.path.join(DATA_DIR, file_name).replace("\\", "/")

            # Konvertierung mit Docling
            result = converter.convert(file_path)
            md_text = result.document.export_to_markdown()

            if len(md_text.strip()) < 100: continue
            docs.append(
                Document(page_content=clean_text(md_text), metadata={"source": file_name, "document_type": "pdf"}))
        except Exception as e:
            print(f"Fehler bei {file_name}: {e}")
    return docs


def load_web_documents() -> List[Document]:
    all_web_docs = []
    for url in URLS_TO_LEARN:
        try:
            for doc in WebBaseLoader(url).load():
                content = clean_text(doc.page_content)
                if len(content.strip()) < 100: continue
                doc.page_content = content
                doc.metadata.update({"source": url, "document_type": "webseite"})
                all_web_docs.append(doc)
        except Exception as e:
            print(f"Fehler bei {url}: {e}")
    return all_web_docs


def split_documents_intelligently(documents: List[Document]) -> List[Document]:
    print("✂️ Teile Dokumente logisch auf (Markdown Header + Recursive Splitter)...")
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
                                             strip_headers=False)
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=160)

    final_chunks = []
    for doc in documents:
        header_splits = md_splitter.split_text(doc.page_content) if doc.metadata.get("document_type") == "pdf" else [
            doc]
        for h_split in header_splits: h_split.metadata.update(doc.metadata)
        final_chunks.extend(recursive_splitter.split_documents(header_splits))
    return final_chunks


def build_expert_database():
    print("🚀 Starte Aufbau via LM Studio API & Docling...")
    all_docs = load_pdf_documents_as_markdown() + load_web_documents()
    if not all_docs: return

    embeddings = LMStudioEmbeddings()
    chunks = split_documents_intelligently(all_docs)

    if os.path.exists(QDRANT_DIR): shutil.rmtree(QDRANT_DIR)

    client = QdrantClient(path=QDRANT_DIR)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=len(embeddings.embed_query("Test")), distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        vector_store.add_documents(chunks[i: i + batch_size])
        print(f"  -> Batch {i // batch_size + 1} von {len(chunks) // batch_size + 1} gespeichert.")

    print("✅ Wissensdatenbank fertig!")


if __name__ == "__main__":
    build_expert_database()