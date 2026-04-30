import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. PDFs laden
pdf_ordner = "daten"
docs = []

# Wir laden alle PDFs, die ihr in den Ordner 'daten' gelegt habt
for file in os.listdir(pdf_ordner):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_ordner, file))
        docs.extend(loader.load())

if not docs:
    print("Keine PDFs im Ordner 'daten' gefunden! Bitte legt dort ein Dokument ab.")
else:
    # 2. Text in Stücke schneiden (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 3. Vektordatenbank erstellen
    # Achtung: Vorher in der CMD 'ollama pull nomic-embed-text' ausführen!
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Hier wird die DB gespeichert
    )
    print(f"Erfolgreich {len(chunks)} Textabschnitte in 'chroma_db' gespeichert.")