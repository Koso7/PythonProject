import os
import os
# --- NEU: User-Agent setzen, um Warnungen und Blockaden von Webseiten zu vermeiden ---
os.environ["USER_AGENT"] = "PflegeAssistentBot/1.0"
# ... (der Rest deines Codes bleibt exakt gleich)
from langchain_community.document_loaders import PyPDFDirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


def build_expert_database():
    print("📚 Starte Wissens-Upload für die KI...")

    all_documents = []

    # --- 1. PDFs aus dem Ordner laden ---
    if os.path.exists("./daten"):
        pdf_loader = PyPDFDirectoryLoader("./daten")
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            all_documents.extend(pdf_docs)
            print(f"📄 {len(pdf_docs)} PDF-Seiten gefunden und gelesen.")
        else:
            print("ℹ️ Keine PDFs im Ordner 'daten' gefunden.")
    else:
        print("⚠️ Ordner 'daten' existiert nicht. Bitte anlegen!")

    # --- 2. Webseiten auslesen ---
    # Hier trägst du einfach die URLs ein, die die KI auswendig lernen soll
    urls_to_learn = [
        "https://www.bundesgesundheitsministerium.de/themen/pflege/pflegebeduerftigkeit/pflegegrade.html",
        "https://www.pflege.de/pflegekasse-pflegerecht/pflegegrade/widerspruch/",
        "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/pflegegrad-abgelehnt-so-wehren-sie-sich-mit-widerspruch-und-klage-11547",
        "https://www.vdk.de/aktuelles/aktuelle-meldungen/artikel/widerspruch-gegen-pflegegrad-lohnt-sich-oft/",
        "https://www.pflege-betreuer.de/de/pflegewissen/pflegerecht-und-ansprueche/widerspruch-gegen-die-pflegegrad-einstufung-einlegen",
        "https://www.verbraucherzentrale.de/wissen/gesundheit-pflege/pflegeantrag-und-leistungen/pflegegrad-beantragen-so-gehts-13413",
        "https://www.pflege.de/pflegekasse-pflegerecht/pflegegrade/beantragen/",
        "https://www.bundesgesundheitsministerium.de/themen/pflege/online-ratgeber-pflege/pflegebeduerftig-was-nun"


        # "Hier kannst du weitere URLs mit Komma getrennt einfügen"
    ]

    if urls_to_learn:
        print("🌐 Lade und lese Webseiten...")
        try:
            web_loader = WebBaseLoader(urls_to_learn)
            web_docs = web_loader.load()
            all_documents.extend(web_docs)
            print(f"🌐 {len(web_docs)} Webseite(n) erfolgreich gelesen.")
        except Exception as e:
            print(f"⚠️ Fehler beim Lesen der Webseiten: {e}")

    # --- 3. Verarbeitung ---
    if not all_documents:
        print("❌ Nichts zu tun. Weder PDFs noch Webseiten gefunden.")
        return

    print(f"✂️ Teile den gesamten Text in häppchengerechte Abschnitte...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)

    print("🧠 Speichere das Wissen dauerhaft in der Vektor-Datenbank (ChromaDB)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("✅ Perfekt! Die KI kennt jetzt sowohl die PDFs als auch die Webseiten.")


if __name__ == "__main__":
    build_expert_database()