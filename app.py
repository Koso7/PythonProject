import datetime
import os
import tempfile
import uuid
from typing import List, Tuple

import requests
import streamlit as st
from docling.document_converter import DocumentConverter  # NEU: IBM Docling Engine
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Pflege-Assistent Pro",
    page_icon="⚖️",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
QDRANT_DIR = os.getenv("QDRANT_DIR", "./qdrant_db")
EMBEDDING_MODEL = "text-embedding-bge-m3"
LLM_MODEL = "mistralai/mistral-nemo-instruct-2407"

MAX_FILE_SIZE_MB = 10
MAX_TOTAL_TEXT_CHARS = 160_000

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "token" not in st.session_state: st.session_state.token = None
if "verify_user" not in st.session_state: st.session_state.verify_user = None
if "messages" not in st.session_state: st.session_state.messages = []
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""
if "user_documents" not in st.session_state: st.session_state.user_documents = []
if "last_user_sources" not in st.session_state: st.session_state.last_user_sources = []
if "last_expert_sources" not in st.session_state: st.session_state.last_expert_sources = []

# ------------------------------------------------------------
# API-HILFSFUNKTIONEN
# ------------------------------------------------------------
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}

def api_get_me():
    try:
        return requests.get(f"{API_URL}/me", headers=auth_headers(), timeout=10)
    except requests.RequestException:
        return None

def logout():
    st.session_state.token = None
    st.session_state.messages = []
    st.session_state.extracted_text = ""
    st.session_state.user_documents = []
    st.session_state.last_user_sources = []
    st.session_state.last_expert_sources = []
    st.rerun()

# ------------------------------------------------------------
# TEXT-HILFSFUNKTIONEN
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\t", " ").replace("  ", " ")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def remove_duplicate_docs(docs: List[Document]) -> List[Document]:
    unique_docs = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "")
        content_preview = clean_text(doc.page_content)[:350]
        key = (source, content_preview)
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs

# ------------------------------------------------------------
# KI-/RAG-RESSOURCEN
# ------------------------------------------------------------
class LMStudioEmbeddings(Embeddings):
    def __init__(self, base_url="http://localhost:1234/v1", model=EMBEDDING_MODEL):
        self.base_url = base_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(f"{self.base_url}/embeddings", json={"input": texts, "model": self.model})
        response.raise_for_status()
        return [data["embedding"] for data in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

@st.cache_resource
def get_embeddings():
    return LMStudioEmbeddings()

@st.cache_resource
def get_expert_database():
    client = QdrantClient(path=QDRANT_DIR)
    return QdrantVectorStore(client=client, collection_name="pflege_fachwissen", embedding=get_embeddings())

def call_lm_studio(system_prompt: str, chat_history: list) -> str:
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "messages": [{"role": "system", "content": system_prompt}] + chat_history,
        "temperature": 0.2,
        "max_tokens": 1500
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200: return response.json()["choices"][0]["message"]["content"]
        return f"Fehler von LM Studio: {response.status_code}"
    except Exception as e:
        return f"Verbindungsfehler zur KI: {e}"

# ------------------------------------------------------------
# PDF-VERARBEITUNG NUTZERDOKUMENTE (Docling Integration)
# ------------------------------------------------------------
def extract_user_documents_from_pdfs(uploaded_files) -> Tuple[str, List[Document]]:
    full_text = ""
    raw_docs = []
    converter = DocumentConverter()  # Docling Initialisierung

    for file in uploaded_files:
        if file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
            raise ValueError(f"Die Datei '{file.name}' ist größer als {MAX_FILE_SIZE_MB} MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            # Layouttreue Markdown Extraktion via Docling
            result = converter.convert(tmp_path)
            md_text = result.document.export_to_markdown()
            md_text = clean_text(md_text)

            if len(md_text) > 30:
                full_text += f"\n\n--- Dokument: {file.name} ---\n{md_text}"
                raw_docs.append(
                    Document(page_content=md_text, metadata={"source": file.name, "document_type": "nutzerdokument"}))
        finally:
            os.remove(tmp_path)

        if len(full_text) > MAX_TOTAL_TEXT_CHARS:
            full_text = full_text[:MAX_TOTAL_TEXT_CHARS]
            full_text += "\n\n[Hinweis: Der Text wurde gekürzt, weil sehr viele Daten hochgeladen wurden.]"
            break

    chunks = split_user_documents(raw_docs)
    return full_text, chunks

def split_user_documents(raw_docs: List[Document]) -> List[Document]:
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=160)

    final_chunks = []
    seen = set()
    for doc in raw_docs:
        header_splits = md_splitter.split_text(doc.page_content)
        for h_split in header_splits: h_split.metadata.update(doc.metadata)

        for chunk in recursive_splitter.split_documents(header_splits):
            content = clean_text(chunk.page_content)
            if len(content) < 80: continue

            key = (chunk.metadata.get("source", ""), content[:300])
            if key in seen: continue
            seen.add(key)

            chunk.page_content = content
            final_chunks.append(chunk)

    return final_chunks

# ------------------------------------------------------------
# RETRIEVAL NUTZERDOKUMENTE (In-Memory Qdrant)
# ------------------------------------------------------------
def search_user_documents(user_documents: List[Document], user_question: str, k: int = 3) -> List[Document]:
    if not user_documents: return []

    temp_db = QdrantVectorStore.from_documents(
        documents=user_documents,
        embedding=get_embeddings(),
        location=":memory:",
        collection_name=f"temp_user_docs_{uuid.uuid4().hex}"
    )

    docs = temp_db.as_retriever(search_kwargs={"k": k}).invoke(user_question)
    return remove_duplicate_docs(docs)

# ------------------------------------------------------------
# RETRIEVAL FACHWISSEN
# ------------------------------------------------------------
def build_expert_search_query(user_question: str, relevant_user_docs: List[Document]) -> str:
    user_doc_excerpt = "\n\n".join(doc.page_content[:900] for doc in relevant_user_docs[:6])
    search_query = f"""
Pflegegrad Widerspruch Pflegekasse Medizinischer Dienst MD Gutachten Begutachtung
Neues Begutachtungsassessment NBA Pflegebedürftigkeitsrichtlinien Begutachtungsrichtlinien
Module Mobilität kognitive kommunikative Fähigkeiten Verhaltensweisen psychische Problemlagen
Selbstversorgung krankheitsbedingte Anforderungen Gestaltung des Alltagslebens soziale Kontakte
Pflegegrad 2 Pflegegrad 3 Höherstufung Widerspruchsbegründung Unstimmigkeiten Gutachten

Nutzerfrage:
{user_question}

Relevante Auszüge aus Nutzerdokumenten:
{user_doc_excerpt}
"""
    return search_query

def search_expert_documents(expert_db, search_query: str, k: int = 3) -> List[Document]:
    docs = expert_db.as_retriever(search_kwargs={"k": k}).invoke(search_query)
    return remove_duplicate_docs(docs)

# ------------------------------------------------------------
# QUELLENFORMATIERUNG
# ------------------------------------------------------------
def format_docs_for_prompt(title: str, docs: List[Document]) -> str:
    if not docs: return f"{title}: Keine relevanten Textstellen gefunden."
    parts = [title]
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        parts.append(f"{title} {index}: {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def build_source_list(docs: List[Document]) -> List[dict]:
    sources = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unbekannte Quelle")
        preview = clean_text(doc.page_content)
        if len(preview) > 550: preview = preview[:550] + "..."

        key = (source, preview[:250])
        if key in seen: continue
        seen.add(key)
        sources.append({"nr": len(sources) + 1, "source": source, "preview": preview})
    return sources

# ------------------------------------------------------------
# ANTWORTGENERIERUNG (Gedächtnis-optimiert)
# ------------------------------------------------------------
def generate_rag_answer(expert_db, user_question: str, user_documents: List[Document], chat_history: list) -> Tuple[str, List[dict], List[dict]]:
    relevant_user_docs = search_user_documents(user_documents=user_documents, user_question=user_question, k=5)
    expert_search_query = build_expert_search_query(user_question=user_question, relevant_user_docs=relevant_user_docs)
    relevant_expert_docs = search_expert_documents(expert_db=expert_db, search_query=expert_search_query, k=5)

    user_context = format_docs_for_prompt(title="RELEVANTE NUTZERDOKUMENTSTELLEN", docs=relevant_user_docs)
    expert_context = format_docs_for_prompt(title="RELEVANTES FACHWISSEN", docs=relevant_expert_docs)

    template = f"""Du bist ein KI-gestützter Assistenzdienst zur strukturierten Vorbereitung eines Pflegegrad-Widerspruchs.

Wichtige Grenzen:
- Du ersetzt keine Rechtsberatung.
- Du ersetzt keine medizinische Begutachtung.
- Du darfst keine Tatsachen erfinden.
- Du darfst nur mit den bereitgestellten Nutzerdokumentstellen und dem bereitgestellten Fachwissen arbeiten.
- Wenn eine Information nicht in den bereitgestellten Texten steht, schreibe ausdrücklich, dass diese Information nicht vorliegt.
- Schreibe vollständig auf Deutsch.
- Verwende eine sachliche, behördentauftliche Sprache.

AUFGABE:
Beantworte exakt den Auftrag des Nutzers.
Wenn der Nutzer eine Prüfung oder Analyse verlangt, erstelle zuerst eine strukturierte Analyse und keinen vollständigen Widerspruchsbrief.
Nur wenn der Nutzer ausdrücklich einen vollständigen Widerspruchsbrief verlangt, erstelle einen vollständigen Brief. Nutze dafür die vorherige Analyse aus dem Chatverlauf.

RELEVANTE NUTZERDOKUMENTSTELLEN:
{user_context}

RELEVANTES FACHWISSEN:
{expert_context}
"""
    answer = call_lm_studio(system_prompt=template, chat_history=chat_history)
    user_sources = build_source_list(relevant_user_docs)
    expert_sources = build_source_list(relevant_expert_docs)
    return answer, user_sources, expert_sources

# ------------------------------------------------------------
# LOGIN & REGISTRIERUNG
# ------------------------------------------------------------
if st.session_state.token is None:
    st.title("🛡️ Pflegehilfe Online - Portal")
    st.caption("KI-gestützte Unterstützung bei der Vorbereitung eines Pflegegrad-Widerspruchs.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Nutzername")
            password = st.text_input("Passwort", type="password")
            submitted = st.form_submit_button("Anmelden", use_container_width=True)

            if submitted:
                try:
                    res = requests.post(f"{API_URL}/login", data={"username": username, "password": password}, timeout=10)
                    if res.status_code == 200:
                        st.session_state.token = res.json()["access_token"]
                        me = api_get_me()
                        if me is not None and me.status_code == 200:
                            st.success("Login erfolgreich.")
                            st.rerun()
                        else:
                            st.session_state.token = None
                            st.error("Login konnte nicht bestätigt werden.")
                    elif res.status_code == 403:
                        st.error("Das Konto ist noch nicht verifiziert.")
                    else:
                        st.error(res.json().get("detail", "Login fehlgeschlagen."))
                except requests.RequestException:
                    st.error("Backend nicht erreichbar. Läuft FastAPI auf Port 8000?")

    with col2:
        if not st.session_state.verify_user:
            st.subheader("Registrierung")
            with st.form("register_form"):
                reg_username = st.text_input("Neuer Nutzername")
                reg_email = st.text_input("E-Mail")
                reg_password = st.text_input("Neues Passwort", type="password")
                registered = st.form_submit_button("Konto erstellen", use_container_width=True)

                if registered:
                    try:
                        res = requests.post(f"{API_URL}/register", json={"username": reg_username, "email": reg_email, "password": reg_password}, timeout=10)
                        if res.status_code == 200:
                            st.session_state.verify_user = reg_username
                            st.success("Registrierung erfolgreich. Code im Backend-Terminal ablesen.")
                            st.rerun()
                        else:
                            st.error(res.json().get("detail", "Registrierung fehlgeschlagen."))
                    except requests.RequestException:
                        st.error("Backend nicht erreichbar. Läuft FastAPI auf Port 8000?")
        else:
            st.subheader("Konto verifizieren")
            st.info(f"Der Verifizierungscode für **{st.session_state.verify_user}** wird im Backend-Terminal angezeigt.")
            code = st.text_input("6-stelliger Code")

            if st.button("Verifizieren", use_container_width=True):
                try:
                    res = requests.post(f"{API_URL}/verify", json={"username": st.session_state.verify_user, "code": code}, timeout=10)
                    if res.status_code == 200:
                        st.success("Konto verifiziert. Bitte jetzt einloggen.")
                        st.session_state.verify_user = None
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Verifizierung fehlgeschlagen."))
                except requests.RequestException:
                    st.error("Backend nicht erreichbar.")

            if st.button("Zurück zur Registrierung"):
                st.session_state.verify_user = None
                st.rerun()

# ------------------------------------------------------------
# HAUPT-APP
# ------------------------------------------------------------
else:
    me_response = api_get_me()

    if me_response is None:
        st.error("Backend nicht erreichbar. Sie wurden aus Sicherheitsgründen ausgeloggt.")
        logout()
    if me_response.status_code != 200:
        st.error("Ihre Sitzung ist abgelaufen oder ungültig. Bitte erneut einloggen.")
        logout()

    current_user = me_response.json()

    with st.sidebar:
        st.success(f"Angemeldet als: **{current_user['username']}**")
        st.warning("⚠️ **Wichtiger Hinweis**\n\nDiese Anwendung erstellt nur KI-gestützte Entwürfe und Hinweise. Sie ersetzt keine rechtliche, medizinische oder pflegefachliche Beratung. Alle generierten Texte müssen sorgfältig geprüft werden.")
        st.divider()
        st.caption("Datenschutz: Hochgeladene persönliche Dokumente werden nur temporär während der Sitzung verarbeitet und nicht dauerhaft in der Wissensdatenbank gespeichert.")
        st.caption(f"LLM: {LLM_MODEL}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")

        if st.button("🚪 Ausloggen", use_container_width=True): logout()

    tab1, tab2, tab3 = st.tabs(["📄 Dokumente", "💬 KI-Assistent", "📅 Fristen"])

    # --- TAB 1: DOKUMENTE ---
    with tab1:
        st.subheader("Persönliche Dokumente hochladen")
        st.info("Laden Sie hier Bescheide, Gutachten, ärztliche Unterlagen oder Pflegetagebücher als PDF hoch. Die Inhalte werden nur für die aktuelle Sitzung verwendet.")

        uploaded_files = st.file_uploader("PDFs auswählen", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            st.write(f"Ausgewählte Dateien: **{len(uploaded_files)}**")
            for file in uploaded_files:
                st.caption(f"- {file.name} ({file.size / (1024 * 1024):.2f} MB)")

            if st.button("🚀 Dokumente einlesen", type="primary"):
                with st.spinner("Dokumente werden gelesen und in temporäre Chunks zerlegt (Docling)..."):
                    try:
                        extracted_text, user_documents = extract_user_documents_from_pdfs(uploaded_files)
                        st.session_state.extracted_text = extracted_text
                        st.session_state.user_documents = user_documents
                        st.session_state.messages = []
                        st.session_state.last_user_sources = []
                        st.session_state.last_expert_sources = []

                        st.success(f"Dokumente wurden erfolgreich eingelesen. Es wurden **{len(user_documents)} temporäre Textabschnitte** erstellt.")
                        with st.expander("Eingelesenen Text anzeigen"):
                            st.text_area("Extrahierter Text", value=st.session_state.extracted_text[:12_000], height=350)
                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten der PDFs: {e}")

        if st.session_state.user_documents:
            st.success(f"Es sind aktuell **{len(st.session_state.user_documents)} temporäre Nutzerdokument-Chunks** geladen.")
            if st.button("Geladene Dokumentdaten aus Sitzung löschen"):
                st.session_state.extracted_text = ""
                st.session_state.user_documents = []
                st.session_state.messages = []
                st.session_state.last_user_sources = []
                st.session_state.last_expert_sources = []
                st.rerun()

    # --- TAB 2: CHAT ---
    with tab2:
        st.subheader("Pflegegrad-Widerspruchsassistent")

        if not st.session_state.user_documents:
            st.info("Sie haben noch keine persönlichen Dokumente hochgeladen. Der Assistent kann allgemeine Fragen beantworten, aber keine individuelle Dokumentenprüfung durchführen.")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input("Beispiel: Prüfe Bescheid, Pflegetagebuch und ärztliche Unterlagen auf mögliche Unstimmigkeiten.")

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Suche relevante Nutzerdokumentstellen und Fachquellen..."):
                    try:
                        expert_db = get_expert_database()
                        # Übergabe der Historie (st.session_state.messages) zur Behebung des Gedächtnisverlusts
                        answer, user_sources, expert_sources = generate_rag_answer(
                            expert_db=expert_db,
                            user_question=user_prompt,
                            user_documents=st.session_state.user_documents,
                            chat_history=st.session_state.messages
                        )
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_user_sources = user_sources
                        st.session_state.last_expert_sources = expert_sources
                    except Exception as e:
                        st.error(f"KI-Fehler: {e}")

        if st.session_state.last_user_sources or st.session_state.last_expert_sources:
            st.divider()

        if st.session_state.last_user_sources:
            st.subheader("Verwendete Nutzerdokumentstellen")
            for source in st.session_state.last_user_sources:
                with st.expander(f"Nutzerdokument {source['nr']}: {source['source']}"):
                    st.write(source["preview"])

        if st.session_state.last_expert_sources:
            st.subheader("Verwendete Fachquellen")
            for source in st.session_state.last_expert_sources:
                with st.expander(f"Fachquelle {source['nr']}: {source['source']}"):
                    st.write(source["preview"])

    # --- TAB 3: FRISTEN ---
    with tab3:
        st.subheader("Fristenrechner")
        st.warning("Der Fristenrechner dient nur als Orientierung. Die genaue Frist sollte im Zweifel rechtlich geprüft werden.")

        received_date = st.date_input("Eingangsdatum des Pflegebescheids", value=datetime.date.today())
        simple_deadline = received_date + datetime.timedelta(days=30)

        st.info(f"Die grob berechnete Widerspruchsfrist endet voraussichtlich am **{simple_deadline.strftime('%d.%m.%Y')}**.")

        if simple_deadline.weekday() == 5:
            adjusted_deadline = simple_deadline + datetime.timedelta(days=2)
            st.warning(f"Das berechnete Datum fällt auf einen Samstag. Der nächste Werktag wäre der **{adjusted_deadline.strftime('%d.%m.%Y')}**.")
        elif simple_deadline.weekday() == 6:
            adjusted_deadline = simple_deadline + datetime.timedelta(days=1)
            st.warning(f"Das berechnete Datum fällt auf einen Sonntag. Der nächste Werktag wäre der **{adjusted_deadline.strftime('%d.%m.%Y')}**.")

        st.caption("Gesetzliche Feiertage werden in diesem Prototyp noch nicht automatisch berücksichtigt.")