import datetime
import os
import tempfile
import uuid
from typing import List, Tuple

import requests
import streamlit as st
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from fpdf import FPDF

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
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
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
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None
if "last_generated_appeal" not in st.session_state: st.session_state.last_generated_appeal = ""


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
    st.session_state.last_generated_appeal = ""
    st.rerun()


# ------------------------------------------------------------
# TEXT- & PDF-HILFSFUNKTIONEN
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


def generate_pdf_letter(absender_name, absender_adresse, kasse_name, kasse_adresse, versichert_name, versichert_nr,
                        bescheid_datum, brief_text):
    pdf = FPDF()
    pdf.add_page()

    # --- NEU: Sonderzeichen-Filter für PDF-Kompatibilität ---
    def clean_pdf_text(text):
        if not text: return ""
        replacements = {
            "–": "-", "—": "-",  # Lange Gedankenstriche zu Bindestrichen
            "„": '"', "“": '"', "”": '"',  # Geschwungene Anführungszeichen zu geraden
            "‘": "'", "’": "'",  # Typografische Apostrophe zu geraden
            "•": "-",  # Bulletpoints zu Bindestrichen
            "€": "Euro"  # Euro-Zeichen sicherheitshalber ausschreiben
        }
        for alt, neu in replacements.items():
            text = text.replace(alt, neu)
        return text

    # Texte vor dem Einfügen bereinigen
    absender_name = clean_pdf_text(absender_name)
    absender_adresse = clean_pdf_text(absender_adresse)
    kasse_name = clean_pdf_text(kasse_name)
    kasse_adresse = clean_pdf_text(kasse_adresse)
    versichert_name = clean_pdf_text(versichert_name)
    versichert_nr = clean_pdf_text(versichert_nr)
    brief_text = clean_pdf_text(brief_text)
    # ---------------------------------------------------------

    # Helvetica ist im Standard-Lieferumfang von fpdf2
    pdf.set_font("Helvetica", size=11)

    # Absender
    pdf.cell(0, 5, text=absender_name, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, text=absender_adresse, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)

    # Empfänger (Pflegekasse)
    pdf.cell(0, 5, text=kasse_name, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, text=kasse_adresse, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(15)

    # Datum rechtsbündig
    heute = datetime.date.today().strftime("%d.%m.%Y")
    pdf.cell(0, 5, text=f"Datum: {heute}", align="R", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # Betreff (Fett)
    pdf.set_font("Helvetica", style="B", size=11)
    betreff = f"Widerspruch gegen den Bescheid vom {bescheid_datum}"
    pdf.cell(0, 5, text=betreff, new_x="LMARGIN", new_y="NEXT")

    # Versicherungsnummer
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 5, text=f"Versicherte Person: {versichert_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, text=f"Versichertennummer: {versichert_nr}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # Brieftext (MultiCell für automatischen Zeilenumbruch)
    pdf.multi_cell(0, 6, text=brief_text)

    pdf.ln(15)

    # Grußformel & Unterschrift
    pdf.cell(0, 5, text="Mit freundlichen Grüßen,", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.cell(0, 5, text="__________________________________________________", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, text=f"(Unterschrift {absender_name})", new_x="LMARGIN", new_y="NEXT")

    return pdf.output()


# ------------------------------------------------------------
# KI-/RAG-RESSOURCEN
# ------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(
        openai_api_base=LM_STUDIO_URL,
        openai_api_key="lm-studio",
        model=EMBEDDING_MODEL,
        check_embedding_ctx_length=False
    )


@st.cache_resource
def get_llm():
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key="lm-studio",
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=2000  # Erhöht für längere Briefe
    )


@st.cache_resource
def get_expert_database():
    client = QdrantClient(path=QDRANT_DIR)
    return QdrantVectorStore(client=client, collection_name="pflege_fachwissen", embedding=get_embeddings())


# ------------------------------------------------------------
# PDF-VERARBEITUNG NUTZERDOKUMENTE
# ------------------------------------------------------------
def extract_user_documents_from_pdfs(uploaded_files) -> Tuple[str, List[Document]]:
    full_text = ""
    raw_docs = []
    converter = DocumentConverter()

    for file in uploaded_files:
        if file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
            raise ValueError(f"Die Datei '{file.name}' ist größer als {MAX_FILE_SIZE_MB} MB.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
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
            full_text += "\n\n[Hinweis: Text gekürzt (Limit erreicht).]"
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
        if not header_splits:
            header_splits = [doc]

        for h_split in header_splits:
            h_split.metadata.update(doc.metadata)

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
# RETRIEVAL LOGIK
# ------------------------------------------------------------
def search_user_documents(user_documents: List[Document], user_question: str, k: int = 4) -> List[Document]:
    if not user_documents: return []
    temp_db = QdrantVectorStore.from_documents(
        documents=user_documents,
        embedding=get_embeddings(),
        location=":memory:",
        collection_name=f"temp_user_docs_{uuid.uuid4().hex}"
    )
    return remove_duplicate_docs(temp_db.as_retriever(search_kwargs={"k": k}).invoke(user_question))


def build_expert_search_query(user_question: str, relevant_user_docs: List[Document]) -> str:
    user_doc_excerpt = "\n\n".join(doc.page_content[:900] for doc in relevant_user_docs[:6])
    return f"""Pflegegrad Widerspruch Musterbrief Pflegekasse Medizinischer Dienst MD Gutachten Begutachtung
Nutzerfrage: {user_question}
Relevante Auszüge aus Nutzerdokumenten: {user_doc_excerpt}"""


def search_expert_documents(expert_db, search_query: str, k: int = 4) -> List[Document]:
    return remove_duplicate_docs(expert_db.as_retriever(search_kwargs={"k": k}).invoke(search_query))


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
        preview = clean_text(doc.page_content)[:250] + "..." if len(doc.page_content) > 250 else doc.page_content
        key = (source, preview[:100])
        if key not in seen:
            seen.add(key)
            sources.append({"nr": len(sources) + 1, "source": source, "preview": preview})
    return sources


def generate_rag_answer(expert_db, user_question: str, user_documents: List[Document], chat_history: list) -> Tuple[
    str, List[dict], List[dict]]:
    relevant_user_docs = search_user_documents(user_documents, user_question, k=6)
    expert_search_query = build_expert_search_query(user_question, relevant_user_docs)
    relevant_expert_docs = search_expert_documents(expert_db, expert_search_query, k=5)

    user_context = format_docs_for_prompt("RELEVANTE NUTZERDOKUMENTSTELLEN", relevant_user_docs)
    expert_context = format_docs_for_prompt("RELEVANTES FACHWISSEN (inkl. Musterbriefe)", relevant_expert_docs)

    system_prompt = f"""Du bist ein KI-gestützter Assistenzdienst zur strukturierten Vorbereitung eines Pflegegrad-Widerspruchs.

Regeln:
- Keine Rechtsberatung ersetzen, keine Tatsachen erfinden.
- Antworte basierend auf den Dokumenten und dem Fachwissen.
- Falls du einen Widerspruchsbrief verfasst, orientiere dich ZWINGEND an eventuell vorhandenen Musterbriefen im Fachwissen (wie Musterbrief_Bescheid_der_Pflegekasse.pdf). 
- Formuliere formell, sachlich und behördentauglich.

NUTZERDOKUMENTE:
{user_context}

FACHWISSEN:
{expert_context}
"""
    llm = get_llm()
    messages = [{"role": "system", "content": system_prompt}] + chat_history

    try:
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        answer = f"Verbindungsfehler zur KI: {e}"

    return answer, build_source_list(relevant_user_docs), build_source_list(relevant_expert_docs)


# ------------------------------------------------------------
# LOGIN & REGISTRIERUNG
# ------------------------------------------------------------
if st.session_state.token is None:
    st.title("🛡️ Pflegehilfe Online - Portal")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Nutzername")
            password = st.text_input("Passwort", type="password")
            if st.form_submit_button("Anmelden", use_container_width=True):
                try:
                    res = requests.post(f"{API_URL}/login", data={"username": username, "password": password},
                                        timeout=10)
                    if res.status_code == 200:
                        st.session_state.token = res.json()["access_token"]
                        if api_get_me() is not None:
                            st.rerun()
                    elif res.status_code == 403:
                        st.error("Konto nicht verifiziert.")
                    else:
                        st.error("Login fehlgeschlagen.")
                except requests.RequestException:
                    st.error("Backend nicht erreichbar.")
    with col2:
        if not st.session_state.verify_user:
            st.subheader("Registrierung")
            with st.form("register_form"):
                reg_username = st.text_input("Neuer Nutzername")
                reg_email = st.text_input("E-Mail")
                reg_password = st.text_input("Neues Passwort", type="password")
                if st.form_submit_button("Konto erstellen", use_container_width=True):
                    try:
                        res = requests.post(f"{API_URL}/register", json={"username": reg_username, "email": reg_email,
                                                                         "password": reg_password}, timeout=10)
                        if res.status_code == 200:
                            st.session_state.verify_user = reg_username
                            st.rerun()
                        else:
                            st.error(res.json().get("detail", "Fehler"))
                    except requests.RequestException:
                        st.error("Backend offline.")
        else:
            st.subheader("Konto verifizieren")
            code = st.text_input("6-stelliger Code (siehe Backend-Terminal)")
            if st.button("Verifizieren", use_container_width=True):
                res = requests.post(f"{API_URL}/verify", json={"username": st.session_state.verify_user, "code": code})
                if res.status_code == 200:
                    st.session_state.verify_user = None
                    st.success("Erfolgreich! Bitte einloggen.")
                else:
                    st.error("Code falsch.")

# ------------------------------------------------------------
# HAUPT-APP
# ------------------------------------------------------------
else:
    me_response = api_get_me()
    if me_response is None or me_response.status_code != 200: logout()

    with st.sidebar:
        st.success(f"Angemeldet als: **{me_response.json()['username']}**")
        if st.button("🚪 Ausloggen", use_container_width=True): logout()
        st.divider()
        st.warning(
            "⚠️ **Wichtiger Hinweis**\nDie KI ersetzt keine rechtliche Beratung. Generierte Texte müssen geprüft werden.")

    tab1, tab2, tab3, tab4 = st.tabs(["📄 Dokumente", "💬 KI-Assistent", "📅 Fristen", "🖨️ PDF Export"])

    # --- TAB 1: DOKUMENTE ---
    with tab1:
        st.subheader("Persönliche Dokumente hochladen")
        uploaded_files = st.file_uploader("Bescheide, Gutachten, Pflegetagebücher als PDF", type="pdf",
                                          accept_multiple_files=True)
        if uploaded_files and st.button("🚀 Dokumente einlesen", type="primary"):
            with st.spinner("Dokumente werden gelesen..."):
                extracted_text, user_documents = extract_user_documents_from_pdfs(uploaded_files)
                st.session_state.extracted_text = extracted_text
                st.session_state.user_documents = user_documents
                st.success(f"{len(user_documents)} Textabschnitte erstellt.")

    # --- TAB 2: CHAT (Überarbeitet für ChatGPT-Like Layout) ---
    with tab2:
        st.subheader("Pflegegrad-Widerspruchsassistent")

        if not st.session_state.user_documents:
            st.info("💡 Bitte laden Sie im ersten Tab Ihre Dokumente hoch, damit der Assistent individuell helfen kann.")

        # Vorgefertigte Buttons (Feature 2)
        st.write("### Schnellauswahl")
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        if col_btn1.button("📊 Differenzanalyse"):
            st.session_state.pending_prompt = "Führe eine detaillierte Differenzanalyse zwischen dem Gutachten des Medizinischen Dienstes (MD) und den eingereichten ärztlichen Unterlagen sowie dem Pflegetagebuch durch. Zeige alle Diskrepanzen auf."

        if col_btn2.button("📖 Dokumente einlesen & prüfen"):
            st.session_state.pending_prompt = "Bitte lies alle hochgeladenen Nutzerdokumente gründlich. Fasse die wichtigsten pflegerelevanten Einschränkungen und Diagnosen kurz zusammen und bereite dich darauf vor, daraus Argumente abzuleiten."

        if col_btn3.button("✍️ Widerspruchsgutachten schreiben"):
            st.session_state.pending_prompt = "Verfasse nun ein vollständiges, formelles Widerspruchsschreiben. Nutze das gesamte RAG-Wissen, achte besonders auf die Struktur des hochgeladenen 'Musterbrief_Bescheid_der_Pflegekasse.pdf' und beziehe dich in der Begründung auf die Diskrepanzen in meinen Nutzerdokumenten."

        st.divider()

        # Scrollbarer Container für den Nachrichtenverlauf (Feature 1)
        chat_container = st.container(height=450)

        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat Input logic
        user_input = st.chat_input("Deine Nachricht oder Frage eingeben...")

        # Trigger either by manual input or predefined button
        prompt_to_execute = st.session_state.pending_prompt if st.session_state.pending_prompt else user_input

        if prompt_to_execute:
            st.session_state.pending_prompt = None  # Reset

            # User Message anzeigen & speichern
            st.session_state.messages.append({"role": "user", "content": prompt_to_execute})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt_to_execute)

                with st.chat_message("assistant"):
                    with st.spinner("KI generiert Antwort..."):
                        expert_db = get_expert_database()
                        answer, user_sources, expert_sources = generate_rag_answer(
                            expert_db=expert_db,
                            user_question=prompt_to_execute,
                            user_documents=st.session_state.user_documents,
                            chat_history=st.session_state.messages
                        )
                        st.markdown(answer)

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.last_user_sources = user_sources
                        st.session_state.last_expert_sources = expert_sources

                        # Letzte KI Antwort merken, falls es wie ein Brief aussieht, fürs PDF
                        st.session_state.last_generated_appeal = answer

        # Quellen unter dem Chat anzeigen
        if st.session_state.last_user_sources or st.session_state.last_expert_sources:
            with st.expander("📚 Verwendete Quellen für die letzte Antwort ansehen"):
                if st.session_state.last_user_sources:
                    st.write("**Eigene Dokumente:**")
                    for s in st.session_state.last_user_sources:
                        st.caption(f"- {s['source']}: {s['preview']}")
                if st.session_state.last_expert_sources:
                    st.write("**Fachwissen:**")
                    for s in st.session_state.last_expert_sources:
                        st.caption(f"- {s['source']}: {s['preview']}")

    # --- TAB 3: FRISTEN ---
    with tab3:
        st.subheader("Fristenrechner")
        received_date = st.date_input("Eingangsdatum des Bescheids", value=datetime.date.today())
        deadline = received_date + datetime.timedelta(days=30)
        st.info(f"Die grob berechnete Frist endet am **{deadline.strftime('%d.%m.%Y')}**.")

    # --- TAB 4: PDF EXPORT (Feature 3) ---
    with tab4:
        st.subheader("Widerspruch als fertiges PDF exportieren")
        st.write(
            "Füllen Sie die fehlenden Daten aus, überprüfen Sie den von der KI verfassten Text und generieren Sie Ihr fertiges PDF zum Ausdrucken und Unterschreiben.")

        col_form1, col_form2 = st.columns(2)

        with col_form1:
            st.write("**Angaben zum Versicherten / Absender**")
            absender_name = st.text_input("Vor- und Nachname (Absender)")
            absender_adresse = st.text_input("Straße, Hausnummer, PLZ, Ort (Absender)")
            versichert_name = st.text_input("Name der pflegebedürftigen Person (falls abweichend)",
                                            help="Leer lassen, wenn Sie selbst betroffen sind.")
            versichert_nr = st.text_input("Versichertennummer")

        with col_form2:
            st.write("**Angaben zur Pflegekasse**")
            kasse_name = st.text_input("Name der Pflegekasse", value="Pflegekasse bei der ...")
            kasse_adresse = st.text_input("Straße, Hausnummer, PLZ, Ort (Pflegekasse)")
            bescheid_datum = st.text_input("Datum des Ablehnungsbescheids", value="TT.MM.JJJJ")

        # Fallback falls Namen identisch sind
        if not versichert_name:
            versichert_name = absender_name

        st.write("**Widerspruchstext (Begründung)**")
        st.caption(
            "Sie können den Text hier noch manuell anpassen, bevor Sie das PDF generieren. Der Text wird automatisch mit der letzten Antwort der KI vorausgefüllt.")

        # Vorausfüllen mit der letzen KI Antwort
        brief_text = st.text_area("Haupttext", value=st.session_state.last_generated_appeal, height=400)

        if st.button("📄 PDF Generieren", type="primary"):
            if not absender_name or not kasse_name or not brief_text:
                st.error("Bitte füllen Sie mindestens Name, Kasse und den Text aus.")
            else:
                try:
                    pdf_bytes = generate_pdf_letter(
                        absender_name=absender_name,
                        absender_adresse=absender_adresse,
                        kasse_name=kasse_name,
                        kasse_adresse=kasse_adresse,
                        versichert_name=versichert_name,
                        versichert_nr=versichert_nr,
                        bescheid_datum=bescheid_datum,
                        brief_text=brief_text
                    )

                    st.success("PDF wurde erfolgreich generiert!")
                    st.download_button(
                        label="📥 PDF Herunterladen",
                        data=pdf_bytes,
                        file_name="Widerspruch_Pflegegrad.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Fehler bei der PDF-Erstellung: {e}")