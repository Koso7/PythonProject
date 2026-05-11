import datetime
import os
import uuid
from typing import List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-nemo")

MAX_FILE_SIZE_MB = 10
MAX_TOTAL_TEXT_CHARS = 160_000

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
if "token" not in st.session_state:
    st.session_state.token = None

if "verify_user" not in st.session_state:
    st.session_state.verify_user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "user_documents" not in st.session_state:
    st.session_state.user_documents = []

if "last_user_sources" not in st.session_state:
    st.session_state.last_user_sources = []

if "last_expert_sources" not in st.session_state:
    st.session_state.last_expert_sources = []


# ------------------------------------------------------------
# API-HILFSFUNKTIONEN
# ------------------------------------------------------------
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_get_me():
    try:
        return requests.get(
            f"{API_URL}/me",
            headers=auth_headers(),
            timeout=10,
        )
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
    text = text.replace("\x00", " ")
    text = text.replace("\t", " ")
    text = text.replace("  ", " ")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


def remove_duplicate_docs(docs: List[Document]) -> List[Document]:
    unique_docs = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content_preview = clean_text(doc.page_content)[:350]

        key = (source, page, content_preview)

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    return unique_docs


# ------------------------------------------------------------
# KI-/RAG-RESSOURCEN
# ------------------------------------------------------------
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource
def get_expert_database():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="pflege_fachwissen",
    )


@st.cache_resource
def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.0,
    )


# ------------------------------------------------------------
# PDF-VERARBEITUNG NUTZERDOKUMENTE
# ------------------------------------------------------------
def extract_user_documents_from_pdfs(uploaded_files) -> Tuple[str, List[Document]]:
    full_text = ""
    page_documents = []

    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Die Datei '{file.name}' ist größer als {MAX_FILE_SIZE_MB} MB."
            )

        reader = PdfReader(file)

        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_text = clean_text(page_text)

            if len(page_text.strip()) < 30:
                continue

            full_text += f"\n\n--- Dokument: {file.name}, Seite {page_index} ---\n"
            full_text += page_text

            page_documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": file.name,
                        "page": page_index,
                        "document_type": "nutzerdokument",
                    },
                )
            )

            if len(full_text) > MAX_TOTAL_TEXT_CHARS:
                full_text = full_text[:MAX_TOTAL_TEXT_CHARS]
                full_text += "\n\n[Hinweis: Der Text wurde gekürzt, weil sehr viele Daten hochgeladen wurden.]"
                break

    chunks = split_user_documents(page_documents)

    return full_text, chunks


def split_user_documents(page_documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=160,
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

    chunks = splitter.split_documents(page_documents)

    cleaned_chunks = []
    seen = set()

    for index, chunk in enumerate(chunks):
        content = clean_text(chunk.page_content)

        if len(content) < 80:
            continue

        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", "")
        key = (source, page, content[:300])

        if key in seen:
            continue

        seen.add(key)

        chunk.page_content = content
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_size"] = len(content)

        cleaned_chunks.append(chunk)

    return cleaned_chunks


# ------------------------------------------------------------
# RETRIEVAL NUTZERDOKUMENTE
# ------------------------------------------------------------
def search_user_documents(
    user_documents: List[Document],
    user_question: str,
    k: int = 8,
) -> List[Document]:
    if not user_documents:
        return []

    embeddings = get_embeddings()

    collection_name = f"user_docs_{uuid.uuid4().hex}"

    temp_db = Chroma.from_documents(
        documents=user_documents,
        embedding=embeddings,
        collection_name=collection_name,
    )

    retriever = temp_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": min(30, max(k, len(user_documents))),
            "lambda_mult": 0.45,
        },
    )

    docs = retriever.invoke(user_question)

    return remove_duplicate_docs(docs)


# ------------------------------------------------------------
# RETRIEVAL FACHWISSEN
# ------------------------------------------------------------
def build_expert_search_query(
    user_question: str,
    relevant_user_docs: List[Document],
) -> str:
    user_doc_excerpt = "\n\n".join(
        doc.page_content[:900] for doc in relevant_user_docs[:6]
    )

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


def search_expert_documents(
    expert_db,
    search_query: str,
    k: int = 10,
) -> List[Document]:
    retriever = expert_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 45,
            "lambda_mult": 0.35,
        },
    )

    docs = retriever.invoke(search_query)

    return remove_duplicate_docs(docs)


# ------------------------------------------------------------
# QUELLENFORMATIERUNG
# ------------------------------------------------------------
def format_docs_for_prompt(title: str, docs: List[Document]) -> str:
    if not docs:
        return f"{title}: Keine relevanten Textstellen gefunden."

    parts = [title]

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        page = doc.metadata.get("page", None)

        if page:
            source_label = f"{title} {index}: {source}, Seite {page}"
        else:
            source_label = f"{title} {index}: {source}"

        parts.append(
            f"{source_label}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def build_source_list(docs: List[Document]) -> List[dict]:
    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unbekannte Quelle")
        page = doc.metadata.get("page", None)

        preview = clean_text(doc.page_content)
        if len(preview) > 550:
            preview = preview[:550] + "..."

        key = (source, page, preview[:250])

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "nr": len(sources) + 1,
                "source": source,
                "page": page,
                "preview": preview,
            }
        )

    return sources


# ------------------------------------------------------------
# ANTWORTGENERIERUNG
# ------------------------------------------------------------
def generate_rag_answer(
    expert_db,
    user_question: str,
    user_documents: List[Document],
) -> Tuple[str, List[dict], List[dict]]:
    relevant_user_docs = search_user_documents(
        user_documents=user_documents,
        user_question=user_question,
        k=8,
    )

    expert_search_query = build_expert_search_query(
        user_question=user_question,
        relevant_user_docs=relevant_user_docs,
    )

    relevant_expert_docs = search_expert_documents(
        expert_db=expert_db,
        search_query=expert_search_query,
        k=10,
    )

    user_context = format_docs_for_prompt(
        title="RELEVANTE NUTZERDOKUMENTSTELLEN",
        docs=relevant_user_docs,
    )

    expert_context = format_docs_for_prompt(
        title="RELEVANTES FACHWISSEN",
        docs=relevant_expert_docs,
    )

    llm = get_llm()

    template = """
Du bist ein KI-gestützter Assistenzdienst zur strukturierten Vorbereitung eines Pflegegrad-Widerspruchs.

Wichtige Grenzen:
- Du ersetzt keine Rechtsberatung.
- Du ersetzt keine medizinische Begutachtung.
- Du darfst keine Tatsachen erfinden.
- Du darfst nur mit den bereitgestellten Nutzerdokumentstellen und dem bereitgestellten Fachwissen arbeiten.
- Wenn eine Information nicht in den bereitgestellten Texten steht, schreibe ausdrücklich, dass diese Information nicht vorliegt.
- Schreibe vollständig auf Deutsch.
- Verwende eine sachliche, behördentaugliche Sprache.

AUFGABE:
Beantworte exakt den Auftrag des Nutzers.

Wenn der Nutzer eine Prüfung oder Analyse verlangt, erstelle zuerst eine strukturierte Analyse und keinen vollständigen Widerspruchsbrief.

Struktur bei Dokumentenprüfung:
1. Kurzfazit
2. Welche relevanten Probleme stehen in den Nutzerdokumenten?
3. Was scheint im Gutachten/Bescheid berücksichtigt worden zu sein?
4. Welche möglichen Unstimmigkeiten oder Lücken gibt es?
5. Welche Pflegegrad-Module sind betroffen?
6. Welche Argumentationsgrundlagen für einen Widerspruch ergeben sich?
7. Welche Informationen fehlen oder sollten noch geprüft werden?
8. Optional: Formulierungsvorschläge für einzelne Widerspruchsargumente

Nur wenn der Nutzer ausdrücklich einen vollständigen Widerspruchsbrief verlangt, erstelle einen vollständigen Brief.

RELEVANTE NUTZERDOKUMENTSTELLEN:
{user_context}

RELEVANTES FACHWISSEN:
{expert_context}

NUTZERFRAGE:
{question}

ANTWORT:
"""

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "user_context": user_context,
            "expert_context": expert_context,
            "question": user_question,
        }
    )

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
                    res = requests.post(
                        f"{API_URL}/login",
                        data={
                            "username": username,
                            "password": password,
                        },
                        timeout=10,
                    )

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
                        try:
                            detail = res.json().get("detail", "Login fehlgeschlagen.")
                        except Exception:
                            detail = "Login fehlgeschlagen."
                        st.error(detail)

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
                        res = requests.post(
                            f"{API_URL}/register",
                            json={
                                "username": reg_username,
                                "email": reg_email,
                                "password": reg_password,
                            },
                            timeout=10,
                        )

                        if res.status_code == 200:
                            st.session_state.verify_user = reg_username
                            st.success("Registrierung erfolgreich. Code im Backend-Terminal ablesen.")
                            st.rerun()
                        else:
                            try:
                                detail = res.json().get("detail", "Registrierung fehlgeschlagen.")
                            except Exception:
                                detail = "Registrierung fehlgeschlagen."
                            st.error(detail)

                    except requests.RequestException:
                        st.error("Backend nicht erreichbar. Läuft FastAPI auf Port 8000?")

        else:
            st.subheader("Konto verifizieren")
            st.info(
                f"Der Verifizierungscode für **{st.session_state.verify_user}** "
                "wird im Backend-Terminal angezeigt."
            )

            code = st.text_input("6-stelliger Code")

            if st.button("Verifizieren", use_container_width=True):
                try:
                    res = requests.post(
                        f"{API_URL}/verify",
                        json={
                            "username": st.session_state.verify_user,
                            "code": code,
                        },
                        timeout=10,
                    )

                    if res.status_code == 200:
                        st.success("Konto verifiziert. Bitte jetzt einloggen.")
                        st.session_state.verify_user = None
                        st.rerun()
                    else:
                        try:
                            detail = res.json().get("detail", "Verifizierung fehlgeschlagen.")
                        except Exception:
                            detail = "Verifizierung fehlgeschlagen."
                        st.error(detail)

                except requests.RequestException:
                    st.error("Backend nicht erreichbar. Läuft FastAPI auf Port 8000?")

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

        st.warning(
            "⚠️ **Wichtiger Hinweis**\n\n"
            "Diese Anwendung erstellt nur KI-gestützte Entwürfe und Hinweise. "
            "Sie ersetzt keine rechtliche, medizinische oder pflegefachliche Beratung. "
            "Alle generierten Texte müssen sorgfältig geprüft werden."
        )

        st.divider()

        st.caption(
            "Datenschutz: Hochgeladene persönliche Dokumente werden nur temporär "
            "während der Sitzung verarbeitet und nicht dauerhaft in der Wissensdatenbank gespeichert."
        )

        st.caption(f"LLM: {LLM_MODEL}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL}")

        if st.button("🚪 Ausloggen", use_container_width=True):
            logout()

    tab1, tab2, tab3 = st.tabs(
        [
            "📄 Dokumente",
            "💬 KI-Assistent",
            "📅 Fristen",
        ]
    )

    # --------------------------------------------------------
    # TAB 1: DOKUMENTE
    # --------------------------------------------------------
    with tab1:
        st.subheader("Persönliche Dokumente hochladen")
        st.info(
            "Laden Sie hier Bescheide, Gutachten, ärztliche Unterlagen oder Pflegetagebücher als PDF hoch. "
            "Die Inhalte werden nur für die aktuelle Sitzung verwendet."
        )

        uploaded_files = st.file_uploader(
            "PDFs auswählen",
            type="pdf",
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"Ausgewählte Dateien: **{len(uploaded_files)}**")

            for file in uploaded_files:
                file_size_mb = file.size / (1024 * 1024)
                st.caption(f"- {file.name} ({file_size_mb:.2f} MB)")

            if st.button("🚀 Dokumente einlesen", type="primary"):
                with st.spinner("Dokumente werden gelesen und in temporäre Chunks zerlegt..."):
                    try:
                        extracted_text, user_documents = extract_user_documents_from_pdfs(uploaded_files)

                        st.session_state.extracted_text = extracted_text
                        st.session_state.user_documents = user_documents
                        st.session_state.messages = []
                        st.session_state.last_user_sources = []
                        st.session_state.last_expert_sources = []

                        st.success(
                            f"Dokumente wurden erfolgreich eingelesen. "
                            f"Es wurden **{len(user_documents)} temporäre Textabschnitte** erstellt."
                        )

                        with st.expander("Eingelesenen Text anzeigen"):
                            st.text_area(
                                "Extrahierter Text",
                                value=st.session_state.extracted_text[:12_000],
                                height=350,
                            )

                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten der PDFs: {e}")

        if st.session_state.user_documents:
            st.success(
                f"Es sind aktuell **{len(st.session_state.user_documents)} temporäre Nutzerdokument-Chunks** geladen."
            )

            if st.button("Geladene Dokumentdaten aus Sitzung löschen"):
                st.session_state.extracted_text = ""
                st.session_state.user_documents = []
                st.session_state.messages = []
                st.session_state.last_user_sources = []
                st.session_state.last_expert_sources = []
                st.rerun()

    # --------------------------------------------------------
    # TAB 2: CHAT
    # --------------------------------------------------------
    with tab2:
        st.subheader("Pflegegrad-Widerspruchsassistent")

        if not st.session_state.user_documents:
            st.info(
                "Sie haben noch keine persönlichen Dokumente hochgeladen. "
                "Der Assistent kann allgemeine Fragen beantworten, aber keine individuelle Dokumentenprüfung durchführen."
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input(
            "Beispiel: Prüfe Bescheid, Pflegetagebuch und ärztliche Unterlagen auf mögliche Unstimmigkeiten."
        )

        if user_prompt:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )

            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Suche relevante Nutzerdokumentstellen und Fachquellen..."):
                    try:
                        expert_db = get_expert_database()

                        answer, user_sources, expert_sources = generate_rag_answer(
                            expert_db=expert_db,
                            user_question=user_prompt,
                            user_documents=st.session_state.user_documents,
                        )

                        st.markdown(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                        st.session_state.last_user_sources = user_sources
                        st.session_state.last_expert_sources = expert_sources

                    except Exception as e:
                        st.error(f"KI-Fehler: {e}")

        if st.session_state.last_user_sources or st.session_state.last_expert_sources:
            st.divider()

        if st.session_state.last_user_sources:
            st.subheader("Verwendete Nutzerdokumentstellen")

            for source in st.session_state.last_user_sources:
                title = f"Nutzerdokument {source['nr']}: {source['source']}"

                if source["page"]:
                    title += f" — Seite {source['page']}"

                with st.expander(title):
                    st.write(source["preview"])

        if st.session_state.last_expert_sources:
            st.subheader("Verwendete Fachquellen")

            for source in st.session_state.last_expert_sources:
                title = f"Fachquelle {source['nr']}: {source['source']}"

                if source["page"]:
                    title += f" — Seite {source['page']}"

                with st.expander(title):
                    st.write(source["preview"])

    # --------------------------------------------------------
    # TAB 3: FRISTEN
    # --------------------------------------------------------
    with tab3:
        st.subheader("Fristenrechner")

        st.warning(
            "Der Fristenrechner dient nur als Orientierung. "
            "Die genaue Frist sollte im Zweifel rechtlich geprüft werden."
        )

        received_date = st.date_input(
            "Eingangsdatum des Pflegebescheids",
            value=datetime.date.today(),
        )

        simple_deadline = received_date + datetime.timedelta(days=30)

        st.info(
            f"Die grob berechnete Widerspruchsfrist endet voraussichtlich am "
            f"**{simple_deadline.strftime('%d.%m.%Y')}**."
        )

        if simple_deadline.weekday() == 5:
            adjusted_deadline = simple_deadline + datetime.timedelta(days=2)
            st.warning(
                "Das berechnete Datum fällt auf einen Samstag. "
                f"Der nächste Werktag wäre der **{adjusted_deadline.strftime('%d.%m.%Y')}**."
            )

        elif simple_deadline.weekday() == 6:
            adjusted_deadline = simple_deadline + datetime.timedelta(days=1)
            st.warning(
                "Das berechnete Datum fällt auf einen Sonntag. "
                f"Der nächste Werktag wäre der **{adjusted_deadline.strftime('%d.%m.%Y')}**."
            )

        st.caption(
            "Gesetzliche Feiertage werden in diesem Prototyp noch nicht automatisch berücksichtigt."
        )