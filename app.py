import datetime
from typing import List, Tuple

import requests
import streamlit as st
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Pflege-Assistent Pro",
    page_icon="⚖️",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000"
MAX_FILE_SIZE_MB = 10
MAX_TOTAL_TEXT_CHARS = 120_000

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

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


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
    st.session_state.last_sources = []
    st.rerun()


# ------------------------------------------------------------
# KI-/RAG-LOGIK
# ------------------------------------------------------------
@st.cache_resource
def get_expert_database():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )


@st.cache_resource
def get_llm():
    return OllamaLLM(
        model="mistral-nemo",
        temperature=0.0,
    )


def get_relevant_docs(expert_db, question: str, k: int = 12) -> List[Document]:
    retriever = expert_db.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)


def format_docs_for_prompt(docs: List[Document]) -> str:
    context_parts = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        page = doc.metadata.get("page", None)

        if page is not None:
            source_label = f"Quelle {index}: {source}, Seite {page + 1}"
        else:
            source_label = f"Quelle {index}: {source}"

        context_parts.append(
            f"{source_label}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_source_list(docs: List[Document]) -> List[dict]:
    sources = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unbekannte Quelle")
        page = doc.metadata.get("page", None)

        preview = doc.page_content.replace("\n", " ").strip()
        if len(preview) > 400:
            preview = preview[:400] + "..."

        sources.append(
            {
                "nr": index,
                "source": source,
                "page": page + 1 if page is not None else None,
                "preview": preview,
            }
        )

    return sources


def generate_rag_answer(
    expert_db,
    user_question: str,
    user_document_text: str,
) -> Tuple[str, List[dict]]:
    docs = get_relevant_docs(expert_db, user_question, k=12)
    context = format_docs_for_prompt(docs)
    sources = build_source_list(docs)

    llm = get_llm()

    template = """
Du bist ein KI-gestützter Assistenzdienst zur strukturierten Vorbereitung eines Pflegegrad-Widerspruchs.

Wichtig:
- Du ersetzt keine Rechtsberatung.
- Du erstellst nur einen überprüfbaren Entwurf.
- Du darfst keine medizinischen oder rechtlichen Tatsachen erfinden.
- Wenn Informationen fehlen, benenne diese Lücken ausdrücklich.
- Schreibe vollständig auf Deutsch.
- Verwende eine sachliche, behördentaugliche Sprache.

Nutze vorrangig die NUTZER-DOKUMENTE.
Nutze das FACHWISSEN nur ergänzend zur Begründung.
Beziehe dich im Text auf die Quellen aus dem Fachwissen, wenn sie wirklich passen.

FACHWISSEN:
{context}

NUTZER-DOKUMENTE:
{bescheid}

AUFTRAG DES NUTZERS:
{question}

Erstelle nun einen strukturierten Entwurf.
Falls der Nutzer keinen Widerspruchsbrief verlangt, beantworte seine Frage passend zum Pflegegrad-Thema.
"""

    prompt = PromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "bescheid": user_document_text,
            "question": user_question,
        }
    )

    return answer, sources


# ------------------------------------------------------------
# PDF-VERARBEITUNG
# ------------------------------------------------------------
def extract_text_from_pdfs(uploaded_files) -> str:
    text = ""

    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Die Datei '{file.name}' ist größer als {MAX_FILE_SIZE_MB} MB."
            )

        reader = PdfReader(file)

        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

        if len(text) > MAX_TOTAL_TEXT_CHARS:
            text = text[:MAX_TOTAL_TEXT_CHARS]
            text += "\n\n[Hinweis: Der Text wurde gekürzt, weil sehr viele Daten hochgeladen wurden.]"
            break

    return text


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
            "Datenschutz: Hochgeladene persönliche Dokumente werden in diesem Prototyp "
            "nur temporär während der Sitzung verarbeitet und nicht dauerhaft in der Wissensdatenbank gespeichert."
        )

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
            "Laden Sie hier Bescheide, Gutachten oder Pflegetagebücher als PDF hoch. "
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
                with st.spinner("Dokumente werden gelesen..."):
                    try:
                        extracted_text = extract_text_from_pdfs(uploaded_files)
                        st.session_state.extracted_text = extracted_text

                        st.success(
                            "Dokumente wurden erfolgreich eingelesen. "
                            "Sie können nun im KI-Assistenten Fragen dazu stellen."
                        )

                        with st.expander("Eingelesenen Text anzeigen"):
                            st.text_area(
                                "Extrahierter Text",
                                value=st.session_state.extracted_text[:10_000],
                                height=300,
                            )

                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten der PDFs: {e}")

        if st.session_state.extracted_text:
            st.success("Es sind aktuell persönliche Dokumente in dieser Sitzung geladen.")

            if st.button("Geladene Dokumentdaten aus Sitzung löschen"):
                st.session_state.extracted_text = ""
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.rerun()

    # --------------------------------------------------------
    # TAB 2: CHAT
    # --------------------------------------------------------
    with tab2:
        st.subheader("Pflegegrad-Widerspruchsassistent")

        if not st.session_state.extracted_text:
            st.info(
                "Sie haben noch keine persönlichen Dokumente hochgeladen. "
                "Der Assistent kann trotzdem allgemeine Fragen beantworten, "
                "aber keine individuelle Dokumentenanalyse durchführen."
            )

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input(
            "Beispiel: Analysiere den Bescheid und erstelle einen Entwurf für einen Widerspruch."
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
                with st.spinner("Analysiere Fachwissen und Dokumente..."):
                    try:
                        expert_db = get_expert_database()

                        user_document_text = (
                            st.session_state.extracted_text
                            if st.session_state.extracted_text
                            else "Der Nutzer hat keine persönlichen Dokumente hochgeladen."
                        )

                        answer, sources = generate_rag_answer(
                            expert_db=expert_db,
                            user_question=user_prompt,
                            user_document_text=user_document_text,
                        )

                        st.markdown(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                        st.session_state.last_sources = sources

                    except Exception as e:
                        st.error(f"KI-Fehler: {e}")

        if st.session_state.last_sources:
            st.divider()
            st.subheader("Verwendete Fachquellen")

            for source in st.session_state.last_sources:
                title = f"Quelle {source['nr']}: {source['source']}"

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