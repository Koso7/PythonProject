import streamlit as st
import requests
import datetime
from pypdf import PdfReader

# --- LANGCHAIN & OLLAMA IMPORTS ---
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="Pflege-Assistent Pro", page_icon="⚖️", layout="wide")
API_URL = "http://127.0.0.1:8000"

# Session State initialisieren
if "token" not in st.session_state: st.session_state.token = None
if "verify_user" not in st.session_state: st.session_state.verify_user = None
if "messages" not in st.session_state: st.session_state.messages = []
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""


# --- 2. KI-LOGIK (Fachwissen + User-Bescheid) ---
@st.cache_resource
def get_expert_database():
    """Lädt das feste Fachwissen (z.B. GKV-Richtlinien) aus dem daten-Ordner."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Greift auf den Ordner zu, den ingest.py erstellt hat
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def get_rag_chain(expert_db, user_bescheid_text):
    """Kombiniert das Fachwissen aus der DB mit dem Bescheid des Nutzers."""
    retriever = expert_db.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model="mistral")

    template = """Du bist ein professioneller KI-Assistent für Pflegegrad-Widersprüche.
    WICHTIGE REGELN: 
    1. Beantworte die Frage basierend auf dem FACHWISSEN und dem PERSÖNLICHEN BESCHEID.
    2. Wenn die Antwort in diesen Texten nicht zu finden ist, antworte zwingend: "Dazu habe ich keine Informationen in den Dokumenten gefunden."
    3. Erfinde niemals eigene Fakten oder Fristen. Dies ist keine Rechtsberatung.

    FACHWISSEN (Gesetze & Richtlinien):
    {context}

    PERSÖNLICHER BESCHEID (Vom Nutzer hochgeladen):
    {bescheid}

    FRAGE DES NUTZERS:
    {question}

    ANTWORT:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Baut die Pipeline zusammen
    chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "bescheid": lambda x: user_bescheid_text
            }
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain


# =====================================================================
# PHASE 1: LOGIN & REGISTRIERUNG
# =====================================================================
if st.session_state.token is None:
    st.title("🛡️ Pflegehilfe Online - Portal")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Login")
        with st.form("l_form"):
            u = st.text_input("Nutzername")
            p = st.text_input("Passwort", type="password")
            if st.form_submit_button("Anmelden", use_container_width=True):
                res = requests.post(f"{API_URL}/login", data={"username": u, "password": p})
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Login fehlgeschlagen. Bitte Daten prüfen.")

    with col2:
        if not st.session_state.verify_user:
            st.subheader("Registrierung")
            with st.form("r_form"):
                ru = st.text_input("Nutzername")
                re = st.text_input("Email")
                rp = st.text_input("Passwort", type="password")
                if st.form_submit_button("Konto erstellen", use_container_width=True):
                    res = requests.post(f"{API_URL}/register", json={"username": ru, "email": re, "password": rp})
                    if res.status_code == 200:
                        st.session_state.verify_user = ru
                        st.rerun()
                    else:
                        st.error("Fehler bei der Registrierung (Name/Mail evtl. schon vergeben).")
        else:
            st.info(f"Code für {st.session_state.verify_user} im Backend-Terminal ablesen!")
            code = st.text_input("6-stelliger Code")
            if st.button("Verifizieren", use_container_width=True):
                res = requests.post(f"{API_URL}/verify", json={"username": st.session_state.verify_user, "code": code})
                if res.status_code == 200:
                    st.success("Erfolg! Bitte jetzt links einloggen.")
                    st.session_state.verify_user = None
                else:
                    st.error("Falscher Code.")

# =====================================================================
# PHASE 2: HAUPT-APP (Nach Login)
# =====================================================================
else:
    # --- DAUERHAFTER HAFTUNGSAUSSCHLUSS ---
    with st.sidebar:
        st.warning(
            "⚠️ **WICHTIGER HINWEIS**\n\nDies ist eine KI-gestützte Analyse-Software und stellt **keine rechtliche oder medizinische Beratung** dar.\n\nAlle generierten Texte müssen von Ihnen auf Richtigkeit geprüft werden. Bei rechtlichen Fragen konsultieren Sie bitte einen Fachanwalt oder Sozialverband.")

        st.divider()
        if st.button("🚪 Ausloggen", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.session_state.extracted_text = ""
            st.rerun()

    # --- NAVIGATION ---
    tab1, tab2, tab3 = st.tabs(["📄 Dokumente (Upload)", "💬 KI-Chat", "📅 Fristen"])

    # --- TAB 1: DOKUMENTE HOCHLADEN ---
    with tab1:
        st.subheader("Ihren Bescheid hochladen")
        st.info("Sie können mehrere PDF-Dokumente gleichzeitig auswählen.")

        uploaded_files = st.file_uploader("PDFs hochladen", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            if st.button("🚀 Dokumente für die KI einlesen", type="primary"):
                with st.spinner("Lese Ihre PDFs..."):
                    try:
                        text = ""
                        for file in uploaded_files:
                            reader = PdfReader(file)
                            for page in reader.pages:
                                text += (page.extract_text() or "") + "\n"

                        # Wir speichern den Text einfach im Session State
                        st.session_state.extracted_text = text
                        st.success(
                            f"✅ {len(uploaded_files)} Dokument(e) erfolgreich gelesen. Sie können jetzt in den Chat wechseln!")
                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten: {e}")

    # --- TAB 2: KI CHAT ---
    with tab2:
        st.subheader("Widerspruchs-Assistent")

        # Chat Historie anzeigen
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Neues Input-Feld
        if prompt := st.chat_input("Fragen Sie etwas (z.B. 'Warum wurde Pflegegrad 2 abgelehnt?')..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analysiere Gesetze und Ihren Bescheid..."):
                    try:
                        # 1. Holt die feste Datenbank
                        expert_db = get_expert_database()
                        # 2. Holt den hochgeladenen Text (oder einen Hinweis, falls leer)
                        bescheid_text = st.session_state.extracted_text if st.session_state.extracted_text else "Der Nutzer hat noch keinen eigenen Bescheid hochgeladen."

                        # 3. Startet die KI mit beidem
                        chain = get_rag_chain(expert_db, bescheid_text)
                        response = chain.invoke(prompt)

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"KI-Fehler: Haben Sie 'ingest.py' ausgeführt? (Details: {e})")

    # --- TAB 3: FRISTEN ---
    with tab3:
        st.subheader("Fristenrechner")
        datum = st.date_input("Eingangsdatum des Pflegebescheids", value=datetime.date.today())
        deadline = datum + datetime.timedelta(days=30)
        st.info(f"📅 Ihre Widerspruchsfrist endet voraussichtlich am **{deadline.strftime('%d.%m.%Y')}**.")
        st.caption(
            "Fällt dieser Tag auf ein Wochenende oder einen Feiertag, verschiebt sich die Frist auf den nächsten Werktag.")