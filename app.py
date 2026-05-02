import streamlit as st
import requests
import datetime
from pypdf import PdfReader

# LangChain & Ollama Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- KONFIGURATION ---
st.set_page_config(page_title="Pflege-Assistent Pro", page_icon="⚖️", layout="wide")
API_URL = "http://127.0.0.1:8000"

# Initialisierung des Session-Speichers
if "token" not in st.session_state: st.session_state.token = None
if "verify_user" not in st.session_state: st.session_state.verify_user = None
if "messages" not in st.session_state: st.session_state.messages = []
if "vector_store" not in st.session_state: st.session_state.vector_store = None


# --- KI-LOGIK (Striktes RAG mit Mistral) ---
def process_documents_to_chroma(files):
    """Liest PDFs, teilt sie in Chunks und speichert sie in Chroma."""
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    # Text in sinnvolle Abschnitte teilen
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # Embeddings (Vektorisierung) via Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Temporäre In-Memory Chroma-Datenbank für diese Sitzung erstellen
    vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


def get_rag_chain(vectorstore):
    """Baut die Pipeline auf, die sicherstellt, dass NUR die Dokumente genutzt werden."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model="mistral")

    # Der strikte System-Prompt
    template = """Du bist ein KI-Assistent für Pflegegrad-Widersprüche.
    WICHTIGE REGEL: Beantworte die Frage AUSSCHLIESSLICH basierend auf dem folgenden Kontext (den Dokumenten des Nutzers).
    Wenn die Antwort NICHT im Kontext enthalten ist, antworte zwingend mit: "Dazu habe ich keine Informationen in den hochgeladenen Dokumenten gefunden."
    Erfinde keine Daten. Dies ist keine Rechtsberatung.

    KONTEXT:
    {context}

    FRAGE:
    {question}

    ANTWORT:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
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
                    st.error("Login fehlgeschlagen. Daten prüfen.")

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
                        detail = res.json().get("detail", "Fehler")
                        st.error(f"Fehler: {detail}")
        else:
            st.info(f"Bitte Code für {st.session_state.verify_user} eingeben (siehe Backend-Terminal)")
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
            "⚠️ **WICHTIGER HINWEIS**\n\nDies ist eine KI-gestützte Analyse-Software und stellt **keine rechtliche oder medizinische Beratung** dar.\n\nAlle generierten Texte (wie z.B. Widerspruchsschreiben) müssen von Ihnen auf Richtigkeit geprüft werden. Bei rechtlichen Fragen konsultieren Sie bitte einen Fachanwalt oder Sozialverband.")

        st.divider()
        if st.button("🚪 Ausloggen", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.rerun()

    # --- NAVIGATION ---
    tab1, tab2, tab3 = st.tabs(["📄 Dokumente (Upload)", "💬 KI-Chat", "📅 Fristen"])

    # --- TAB 1: DOKUMENTE HOCHLADEN ---
    with tab1:
        st.subheader("Ihre Bescheide & Gutachten")
        st.info("Sie können mehrere PDF-Dokumente gleichzeitig auswählen.")

        # accept_multiple_files=True erlaubt mehrere Dateien
        uploaded_files = st.file_uploader("PDFs hochladen", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            if st.button("🚀 Dokumente analysieren & für KI aufbereiten", type="primary"):
                with st.spinner("Lese Dokumente und füttere die Datenbank..."):
                    try:
                        # Dokumente in ChromaDB laden
                        vs = process_documents_to_chroma(uploaded_files)
                        st.session_state.vector_store = vs
                        st.success(
                            f"✅ {len(uploaded_files)} Dokument(e) erfolgreich verarbeitet. Sie können jetzt im Chat-Tab Fragen dazu stellen.")
                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten: {e}")

    # --- TAB 2: KI CHAT ---
    with tab2:
        st.subheader("KI-Widerspruchs-Assistent")

        if st.session_state.vector_store is None:
            st.warning("Bitte laden Sie zuerst im Tab 'Dokumente' Ihre PDFs hoch und klicken Sie auf Analysieren.")
        else:
            # Chat Historie anzeigen
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Neues Input-Feld
            if prompt := st.chat_input("Fragen Sie etwas zu Ihren hochgeladenen Bescheiden..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Suche in Ihren Dokumenten..."):
                        chain = get_rag_chain(st.session_state.vector_store)
                        response = chain.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

    # --- TAB 3: FRISTEN ---
    with tab3:
        st.subheader("Fristenrechner")
        datum = st.date_input("Eingangsdatum des Pflegebescheids", value=datetime.date.today())
        deadline = datum + datetime.timedelta(days=30)
        st.info(f"📅 Ihre Widerspruchsfrist endet voraussichtlich am **{deadline.strftime('%d.%m.%Y')}**.")
        st.caption(
            "Fällt dieser Tag auf ein Wochenende oder einen Feiertag, verschiebt sich die Frist auf den nächsten Werktag.")