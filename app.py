import streamlit as st
import requests
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import datetime

# --- 1. GLOBALE KONFIGURATION ---
st.set_page_config(page_title="Pflege-Assistent Pro", page_icon="⚖️", layout="wide")
API_URL = "http://127.0.0.1:8000"

# Initialisierung des Session-Speichers
if "token" not in st.session_state: st.session_state.token = None
if "username" not in st.session_state: st.session_state.username = ""
if "verify_mode" not in st.session_state: st.session_state.verify_mode = False
if "temp_user" not in st.session_state: st.session_state.temp_user = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""


# --- 2. KI-ENGINE (Caching für Geschwindigkeit) ---
@st.cache_resource
def get_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model="mistral")

    template = """Du bist ein Experte für Pflegegrad-Widersprüche.
    NUTZER-DOKUMENTE: {bescheid_text}
    GESETZES-KONTEXT: {context}
    FRAGE: {question}
    ANTWORT:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

    return ({"context": retriever | format_docs, "question": RunnablePassthrough(),
             "bescheid_text": lambda x: x["bescheid_text"]}
            | prompt | llm | StrOutputParser())


# =====================================================================
# PHASE 1: LOGIN & REGISTRIERUNG (Vorgeschaltet)
# =====================================================================
if st.session_state.token is None:
    st.title("🛡️ Pflegehilfe Online - Portal")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Anmeldung")
        with st.form("login_form"):
            u = st.text_input("Benutzername")
            p = st.text_input("Passwort", type="password")
            if st.form_submit_button("Einloggen", use_container_width=True):
                try:
                    res = requests.post(f"{API_URL}/login", data={"username": u, "password": p})
                    if res.status_code == 200:
                        st.session_state.token = res.json().get("access_token")
                        st.session_state.username = u
                        st.rerun()
                    else:
                        detail = res.json().get("detail", "Login fehlgeschlagen.")
                        st.error(detail)
                except Exception as e:
                    st.error(f"Verbindung zum Backend fehlgeschlagen. Läuft uvicorn? (Fehler: {e})")

    with col_b:
        st.subheader("Registrierung")
        if not st.session_state.verify_mode:
            with st.form("reg_form"):
                ru = st.text_input("Benutzername wählen")
                re = st.text_input("E-Mail")
                rp = st.text_input("Passwort", type="password")
                if st.form_submit_button("Konto erstellen"):
                    try:
                        res = requests.post(f"{API_URL}/register", json={"username": ru, "email": re, "password": rp})
                        if res.status_code == 200:
                            st.session_state.temp_user = ru
                            st.session_state.verify_mode = True
                            st.rerun()
                        else:
                            # Robuste Fehlerbehandlung gegen JSONDecodeError
                            try:
                                detail = res.json().get("detail", "Unbekannter Fehler")
                                st.error(f"Server-Info: {detail}")
                            except:
                                st.error(f"Backend-Fehler (Status: {res.status_code}). Bitte uvicorn-Terminal prüfen!")
                    except Exception as e:
                        st.error(f"Anfrage fehlgeschlagen: {e}")
        else:
            st.warning(f"Bitte Code für {st.session_state.temp_user} eingeben:")
            v_code = st.text_input("6-stelliger Code (siehe Backend-Terminal)")
            if st.button("Code verifizieren"):
                try:
                    res = requests.post(f"{API_URL}/verify",
                                        json={"username": st.session_state.temp_user, "code": v_code})
                    if res.status_code == 200:
                        st.success("Erfolgreich! Bitte jetzt links einloggen.")
                        st.session_state.verify_mode = False
                    else:
                        st.error(res.json().get("detail", "Falscher Code."))
                except:
                    st.error("Verbindung zum Server unterbrochen.")

# =====================================================================
# PHASE 2: DIE HAUPT-APP (Nach Login)
# =====================================================================
else:
    # Sidebar für Status und Logout
    with st.sidebar:
        st.title("User-Panel")
        st.write(f"Eingeloggt als: **{st.session_state.username}**")
        if st.button("🚪 Abmelden", use_container_width=True):
            st.session_state.token = None
            st.session_state.username = ""
            st.rerun()
        st.divider()
        st.info("Ihre Daten werden DSGVO-konform verarbeitet.")

    # Die Tab-Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 KI-Chatbot",
        "📄 Dokumenten-Management",
        "⏳ Fristen & Recht",
        "⚙️ Profil-Einstellungen"
    ])

    # --- TAB 1: CHATBOT ---
    with tab1:
        st.subheader("KI-Analyse & Widerspruchs-Hilfe")
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Fragen Sie etwas zu Ihrem Bescheid..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                txt = st.session_state.extracted_text
                if txt:
                    chain = get_rag_chain()
                    with st.spinner("Analysiere Daten..."):
                        response = chain.invoke({"question": prompt, "bescheid_text": txt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Bitte laden Sie im Tab 'Dokumenten-Management' zuerst Ihre PDFs hoch!")

    # --- TAB 2: DOKUMENTE ---
    with tab2:
        st.subheader("Upload von Bescheiden (PDF)")
        files = st.file_uploader("PDF-Dateien auswählen", type="pdf", accept_multiple_files=True)
        if files:
            combined_text = ""
            for f in files:
                reader = PdfReader(f)
                for page in reader.pages: combined_text += (page.extract_text() or "")
            st.session_state.extracted_text = combined_text
            st.success(f"{len(files)} Dokument(e) für die KI bereitgestellt.")

    # --- TAB 3: FRISTEN ---
    with tab3:
        st.subheader("Widerspruchs-Fristenrechner")
        datum = st.date_input("Eingangsdatum des Bescheids", value=datetime.date.today())
        frist = datum + datetime.timedelta(days=30)
        st.metric("Ihre Deadline", frist.strftime("%d.%m.%Y"))

        if st.button("Fristverlängerung anfordern"):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            res = requests.post(f"{API_URL}/users/extend", headers=headers)
            st.success("Anfrage zur Verlängerung wurde übermittelt.")

    # --- TAB 4: EINSTELLUNGEN ---
    with tab4:
        st.subheader("Account-Verwaltung")
        if st.button("🚨 Konto unwiderruflich löschen"):
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            requests.delete(f"{API_URL}/users/me", headers=headers)
            st.session_state.token = None
            st.warning("Account gelöscht.")
            st.rerun()