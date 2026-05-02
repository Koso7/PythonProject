import streamlit as st
import requests
import datetime
from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. KONFIGURATION ---
st.set_page_config(page_title="Pflege-Assistent Pro", page_icon="⚖️", layout="wide")
API_URL = "http://127.0.0.1:8000"

if "token" not in st.session_state: st.session_state.token = None
if "verify_user" not in st.session_state: st.session_state.verify_user = None
if "messages" not in st.session_state: st.session_state.messages = []
if "extracted_text" not in st.session_state: st.session_state.extracted_text = ""


# --- 2. KI-LOGIK (Erweitert für komplexe Widersprüche) ---
@st.cache_resource
def get_expert_database():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def get_rag_chain(expert_db, user_bescheid_text):
    # k=15: Zieht jetzt 15 relevante Abschnitte aus den Gesetzen/Webseiten (viel mehr Kontext!)
    retriever = expert_db.as_retriever(search_kwargs={"k": 15})

    # Wir nutzen Llama 3, da es für strukturierte juristische Texte besser geeignet ist
    llm = OllamaLLM(model="mistral-nemo", temperature=0.0)

    # Der Chain-of-Thought Experten-Prompt
    template = """Du bist ein hochqualifizierter Experte für deutsche Pflegegrad-Widersprüche.
        Analysiere die NUTZER-DOKUMENTE, gleiche sie mit dem FACHWISSEN ab und verfasse einen strukturierten, sachlichen Widerspruchsbrief.

        ---
        FACHWISSEN (Gesetze, Richtlinien, Urteile):
        {context}

        NUTZER-DOKUMENTE (Bescheid, Pflegetagebuch, etc.):
        {bescheid}

        AUFTRAG DES NUTZERS:
        {question}
        ---

        WICHTIGE UND UNUMSTÖSSLICHE REGELN:
        1. PERSPEKTIVE: Schreibe den Brief zwingend aus der Sicht eines ANGEHÖRIGEN / BEVOLLMÄCHTIGTEN, der für die pflegebedürftige Person spricht (z.B. "hiermit lege ich gegen den Bescheid meiner Mutter Widerspruch ein...").
        2. BELEGE: Nenne in deiner Begründung aktiv Passagen aus dem FACHWISSEN (z.B. "Gemäß den Pflegebedürftigkeits-Richtlinien (BRi)..."), um deine Argumente zu stützen.
        3. SPRACHE: Dieser Brief geht an eine deutsche Behörde. Du musst den KOMPLETTEN Text zwingend auf DEUTSCH schreiben. Kein einziges englisches Wort!

        DEIN FERTIGER WIDERSPRUCH (KOMPLETT AUF DEUTSCH UND ALS ANGEHÖRIGER):"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        # Wir fügen die gefundenen Texte zusammen
        context_text = "\n\n".join(doc.page_content for doc in docs)

        # SPION-FUNKTION: Wir drucken die ersten 500 Zeichen des gefundenen
        # Fachwissens in dein PyCharm-Terminal, damit DU prüfen kannst,
        # was die KI gerade liest!
        print("\n" + "=" * 50)
        print("🔍 LIVE-CHECK: DIESES FACHWISSEN WURDE GERADE GEFUNDEN:")
        print("=" * 50)
        print(context_text[:800] + "\n... [Text geht weiter] ...\n")

        return context_text

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
    with st.sidebar:
        st.warning(
            "⚠️ **WICHTIGER HINWEIS**\n\nDies ist eine KI-gestützte Analyse-Software und stellt **keine rechtliche oder medizinische Beratung** dar.\n\nAlle generierten Texte (wie z.B. Widerspruchsschreiben) müssen von Ihnen auf Richtigkeit geprüft werden. Bei rechtlichen Fragen konsultieren Sie bitte einen Fachanwalt oder Sozialverband.")
        st.divider()
        if st.button("🚪 Ausloggen", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.session_state.extracted_text = ""
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["📄 Ihre Dokumente", "💬 KI-Expertenchat", "📅 Fristen"])

    with tab1:
        st.subheader("Pflegetagebuch / Bescheid hochladen")
        st.info("Laden Sie hier alle relevanten persönlichen Dokumente für den Widerspruch hoch.")

        uploaded_files = st.file_uploader("PDFs auswählen", type="pdf", accept_multiple_files=True)

        if uploaded_files:
            if st.button("🚀 Dokumente für die KI einlesen", type="primary"):
                with st.spinner("Lese Ihre Dokumente..."):
                    try:
                        text = ""
                        for file in uploaded_files:
                            reader = PdfReader(file)
                            for page in reader.pages:
                                text += (page.extract_text() or "") + "\n"
                        st.session_state.extracted_text = text
                        st.success(
                            f"✅ {len(uploaded_files)} Dokument(e) erfolgreich gelesen. Wechseln Sie jetzt in den Chat!")
                    except Exception as e:
                        st.error(f"Fehler beim Verarbeiten: {e}")

    with tab2:
        st.subheader("Widerspruchs-Assistent")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Beispiel: Analysiere mein Pflegetagebuch und schreibe einen Widerspruch..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analysiere Gesetze, Urteile und Ihre Dokumente..."):
                    try:
                        expert_db = get_expert_database()
                        bescheid_text = st.session_state.extracted_text if st.session_state.extracted_text else "Der Nutzer hat noch keine eigenen Dokumente hochgeladen."

                        chain = get_rag_chain(expert_db, bescheid_text)
                        response = chain.invoke(prompt)

                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"KI-Fehler: {e}")

    with tab3:
        st.subheader("Fristenrechner")
        datum = st.date_input("Eingangsdatum des Pflegebescheids", value=datetime.date.today())
        deadline = datum + datetime.timedelta(days=30)
        st.info(f"📅 Ihre Widerspruchsfrist endet voraussichtlich am **{deadline.strftime('%d.%m.%Y')}**.")
        st.caption(
            "Fällt dieser Tag auf ein Wochenende oder Feiertag, verschiebt sich die Frist auf den nächsten Werktag.")