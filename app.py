import streamlit as st
import requests
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import datetime
from dateutil.relativedelta import relativedelta

# --- 1. SEITEN-SETUP & KONFIGURATION ---
st.set_page_config(page_title="Pflege-Assistent", page_icon="🏥", layout="centered")
API_URL = "http://127.0.0.1:8000"

# Initialisierung der Zustände
if "token" not in st.session_state: st.session_state.token = None
if "verify_mode" not in st.session_state: st.session_state.verify_mode = False
if "temp_username" not in st.session_state: st.session_state.temp_username = ""


# --- 2. KI LADE-FUNKTION (wird erst benötigt, wenn eingeloggt) ---
@st.cache_resource
def get_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = OllamaLLM(model="mistral")
    return retriever, llm


def analyze_documents(bescheid_text, user_query):
    retriever, llm = get_rag_chain()
    template = """Du bist ein Experte für Pflegegrad-Widersprüche.
    QUELLE 1 (Nutzer-Dokumente): {bescheid_text}
    QUELLE 2 (Richtlinien): {context}
    FRAGE: {question}
    ANTWORT:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough(),
              "bescheid_text": lambda x: bescheid_text} | prompt | llm | StrOutputParser())
    return chain.invoke(user_query)


# =====================================================================
# ANSICHT 1: DAS LOGIN-PORTAL (Nur sichtbar, wenn NICHT eingeloggt)
# =====================================================================
if st.session_state.token is None:
    st.title("🏥 Willkommen beim Pflege-Assistenten")
    st.markdown("### Ihr digitaler Begleiter für Pflegegrad-Widersprüche")
    st.write("Um Ihre sensiblen Gesundheitsdaten zu schützen, melden Sie sich bitte an oder registrieren Sie sich.")
    st.divider()

    tab_login, tab_reg = st.tabs(["🔑 Anmelden", "📝 Neu Registrieren"])

    with tab_login:
        st.subheader("Anmeldung")
        st.write("Bitte geben Sie Ihre E-Mail-Adresse (oder Benutzernamen) und Ihr Passwort ein.")
        log_user = st.text_input("E-Mail oder Benutzername", key="log_user")
        log_pass = st.text_input("Passwort", type="password", key="log_pass")

        if st.button("Jetzt Anmelden", use_container_width=True, type="primary"):
            try:
                response = requests.post(f"{API_URL}/login", data={"username": log_user, "password": log_pass})
                if response.status_code == 200:
                    st.session_state.token = response.json().get("access_token")
                    st.rerun()  # Schaltet die Seite auf die App um!
                else:
                    st.error(f"Fehler: {response.json().get('detail')}")
            except requests.exceptions.ConnectionError:
                st.error("Der Server ist momentan nicht erreichbar.")

    with tab_reg:
        if not st.session_state.verify_mode:
            st.subheader("Neues Konto erstellen")
            reg_user = st.text_input("Wählen Sie einen Benutzernamen")
            reg_mail = st.text_input("Ihre E-Mail-Adresse")
            reg_pass = st.text_input("Wählen Sie ein sicheres Passwort", type="password")

            if st.button("Konto erstellen & Code anfordern", use_container_width=True):
                if not reg_user or not reg_mail or not reg_pass:
                    st.warning("Bitte füllen Sie alle Felder aus.")
                else:
                    try:
                        res = requests.post(f"{API_URL}/register",
                                            json={"username": reg_user, "email": reg_mail, "password": reg_pass})
                        if res.status_code == 200:
                            st.session_state.temp_username = reg_user
                            st.session_state.verify_mode = True
                            st.rerun()
                        else:
                            st.error(res.json().get("detail"))
                    except requests.exceptions.ConnectionError:
                        st.error("Der Server ist momentan nicht erreichbar.")

        else:
            # Ansicht für die Code-Eingabe
            st.subheader("📧 E-Mail bestätigen")
            st.success("Wir haben Ihnen einen 6-stelligen Code gesendet. (Pst: Schau ins PyCharm Backend-Terminal!)")
            code_input = st.text_input("Bitte geben Sie hier Ihren Code ein:")

            if st.button("Code bestätigen", use_container_width=True, type="primary"):
                res = requests.post(f"{API_URL}/verify",
                                    json={"username": st.session_state.temp_username, "code": code_input})
                if res.status_code == 200:
                    st.success("Erfolgreich bestätigt! Sie können sich nun im Tab 'Anmelden' einloggen.")
                    st.session_state.verify_mode = False
                    st.session_state.temp_username = ""
                else:
                    st.error("Der Code ist leider falsch.")


# =====================================================================
# ANSICHT 2: DIE EIGENTLICHE APP (Nur sichtbar, wenn eingeloggt)
# =====================================================================
else:
    # Damit die App breiter ist als das Login-Fenster
    st.markdown("<style> .block-container { max-width: 1200px; } </style>", unsafe_allow_html=True)

    with st.sidebar:
        st.success("✅ Sicher eingeloggt.")
        if st.button("🚪 Abmelden"):
            st.session_state.token = None
            st.rerun()

        st.divider()
        st.header("📂 1. Dokumente hochladen")
        st.write("Laden Sie hier Briefe von Ärzten oder der Pflegekasse hoch.")
        einwilligung = st.checkbox("Ich erlaube die temporäre Auswertung dieser Dokumente.", value=False)

        bescheid_inhalt = ""
        if einwilligung:
            uploaded_files = st.file_uploader("PDF-Dateien hier ablegen", type=["pdf"], accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    bescheid_inhalt += f"\n\n=== {file.name} ===\n\n"
                    reader = PdfReader(file)
                    for p in reader.pages: bescheid_inhalt += (p.extract_text() or "")
                st.success(f"{len(uploaded_files)} Dokument(e) bereit.")

    st.title("🏥 Ihr Pflege-Assistent")
    st.write("Wie kann ich Ihnen heute bei Ihrem Pflegegrad helfen?")

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Fragen stellen", "📝 Brief erstellen", "⏱️ Fristen", "👤 Profil"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Guten Tag! Haben Sie Fragen zu Ihren Unterlagen?"}]
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
        if user_input := st.chat_input("Ihre Frage..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                if 'bescheid_inhalt' in locals() and bescheid_inhalt:
                    with st.spinner("Ich lese die Unterlagen..."):
                        response = analyze_documents(bescheid_inhalt, user_input)
                else:
                    response = "Bitte laden Sie links zuerst Ihre Dokumente hoch."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    with tab2:
        st.header("Widerspruch verfassen")
        if st.button("Brief jetzt entwerfen", type="primary"):
            if 'bescheid_inhalt' in locals() and bescheid_inhalt:
                with st.spinner("Ich formuliere den Brief für Sie..."):
                    draft = analyze_documents(bescheid_inhalt,
                                              "Schreibe einen formellen Widerspruchsbrief. Analysiere Dokumente auf Differenzen und begründe mit NBA-Richtlinien.")
                    st.text_area("Ihr fertiger Brief:", value=draft, height=400)
            else:
                st.warning("Bitte laden Sie zuerst Dokumente hoch.")

    with tab3:
        st.header("Fristen-Kalender")
        bescheid_datum = st.date_input("An welchem Tag lag der Bescheid in Ihrem Briefkasten?", format="DD.MM.YYYY")
        if bescheid_datum:
            frist_ende = bescheid_datum + relativedelta(months=1)
            if frist_ende.weekday() >= 5: frist_ende += datetime.timedelta(days=(7 - frist_ende.weekday()))
            st.error(f"🚨 **Wichtig: Ihr Brief muss bis zum {frist_ende.strftime('%d.%m.%Y')} bei der Kasse sein!**")

    with tab4:
        st.header("Ihre Daten")
        auth_headers = {"Authorization": f"Bearer {st.session_state.token}"}
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏳ Account-Löschung um 1 Woche verschieben"):
                requests.post(f"{API_URL}/users/extend", headers=auth_headers)
                st.success("Frist erfolgreich verlängert.")
        with col2:
            if st.button("🗑️ Account jetzt löschen"):
                requests.delete(f"{API_URL}/users/me", headers=auth_headers)
                st.session_state.token = None
                st.rerun()