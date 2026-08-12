"""Pflegehilfe Online - Unterstützung beim Widerspruch gegen einen Pflegegradbescheid.

Die Oberfläche ist auf gute Lesbarkeit und einfache Bedienung ausgelegt:
große Schrift, hoher Kontrast, klar sichtbare Tastatur-Markierung, große
Schaltflächen und durchgehend verständliche Sprache.

Datenschutz: Es gibt keine Registrierung. Eine Sitzung wird ausschließlich über
einen zufälligen Zugangscode identifiziert. Alle Verarbeitung findet örtlich statt.
"""

from __future__ import annotations

import datetime
import os
from typing import List, Optional

import requests
import streamlit as st
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_core.documents import Document

import pflege_pdf
import pflege_rag

load_dotenv()

st.set_page_config(
    page_title="Pflegehilfe Online",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# Eingescannte Gutachten des Medizinischen Dienstes sind erfahrungsgemäß groß
# (20 MB und mehr). Eine zu enge Grenze würde ausgerechnet das wichtigste
# Dokument abweisen.
MAX_FILE_SIZE_MB = 30
MAX_DOCUMENTS = 15
SESSION_DAYS = 28

# Diese Angaben werden verschlüsselt beim Zugangscode gespeichert und beim
# Wiedereinstieg vollständig wiederhergestellt.
SYNCED_STATE_KEYS = [
    "messages", "user_documents", "document_names", "last_generated_letter",
    "absender_name", "absender_strasse", "absender_plz_ort", "absender_ort",
    "kasse_name", "kasse_strasse", "kasse_plz_ort",
    "versichert_name", "versichert_nr", "aktenzeichen", "bescheid_datum",
    "letter_text", "font_scale", "high_contrast",
]

# Freitextfelder des PDF-Formulars. Sie werden ausschließlich unter ihrem
# Widget-Schlüssel ("w_" + Name) im Sitzungszustand gehalten. Streamlit-Widgets
# dürfen nicht gleichzeitig über `value=` und über den Sitzungszustand gesetzt
# werden - sonst gewinnt der zuvor gespeicherte Widget-Zustand und ein
# programmgesteuertes Vorbelegen bleibt wirkungslos.
TEXT_FIELDS = [
    "absender_name", "absender_strasse", "absender_plz_ort", "absender_ort",
    "kasse_name", "kasse_strasse", "kasse_plz_ort",
    "versichert_name", "versichert_nr", "aktenzeichen", "bescheid_datum",
    "letter_text",
]

DEFAULT_STATE = {
    "messages": [],
    "user_documents": [],
    "document_names": [],
    "last_generated_letter": "",
    "last_user_sources": [],
    "last_expert_sources": [],
    "font_scale": "Normal",
    "high_contrast": False,
}

FONT_SCALES = {"Normal": 18, "Groß": 21, "Sehr groß": 24}


# ---------------------------------------------------------------------------
# ZUSTAND
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("expires_at", None)
    for key, default in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = list(default) if isinstance(default, list) else default
    for name in TEXT_FIELDS:
        st.session_state.setdefault(f"w_{name}", "")


def get_field(name: str) -> str:
    """Liest ein Formularfeld aus dem Sitzungszustand."""
    return st.session_state.get(f"w_{name}", "")


def set_field(name: str, value: str) -> None:
    """Belegt ein Formularfeld vor. Muss vor dem Erzeugen des Widgets geschehen."""
    st.session_state[f"w_{name}"] = value or ""


def reset_local_state() -> None:
    """Setzt die Anzeige zurück, ohne die gespeicherten Daten anzutasten."""
    st.session_state.token = None
    st.session_state.expires_at = None
    for key, default in DEFAULT_STATE.items():
        st.session_state[key] = list(default) if isinstance(default, list) else default
    st.session_state.pop("user_store", None)
    st.session_state.pop("pdf_bytes", None)
    # Auch die internen Zustände der Eingabefelder entfernen, damit keine
    # Angaben aus einer vorherigen Sitzung stehen bleiben.
    for key in [k for k in list(st.session_state.keys()) if str(k).startswith("w_")]:
        del st.session_state[key]


def utcnow() -> datetime.datetime:
    """Aktuelle UTC-Zeit ohne Zeitzonenangabe (passend zur Speicherung im Dienst)."""
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def parse_datetime(value) -> Optional[datetime.datetime]:
    if isinstance(value, datetime.datetime):
        return value
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# GESTALTUNG
# ---------------------------------------------------------------------------
def inject_css() -> None:
    base_px = FONT_SCALES.get(st.session_state.font_scale, 18)
    if st.session_state.high_contrast:
        text, bg, panel, border, primary, muted = (
            "#000000", "#FFFFFF", "#FFFFFF", "#000000", "#00293D", "#000000",
        )
    else:
        text, bg, panel, border, primary, muted = (
            "#16202A", "#FFFFFF", "#F1F4F7", "#C6D2DC", "#1B4965", "#41525F",
        )

    st.markdown(
        f"""
        <style>
        :root {{
            --text: {text};
            --bg: {bg};
            --panel: {panel};
            --border: {border};
            --primary: {primary};
            --muted: {muted};
        }}

        /* Grundschriftgröße: skaliert die gesamte Oberfläche mit. */
        html {{ font-size: {base_px}px; }}

        .stApp {{ background: var(--bg); }}

        /* Begrenzte Zeilenlänge - sehr lange Zeilen sind schwer zu lesen. */
        .block-container {{
            max-width: 1120px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }}

        h1 {{ font-size: 2.1rem !important; font-weight: 700 !important; line-height: 1.25 !important; }}
        h2 {{ font-size: 1.6rem !important; font-weight: 700 !important; }}
        h3 {{ font-size: 1.25rem !important; font-weight: 700 !important; }}

        p, li, label, .stMarkdown {{
            font-size: 1rem;
            line-height: 1.65;
            color: var(--text);
        }}

        /* Tastaturbedienung muss jederzeit klar erkennbar sein. */
        *:focus-visible {{
            outline: 3px solid var(--primary) !important;
            outline-offset: 2px !important;
            border-radius: 4px;
        }}

        /* Große, gut treffbare Schaltflächen. */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            min-height: 3.1rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            border: 2px solid var(--primary);
            padding: 0.5rem 1.1rem;
        }}
        .stButton > button[kind="primary"],
        .stDownloadButton > button,
        .stFormSubmitButton > button {{
            background: var(--primary);
            color: #FFFFFF;
        }}
        .stButton > button[kind="secondary"] {{
            background: #FFFFFF;
            color: var(--primary);
        }}
        .stButton > button:disabled {{
            opacity: 0.55;
            border-color: var(--muted);
        }}

        /* Reiter deutlich größer, aktiver Reiter klar markiert. */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            border-bottom: 2px solid var(--border);
        }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 1.02rem;
            font-weight: 600;
            padding: 0.85rem 1.1rem;
            color: var(--muted);
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--primary) !important;
            border-bottom: 4px solid var(--primary) !important;
        }}

        /* Eingabefelder mit sichtbarem Rahmen. */
        .stTextInput input, .stTextArea textarea {{
            font-size: 1rem !important;
            border: 2px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text) !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: var(--primary) !important;
        }}
        .stTextInput label, .stTextArea label, .stSelectbox label,
        .stRadio label, .stFileUploader label, .stCheckbox label {{
            font-weight: 600 !important;
            font-size: 1rem !important;
            color: var(--text) !important;
        }}

        /* Hinweisfelder mit kräftiger Umrandung. */
        [data-testid="stAlert"] {{
            border-radius: 8px;
            border-left: 6px solid var(--primary);
            font-size: 1rem;
        }}

        /* Chatblasen */
        [data-testid="stChatMessage"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }}
        [data-testid="stChatInput"] textarea {{ font-size: 1rem !important; }}

        /* Karten für klare Gliederung */
        .karte {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 1rem;
        }}
        .karte h3 {{ margin-top: 0 !important; }}

        .schritt {{
            display: inline-block;
            background: var(--primary);
            color: #FFFFFF;
            font-weight: 700;
            border-radius: 50%;
            width: 1.9rem;
            height: 1.9rem;
            line-height: 1.9rem;
            text-align: center;
            margin-right: 0.5rem;
        }}

        /* Bewegung reduzieren, wenn das Betriebssystem das verlangt. */
        @media (prefers-reduced-motion: reduce) {{
            * {{ animation: none !important; transition: none !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# VERBINDUNG ZUM ÖRTLICHEN SITZUNGSDIENST
# ---------------------------------------------------------------------------
def api_create_session():
    try:
        response = requests.post(f"{API_URL}/session", timeout=10)
        return response if response.status_code == 200 else None
    except requests.RequestException:
        return None


def api_load_session(token: str):
    try:
        return requests.get(f"{API_URL}/session/{token}", timeout=20)
    except requests.RequestException:
        return None


def api_sync_session(token: str, data: dict) -> None:
    try:
        requests.put(f"{API_URL}/session/{token}", json={"data": data}, timeout=25)
    except requests.RequestException:
        # Ein fehlgeschlagener Abgleich darf die Bedienung nicht blockieren;
        # beim nächsten Durchlauf wird erneut versucht zu speichern.
        pass


def api_extend_session(token: str):
    try:
        return requests.post(f"{API_URL}/session/{token}/extend", timeout=10)
    except requests.RequestException:
        return None


def api_delete_session(token: str) -> bool:
    try:
        return requests.delete(f"{API_URL}/session/{token}", timeout=20).status_code == 200
    except requests.RequestException:
        return False


def serialize_documents(docs: List[Document]) -> list:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def deserialize_documents(raw: list) -> List[Document]:
    return [
        Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {}))
        for item in (raw or [])
    ]


def sync_session() -> None:
    if not st.session_state.token:
        return
    payload = {}
    for key in SYNCED_STATE_KEYS:
        if key in TEXT_FIELDS:
            payload[key] = get_field(key)
        elif key == "user_documents":
            payload[key] = serialize_documents(st.session_state.user_documents)
        else:
            payload[key] = st.session_state.get(key)
    api_sync_session(st.session_state.token, payload)


def apply_loaded_data(data: dict) -> None:
    for key in SYNCED_STATE_KEYS:
        if key not in data:
            continue
        if key in TEXT_FIELDS:
            set_field(key, data[key])
        elif key == "user_documents":
            st.session_state.user_documents = deserialize_documents(data[key])
        else:
            st.session_state[key] = data[key]
    st.session_state.document_names = collect_document_names()


def collect_document_names() -> List[str]:
    return sorted(
        {
            doc.metadata.get("source", "")
            for doc in st.session_state.user_documents
            if doc.metadata.get("source")
        }
    )


# ---------------------------------------------------------------------------
# ZWISCHENGESPEICHERTE RESSOURCEN (alle örtlich)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return pflege_rag.create_embeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return pflege_rag.create_llm()


@st.cache_resource(show_spinner=False)
def get_expert_database():
    try:
        return pflege_rag.open_expert_database(get_embeddings())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_converter():
    return DocumentConverter()


def rebuild_user_store() -> None:
    """Baut den Suchindex der hochgeladenen Unterlagen neu auf (nur im Arbeitsspeicher)."""
    docs = st.session_state.user_documents
    st.session_state.user_store = (
        pflege_rag.build_user_vector_store(docs, get_embeddings()) if docs else None
    )


def get_user_store():
    if "user_store" not in st.session_state:
        rebuild_user_store()
    return st.session_state.user_store


# ---------------------------------------------------------------------------
# STARTSEITE
# ---------------------------------------------------------------------------
def render_start_page() -> None:
    st.title("⚖️ Pflegehilfe Online")
    st.markdown(
        "#### Unterstützung beim Widerspruch gegen einen Pflegegradbescheid\n"
        "Diese Anwendung hilft Ihnen, Ihre Pflegeunterlagen zu prüfen und einen Widerspruch "
        "vorzubereiten. **Sie brauchen sich nicht anzumelden.** Wir fragen weder Ihren Namen "
        "noch Ihre E-Mail-Adresse ab."
    )

    st.markdown("---")
    links, rechts = st.columns(2, gap="large")

    with links:
        st.markdown(
            '<div class="karte"><h3>🆕 Neu anfangen</h3>'
            "<p>Sie starten eine neue Sitzung. Danach erhalten Sie einen persönlichen "
            "Zugangscode. Mit diesem Code können Sie später an derselben Stelle weiterarbeiten.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Neu anfangen", type="primary", use_container_width=True):
            response = api_create_session()
            if response is None:
                st.error(
                    "Die Anwendung ist gerade nicht erreichbar. Bitte prüfen Sie, ob der "
                    "Hintergrunddienst läuft, und versuchen Sie es noch einmal."
                )
            else:
                payload = response.json()
                st.session_state.token = payload["token"]
                st.session_state.expires_at = payload["expires_at"]
                st.rerun()

    with rechts:
        st.markdown(
            '<div class="karte"><h3>🔑 Mit Zugangscode fortsetzen</h3>'
            "<p>Sie haben bereits einen Zugangscode? Dann geben Sie ihn hier ein. Ihre Unterlagen, "
            "Ihr Gesprächsverlauf und Ihr Schreiben sind dann wieder da.</p></div>",
            unsafe_allow_html=True,
        )
        with st.form("token_form"):
            token_input = st.text_input(
                "Ihr Zugangscode",
                help="Der lange Code, den Sie beim letzten Mal erhalten und aufgeschrieben haben.",
            )
            if st.form_submit_button("Weiterarbeiten", use_container_width=True):
                token = token_input.strip()
                if not token:
                    st.error("Bitte geben Sie zuerst Ihren Zugangscode ein.")
                else:
                    response = api_load_session(token)
                    if response is None:
                        st.error("Die Anwendung ist gerade nicht erreichbar.")
                    elif response.status_code == 200:
                        payload = response.json()
                        st.session_state.token = payload["token"]
                        st.session_state.expires_at = payload["expires_at"]
                        apply_loaded_data(payload.get("data", {}))
                        rebuild_user_store()
                        st.rerun()
                    elif response.status_code == 410:
                        st.error(
                            "Dieser Zugangscode ist abgelaufen. Aus Datenschutzgründen wurden alle "
                            "zugehörigen Daten bereits vollständig gelöscht."
                        )
                    else:
                        st.error(
                            "Dieser Zugangscode ist nicht bekannt. Bitte prüfen Sie Ihre Eingabe "
                            "auf Tippfehler."
                        )

    st.markdown("---")
    st.markdown("### 🔒 Ihre Daten bleiben auf diesem Rechner")
    spalte1, spalte2 = st.columns(2, gap="large")
    with spalte1:
        st.markdown(
            "- Ihre Unterlagen werden **ausschließlich auf diesem Rechner** verarbeitet.\n"
            "- Es werden **keine Daten an Unternehmen im Internet** übertragen.\n"
            "- Gespeicherte Daten sind **verschlüsselt**."
        )
    with spalte2:
        st.markdown(
            f"- Nach **{SESSION_DAYS // 7} Wochen** wird alles **automatisch und vollständig gelöscht**.\n"
            "- Sie können die Frist jederzeit um 3 Tage verlängern.\n"
            "- Sie können alles jederzeit **sofort selbst löschen**."
        )
    st.warning(
        "**Wichtiger Hinweis:** Diese Anwendung ersetzt keine Rechtsberatung. Von der künstlichen "
        "Intelligenz erstellte Texte müssen Sie vor dem Absenden immer selbst sorgfältig prüfen."
    )


# ---------------------------------------------------------------------------
# REITER 1: DATEN HOCHLADEN
# ---------------------------------------------------------------------------
def render_upload_tab() -> None:
    st.header("Ihre Unterlagen hochladen")
    st.markdown(
        "Laden Sie hier alle Unterlagen hoch, die für den Widerspruch wichtig sind. "
        "Sie können **mehrere Dateien gleichzeitig** auswählen."
    )
    st.info(
        "**Das hilft besonders:** der Pflegegradbescheid, das Gutachten des Medizinischen "
        "Dienstes, Ihr Pflegetagebuch, Arztberichte und Krankenhausberichte.",
        icon="💡",
    )

    with st.expander("🔒 Datenschutzhinweis – bitte einmal lesen"):
        st.markdown(
            """
**Was mit Ihren Unterlagen passiert**

Ihre Dateien werden ausschließlich auf diesem Rechner gelesen und ausgewertet. Sie werden
**nicht** an ein Unternehmen im Internet übertragen und **nicht** zum Trainieren von
künstlicher Intelligenz verwendet. Auch das verwendete Sprachmodell läuft örtlich auf
diesem Rechner.

**Wie gespeichert wird**

Ihr Arbeitsstand bleibt nur erhalten, wenn Sie Ihren Zugangscode aufbewahren. Alles
Gespeicherte ist verschlüsselt. Ohne den Zugangscode kann niemand darauf zugreifen.

**Wie gelöscht wird**

Spätestens 4 Wochen nach dem Beginn wird Ihre Sitzung automatisch und vollständig
gelöscht: der Gesprächsverlauf, die Inhalte Ihrer Unterlagen und Ihr Schreiben. Im Reiter
**Einstellungen** können Sie alles auch jederzeit sofort selbst löschen.

**Empfehlung**

Laden Sie nur die Unterlagen hoch, die Sie wirklich brauchen. Je weniger sensible Daten
verarbeitet werden, desto besser.
            """
        )

    st.markdown(f"**Erlaubt:** PDF-Dateien, höchstens {MAX_FILE_SIZE_MB} MB je Datei.")
    uploaded_files = st.file_uploader(
        "Dateien auswählen",
        type="pdf",
        accept_multiple_files=True,
        help="Mit gedrückter Strg-Taste können Sie mehrere Dateien auf einmal auswählen.",
    )

    if uploaded_files:
        st.markdown(
            f"**{len(uploaded_files)} Datei(en) ausgewählt.** "
            "Klicken Sie jetzt auf „Unterlagen einlesen“."
        )
        if st.button("📥 Unterlagen einlesen", type="primary"):
            process_uploads(uploaded_files)

    st.markdown("---")
    render_document_list()


def process_uploads(uploaded_files) -> None:
    """Liest die hochgeladenen PDF-Dateien ein und baut den Suchindex neu auf."""
    if len(uploaded_files) + len(st.session_state.document_names) > MAX_DOCUMENTS:
        st.error(
            f"Es können höchstens {MAX_DOCUMENTS} Dokumente gleichzeitig verarbeitet werden. "
            "Bitte entfernen Sie zuerst nicht benötigte Unterlagen."
        )
        return

    converter = get_converter()
    fortschritt = st.progress(0.0, text="Die Unterlagen werden gelesen …")
    neue_dokumente: List[Document] = []
    hinweise: List[str] = []
    vorhanden = set(st.session_state.document_names)

    for index, datei in enumerate(uploaded_files, start=1):
        fortschritt.progress(
            (index - 1) / len(uploaded_files),
            text=f"Datei {index} von {len(uploaded_files)} wird gelesen: {datei.name}",
        )

        if datei.name in vorhanden:
            hinweise.append(f"„{datei.name}“ wurde übersprungen, weil sie bereits eingelesen ist.")
            continue
        if datei.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
            hinweise.append(f"„{datei.name}“ ist größer als {MAX_FILE_SIZE_MB} MB und wurde übersprungen.")
            continue

        try:
            dokument = pflege_rag.extract_document_from_pdf(datei.getvalue(), datei.name, converter)
        except Exception:
            # Der technische Fehlertext wird bewusst nicht angezeigt, damit keine
            # Inhalte aus dem Dokument in einer Meldung erscheinen können.
            hinweise.append(f"„{datei.name}“ konnte nicht gelesen werden. Ist die Datei beschädigt?")
            continue

        if dokument is None:
            hinweise.append(
                f"„{datei.name}“ enthält keinen lesbaren Text. Eingescannte Unterlagen sollten "
                "in guter Qualität vorliegen."
            )
            continue

        neue_dokumente.append(dokument)
        vorhanden.add(datei.name)

    if neue_dokumente:
        fortschritt.progress(1.0, text="Die Unterlagen werden durchsuchbar gemacht …")
        st.session_state.user_documents.extend(pflege_rag.split_documents(neue_dokumente))
        st.session_state.document_names = collect_document_names()
        rebuild_user_store()

    fortschritt.empty()

    if neue_dokumente:
        st.success(
            f"**{len(neue_dokumente)} Dokument(e) erfolgreich eingelesen.** "
            "Sie können jetzt im Reiter „KI-Assistent“ Fragen stellen.",
            icon="✅",
        )
    for meldung in hinweise:
        st.warning(meldung, icon="⚠️")
    if not neue_dokumente and not hinweise:
        st.warning("Es konnten keine neuen Unterlagen eingelesen werden.", icon="⚠️")


def render_document_list() -> None:
    st.subheader("Eingelesene Unterlagen")
    if not st.session_state.document_names:
        st.info("Sie haben noch keine Unterlagen hochgeladen.", icon="📄")
        return

    for name in st.session_state.document_names:
        anzahl = sum(
            1 for doc in st.session_state.user_documents if doc.metadata.get("source") == name
        )
        spalte_name, spalte_knopf = st.columns([5, 1])
        with spalte_name:
            st.markdown(f"**📄 {name}**  \n{anzahl} Textabschnitte")
        with spalte_knopf:
            if st.button("Entfernen", key=f"remove_{name}", help=f"„{name}“ wieder entfernen"):
                st.session_state.user_documents = [
                    doc for doc in st.session_state.user_documents
                    if doc.metadata.get("source") != name
                ]
                st.session_state.document_names = collect_document_names()
                rebuild_user_store()
                st.rerun()


# ---------------------------------------------------------------------------
# REITER 2: KI-ASSISTENT
# ---------------------------------------------------------------------------
def render_chat_tab() -> None:
    st.header("KI-Assistent")

    if not st.session_state.user_documents:
        st.warning(
            "Sie haben noch keine Unterlagen hochgeladen. Der Assistent kann Ihnen erst dann "
            "richtig helfen. Wechseln Sie dazu in den Reiter **„Daten hochladen“**.",
            icon="⚠️",
        )

    st.markdown("**Was möchten Sie tun?** Wählen Sie eine Aufgabe oder schreiben Sie unten Ihre eigene Frage.")

    aktion: Optional[str] = None
    spalte1, spalte2 = st.columns(2, gap="small")
    spalte3, spalte4 = st.columns(2, gap="small")

    with spalte1:
        if st.button("📖 Daten einlesen", use_container_width=True,
                     help="Der Assistent verschafft sich einen Überblick über Ihre Unterlagen."):
            aktion = pflege_rag.QUICK_ACTIONS["einlesen"]
    with spalte2:
        if st.button("🔍 Differenzanalyse", use_container_width=True,
                     help="Vergleicht das Gutachten des Medizinischen Dienstes mit Ihren übrigen Unterlagen."):
            aktion = pflege_rag.QUICK_ACTIONS["differenz"]
    with spalte3:
        if st.button("💬 Argumente sammeln", use_container_width=True,
                     help="Sammelt begründete Argumente für Ihren Widerspruch."):
            aktion = pflege_rag.QUICK_ACTIONS["argumente"]
    with spalte4:
        if st.button("✍️ Widerspruch schreiben", use_container_width=True,
                     help="Verfasst die Begründung für Ihr Widerspruchsschreiben."):
            aktion = pflege_rag.QUICK_ACTIONS["schreiben"]

    verlauf = st.container(height=430, border=True)
    with verlauf:
        if not st.session_state.messages:
            st.markdown(
                "**Der Gesprächsverlauf erscheint hier.**  \n"
                "Beginnen Sie mit einer der Aufgaben oben oder stellen Sie unten eine eigene Frage."
            )
        for nachricht in st.session_state.messages:
            with st.chat_message(nachricht["role"], avatar="🧑" if nachricht["role"] == "user" else "⚖️"):
                st.markdown(nachricht["content"])

    eingabe = st.chat_input("Schreiben Sie hier Ihre Frage …")
    prompt = aktion or eingabe

    if prompt:
        with verlauf:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant", avatar="⚖️"):
                antwort = generate_answer(prompt)

        if antwort:
            st.session_state.messages.append({"role": "assistant", "content": antwort})
            st.session_state.last_generated_letter = antwort

    render_sources()

    if st.session_state.messages:
        if st.button("🗑️ Gesprächsverlauf löschen",
                     help="Löscht nur das Gespräch. Ihre Unterlagen bleiben erhalten."):
            st.session_state.messages = []
            st.session_state.last_user_sources = []
            st.session_state.last_expert_sources = []
            st.rerun()


def generate_answer(prompt: str) -> str:
    """Sucht passende Textstellen und gibt die Antwort des Sprachmodells fortlaufend aus."""
    try:
        with st.spinner("Ihre Unterlagen werden durchsucht …"):
            system_prompt, user_docs, expert_docs = pflege_rag.prepare_context(
                get_expert_database(), get_user_store(), prompt
            )
        st.session_state.last_user_sources = pflege_rag.build_source_list(user_docs)
        st.session_state.last_expert_sources = pflege_rag.build_source_list(expert_docs)

        nachrichten = pflege_rag.build_messages(system_prompt, st.session_state.messages)
        return st.write_stream(pflege_rag.stream_answer(get_llm(), nachrichten))
    except Exception:
        st.error(
            "**Die künstliche Intelligenz ist gerade nicht erreichbar.**\n\n"
            "Bitte prüfen Sie, ob das Programm LM Studio läuft und das Sprachmodell geladen ist. "
            "Versuchen Sie es danach noch einmal."
        )
        return ""


def render_sources() -> None:
    if not (st.session_state.last_user_sources or st.session_state.last_expert_sources):
        return
    with st.expander("📚 Worauf sich die letzte Antwort stützt"):
        if st.session_state.last_user_sources:
            st.markdown("**Aus Ihren eigenen Unterlagen:**")
            for quelle in st.session_state.last_user_sources:
                st.markdown(f"- **{quelle['source']}**: {quelle['preview']}")
        if st.session_state.last_expert_sources:
            st.markdown("**Aus dem geprüften Fachwissen:**")
            for quelle in st.session_state.last_expert_sources:
                st.markdown(f"- **{quelle['source']}**: {quelle['preview']}")


# ---------------------------------------------------------------------------
# REITER 3: PDF ERSTELLEN
# ---------------------------------------------------------------------------
def feld(label: str, key: str, hilfe: str = "", platzhalter: str = "") -> None:
    """Eingabefeld, dessen Inhalt beim Zugangscode gespeichert wird.

    Der Wert wird bewusst nur über `key` verwaltet (kein `value=`), damit ein
    programmgesteuertes Vorbelegen zuverlässig wirkt.
    """
    st.text_input(
        label,
        key=f"w_{key}",
        help=hilfe or None,
        placeholder=platzhalter or None,
    )


def render_pdf_tab() -> None:
    st.header("Widerspruch als PDF erstellen")
    st.markdown(
        "Hier entsteht Ihr fertiges Widerspruchsschreiben zum Ausdrucken und Unterschreiben. "
        "Der Aufbau folgt dem Musterbrief der Verbraucherzentrale."
    )

    st.markdown('<p><span class="schritt">1</span><strong>Ihre Angaben</strong></p>',
                unsafe_allow_html=True)
    links, rechts = st.columns(2, gap="large")
    with links:
        st.markdown("**Absender – Ihre Anschrift**")
        feld("Ihr Vor- und Nachname", "absender_name", platzhalter="Michaela Muster")
        feld("Ihre Straße und Hausnummer", "absender_strasse", platzhalter="Musterweg 1")
        feld("Ihre Postleitzahl und Ihr Ort", "absender_plz_ort", platzhalter="99999 Musterstadt")
        feld("Ort für die Datumszeile", "absender_ort",
             hilfe="Erscheint oben rechts vor dem Datum. Kann leer bleiben.",
             platzhalter="Musterstadt")
    with rechts:
        st.markdown("**Empfänger – Ihre Pflegekasse**")
        # Die Bezeichnungen unterscheiden sich bewusst von denen des Absenders:
        # Gleichlautende Feldbezeichnungen werden von Vorleseprogrammen nicht
        # unterscheidbar angesagt.
        feld("Name der Pflegekasse", "kasse_name",
             platzhalter="Pflegekasse bei der Musterkrankenkasse")
        feld("Straße und Hausnummer der Pflegekasse", "kasse_strasse")
        feld("Postleitzahl und Ort der Pflegekasse", "kasse_plz_ort")

    st.markdown("**Angaben zum Bescheid**")
    spalte1, spalte2, spalte3 = st.columns(3)
    with spalte1:
        feld("Datum des Bescheids", "bescheid_datum",
             hilfe="Das Datum, das oben auf Ihrem Bescheid steht.", platzhalter="14.03.2026")
    with spalte2:
        feld("Aktenzeichen", "aktenzeichen",
             hilfe="Steht meist oben auf dem Bescheid. Kann leer bleiben.")
    with spalte3:
        feld("Versichertennummer", "versichert_nr")
    feld("Name der pflegebedürftigen Person", "versichert_name",
         hilfe="Nur ausfüllen, wenn Sie den Widerspruch für eine andere Person stellen.")

    st.markdown("---")
    st.markdown('<p><span class="schritt">2</span><strong>Begründung des Widerspruchs</strong></p>',
                unsafe_allow_html=True)

    if st.session_state.last_generated_letter:
        if st.button("⬇️ Text aus dem KI-Gespräch übernehmen", type="secondary"):
            set_field(
                "letter_text",
                pflege_pdf.prepare_begruendung(st.session_state.last_generated_letter),
            )
            st.rerun()
    else:
        st.info(
            "Noch kein Text vorhanden. Nutzen Sie im Reiter **„KI-Assistent“** die Aufgabe "
            "**„Widerspruch schreiben“**. Der Text erscheint dann hier zum Übernehmen.",
            icon="💡",
        )

    st.text_area(
        "Begründung (Sie können den Text hier frei bearbeiten)",
        key="w_letter_text",
        height=320,
        help="Anrede, Betreff und Grußformel ergänzt die Vorlage automatisch. "
             "Schreiben Sie hier nur die Begründung.",
    )

    # Sprachmodelle setzen Platzhalter ein, wenn ihnen eine Angabe fehlt. Ein
    # Schreiben mit solchen Lücken darf nicht bei der Pflegekasse landen.
    luecken = pflege_pdf.find_placeholders(get_field("letter_text"))
    if luecken:
        st.warning(
            "**Im Text stehen noch Lücken, die Sie ausfüllen sollten.** Die künstliche "
            "Intelligenz konnte diese Angaben nicht aus Ihren Unterlagen entnehmen:\n\n"
            + "\n".join(f"- `{luecke}`" for luecke in luecken[:8])
            + "\n\nBitte ersetzen Sie diese Stellen, bevor Sie das Schreiben abschicken.",
            icon="✏️",
        )

    begruendung_folgt = st.checkbox(
        "Ich reiche die Begründung später nach (Widerspruch zunächst nur fristwahrend einlegen)",
        help="Damit wird die Frist gewahrt. Im Schreiben steht dann, dass die Begründung in Kürze folgt.",
    )

    st.markdown("---")
    st.markdown('<p><span class="schritt">3</span><strong>PDF erstellen und herunterladen</strong></p>',
                unsafe_allow_html=True)

    daten = pflege_pdf.LetterData(
        absender_name=get_field("absender_name"),
        absender_strasse=get_field("absender_strasse"),
        absender_plz_ort=get_field("absender_plz_ort"),
        kasse_name=get_field("kasse_name"),
        kasse_strasse=get_field("kasse_strasse"),
        kasse_plz_ort=get_field("kasse_plz_ort"),
        versichert_name=get_field("versichert_name") or get_field("absender_name"),
        versichert_nr=get_field("versichert_nr"),
        aktenzeichen=get_field("aktenzeichen"),
        bescheid_datum=get_field("bescheid_datum"),
        begruendung=get_field("letter_text"),
        begruendung_folgt=begruendung_folgt,
        ort=get_field("absender_ort"),
    )

    fehlend = pflege_pdf.validate(daten)
    if fehlend:
        st.warning(
            "Bitte ergänzen Sie noch folgende Angaben:\n\n"
            + "\n".join(f"- {eintrag}" for eintrag in fehlend),
            icon="⚠️",
        )

    if st.button("📄 PDF jetzt erstellen", type="primary", disabled=bool(fehlend)):
        try:
            st.session_state.pdf_bytes = pflege_pdf.build_letter_pdf(daten)
            st.success("Ihr Widerspruchsschreiben wurde erstellt.", icon="✅")
        except Exception:
            st.session_state.pop("pdf_bytes", None)
            st.error("Das PDF konnte nicht erstellt werden. Bitte prüfen Sie Ihre Eingaben.")

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            "⬇️ PDF herunterladen",
            data=st.session_state.pdf_bytes,
            file_name="Widerspruch_Pflegegrad.pdf",
            mime="application/pdf",
        )
        st.info(
            "**So geht es weiter:** Drucken Sie das Schreiben aus und **unterschreiben Sie es von Hand**. "
            "Schicken Sie es per Post – am besten als Einwurfeinschreiben – oder per Fax an Ihre "
            "Pflegekasse. Eine E-Mail wahrt die Frist nicht.",
            icon="📮",
        )


# ---------------------------------------------------------------------------
# REITER 4: EINSTELLUNGEN
# ---------------------------------------------------------------------------
def render_settings_tab() -> None:
    st.header("Einstellungen")

    st.subheader("🔑 Ihr Zugangscode")
    st.markdown(
        "Mit diesem Code arbeiten Sie später weiter. **Schreiben Sie ihn sich auf** und bewahren "
        "Sie ihn sicher auf. Er kann nicht wiederhergestellt werden."
    )
    st.code(st.session_state.token, language=None)

    ablauf = parse_datetime(st.session_state.expires_at)
    if ablauf:
        verbleibend = ablauf - utcnow()
        tage = max(verbleibend.days, 0)
        stunden = max(verbleibend.seconds // 3600, 0) if verbleibend.days >= 0 else 0
        st.markdown(
            f"### ⏳ Noch **{tage} Tage und {stunden} Stunden** gültig\n"
            f"Ihre Sitzung wird am **{ablauf.strftime('%d.%m.%Y um %H:%M')} Uhr** automatisch "
            "und vollständig gelöscht."
        )
        st.progress(min(max(tage / SESSION_DAYS, 0.0), 1.0))

    if st.button("➕ Um 3 Tage verlängern", type="primary"):
        antwort = api_extend_session(st.session_state.token)
        if antwort is not None and antwort.status_code == 200:
            st.session_state.expires_at = antwort.json()["expires_at"]
            st.success("Ihre Sitzung wurde um 3 Tage verlängert.", icon="✅")
            st.rerun()
        else:
            st.error("Die Verlängerung hat nicht geklappt. Bitte versuchen Sie es noch einmal.")

    st.markdown("---")
    st.subheader("👁️ Darstellung")
    st.markdown("Passen Sie die Anzeige an Ihre Bedürfnisse an.")
    spalte1, spalte2 = st.columns(2)
    with spalte1:
        auswahl = st.radio(
            "Schriftgröße",
            list(FONT_SCALES.keys()),
            index=list(FONT_SCALES.keys()).index(st.session_state.font_scale),
            horizontal=True,
        )
        if auswahl != st.session_state.font_scale:
            st.session_state.font_scale = auswahl
            st.rerun()
    with spalte2:
        kontrast = st.toggle(
            "Hoher Kontrast",
            value=st.session_state.high_contrast,
            help="Verstärkt Schwarz-Weiß-Kontraste und Umrandungen.",
        )
        if kontrast != st.session_state.high_contrast:
            st.session_state.high_contrast = kontrast
            st.rerun()

    st.markdown("---")
    st.subheader("🗑️ Alles löschen und beenden")
    st.markdown(
        "Hiermit werden **sofort und unwiderruflich** gelöscht: Ihr Zugangscode, die Inhalte Ihrer "
        "Unterlagen, Ihr Gesprächsverlauf und Ihr Schreiben. Danach ist ein Wiedereinstieg nicht "
        "mehr möglich."
    )
    bestaetigt = st.checkbox("Ja, ich möchte wirklich alles endgültig löschen.")
    if st.button("🗑️ Jetzt alles löschen und beenden", disabled=not bestaetigt):
        erfolgreich = api_delete_session(st.session_state.token)
        reset_local_state()
        st.session_state.deletion_done = erfolgreich
        st.rerun()

    st.markdown("---")
    st.subheader("🚪 Nur abmelden")
    st.markdown(
        "Sie schließen die Sitzung auf diesem Bildschirm. Ihre Daten bleiben gespeichert und Sie "
        "können mit Ihrem Zugangscode jederzeit weiterarbeiten."
    )
    if st.button("🚪 Abmelden"):
        reset_local_state()
        st.rerun()


# ---------------------------------------------------------------------------
# HAUPTANSICHT
# ---------------------------------------------------------------------------
def render_app() -> None:
    kopf_links, kopf_rechts = st.columns([3, 1])
    with kopf_links:
        st.title("⚖️ Pflegehilfe Online")
    with kopf_rechts:
        ablauf = parse_datetime(st.session_state.expires_at)
        if ablauf:
            tage = max((ablauf - utcnow()).days, 0)
            st.markdown(
                "<div class='karte' style='text-align:center;padding:0.8rem'>"
                f"<strong>Sitzung aktiv</strong><br>noch {tage} Tage gültig</div>",
                unsafe_allow_html=True,
            )

    if not st.session_state.user_documents:
        st.info(
            "**So gehen Sie vor:** 1. Unterlagen hochladen → 2. mit dem KI-Assistenten prüfen → "
            "3. Widerspruch als PDF erstellen.",
            icon="🧭",
        )

    reiter = st.tabs([
        "📁  Daten hochladen",
        "💬  KI-Assistent",
        "📄  PDF erstellen",
        "⚙️  Einstellungen",
    ])
    with reiter[0]:
        render_upload_tab()
    with reiter[1]:
        render_chat_tab()
    with reiter[2]:
        render_pdf_tab()
    with reiter[3]:
        render_settings_tab()

    st.markdown("---")
    st.caption(
        "Diese Anwendung ersetzt keine Rechtsberatung. Alle von der künstlichen Intelligenz "
        "erstellten Texte müssen vor dem Absenden geprüft werden. Ihre Daten werden "
        "ausschließlich örtlich verarbeitet."
    )


def main() -> None:
    init_state()
    inject_css()

    if st.session_state.pop("deletion_done", None) is not None:
        st.success(
            "**Alle Ihre Daten wurden vollständig gelöscht.** Vielen Dank für Ihr Vertrauen.",
            icon="✅",
        )

    if st.session_state.token is None:
        render_start_page()
    else:
        render_app()
        sync_session()


main()
