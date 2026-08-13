"""Pflegehilfe Online – Weboberfläche des Assistenten für den Pflegegrad-Widerspruch.

Gestaltungsleitlinien (nach Don Norman):

* Sichtbarkeit  – Eine Fortschrittsanzeige zeigt jederzeit, in welchem der drei
                  Arbeitsschritte man steht und was bereits erledigt ist.
* Rückmeldung   – Jede Aktion bestätigt sich sofort: Fortschrittsbalken beim
                  Einlesen, fortlaufender Text bei der Antwort, klare Meldungen.
* Signifier     – Beschriftungen sagen, was passieren wird, nicht was ein Ding ist.
* Constraints   – Was noch nicht möglich ist, bleibt gesperrt und erklärt sich.
                  Fehler werden dadurch verhindert statt hinterher gemeldet.
* Mapping       – Die Reihenfolge der Reiter entspricht der Reihenfolge der Arbeit.

Datenschutz: keine Registrierung, Zugriff nur über einen zufälligen Zugangscode,
alle Verarbeitung ausschließlich örtlich.
"""

from __future__ import annotations

import datetime
import os
from typing import List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document

import pflege_pdf
import pflege_rag

load_dotenv()

st.set_page_config(
    page_title="Pflegehilfe Online",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# Eingescannte Gutachten des Medizinischen Dienstes sind erfahrungsgemäß groß.
MAX_FILE_SIZE_MB = 30
MAX_DOCUMENTS = 15
SESSION_DAYS = 28

TEXT_FIELDS = [
    "absender_name", "absender_strasse", "absender_plz_ort",
    "kasse_name", "kasse_strasse", "kasse_plz_ort",
    "versichert_name", "versichert_nr", "aktenzeichen", "bescheid_datum",
    "letter_text",
]

SYNCED_STATE_KEYS = [
    *TEXT_FIELDS,
    "messages", "user_documents", "document_names", "last_generated_letter",
    "last_sources", "font_scale",
]

DEFAULT_STATE = {
    "messages": [],
    "user_documents": [],
    "document_names": [],
    "last_generated_letter": "",
    "last_sources": [],
    "font_scale": "Normal",
}

FONT_SCALES = {"Normal": 18, "Groß": 21, "Sehr groß": 24}


# ---------------------------------------------------------------------------
# ZUSTAND
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("token", None)
    st.session_state.setdefault("expires_at", None)
    for schluessel, vorgabe in DEFAULT_STATE.items():
        if schluessel not in st.session_state:
            st.session_state[schluessel] = list(vorgabe) if isinstance(vorgabe, list) else vorgabe
    for name in TEXT_FIELDS:
        st.session_state.setdefault(f"w_{name}", "")


def get_field(name: str) -> str:
    return st.session_state.get(f"w_{name}", "")


def set_field(name: str, wert: str) -> None:
    """Belegt ein Formularfeld vor. Muss vor dem Erzeugen des Widgets geschehen."""
    st.session_state[f"w_{name}"] = wert or ""


def reset_local_state() -> None:
    """Setzt die Anzeige zurück, ohne gespeicherte Daten anzutasten."""
    st.session_state.token = None
    st.session_state.expires_at = None
    for schluessel, vorgabe in DEFAULT_STATE.items():
        st.session_state[schluessel] = list(vorgabe) if isinstance(vorgabe, list) else vorgabe
    st.session_state.pop("user_index", None)
    st.session_state.pop("pdf_bytes", None)
    for schluessel in [k for k in list(st.session_state.keys()) if str(k).startswith("w_")]:
        del st.session_state[schluessel]


def utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def parse_datetime(wert) -> Optional[datetime.datetime]:
    if isinstance(wert, datetime.datetime):
        return wert
    if not wert:
        return None
    try:
        return datetime.datetime.fromisoformat(str(wert).replace("Z", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# GESTALTUNG
# ---------------------------------------------------------------------------
def inject_css() -> None:
    grundgroesse = FONT_SCALES.get(st.session_state.font_scale, 18)
    # Gemessen auf Weiß: Fließtext 16,5:1, Schaltflächen 9,6:1 - beides weit
    # über den 7:1, die WCAG auf der strengsten Stufe verlangt. Eine eigene
    # Hochkontrast-Umschaltung brachte darüber hinaus nichts und wurde entfernt.
    text, panel, rand, primaer, gedaempft = "#16202A", "#F1F4F7", "#C6D2DC", "#1B4965", "#41525F"
    erfolg, warnung = "#1B5E20", "#8B4000"

    st.markdown(
        f"""
        <style>
        :root {{
            --text: {text}; --panel: {panel}; --rand: {rand};
            --primaer: {primaer}; --gedaempft: {gedaempft};
            --erfolg: {erfolg}; --warnung: {warnung};
        }}
        html {{ font-size: {grundgroesse}px; }}
        .stApp {{ background: #FFFFFF; }}
        .block-container {{ max-width: 1140px; padding-top: 1.6rem; padding-bottom: 3rem; }}

        h1 {{ font-size: 2rem !important; font-weight: 700 !important; line-height: 1.25 !important; }}
        h2 {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
        h3 {{ font-size: 1.2rem !important; font-weight: 700 !important; }}
        p, li, label, .stMarkdown {{ font-size: 1rem; line-height: 1.65; color: var(--text); }}

        *:focus-visible {{
            outline: 3px solid var(--primaer) !important;
            outline-offset: 2px !important; border-radius: 4px;
        }}

        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            min-height: 3.1rem; font-size: 1rem; font-weight: 600;
            border-radius: 8px; border: 2px solid var(--primaer); padding: 0.5rem 1.1rem;
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button,
        .stFormSubmitButton > button {{ background: var(--primaer); color: #FFFFFF; }}
        .stButton > button[kind="secondary"] {{ background: #FFFFFF; color: var(--primaer); }}
        .stButton > button:disabled {{
            opacity: 0.5; border-color: var(--gedaempft); cursor: not-allowed;
        }}

        .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; border-bottom: 2px solid var(--rand); }}
        .stTabs [data-baseweb="tab"] {{
            font-size: 1.02rem; font-weight: 600; padding: 0.85rem 1.1rem; color: var(--gedaempft);
        }}
        .stTabs [aria-selected="true"] {{
            color: var(--primaer) !important; border-bottom: 4px solid var(--primaer) !important;
        }}

        .stTextInput input, .stTextArea textarea {{
            font-size: 1rem !important; border: 2px solid var(--rand) !important;
            border-radius: 8px !important; color: var(--text) !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{ border-color: var(--primaer) !important; }}
        .stTextInput label, .stTextArea label, .stSelectbox label,
        .stRadio label, .stFileUploader label, .stCheckbox label {{
            font-weight: 600 !important; font-size: 1rem !important; color: var(--text) !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 8px; border-left: 6px solid var(--primaer); font-size: 1rem;
        }}
        [data-testid="stChatMessage"] {{
            background: var(--panel); border: 1px solid var(--rand);
            border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.8rem;
        }}
        [data-testid="stChatInput"] textarea {{ font-size: 1rem !important; }}

        /* --- Kopfzeile mit Systemzustand --- */
        .kopf {{
            display: flex; justify-content: space-between; align-items: center;
            gap: 1rem; flex-wrap: wrap; border-bottom: 2px solid var(--rand);
            padding-bottom: 0.8rem; margin-bottom: 1rem;
        }}
        .kopf-titel {{ font-size: 1.6rem; font-weight: 700; color: var(--primaer); }}
        .zustand {{ display: flex; gap: 0.5rem; flex-wrap: wrap; }}
        .plakette {{
            background: var(--panel); border: 1px solid var(--rand); border-radius: 999px;
            padding: 0.3rem 0.85rem; font-size: 0.82rem; font-weight: 600; color: var(--text);
            white-space: nowrap;
        }}
        .plakette.aktiv {{ border-color: var(--erfolg); color: var(--erfolg); }}
        .plakette.offen {{ border-color: var(--warnung); color: var(--warnung); }}

        /* --- Fortschrittsanzeige über die drei Arbeitsschritte --- */
        .schritte {{ display: flex; gap: 0.5rem; margin: 0.2rem 0 1.4rem 0; flex-wrap: wrap; }}
        .schritt {{
            flex: 1 1 200px; background: var(--panel); border: 2px solid var(--rand);
            border-left: 6px solid var(--rand); border-radius: 8px; padding: 0.65rem 0.9rem;
        }}
        .schritt.fertig {{ border-left-color: var(--erfolg); }}
        .schritt.laufend {{ border-left-color: var(--primaer); background: #FFFFFF; }}
        .schritt-kopf {{ font-size: 0.8rem; font-weight: 700; color: var(--gedaempft); letter-spacing: 0.03em; }}
        .schritt-text {{ font-size: 0.98rem; font-weight: 600; color: var(--text); }}
        .schritt-status {{ font-size: 0.82rem; color: var(--gedaempft); }}
        .schritt.fertig .schritt-status {{ color: var(--erfolg); font-weight: 600; }}

        /* --- Karten und Quellenangaben --- */
        .karte {{
            background: var(--panel); border: 1px solid var(--rand);
            border-radius: 10px; padding: 1.1rem 1.25rem; margin-bottom: 1rem;
        }}
        .karte h3 {{ margin-top: 0 !important; }}

        /* Die drei Schritte auf der Startseite.
           Als Raster statt als Streamlit-Spalten: Rasterfelder sind von Haus
           aus gleich hoch, Spalten richten sich nach ihrem eigenen Inhalt -
           bei unterschiedlich langen Texten stehen die Kästen sonst
           verschieden tief. */
        .schrittkarten {{
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 1rem; margin-bottom: 1rem;
        }}
        .schrittkarten .karte {{ margin-bottom: 0; height: 100%; }}
        @media (max-width: 800px) {{
            .schrittkarten {{ grid-template-columns: 1fr; }}
        }}
        .quelle {{
            border-left: 4px solid var(--primaer); background: var(--panel);
            border-radius: 6px; padding: 0.7rem 0.95rem; margin-bottom: 0.7rem;
        }}
        .quelle-kopf {{ font-weight: 700; font-size: 0.95rem; color: var(--text); }}
        .quelle-art {{
            font-size: 0.78rem; color: var(--gedaempft); font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.04em;
        }}
        .quelle-text {{ font-size: 0.94rem; color: var(--text); margin-top: 0.35rem; }}
        .ziffer {{
            display: inline-block; min-width: 1.6rem; height: 1.6rem; line-height: 1.6rem;
            text-align: center; background: var(--primaer); color: #FFFFFF;
            border-radius: 50%; font-size: 0.85rem; font-weight: 700; margin-right: 0.5rem;
        }}

        @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# VERBINDUNG ZUM ÖRTLICHEN SITZUNGSDIENST
# ---------------------------------------------------------------------------
def api_create_session():
    try:
        antwort = requests.post(f"{API_URL}/session", timeout=10)
        return antwort if antwort.status_code == 200 else None
    except requests.RequestException:
        return None


def api_load_session(token: str):
    try:
        return requests.get(f"{API_URL}/session/{token}", timeout=25)
    except requests.RequestException:
        return None


def api_sync_session(token: str, daten: dict) -> None:
    try:
        requests.put(f"{API_URL}/session/{token}", json={"data": daten}, timeout=30)
    except requests.RequestException:
        pass  # Beim nächsten Durchlauf wird erneut gespeichert.


def api_extend_session(token: str):
    try:
        return requests.post(f"{API_URL}/session/{token}/extend", timeout=10)
    except requests.RequestException:
        return None


def api_delete_session(token: str) -> bool:
    try:
        return requests.delete(f"{API_URL}/session/{token}", timeout=25).status_code == 200
    except requests.RequestException:
        return False


def serialize_documents(docs: List[Document]) -> list:
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]


def deserialize_documents(roh: list) -> List[Document]:
    return [
        Document(page_content=e.get("page_content", ""), metadata=e.get("metadata", {}))
        for e in (roh or [])
    ]


def sync_session() -> None:
    if not st.session_state.token:
        return
    nutzdaten = {}
    for schluessel in SYNCED_STATE_KEYS:
        if schluessel in TEXT_FIELDS:
            nutzdaten[schluessel] = get_field(schluessel)
        elif schluessel == "user_documents":
            nutzdaten[schluessel] = serialize_documents(st.session_state.user_documents)
        else:
            nutzdaten[schluessel] = st.session_state.get(schluessel)
    api_sync_session(st.session_state.token, nutzdaten)


def apply_loaded_data(daten: dict) -> None:
    for schluessel in SYNCED_STATE_KEYS:
        if schluessel not in daten:
            continue
        if schluessel in TEXT_FIELDS:
            set_field(schluessel, daten[schluessel])
        elif schluessel == "user_documents":
            st.session_state.user_documents = deserialize_documents(daten[schluessel])
        else:
            st.session_state[schluessel] = daten[schluessel]
    st.session_state.document_names = collect_document_names()


def collect_document_names() -> List[str]:
    return sorted(
        {
            d.metadata.get("source", "")
            for d in st.session_state.user_documents
            if d.metadata.get("source")
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
def get_reranker():
    """Lädt den Cross-Encoder. Beim ersten Aufruf dauert das etwa 20 Sekunden."""
    try:
        return pflege_rag.create_reranker()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_expert_index():
    """Öffnet die Wissensdatenbank und baut den Stichwortindex dazu auf."""
    try:
        speicher = pflege_rag.open_expert_database(get_embeddings())
        abschnitte = pflege_rag.load_all_expert_chunks(speicher)
        return pflege_rag.HybridIndex(speicher, abschnitte)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_converters():
    """Zwei Umwandler: schnell für digitale PDFs, mit Texterkennung für Scans."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    schnell = PdfPipelineOptions()
    schnell.do_ocr = False
    schnell.do_table_structure = True
    mit_ocr = PdfPipelineOptions()
    mit_ocr.do_ocr = True
    mit_ocr.do_table_structure = True
    return (
        DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=schnell)}),
        DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=mit_ocr)}),
    )


def rebuild_user_index() -> None:
    """Baut den Suchindex der hochgeladenen Unterlagen neu auf (nur im Arbeitsspeicher)."""
    dokumente = st.session_state.user_documents
    if not dokumente:
        st.session_state.user_index = None
        return
    from langchain_qdrant import QdrantVectorStore

    speicher = QdrantVectorStore.from_documents(
        documents=dokumente,
        embedding=get_embeddings(),
        location=":memory:",
        collection_name="nutzerdokumente",
    )
    st.session_state.user_index = pflege_rag.HybridIndex(speicher, dokumente)


def get_user_index():
    if "user_index" not in st.session_state:
        rebuild_user_index()
    return st.session_state.user_index


# ---------------------------------------------------------------------------
# BAUSTEINE DER OBERFLÄCHE
# ---------------------------------------------------------------------------
def anzahl_dokumente() -> int:
    return len(st.session_state.document_names)


def hat_antworten() -> bool:
    return any(n["role"] == "assistant" for n in st.session_state.messages)


def pdf_bereit() -> bool:
    return bool(get_field("absender_name") and get_field("kasse_name") and get_field("letter_text"))


def render_kopfzeile() -> None:
    """Zeigt jederzeit, in welchem Zustand die Sitzung ist."""
    ablauf = parse_datetime(st.session_state.expires_at)
    tage = max((ablauf - utcnow()).days, 0) if ablauf else None

    plaketten = ['<span class="plakette aktiv">Sitzung aktiv</span>']
    if tage is not None:
        plaketten.append(f'<span class="plakette">noch {tage} Tage gültig</span>')
    anzahl = anzahl_dokumente()
    klasse = "aktiv" if anzahl else "offen"
    text = f"{anzahl} Unterlage(n)" if anzahl else "keine Unterlagen"
    plaketten.append(f'<span class="plakette {klasse}">{text}</span>')

    st.markdown(
        '<div class="kopf"><div class="kopf-titel">Pflegehilfe Online</div>'
        f'<div class="zustand">{"".join(plaketten)}</div></div>',
        unsafe_allow_html=True,
    )


def render_fortschritt() -> None:
    """Fortschrittsanzeige über die drei Arbeitsschritte."""
    schritte = [
        ("Schritt 1", "Unterlagen hochladen",
         anzahl_dokumente() > 0,
         f"{anzahl_dokumente()} eingelesen" if anzahl_dokumente() else "noch offen"),
        ("Schritt 2", "Mit dem Assistenten prüfen",
         hat_antworten(),
         "Auswertung liegt vor" if hat_antworten() else "noch offen"),
        ("Schritt 3", "Widerspruch als PDF",
         bool(st.session_state.get("pdf_bytes")),
         "PDF erstellt" if st.session_state.get("pdf_bytes")
         else ("bereit" if pdf_bereit() else "noch offen")),
    ]

    laufend_gesetzt = False
    teile = []
    for kopf, titel, fertig, status in schritte:
        if fertig:
            klasse = "schritt fertig"
            zeichen = "✓ "
        elif not laufend_gesetzt:
            klasse = "schritt laufend"
            zeichen = "→ "
            laufend_gesetzt = True
        else:
            klasse = "schritt"
            zeichen = ""
        teile.append(
            f'<div class="{klasse}"><div class="schritt-kopf">{kopf}</div>'
            f'<div class="schritt-text">{titel}</div>'
            f'<div class="schritt-status">{zeichen}{status}</div></div>'
        )
    st.markdown(f'<div class="schritte">{"".join(teile)}</div>', unsafe_allow_html=True)


def render_quellen(quellen: List[dict]) -> None:
    """Zeigt die Belegstellen zur letzten Antwort."""
    if not quellen:
        return
    eigene = [q for q in quellen if q.get("herkunft") == "nutzer"]
    fachwissen = [q for q in quellen if q.get("herkunft") == "fachwissen"]

    with st.expander(f"📚 Verwendete Quellen ({len(quellen)}) – anzeigen", expanded=False):
        st.caption(
            "Die hochgestellten Ziffern in der Antwort verweisen auf diese Abschnitte. "
            "Angezeigt wird jeweils die Textstelle, die tatsächlich verwendet wurde."
        )
        for titel, gruppe in (("Aus Ihren eigenen Unterlagen", eigene),
                              ("Aus dem geprüften Fachwissen", fachwissen)):
            if not gruppe:
                continue
            st.markdown(f"**{titel}**")
            for quelle in gruppe:
                ueberschrift = f" · {quelle['ueberschrift']}" if quelle.get("ueberschrift") else ""
                st.markdown(
                    f'<div class="quelle">'
                    f'<div class="quelle-art">{quelle["art"]}{ueberschrift}</div>'
                    f'<div class="quelle-kopf"><span class="ziffer">{quelle["nummer"]}</span>'
                    f'{quelle["quelle"]}</div>'
                    f'<div class="quelle-text">{quelle["ausschnitt"]}</div></div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# STARTSEITE
# ---------------------------------------------------------------------------
def render_start_page() -> None:
    st.title("Pflegehilfe Online")
    st.markdown(
        "#### Unterstützung beim Widerspruch gegen einen Pflegegradbescheid\n"
        "Dieser Assistent prüft Ihre Pflegeunterlagen und hilft Ihnen, einen begründeten "
        "Widerspruch zu verfassen. **Eine Anmeldung ist nicht nötig.**"
    )

    st.markdown("##### So läuft es ab")
    schritte = (
        ("1", "Unterlagen hochladen",
         "Bescheid, Gutachten, Pflegetagebuch und Arztberichte als PDF."),
        ("2", "Prüfen lassen",
         "Der Assistent vergleicht das Gutachten mit Ihren Unterlagen."),
        ("3", "Widerspruch erstellen",
         "Fertiges Schreiben zum Ausdrucken und Unterschreiben."),
    )
    karten = "".join(
        f'<div class="karte"><div class="quelle-art">Schritt {nummer}</div>'
        f"<h3>{titel}</h3><p>{text}</p></div>"
        for nummer, titel, text in schritte
    )
    st.markdown(f'<div class="schrittkarten">{karten}</div>', unsafe_allow_html=True)

    st.divider()
    links, rechts = st.columns(2, gap="large")

    with links:
        st.markdown("### 🆕 Neu anfangen")
        st.markdown(
            "Sie starten mit einer leeren Sitzung. Danach erhalten Sie einen persönlichen "
            "Zugangscode, mit dem Sie später weiterarbeiten können."
        )
        if st.button("Neue Sitzung starten", type="primary", use_container_width=True):
            with st.spinner("Sitzung wird angelegt …"):
                antwort = api_create_session()
            if antwort is None:
                st.error(
                    "Der Hintergrunddienst ist nicht erreichbar. Bitte starten Sie ihn und "
                    "versuchen Sie es noch einmal."
                )
            else:
                nutzdaten = antwort.json()
                st.session_state.token = nutzdaten["token"]
                st.session_state.expires_at = nutzdaten["expires_at"]
                st.rerun()

    with rechts:
        st.markdown("### 🔑 Mit Zugangscode fortsetzen")
        st.markdown(
            "Sie haben schon einen Zugangscode? Dann geht es genau dort weiter, wo Sie "
            "aufgehört haben – mit Unterlagen, Gesprächsverlauf und Schreiben."
        )
        with st.form("token_form"):
            eingabe = st.text_input(
                "Ihr Zugangscode",
                help="Der lange Code, den Sie beim letzten Mal aufgeschrieben haben.",
            )
            if st.form_submit_button("Weiterarbeiten", use_container_width=True):
                token = eingabe.strip()
                if not token:
                    st.error("Bitte geben Sie zuerst Ihren Zugangscode ein.")
                else:
                    with st.spinner("Ihr Arbeitsstand wird geladen …"):
                        antwort = api_load_session(token)
                    if antwort is None:
                        st.error("Der Hintergrunddienst ist nicht erreichbar.")
                    elif antwort.status_code == 200:
                        nutzdaten = antwort.json()
                        st.session_state.token = nutzdaten["token"]
                        st.session_state.expires_at = nutzdaten["expires_at"]
                        apply_loaded_data(nutzdaten.get("data", {}))
                        with st.spinner("Ihre Unterlagen werden wieder durchsuchbar gemacht …"):
                            rebuild_user_index()
                        st.rerun()
                    elif antwort.status_code == 410:
                        st.error(
                            "Dieser Zugangscode ist abgelaufen. Aus Datenschutzgründen wurden "
                            "alle zugehörigen Daten bereits vollständig gelöscht."
                        )
                    else:
                        st.error("Dieser Zugangscode ist unbekannt. Bitte prüfen Sie ihn auf Tippfehler.")

    st.divider()
    st.markdown("### 🔒 Ihre Daten bleiben auf diesem Rechner")
    eins, zwei = st.columns(2, gap="large")
    with eins:
        st.markdown(
            "- Verarbeitung **ausschließlich auf diesem Rechner**\n"
            "- **Keine Weiterverarbeitung** der Daten\n"
            "- Gespeicherte Daten sind **verschlüsselt**"
        )
    with zwei:
        st.markdown(
            f"- Nach **{SESSION_DAYS // 7} Wochen** wird alles **vollständig gelöscht**\n"
            "- Verlängerung um 3 Tage jederzeit möglich\n"
            "- Sofortige Löschung auf Knopfdruck"
        )
    st.warning(
        "**Wichtiger Hinweis:** Dieser Assistent ersetzt keine Rechtsberatung. Prüfen Sie alle "
        "erstellten Texte vor dem Absenden sorgfältig selbst."
    )


# ---------------------------------------------------------------------------
# REITER 1: UNTERLAGEN
# ---------------------------------------------------------------------------
def render_upload_tab() -> None:
    st.header("Schritt 1 – Ihre Unterlagen hochladen")
    st.markdown(
        "Laden Sie alle Unterlagen hoch, die für den Widerspruch wichtig sind. "
        "Sie können **mehrere Dateien gleichzeitig** auswählen."
    )
    st.info(
        "**Besonders hilfreich:** der Pflegegradbescheid, das Gutachten des Medizinischen "
        "Dienstes, Ihr Pflegetagebuch sowie Arzt- und Krankenhausberichte.",
        icon="💡",
    )

    with st.expander("🔒 Datenschutzhinweis – bitte einmal lesen"):
        st.markdown(
            """
**Was mit Ihren Unterlagen passiert**

Ihre Dateien werden ausschließlich auf diesem Rechner gelesen und ausgewertet. Sie werden
**nicht** an ein Unternehmen im Internet übertragen und **nicht** zum Trainieren von
künstlicher Intelligenz verwendet. Auch das Sprachmodell läuft örtlich.

**Wie gespeichert wird**

Ihr Arbeitsstand bleibt nur erhalten, solange Sie Ihren Zugangscode haben. Sie finden ihn
jederzeit im Reiter **Einstellungen** und können ihn sich dort kopieren. Alles Gespeicherte
ist verschlüsselt; ohne den Code kann niemand darauf zugreifen.

**Wie gelöscht wird**

Spätestens 4 Wochen nach Beginn wird die Sitzung vollständig gelöscht – Gesprächsverlauf,
Inhalte Ihrer Unterlagen und Ihr Schreiben. Im Reiter **Einstellungen** können Sie jederzeit
sofort selbst löschen.
            """
        )

    st.markdown(f"**Zulässig:** PDF-Dateien, höchstens {MAX_FILE_SIZE_MB} MB je Datei, "
                f"bis zu {MAX_DOCUMENTS} Dokumente.")
    dateien = st.file_uploader(
        "Dateien auswählen",
        type="pdf",
        accept_multiple_files=True,
        help="Mit gedrückter Strg-Taste wählen Sie mehrere Dateien auf einmal aus.",
    )

    if dateien:
        st.success(f"**{len(dateien)} Datei(en) ausgewählt.** Klicken Sie jetzt auf „Unterlagen einlesen“.")
        if st.button("📥 Unterlagen einlesen", type="primary"):
            process_uploads(dateien)
            st.rerun()

    st.divider()
    render_document_list()


def process_uploads(dateien) -> None:
    """Liest hochgeladene PDF-Dateien ein und meldet jeden Schritt zurück."""
    vorhanden = set(st.session_state.document_names)
    if len(dateien) + len(vorhanden) > MAX_DOCUMENTS:
        st.error(
            f"Es sind höchstens {MAX_DOCUMENTS} Dokumente möglich. Bitte entfernen Sie zuerst "
            "Unterlagen, die Sie nicht mehr brauchen."
        )
        return

    schnell, mit_ocr = get_converters()
    balken = st.progress(0.0, text="Die Unterlagen werden gelesen …")
    neue: List[Document] = []
    hinweise: List[str] = []

    for nummer, datei in enumerate(dateien, start=1):
        balken.progress(
            (nummer - 1) / len(dateien),
            text=f"Datei {nummer} von {len(dateien)}: {datei.name} wird gelesen …",
        )

        if datei.name in vorhanden:
            hinweise.append(f"„{datei.name}“ war bereits eingelesen und wurde übersprungen.")
            continue
        if datei.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
            hinweise.append(f"„{datei.name}“ ist größer als {MAX_FILE_SIZE_MB} MB und wurde übersprungen.")
            continue

        try:
            dokument = pflege_rag.extract_document_from_pdf(
                datei.getvalue(), datei.name, schnell, ocr_converter=mit_ocr
            )
        except Exception:
            # Kein technischer Fehlertext, damit keine Dokumentinhalte erscheinen.
            hinweise.append(f"„{datei.name}“ konnte nicht gelesen werden. Ist die Datei beschädigt?")
            continue

        if dokument is None:
            hinweise.append(
                f"„{datei.name}“ enthält keinen lesbaren Text. Bei eingescannten Unterlagen "
                "hilft eine bessere Scanqualität."
            )
            continue
        neue.append(dokument)
        vorhanden.add(datei.name)

    if neue:
        balken.progress(0.9, text="Die Unterlagen werden durchsuchbar gemacht …")
        st.session_state.user_documents.extend(pflege_rag.split_documents(neue))
        st.session_state.document_names = collect_document_names()
        rebuild_user_index()

    balken.empty()

    if neue:
        abschnitte = sum(
            1 for d in st.session_state.user_documents
            if d.metadata.get("source") in {n.metadata["source"] for n in neue}
        )
        st.success(
            f"**{len(neue)} Dokument(e) eingelesen**, daraus {abschnitte} durchsuchbare Textabschnitte. "
            "Weiter geht es im Reiter „KI-Assistent“.",
            icon="✅",
        )
    for meldung in hinweise:
        st.warning(meldung, icon="⚠️")
    if not neue and not hinweise:
        st.warning("Es konnten keine neuen Unterlagen eingelesen werden.", icon="⚠️")


def render_document_list() -> None:
    st.subheader("Eingelesene Unterlagen")
    if not st.session_state.document_names:
        st.info(
            "Noch keine Unterlagen vorhanden. Ohne Unterlagen kann der Assistent Ihren Fall "
            "nicht prüfen.",
            icon="📄",
        )
        return

    for name in st.session_state.document_names:
        abschnitte = [d for d in st.session_state.user_documents if d.metadata.get("source") == name]
        art = abschnitte[0].metadata.get("doc_kind", "Dokument") if abschnitte else "Dokument"
        spalte_text, spalte_knopf = st.columns([5, 1])
        with spalte_text:
            st.markdown(
                f'<div class="quelle"><div class="quelle-art">{art}</div>'
                f'<div class="quelle-kopf">📄 {name}</div>'
                f'<div class="quelle-text">{len(abschnitte)} durchsuchbare Textabschnitte</div></div>',
                unsafe_allow_html=True,
            )
        with spalte_knopf:
            if st.button("Entfernen", key=f"remove_{name}", help=f"„{name}“ aus der Sitzung entfernen"):
                st.session_state.user_documents = [
                    d for d in st.session_state.user_documents if d.metadata.get("source") != name
                ]
                st.session_state.document_names = collect_document_names()
                rebuild_user_index()
                st.rerun()


# ---------------------------------------------------------------------------
# REITER 2: KI-ASSISTENT
# ---------------------------------------------------------------------------
def render_chat_tab() -> None:
    st.header("Schritt 2 – Mit dem Assistenten prüfen")
    hat_dokumente = anzahl_dokumente() > 0

    if st.session_state.pop("entwurf_fertig", False):
        st.success(
            "**Der Entwurf steht.** Übernehmen Sie ihn im Reiter **„PDF erstellen“** "
            "in Ihr Schreiben.",
            icon="✅",
        )

    if not hat_dokumente:
        st.warning(
            "**Die Aufgaben sind noch gesperrt.** Der Assistent braucht zuerst Ihre Unterlagen. "
            "Wechseln Sie dafür in den Reiter **„Unterlagen“**.",
            icon="🔒",
        )

    st.markdown("**Was möchten Sie tun?**")
    aktion = None
    spalten = st.columns(2, gap="small") + st.columns(2, gap="small")
    for spalte, schnellaktion in zip(spalten, pflege_rag.QUICK_ACTIONS):
        with spalte:
            if st.button(
                schnellaktion.titel,
                use_container_width=True,
                disabled=not hat_dokumente,
                help=schnellaktion.beschreibung if hat_dokumente
                else "Zuerst Unterlagen hochladen.",
                key=f"aktion_{schnellaktion.schluessel}",
            ):
                aktion = schnellaktion

    verlauf = st.container(height=440, border=True)
    with verlauf:
        if not st.session_state.messages:
            st.markdown(
                "**Hier erscheint Ihr Gespräch.**  \n"
                "Wählen Sie oben eine Aufgabe oder stellen Sie unten Ihre eigene Frage."
            )
        for nachricht in st.session_state.messages:
            with st.chat_message(nachricht["role"], avatar="🧑" if nachricht["role"] == "user" else "⚖️"):
                st.markdown(nachricht["content"])

    eingabe = st.chat_input(
        "Ihre Frage eingeben …" if hat_dokumente else "Zuerst Unterlagen hochladen",
        disabled=not hat_dokumente,
    )

    # Im Verlauf steht die kurze, natürliche Formulierung; an das Sprachmodell
    # geht die ausführliche Anweisung. Die kurze Fassung ist zugleich die
    # bessere Suchanfrage, weil Formatvorgaben die Bedeutung verwässern.
    if aktion is not None:
        anzeige, anweisung, zusatzfragen = aktion.nutzertext, aktion.prompt, aktion.zusatzfragen
    elif eingabe:
        anzeige = anweisung = eingabe
        zusatzfragen = ()
    else:
        anzeige = anweisung = None
        zusatzfragen = ()

    if anzeige:
        with verlauf:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(anzeige)
            st.session_state.messages.append({"role": "user", "content": anzeige})
            with st.chat_message("assistant", avatar="⚖️"):
                antwort, quellen = generate_answer(anzeige, anweisung, zusatzfragen)
        if antwort:
            st.session_state.messages.append({"role": "assistant", "content": antwort})
            st.session_state.last_sources = quellen
            if aktion is not None and aktion.schluessel == "schreiben":
                st.session_state.last_generated_letter = antwort
                st.session_state.entwurf_fertig = True
            # Neu zeichnen, damit die Fortschrittsanzeige den erreichten Schritt
            # sofort bestätigt. Sie steht oberhalb des Chats und wäre sonst noch
            # auf dem Stand von vor der Antwort.
            st.rerun()

    render_quellen(st.session_state.last_sources)

    if st.session_state.messages:
        if st.button("🗑️ Gespräch löschen", help="Löscht nur den Verlauf. Ihre Unterlagen bleiben."):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()


def generate_answer(suchfrage: str, anweisung: str, zusatzfragen=()) -> tuple[str, List[dict]]:
    """Sucht Belege, erzeugt die Antwort und versieht sie mit Hochziffern.

    ``suchfrage`` ist die kurze Formulierung für die Suche, ``anweisung`` der
    ausführliche Auftrag an das Sprachmodell.
    """
    try:
        with st.spinner("Der Assistent wird vorbereitet (nur beim ersten Mal) …"):
            reranker = get_reranker()

        # Anschlussfragen wie "Und was heißt das für mich?" enthalten für sich
        # keine suchbaren Begriffe; sie werden zuvor aufgelöst.
        eigenstaendig = pflege_rag.condense_question(
            get_llm(), suchfrage, st.session_state.messages[:-1]
        )
        if eigenstaendig != suchfrage:
            st.caption(f"🔎 Gesucht wurde nach: „{eigenstaendig}“")

        with st.spinner("Ihre Unterlagen und das Fachwissen werden durchsucht …"):
            ergebnis = pflege_rag.prepare_context(
                get_expert_index(), get_user_index(), eigenstaendig,
                reranker=reranker, extra_queries=zusatzfragen,
            )

        # Themenfremde Fragen gar nicht erst an das Sprachmodell geben. Es
        # antwortet sonst mit dem, was zufällig im Kontext steht - bei der
        # Frage nach dem Wetter kam der Inhalt des Pflegegutachtens heraus,
        # samt Belegziffern. Nur bei freien Fragen prüfen: Die vorbereiteten
        # Aufgaben sind immer im Thema, suchen aber breit über alle Module und
        # erreichen deshalb nicht immer eine hohe Einzelbewertung.
        if not zusatzfragen and ergebnis.themenfremd:
            st.markdown(pflege_rag.ABLEHNUNG_THEMENFREMD)
            return pflege_rag.ABLEHNUNG_THEMENFREMD, []

        # Die letzte Nachricht im Verlauf ist die Kurzfassung; für das
        # Sprachmodell wird sie durch die vollständige Anweisung ersetzt.
        verlauf = list(st.session_state.messages[:-1]) + [{"role": "user", "content": anweisung}]
        nachrichten = pflege_rag.build_messages(ergebnis.system_prompt, verlauf)
        platz = st.empty()
        gesammelt = ""
        for teil in pflege_rag.stream_answer(get_llm(), nachrichten):
            gesammelt += teil
            platz.markdown(gesammelt)

        # Belegnummern in Hochziffern wandeln und nur die tatsächlich
        # verwendeten Quellen anzeigen.
        angezeigt = pflege_rag.render_citations(gesammelt, ergebnis.nummern)
        verwendet = set(pflege_rag.cited_numbers(gesammelt)) & set(ergebnis.nummern)

        # Sicherheitsnetz: Belegt die Antwort nichts, stammt sie aus dem
        # allgemeinen Wissen des Sprachmodells und kann sachlich falsch sein.
        # Im Test erfand das Modell so eine falsche Zuständigkeit. Der Hinweis
        # wird Teil der Nachricht, damit er auch später im Verlauf sichtbar ist.
        if not verwendet:
            angezeigt = (
                "> ⚠️ **Diese Antwort stützt sich auf keine Textstelle** aus Ihren Unterlagen "
                "oder dem geprüften Fachwissen. Sie kann daher sachlich falsch sein – "
                "besonders bei Zuständigkeiten, Fristen und Punktzahlen. Bitte prüfen Sie sie "
                "unbedingt nach oder laden Sie passendere Unterlagen hoch.\n\n"
            ) + angezeigt

        platz.markdown(angezeigt)
        quellen = [q.als_dict() for q in ergebnis.quellen if not verwendet or q.nummer in verwendet]
        return angezeigt, quellen

    except Exception:
        st.error(
            "**Der Assistent ist gerade nicht erreichbar.**\n\n"
            "Bitte prüfen Sie, ob LM Studio läuft und das Sprachmodell geladen ist. "
            "Versuchen Sie es danach noch einmal."
        )
        return "", []


# ---------------------------------------------------------------------------
# REITER 3: PDF ERSTELLEN
# ---------------------------------------------------------------------------
def feld(label: str, key: str, hilfe: str = "", platzhalter: str = "") -> None:
    """Eingabefeld, dessen Inhalt beim Zugangscode gespeichert wird."""
    st.text_input(label, key=f"w_{key}", help=hilfe or None, placeholder=platzhalter or None)


def render_pdf_tab() -> None:
    st.header("Schritt 3 – Widerspruch als PDF erstellen")
    st.markdown(
        "Hier entsteht Ihr fertiges Widerspruchsschreiben zum Ausdrucken und Unterschreiben. "
        "Der Aufbau folgt dem Musterbrief der Verbraucherzentrale."
    )

    st.subheader("1. Ihre Angaben")
    links, rechts = st.columns(2, gap="large")
    with links:
        st.markdown("**Absender – Ihre Anschrift**")
        feld("Ihr Vor- und Nachname", "absender_name", platzhalter="Michaela Muster")
        feld("Ihre Straße und Hausnummer", "absender_strasse", platzhalter="Musterweg 1")
        feld("Ihre Postleitzahl und Ihr Ort", "absender_plz_ort", platzhalter="99999 Musterstadt",
             hilfe="Der Ort erscheint auch oben rechts in der Datumszeile.")
    with rechts:
        st.markdown("**Empfänger – Ihre Pflegekasse**")
        feld("Name der Pflegekasse", "kasse_name", platzhalter="Pflegekasse bei der Musterkrankenkasse")
        feld("Straße und Hausnummer der Pflegekasse", "kasse_strasse")
        feld("Postleitzahl und Ort der Pflegekasse", "kasse_plz_ort")

    st.markdown("**Angaben zum Bescheid**")
    eins, zwei, drei = st.columns(3)
    with eins:
        feld("Datum des Bescheids", "bescheid_datum",
             hilfe="Das Datum oben auf Ihrem Bescheid.", platzhalter="14.03.2026")
    with zwei:
        feld("Aktenzeichen", "aktenzeichen", hilfe="Steht meist oben auf dem Bescheid. Kann leer bleiben.")
    with drei:
        feld("Versichertennummer", "versichert_nr")
    feld("Name der pflegebedürftigen Person", "versichert_name",
         hilfe="Nur ausfüllen, wenn Sie den Widerspruch für eine andere Person stellen.")

    st.divider()
    st.subheader("2. Begründung des Widerspruchs")

    if st.session_state.last_generated_letter:
        if st.button("⬇️ Entwurf aus dem Gespräch übernehmen", type="secondary"):
            set_field(
                "letter_text",
                pflege_pdf.prepare_begruendung(st.session_state.last_generated_letter),
            )
            st.rerun()
    else:
        st.info(
            "Noch kein Entwurf vorhanden. Nutzen Sie im Reiter **„KI-Assistent“** die Aufgabe "
            "**„Widerspruch schreiben“**. Der Text erscheint dann hier zum Übernehmen.",
            icon="💡",
        )

    st.text_area(
        "Begründung (Sie können den Text frei bearbeiten)",
        key="w_letter_text",
        height=320,
        help="Anrede, Betreff und Grußformel ergänzt die Vorlage automatisch. "
             "Schreiben Sie hier nur die Begründung.",
    )

    luecken = pflege_pdf.find_placeholders(get_field("letter_text"))
    if luecken:
        st.warning(
            "**Im Text stehen noch Lücken.** Diese Angaben konnte der Assistent nicht aus Ihren "
            "Unterlagen entnehmen:\n\n"
            + "\n".join(f"- `{luecke}`" for luecke in luecken[:8])
            + "\n\nBitte ersetzen Sie sie, bevor Sie das Schreiben abschicken.",
            icon="✏️",
        )

    # Der Brief soll nicht verraten, wer ihn schreibt. Das Sprachmodell hält
    # sich nicht immer daran; automatisch umschreiben lässt sich das nicht
    # gefahrlos, deshalb hier ein Hinweis mit den konkreten Wörtern.
    persoenlich = pflege_pdf.find_personal_wording(get_field("letter_text"))
    if persoenlich:
        st.warning(
            "**Im Text stehen persönliche Wörter.** Das Schreiben ist bewusst neutral gehalten, "
            "damit es unverändert passt – ganz gleich, ob die pflegebedürftige Person es selbst "
            "abschickt oder jemand anderes für sie. Gefunden wurde: "
            + ", ".join(f"`{wort}`" for wort in persoenlich[:8])
            + ".\n\nBeispiel: statt „Mit Schreiben vom … haben Sie … eingestuft“ besser "
            "„Mit Bescheid vom … wurde … eingestuft“.",
            icon="✏️",
        )

    fristwahrend = st.checkbox(
        "Begründung später nachreichen (Widerspruch zunächst nur fristwahrend einlegen)",
        help="Wahrt die Frist. Im Schreiben steht dann, dass die Begründung in Kürze folgt.",
    )

    st.divider()
    st.subheader("3. PDF erstellen und herunterladen")

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
        begruendung_folgt=fristwahrend,
        # ort bleibt leer: Die Briefvorlage nimmt dann den Ort aus der
        # Absenderanschrift. Ihn zweimal einzutippen war unnötige Arbeit.
    )

    fehlend = pflege_pdf.validate(daten)
    if fehlend:
        st.warning(
            "**Es fehlen noch Angaben.** Sobald alles ausgefüllt ist, wird der Knopf frei:\n\n"
            + "\n".join(f"- {eintrag}" for eintrag in fehlend),
            icon="⚠️",
        )
    else:
        st.success("Alle Pflichtangaben liegen vor.", icon="✅")

    if st.session_state.pop("pdf_fertig", False):
        st.success("Ihr Widerspruchsschreiben ist fertig.", icon="✅")

    if st.button("📄 PDF jetzt erstellen", type="primary", disabled=bool(fehlend)):
        try:
            with st.spinner("Ihr Schreiben wird erstellt …"):
                st.session_state.pdf_bytes = pflege_pdf.build_letter_pdf(daten)
            st.session_state.pdf_fertig = True
            # Neu zeichnen, damit die Fortschrittsanzeige Schritt 3 als erledigt
            # bestätigt - sie steht oberhalb dieses Reiters.
            st.rerun()
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
            "**So geht es weiter:** Ausdrucken und **von Hand unterschreiben**. Versenden Sie das "
            "Schreiben per Post – am besten als Einwurfeinschreiben – oder per Fax. "
            "Eine E-Mail wahrt die Frist nicht.",
            icon="📮",
        )


# ---------------------------------------------------------------------------
# REITER 4: EINSTELLUNGEN
# ---------------------------------------------------------------------------
def render_settings_tab() -> None:
    st.header("Einstellungen")

    st.subheader("🔑 Ihr Zugangscode")
    st.markdown(
        "Mit diesem Code arbeiten Sie später weiter. **Schreiben Sie ihn sich auf.** "
        "Er lässt sich nicht wiederherstellen."
    )
    st.code(st.session_state.token, language=None)

    ablauf = parse_datetime(st.session_state.expires_at)
    if ablauf:
        verbleibend = ablauf - utcnow()
        tage = max(verbleibend.days, 0)
        stunden = max(verbleibend.seconds // 3600, 0) if verbleibend.days >= 0 else 0
        st.markdown(f"### ⏳ Noch **{tage} Tage und {stunden} Stunden** gültig")
        st.progress(min(max(tage / SESSION_DAYS, 0.0), 1.0))
        st.caption(
            f"Automatische, vollständige Löschung am {ablauf.strftime('%d.%m.%Y um %H:%M')} Uhr."
        )

    if st.button("➕ Um 3 Tage verlängern", type="primary"):
        antwort = api_extend_session(st.session_state.token)
        if antwort is not None and antwort.status_code == 200:
            st.session_state.expires_at = antwort.json()["expires_at"]
            st.success("Ihre Sitzung wurde um 3 Tage verlängert.", icon="✅")
            st.rerun()
        else:
            st.error("Die Verlängerung hat nicht geklappt. Bitte versuchen Sie es erneut.")

    st.divider()
    st.subheader("👁️ Darstellung")
    auswahl = st.radio(
        "Schriftgröße", list(FONT_SCALES.keys()),
        index=list(FONT_SCALES.keys()).index(st.session_state.font_scale), horizontal=True,
        help="Vergrößert die gesamte Oberfläche, nicht nur den Fließtext.",
    )
    if auswahl != st.session_state.font_scale:
        st.session_state.font_scale = auswahl
        st.rerun()

    st.divider()
    st.subheader("⚙️ Technische Angaben")
    st.caption(
        f"Neubewertung der Suchtreffer läuft auf: **{pflege_rag.reranker_backend()}**. "
        "Sprachmodell und Einbettungen laufen über LM Studio. Alle Verarbeitung findet "
        "ausschließlich auf diesem Rechner statt."
    )

    st.divider()
    st.subheader("🚪 Sitzung verlassen")
    st.markdown(
        "Der Bildschirm wird geleert, Ihre Daten bleiben gespeichert. Mit Ihrem Zugangscode "
        "kommen Sie jederzeit zurück."
    )
    if st.button("Abmelden"):
        reset_local_state()
        st.rerun()

    st.divider()
    st.subheader("🗑️ Alles löschen und beenden")
    st.markdown(
        "Löscht **sofort und unwiderruflich**: Zugangscode, Inhalte Ihrer Unterlagen, "
        "Gesprächsverlauf und Schreiben. Ein Wiedereinstieg ist danach nicht mehr möglich."
    )
    bestaetigt = st.checkbox("Ja, ich möchte wirklich alles endgültig löschen.")
    if st.button("Jetzt alles löschen und beenden", disabled=not bestaetigt):
        with st.spinner("Alle Daten werden gelöscht …"):
            erfolgreich = api_delete_session(st.session_state.token)
        reset_local_state()
        st.session_state.deletion_done = erfolgreich
        st.rerun()


# ---------------------------------------------------------------------------
# HAUPTANSICHT
# ---------------------------------------------------------------------------
def render_app() -> None:
    render_kopfzeile()

    # Bei einer frischen Installation fehlt die Wissensdatenbank. Ohne diesen
    # Hinweis antwortet der Assistent scheinbar normal, aber ohne Fachwissen.
    index = get_expert_index()
    if index is None or not index.documents:
        st.error(
            "**Die Wissensdatenbank fehlt oder ist leer.** Der Assistent kann sich dann nur auf "
            "Ihre eigenen Unterlagen stützen, nicht auf das geprüfte Fachwissen.\n\n"
            "Bitte einmalig im Projektordner ausführen (die Weboberfläche muss dafür beendet sein):\n"
            "```\npython ingest.py\n```",
            icon="🗄️",
        )

    render_fortschritt()

    reiter = st.tabs([
        "📁  Unterlagen",
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

    st.divider()
    st.caption(
        "Dieser Assistent ersetzt keine Rechtsberatung. Alle erstellten Texte müssen vor dem "
        "Absenden geprüft werden. Die Verarbeitung findet ausschließlich örtlich statt."
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
