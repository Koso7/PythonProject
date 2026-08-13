"""Erzeugung des Widerspruchsschreibens als PDF.

Der Aufbau folgt dem Musterbrief "Widerspruch gegen Bescheid der Pflegekasse"
der Verbraucherzentrale, gesetzt im Format eines Geschäftsbriefs nach DIN 5008
(Form B):

    Falzmarken und Lochmarke am linken Rand
    Rücksendeangabe über dem Anschriftfeld
    Anschriftfeld ab 45 mm von der Blattoberkante
    Informationsblock rechts mit Ort und Datum
    Betreffzeile ab 98,4 mm, fett
    Anrede, Fließtext, Grußformel, Unterschriftsfeld

Damit sieht das Schreiben aus wie ein Brief, den eine Behörde erwartet - und
nicht wie ein maschinell zusammengesetztes Dokument.

Das Modul kennt weder Dienst noch Oberfläche und ist einzeln testbar.
"""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass, field

from fpdf import FPDF

# ---------------------------------------------------------------------------
# MASSE NACH DIN 5008 (FORM B), IN MILLIMETERN
# ---------------------------------------------------------------------------
RAND_LINKS = 25.0
RAND_RECHTS = 20.0
RAND_UNTEN = 25.0

RUECKSENDEANGABE_OBEN = 45.0      # Oberkante des Anschriftfeldes
ANSCHRIFT_OBEN = 51.0             # Empfängeranschrift beginnt darunter
INFOBLOCK_LINKS = 125.0           # Ort und Datum rechts
BETREFF_OBEN = 98.4               # genormte Position der Betreffzeile
FALZMARKE_OBEN = 87.0
FALZMARKE_UNTEN = 192.0
LOCHMARKE = 148.5

ZEILENHOEHE = 5.0
SCHRIFTGROESSE = 11
SEITENBREITE = 210.0

# Schriften in der Reihenfolge der Bevorzugung. Die eingebauten PDF-Schriften
# wirken altmodisch; eine echte Schrift macht den Unterschied zwischen
# "maschinell erzeugt" und "ordentlicher Geschäftsbrief".
SCHRIFT_KANDIDATEN = [
    ("Carlito", "C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/calibrib.ttf",
     "C:/Windows/Fonts/calibrii.ttf"),
    ("Arial", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf",
     "C:/Windows/Fonts/ariali.ttf"),
    ("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
    ("Liberation", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"),
]

# Zeichen, die die eingebaute Ersatzschrift (Latin-1) nicht darstellen kann.
CHAR_REPLACEMENTS = {
    "–": "-", "—": "-", "‑": "-", "−": "-",
    "„": '"', "“": '"', "”": '"', "»": '"', "«": '"',
    "‘": "'", "’": "'", "‚": "'", "›": "'", "‹": "'",
    "•": "-", "▪": "-", "●": "-", "·": "-",
    "…": "...", "€": "Euro",
    " ": " ", " ": " ", " ": " ", "​": "",
    "≥": ">=", "≤": "<=", "→": "->", "×": "x",
    "№": "Nr.", "™": "", "®": "", "©": "",
}

# ---------------------------------------------------------------------------
# TEXTAUFBEREITUNG
# ---------------------------------------------------------------------------
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)

_SALUTATION = re.compile(r"^\s*sehr\s+geehrte[^\n]*\n+", re.IGNORECASE)
_GRUSS = r"gr[üu]e?(?:ß|ss?)en?"
_CLOSING = re.compile(
    rf"\n+\s*(?:mit\s+freundlichen\s+{_GRUSS}|mit\s+besten\s+{_GRUSS}|"
    rf"freundliche\s+{_GRUSS}|herzliche\s+{_GRUSS}).*$",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_HEADING = re.compile(r"^\s*begr[üu]e?ndung\b[^\n]{0,90}?:?\s*\n+", re.IGNORECASE)
_LEADING_FORMAL = re.compile(
    r"^\s*hiermit\s+lege\s+ich[^\n]*?(form-\s*und\s+fristgerecht)?\s*"
    r"(widerspruch)?\s*(ein\.?)?\s*\n+",
    re.IGNORECASE,
)

_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_CITATION_MARKS = re.compile(rf"\s*[{_SUPERSCRIPTS}]+")

_PLACEHOLDER_PATTERNS = [
    re.compile(r"\[[^\]\n]{2,60}\]"),
    re.compile(r"\((?:bitte\s+)?[^)\n]{0,40}(?:einsetzen|eintragen|ergänzen)[^)\n]{0,20}\)", re.IGNORECASE),
    re.compile(r"\bTT\.MM\.JJJJ\b"),
    re.compile(r"\bXXX+\b"),
    re.compile(r"_{3,}"),
]


def sanitize(text: str) -> str:
    """Ersetzt Zeichen, die die Ersatzschrift nicht darstellen kann."""
    if not text:
        return ""
    for alt, neu in CHAR_REPLACEMENTS.items():
        text = text.replace(alt, neu)
    return text


def strip_markdown(text: str) -> str:
    """Entfernt Markdown-Auszeichnungen aus einem erzeugten Text."""
    if not text:
        return ""
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_BULLET.sub("", text)
    return text


def strip_citations(text: str) -> str:
    """Entfernt Belegziffern; in einem Behördenschreiben haben sie nichts zu suchen."""
    if not text:
        return ""
    text = _CITATION_MARKS.sub("", text)
    text = re.sub(r"\s*\[\d{1,2}\]", "", text)
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def strip_letter_boilerplate(text: str) -> str:
    """Entfernt Anrede, Einleitungsformel und Grußformel.

    Die Briefvorlage steuert diese Bestandteile selbst bei; ohne die
    Bereinigung stünden sie doppelt im Schreiben.
    """
    if not text:
        return ""
    text = text.strip()
    text = _SALUTATION.sub("", text)
    text = _LEADING_FORMAL.sub("", text)
    text = _LEADING_HEADING.sub("", text)
    text = _CLOSING.sub("", text)
    return text.strip()


# Das Sprachmodell übernimmt gelegentlich die technischen Trennzeilen des
# Suchkontexts in seine Antwort. Im Brief an die Pflegekasse wäre das ein
# grober Fehler, deshalb wird hier zusätzlich abgesichert - unabhängig davon,
# was die Anzeige im Chat bereits bereinigt hat.
_CONTEXT_HEADER_RE = re.compile(
    r"-{3,}\s*(?:\[?\d{1,2}\]?\s*-{3,}\s*)?Herkunft:[^\n]*", re.IGNORECASE
)
_LEFTOVER_SEPARATOR_RE = re.compile(r"^\s*-{3,}\s*\[?\d{1,2}\]?\s*-{3,}\s*$", re.MULTILINE)

# Gliederungsmarken, die das Sprachmodell aus der Aufgabenbeschreibung
# übernimmt ("[Einleitung]", "[Hauptteil]"). Sie stehen allein auf einer Zeile
# und gehören nicht in einen Brief.
_ABSCHNITTSMARKE_RE = re.compile(r"^\s*\[[^\]\n]{2,40}\]\s*:?\s*$", re.MULTILINE)


def strip_context_headers(text: str) -> str:
    """Entfernt übernommene Trennzeilen und Gliederungsmarken."""
    if not text:
        return ""
    text = _CONTEXT_HEADER_RE.sub("", text)
    text = _LEFTOVER_SEPARATOR_RE.sub("", text)
    text = _ABSCHNITTSMARKE_RE.sub("", text)
    return text.strip()


def prepare_begruendung(text: str) -> str:
    """Bereitet einen Chat-Text als Begründung für den Brief auf.

    Die Trennzeilen werden zuerst entfernt: Stünde davor noch eine, läge die
    Anrede nicht mehr am Textanfang und bliebe unerkannt stehen.
    """
    ohne_technik = strip_context_headers(text or "")
    fertig = strip_letter_boilerplate(strip_citations(strip_markdown(ohne_technik))).strip()
    # Nach dem Entfernen der Anrede beginnt der Text oft klein ("mit Schreiben
    # vom ..."), weil er als Fortsetzung der Anrede gedacht war.
    return fertig[:1].upper() + fertig[1:] if fertig else ""


def find_placeholders(text: str) -> list[str]:
    """Findet noch offene Platzhalter im Begründungstext."""
    if not text:
        return []
    gefunden: list[str] = []
    for muster in _PLACEHOLDER_PATTERNS:
        for treffer in muster.findall(text):
            wert = treffer.strip()
            if wert and wert not in gefunden:
                gefunden.append(wert)
    return gefunden


# ---------------------------------------------------------------------------
# BRIEFDATEN
# ---------------------------------------------------------------------------
SELBST = "selbst"
ANGEHOERIGE = "angehoerige"


@dataclass
class LetterData:
    """Alle Angaben, die im Widerspruchsschreiben erscheinen."""

    absender_name: str = ""
    absender_strasse: str = ""
    absender_plz_ort: str = ""
    kasse_name: str = ""
    kasse_strasse: str = ""
    kasse_plz_ort: str = ""
    versichert_name: str = ""
    versichert_nr: str = ""
    aktenzeichen: str = ""
    bescheid_datum: str = ""
    begruendung: str = ""
    begruendung_folgt: bool = False
    ort: str = ""
    datum: str = field(default_factory=lambda: datetime.date.today().strftime("%d.%m.%Y"))
    # Wer den Widerspruch einlegt: die betroffene Person selbst oder eine
    # angehörige Person für sie.
    perspektive: str = SELBST
    verhaeltnis: str = ""
    # Beigefügte Unterlagen, eine je Zeile. Bleibt das Feld leer, entfällt der
    # Anlagenblock ganz.
    anlagen: str = ""

    @property
    def anlagenliste(self) -> List[str]:
        """Die Anlagen als bereinigte Liste, ohne Aufzählungszeichen."""
        zeilen = []
        for zeile in (self.anlagen or "").splitlines():
            sauber = zeile.strip().lstrip("-•*·–— ").strip()
            if sauber:
                zeilen.append(sauber)
        return zeilen

    @property
    def schreibt_selbst(self) -> bool:
        return self.perspektive != ANGEHOERIGE

    @property
    def name_versicherte(self) -> str:
        """Name der versicherten Person - bei Eigenantrag der Absender."""
        return (self.versichert_name or self.absender_name).strip()

    @property
    def einleitungssatz(self) -> str:
        """Der Satz, mit dem der Widerspruch eingelegt wird."""
        bezug = f"gegen den Bescheid vom {self.bescheid_datum}"
        if self.aktenzeichen.strip():
            bezug += f" mit dem Aktenzeichen {self.aktenzeichen.strip()}"
        # Das hervorgehobene Wort steht mitten im Satz. Früher stand es
        # zentriert auf einer eigenen Zeile, wodurch das "ein." allein in der
        # nächsten Zeile hing - das sah wie ein Satzfehler aus.
        if self.schreibt_selbst:
            return f"hiermit lege ich {bezug} form- und fristgerecht **Widerspruch** ein."
        name = self.name_versicherte or "die versicherte Person"
        verhaeltnis = self.verhaeltnis.strip()
        wer = f"meine {verhaeltnis}, {name}," if verhaeltnis else f"{name},"
        return f"hiermit lege ich für {wer} {bezug} form- und fristgerecht **Widerspruch** ein."

    @property
    def unterschrift_hinweis(self) -> str:
        """Hinweis unter der Unterschriftszeile.

        ``verhaeltnis`` beschreibt, wie die betroffene Person zur schreibenden
        steht ("Mutter"). Unter der Unterschrift stünde damit fälschlich die
        Rolle der betroffenen statt der unterschreibenden Person, deshalb hier
        eine neutrale Angabe.
        """
        if self.schreibt_selbst:
            return "(Unterschrift der pflegebedürftigen Person)"
        return "(Unterschrift der bevollmächtigten Person)"


def validate(data: LetterData) -> list[str]:
    """Prüft die Pflichtangaben und liefert verständliche Fehlermeldungen."""
    fehlt = []
    if not data.absender_name.strip():
        fehlt.append("Ihr Vor- und Nachname")
    if not data.absender_strasse.strip():
        fehlt.append("Ihre Straße und Hausnummer")
    if not data.absender_plz_ort.strip():
        fehlt.append("Ihre Postleitzahl und Ihr Ort")
    if not data.kasse_name.strip():
        fehlt.append("Der Name der Pflegekasse")
    if not data.bescheid_datum.strip() or data.bescheid_datum.strip() == "TT.MM.JJJJ":
        fehlt.append("Das Datum des Bescheids")
    if not data.begruendung.strip() and not data.begruendung_folgt:
        fehlt.append("Die Begründung des Widerspruchs")
    if not data.schreibt_selbst and not data.versichert_name.strip():
        fehlt.append("Der Name der pflegebedürftigen Person")
    return fehlt


# ---------------------------------------------------------------------------
# PDF-ERZEUGUNG
# ---------------------------------------------------------------------------
class _Geschaeftsbrief(FPDF):
    """Geschäftsbrief mit Falzmarken, Lochmarke und Fußzeile."""

    def __init__(self, schriftname: str):
        super().__init__(format="A4", unit="mm")
        self.schriftname = schriftname

    def header(self) -> None:
        """Falz- und Lochmarken am linken Blattrand."""
        self.set_draw_color(150, 150, 150)
        self.set_line_width(0.2)
        for hoehe in (FALZMARKE_OBEN, FALZMARKE_UNTEN):
            self.line(5, hoehe, 9, hoehe)
        # Lochmarke etwas länger, damit sie unterscheidbar bleibt.
        self.line(5, LOCHMARKE, 11, LOCHMARKE)
        self.set_draw_color(0, 0, 0)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font(self.schriftname, size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 4, f"Seite {self.page_no()} von {{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def _lade_schrift(pdf: FPDF) -> str:
    """Bindet die erste verfügbare Schrift ein, sonst die eingebaute Ersatzschrift."""
    for name, regular, fett, kursiv in SCHRIFT_KANDIDATEN:
        if not os.path.exists(regular):
            continue
        try:
            pdf.add_font(name, "", regular)
            if os.path.exists(fett):
                pdf.add_font(name, "B", fett)
            if os.path.exists(kursiv):
                pdf.add_font(name, "I", kursiv)
            return name
        except Exception:
            continue
    return "Helvetica"


def _text(pdf: FPDF, schrift: str, inhalt: str, groesse: int = SCHRIFTGROESSE,
          stil: str = "", hoehe: float = ZEILENHOEHE, align: str = "L") -> None:
    """Schreibt eine Zeile. Bei der Ersatzschrift werden Sonderzeichen ersetzt."""
    pdf.set_font(schrift, style=stil, size=groesse)
    pdf.cell(0, hoehe, text=inhalt if schrift != "Helvetica" else sanitize(inhalt),
             new_x="LMARGIN", new_y="NEXT", align=align)


def _absatz(pdf: FPDF, schrift: str, inhalt: str, groesse: int = SCHRIFTGROESSE,
            stil: str = "", hoehe: float = ZEILENHOEHE, markdown: bool = False) -> None:
    """Schreibt einen Fließtextabsatz.

    Mit ``markdown`` werden ``**Sternchen**`` als Fettdruck gesetzt. Das wird
    nur für die Einleitung genutzt; der vom Modell erzeugte Text durchläuft
    vorher ``strip_markdown`` und enthält keine Auszeichnungen mehr.
    """
    pdf.set_font(schrift, style=stil, size=groesse)
    pdf.multi_cell(0, hoehe, text=inhalt if schrift != "Helvetica" else sanitize(inhalt),
                   new_x="LMARGIN", new_y="NEXT", markdown=markdown)


def build_letter_pdf(data: LetterData) -> bytes:
    """Erzeugt das vollständige Widerspruchsschreiben als PDF."""
    pdf = _Geschaeftsbrief("Helvetica")
    pdf.set_margins(RAND_LINKS, 20, RAND_RECHTS)
    pdf.set_auto_page_break(auto=True, margin=RAND_UNTEN)
    pdf.alias_nb_pages()
    pdf.set_title("Widerspruch gegen den Bescheid der Pflegekasse")
    # Keine Angaben zu Erzeuger oder Person in den Metadaten.
    pdf.set_author("")
    pdf.set_creator("")
    pdf.set_producer("")

    schrift = _lade_schrift(pdf)
    pdf.schriftname = schrift
    pdf.add_page()

    # --- Rücksendeangabe über dem Anschriftfeld ---------------------------
    pdf.set_y(RUECKSENDEANGABE_OBEN)
    ruecksende = " · ".join(
        teil for teil in (data.absender_name, data.absender_strasse, data.absender_plz_ort)
        if teil.strip()
    )
    pdf.set_font(schrift, size=7)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 3.5, text=ruecksende if schrift != "Helvetica" else sanitize(ruecksende),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # --- Anschriftfeld ----------------------------------------------------
    pdf.set_y(ANSCHRIFT_OBEN)
    _text(pdf, schrift, data.kasse_name)
    if data.kasse_strasse.strip():
        _text(pdf, schrift, data.kasse_strasse)
    if data.kasse_plz_ort.strip():
        _text(pdf, schrift, data.kasse_plz_ort)

    # --- Informationsblock rechts: Ort und Datum --------------------------
    pdf.set_y(ANSCHRIFT_OBEN)
    pdf.set_x(INFOBLOCK_LINKS)
    ort_datum = f"{data.ort.strip()}, den {data.datum}" if data.ort.strip() else data.datum
    pdf.set_font(schrift, size=SCHRIFTGROESSE)
    pdf.cell(SEITENBREITE - INFOBLOCK_LINKS - RAND_RECHTS, ZEILENHOEHE,
             text=ort_datum if schrift != "Helvetica" else sanitize(ort_datum),
             new_x="LMARGIN", new_y="NEXT", align="R")

    # --- Betreff ----------------------------------------------------------
    pdf.set_y(BETREFF_OBEN)
    betreff = f"Widerspruch gegen den Bescheid vom {data.bescheid_datum}"
    if data.aktenzeichen.strip():
        betreff += f", Aktenzeichen {data.aktenzeichen.strip()}"
    _absatz(pdf, schrift, betreff, stil="B")

    pdf.ln(1.5)
    angaben = []
    if data.name_versicherte:
        angaben.append(f"Versicherte Person: {data.name_versicherte}")
    if data.versichert_nr.strip():
        angaben.append(f"Versichertennummer: {data.versichert_nr.strip()}")
    for zeile in angaben:
        _text(pdf, schrift, zeile, groesse=10)

    # --- Anrede -----------------------------------------------------------
    pdf.ln(6)
    _text(pdf, schrift, "Sehr geehrte Damen und Herren,")
    pdf.ln(3)

    # --- Einleitung mit hervorgehobenem "Widerspruch" ---------------------
    _absatz(pdf, schrift, data.einleitungssatz, markdown=True)
    pdf.ln(4)

    # --- Begründung -------------------------------------------------------
    if data.begruendung_folgt or not data.begruendung.strip():
        _absatz(pdf, schrift,
                "Eine ausführliche Begründung des Widerspruchs reiche ich in Kürze nach.")
    else:
        _text(pdf, schrift, "Begründung", stil="B")
        pdf.ln(2)
        for absatz in prepare_begruendung(data.begruendung).split("\n"):
            if absatz.strip():
                _absatz(pdf, schrift, absatz.strip())
                pdf.ln(2)

    pdf.ln(2)
    # Nicht "Ich bitte Sie ...": die Begründung endet häufig selbst mit einer
    # Bitte, zwei gleich beginnende Sätze hintereinander lesen sich holprig.
    _absatz(pdf, schrift, "Bitte bestätigen Sie mir den Eingang dieses Widerspruchs.")

    # --- Grußformel und Unterschrift --------------------------------------
    pdf.ln(8)
    _text(pdf, schrift, "Mit freundlichen Grüßen")
    pdf.ln(16)
    pdf.set_draw_color(120, 120, 120)
    pdf.line(RAND_LINKS, pdf.get_y(), RAND_LINKS + 65, pdf.get_y())
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(1)
    _text(pdf, schrift, data.absender_name)
    pdf.set_text_color(110, 110, 110)
    _text(pdf, schrift, data.unterschrift_hinweis, groesse=8, hoehe=4)
    pdf.set_text_color(0, 0, 0)

    # --- Anlagen ----------------------------------------------------------
    # Steht nach DIN 5008 unter der Unterschrift. Ohne Eintrag entfällt der
    # Block, damit kein leerer Abschnitt im Brief steht.
    anlagen = data.anlagenliste
    if anlagen:
        pdf.ln(8)
        _text(pdf, schrift, "Anlagen", stil="B")
        pdf.ln(1)
        for eintrag in anlagen:
            # Ohne Aufzählungszeichen: so steht es in Geschäftsbriefen.
            _absatz(pdf, schrift, eintrag)

    return bytes(pdf.output())
