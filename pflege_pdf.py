"""Erzeugung des Widerspruchsschreibens als PDF.

Der Aufbau folgt dem Musterbrief "Widerspruch gegen Bescheid der Pflegekasse"
der Verbraucherzentrale (siehe daten/Musterbrief_Widerspruch_Bescheid_der_Pflegekasse.pdf):

    Absender
    Anschrift der Pflegekasse
    Datum (rechtsbündig)
    Betreff: Widerspruch gegen den Bescheid vom ... mit dem Aktenzeichen ...
    Versicherungsnehmer / Versichertennummer
    Anrede
    "... lege ich ... form- und fristgerecht"  ->  Widerspruch  ->  "ein."
    Begründung
    Bitte um Eingangsbestätigung
    Grußformel und Unterschriftszeile

Das Modul ist bewusst frei von Streamlit-Abhängigkeiten, damit es einzeln
getestet werden kann.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

from fpdf import FPDF

# Seitenränder in Millimetern, angelehnt an DIN 5008 für Geschäftsbriefe.
MARGIN_LEFT = 25
MARGIN_RIGHT = 20
MARGIN_TOP = 20
BOTTOM_MARGIN = 20

FONT = "Helvetica"
FONT_SIZE = 11
LINE_HEIGHT = 5.6

# Zeichen, die die Standardschriften von FPDF (Latin-1) nicht darstellen können.
# Ohne diese Ersetzungen bricht die PDF-Erzeugung bei Texten ab, die ein
# Sprachmodell typischerweise erzeugt (Gedankenstriche, typografische
# Anführungszeichen, Aufzählungspunkte).
CHAR_REPLACEMENTS = {
    "–": "-", "—": "-", "‑": "-", "−": "-",
    "„": '"', "“": '"', "”": '"', "»": '"', "«": '"',
    "‘": "'", "’": "'", "‚": "'", "›": "'", "‹": "'",
    "•": "-", "▪": "-", "●": "-", "·": "-",
    "…": "...",
    "€": "Euro",
    " ": " ", " ": " ", " ": " ", "​": "",
    "≥": ">=", "≤": "<=", "→": "->", "×": "x",
    "№": "Nr.", "™": "", "®": "", "©": "",
}

# Markdown-Reste, die ein Sprachmodell gerne erzeugt, im Brief aber nichts
# zu suchen haben.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_MD_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", re.DOTALL)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s{0,3}[-*+]\s+", re.MULTILINE)

# Anrede und Grußformel liefert die Vorlage selbst. Erzeugt das Sprachmodell
# sie trotzdem, würden sie doppelt im Brief stehen.
_SALUTATION = re.compile(r"^\s*sehr\s+geehrte[^\n]*\n+", re.IGNORECASE)
# Deckt die gängigen Schreibweisen ab: Grüßen, Grüssen, Gruessen, Grussen.
_GRUSS = r"gr[üu]e?(?:ß|ss?)en?"
_CLOSING = re.compile(
    rf"\n+\s*(?:mit\s+freundlichen\s+{_GRUSS}|mit\s+besten\s+{_GRUSS}|"
    rf"freundliche\s+{_GRUSS}|herzliche\s+{_GRUSS}).*$",
    re.IGNORECASE | re.DOTALL,
)
# Eine vom Sprachmodell vorangestellte Überschrift wie "Begründung" oder
# "Begründung für den Widerspruch gegen die Einstufung:" würde die Überschrift
# der Briefvorlage doppeln.
_LEADING_HEADING = re.compile(
    r"^\s*begr[üu]e?ndung\b[^\n]{0,90}?:?\s*\n+", re.IGNORECASE
)
# Ein vom Modell mitgeliefertes "Widerspruch ein."-Vorspann-Konstrukt.
_LEADING_FORMAL = re.compile(
    r"^\s*hiermit\s+lege\s+ich[^\n]*?(form-\s*und\s+fristgerecht)?\s*"
    r"(widerspruch)?\s*(ein\.?)?\s*\n+",
    re.IGNORECASE,
)


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


def sanitize(text: str) -> str:
    """Ersetzt Zeichen, die die PDF-Standardschrift nicht darstellen kann."""
    if not text:
        return ""
    for old, new in CHAR_REPLACEMENTS.items():
        text = text.replace(old, new)
    # Letzte Absicherung: alles, was Latin-1 nicht kennt, wird ersetzt statt
    # einen Abbruch der PDF-Erzeugung auszulösen.
    return text.encode("latin-1", "replace").decode("latin-1")


def strip_markdown(text: str) -> str:
    """Entfernt Markdown-Auszeichnungen aus einem vom Sprachmodell erzeugten Text."""
    if not text:
        return ""
    text = _MD_HEADING.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_ITALIC.sub(r"\1", text)
    text = _MD_BULLET.sub("- ", text)
    return text


def strip_letter_boilerplate(text: str) -> str:
    """Entfernt Anrede, Einleitungsformel und Grußformel aus einem Text.

    Die Briefvorlage steuert diese Bestandteile selbst bei. Ohne diese
    Bereinigung stünden sie doppelt im fertigen Schreiben, sobald das
    Sprachmodell einen vollständigen Brief statt nur der Begründung liefert.
    """
    if not text:
        return ""
    text = text.strip()
    text = _SALUTATION.sub("", text)
    text = _LEADING_FORMAL.sub("", text)
    text = _LEADING_HEADING.sub("", text)
    text = _CLOSING.sub("", text)
    return text.strip()


# Belegziffern der Chatantwort. In einem Schreiben an die Pflegekasse haben sie
# nichts zu suchen - dort gibt es keine Quellenliste, auf die sie verweisen.
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_CITATION_MARKS = re.compile(rf"\s*[{_SUPERSCRIPTS}]+")


def strip_citations(text: str) -> str:
    """Entfernt Belegziffern (hochgestellt und in eckigen Klammern)."""
    if not text:
        return ""
    text = _CITATION_MARKS.sub("", text)
    text = re.sub(r"\s*\[\d{1,2}\]", "", text)
    # Vor Satzzeichen darf kein Leerzeichen zurückbleiben.
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def prepare_begruendung(text: str) -> str:
    """Bereitet einen Chat-Text als Begründung für den Brief auf."""
    return strip_letter_boilerplate(strip_citations(strip_markdown(text or ""))).strip()


# Lückenhafte Stellen, die ein Sprachmodell erzeugt, wenn ihm eine Angabe fehlt.
# Ein Brief mit solchen Platzhaltern darf nicht ungeprüft an die Pflegekasse gehen.
_PLACEHOLDER_PATTERNS = [
    re.compile(r"\[[^\]\n]{2,60}\]"),
    re.compile(r"\((?:bitte\s+)?[^)\n]{0,40}(?:einsetzen|eintragen|ergänzen)[^)\n]{0,20}\)", re.IGNORECASE),
    re.compile(r"\bTT\.MM\.JJJJ\b"),
    re.compile(r"\bXXX+\b"),
    re.compile(r"_{3,}"),
]


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


class _LetterPDF(FPDF):
    """FPDF mit Fußzeile inklusive Seitenzahl."""

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font(FONT, size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, f"Seite {self.page_no()} von {{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def _line(pdf: FPDF, text: str = "", height: float = LINE_HEIGHT, **kwargs) -> None:
    pdf.cell(0, height, text=sanitize(text), new_x="LMARGIN", new_y="NEXT", **kwargs)


def validate(data: LetterData) -> list[str]:
    """Prüft die Pflichtangaben und liefert verständliche Fehlermeldungen."""
    missing = []
    if not data.absender_name.strip():
        missing.append("Ihr Vor- und Nachname")
    if not data.absender_strasse.strip():
        missing.append("Ihre Straße und Hausnummer")
    if not data.absender_plz_ort.strip():
        missing.append("Ihre Postleitzahl und Ihr Ort")
    if not data.kasse_name.strip():
        missing.append("Der Name der Pflegekasse")
    if not data.bescheid_datum.strip() or data.bescheid_datum.strip() == "TT.MM.JJJJ":
        missing.append("Das Datum des Bescheids")
    if not data.begruendung.strip() and not data.begruendung_folgt:
        missing.append("Die Begründung des Widerspruchs")
    return missing


def build_letter_pdf(data: LetterData) -> bytes:
    """Erzeugt das vollständige Widerspruchsschreiben als PDF."""
    pdf = _LetterPDF(format="A4", unit="mm")
    pdf.set_margins(MARGIN_LEFT, MARGIN_TOP, MARGIN_RIGHT)
    pdf.set_auto_page_break(auto=True, margin=BOTTOM_MARGIN)
    pdf.alias_nb_pages()
    pdf.set_title("Widerspruch gegen den Bescheid der Pflegekasse")
    # Keine Autoren-/Erzeugerangaben: Die PDF-Metadaten sollen keine Hinweise
    # auf die verwendete Software oder die Person enthalten.
    pdf.set_author("")
    pdf.set_creator("")
    pdf.add_page()
    pdf.set_font(FONT, size=FONT_SIZE)

    # --- Absender -------------------------------------------------------
    _line(pdf, data.absender_name)
    if data.absender_strasse.strip():
        _line(pdf, data.absender_strasse)
    if data.absender_plz_ort.strip():
        _line(pdf, data.absender_plz_ort)

    pdf.ln(12)

    # --- Empfänger ------------------------------------------------------
    _line(pdf, "An")
    _line(pdf, data.kasse_name)
    if data.kasse_strasse.strip():
        _line(pdf, data.kasse_strasse)
    if data.kasse_plz_ort.strip():
        _line(pdf, data.kasse_plz_ort)

    pdf.ln(10)

    # --- Datum, rechtsbündig -------------------------------------------
    ort_datum = f"{data.ort.strip()}, {data.datum}" if data.ort.strip() else data.datum
    _line(pdf, ort_datum, align="R")

    pdf.ln(8)

    # --- Betreff --------------------------------------------------------
    pdf.set_font(FONT, style="B", size=FONT_SIZE)
    betreff = f"Widerspruch gegen den Bescheid vom {data.bescheid_datum}"
    if data.aktenzeichen.strip():
        betreff += f" mit dem Aktenzeichen {data.aktenzeichen.strip()}"
    pdf.multi_cell(0, LINE_HEIGHT, text=sanitize(betreff), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(FONT, size=FONT_SIZE)

    pdf.ln(4)

    # --- Angaben zur versicherten Person --------------------------------
    if data.versichert_name.strip():
        _line(pdf, f"Versicherungsnehmer: {data.versichert_name.strip()}")
    if data.versichert_nr.strip():
        _line(pdf, f"Versichertennummer: {data.versichert_nr.strip()}")

    pdf.ln(8)

    # --- Anrede ---------------------------------------------------------
    _line(pdf, "Sehr geehrte Damen und Herren,")
    pdf.ln(4)

    # --- Einleitung -----------------------------------------------------
    einleitung = f"hiermit lege ich gegen den Bescheid vom {data.bescheid_datum}"
    if data.aktenzeichen.strip():
        einleitung += f" mit dem Aktenzeichen {data.aktenzeichen.strip()}"
    einleitung += " form- und fristgerecht"
    pdf.multi_cell(0, LINE_HEIGHT, text=sanitize(einleitung), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)
    pdf.set_font(FONT, style="B", size=13)
    _line(pdf, "Widerspruch", height=7, align="C")
    pdf.set_font(FONT, size=FONT_SIZE)
    pdf.ln(3)

    _line(pdf, "ein.")
    pdf.ln(5)

    # --- Begründung -----------------------------------------------------
    if data.begruendung_folgt or not data.begruendung.strip():
        pdf.multi_cell(
            0, LINE_HEIGHT,
            text=sanitize("Eine Begründung des Widerspruchs geht Ihnen in Kürze zu."),
            new_x="LMARGIN", new_y="NEXT",
        )
    else:
        pdf.set_font(FONT, style="B", size=FONT_SIZE)
        _line(pdf, "Begründung")
        pdf.set_font(FONT, size=FONT_SIZE)
        pdf.ln(2)
        for absatz in prepare_begruendung(data.begruendung).split("\n"):
            if absatz.strip():
                pdf.multi_cell(
                    0, LINE_HEIGHT, text=sanitize(absatz.strip()),
                    new_x="LMARGIN", new_y="NEXT",
                )
            else:
                pdf.ln(LINE_HEIGHT / 2)

    pdf.ln(5)
    pdf.multi_cell(
        0, LINE_HEIGHT,
        text=sanitize("Ich bitte, mir den Eingang des Widerspruchs zu bestätigen."),
        new_x="LMARGIN", new_y="NEXT",
    )

    # --- Grußformel und Unterschrift ------------------------------------
    pdf.ln(9)
    _line(pdf, "Mit freundlichen Grüßen")
    pdf.ln(18)
    _line(pdf, "______________________________________")
    _line(pdf, data.absender_name)
    pdf.set_font(FONT, size=8)
    pdf.set_text_color(110, 110, 110)
    _line(pdf, "(Unterschrift der pflegebedürftigen Person bzw. der bevollmächtigten Person)", height=4)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(FONT, size=FONT_SIZE)

    return bytes(pdf.output())
