"""Tests für die Erzeugung des Widerspruchsschreibens.

Prüft die Textaufbereitung (die entscheidet, was im Brief an die Pflegekasse
landet) und den Aufbau des PDF nach dem Musterbrief.
"""

from __future__ import annotations

import pytest
from pypdf import PdfReader

import pflege_pdf as pdf


# ---------------------------------------------------------------------------
# TEXTAUFBEREITUNG
# ---------------------------------------------------------------------------
class TestSanitize:
    """Zeichen, die die PDF-Standardschrift nicht kennt."""

    @pytest.mark.parametrize(
        "eingabe, erwartet",
        [
            ("Modul 4 – Selbstversorgung", "Modul 4 - Selbstversorgung"),
            ("„Pflegetagebuch“", '"Pflegetagebuch"'),
            ("100 € monatlich", "100 Euro monatlich"),
            ("• Erster Punkt", "- Erster Punkt"),
            ("Umlaute äöüß bleiben", "Umlaute äöüß bleiben"),
        ],
    )
    def test_ersetzt_sonderzeichen(self, eingabe, erwartet):
        assert pdf.sanitize(eingabe) == erwartet

    def test_bricht_bei_unbekannten_zeichen_nicht_ab(self):
        # Ohne Absicherung würde die PDF-Erzeugung hier scheitern.
        assert pdf.sanitize("Chinesisch: 中文 Emoji: 🙂") is not None


class TestStripMarkdown:
    @pytest.mark.parametrize(
        "eingabe, erwartet",
        [
            ("**fett**", "fett"),
            ("## Überschrift", "Überschrift"),
            ("* Aufzählung", "Aufzählung"),
            ("Ein *betontes* Wort", "Ein betontes Wort"),
        ],
    )
    def test_entfernt_auszeichnungen(self, eingabe, erwartet):
        assert pdf.strip_markdown(eingabe).strip() == erwartet


class TestStripCitations:
    """Belegziffern gehören nicht in ein Behördenschreiben."""

    def test_entfernt_hochziffern(self):
        assert "¹" not in pdf.strip_citations("Ich bin unselbständig ¹.")

    def test_entfernt_eckige_klammern(self):
        assert pdf.strip_citations("Beleg [3] und [12].") == "Beleg und."

    def test_laesst_kein_leerzeichen_vor_satzzeichen(self):
        assert " ." not in pdf.strip_citations("Satzende ⁸.")

    def test_laesst_normale_zahlen_stehen(self):
        text = "Im Modul 4 wurden 20 Punkte vergeben."
        assert pdf.strip_citations(text) == text


class TestStripLetterBoilerplate:
    """Anrede und Grußformel liefert die Vorlage selbst."""

    @pytest.mark.parametrize(
        "gruss",
        ["Mit freundlichen Grüßen", "Mit freundlichen Gruessen",
         "Mit freundlichen Grüssen", "Herzliche Grüße"],
    )
    def test_entfernt_grussformel_in_allen_schreibweisen(self, gruss):
        roh = f"Sehr geehrte Damen und Herren,\n\nDer Inhalt.\n\n{gruss}\nMax Muster"
        sauber = pdf.strip_letter_boilerplate(roh)
        assert sauber == "Der Inhalt."

    def test_entfernt_vorangestellte_ueberschrift(self):
        roh = "Begründung für den Widerspruch gegen die Einstufung:\n\nDer Inhalt."
        assert pdf.strip_letter_boilerplate(roh) == "Der Inhalt."

    def test_zerstoert_keinen_fliesstext(self):
        # "Begründung" mitten im Satz darf nicht als Überschrift gelten.
        text = "Begründung ist erforderlich, weil der Bescheid unvollständig ist."
        assert pdf.strip_letter_boilerplate(text) == text


class TestFindPlaceholders:
    """Lücken im Brief müssen auffallen, bevor er abgeschickt wird."""

    @pytest.mark.parametrize(
        "text",
        ["Es sind nur [Anzahl der Punkte] Punkte.",
         "Bescheid vom (Datum einsetzen)",
         "Datum: TT.MM.JJJJ",
         "Nummer XXXX",
         "Unterschrift: _______"],
    )
    def test_erkennt_luecken(self, text):
        assert pdf.find_placeholders(text)

    @pytest.mark.parametrize(
        "text",
        ["Ein sauberer Text ohne Lücken.",
         "Modul 4 (Selbstversorgung) wurde geprüft.",
         "Der Bescheid vom 14.03.2026 ist unvollständig."],
    )
    def test_meldet_keine_falschen_treffer(self, text):
        assert pdf.find_placeholders(text) == []


class TestPrepareBegruendung:
    """Zusammenspiel aller Aufbereitungsschritte."""

    def test_bereitet_typische_modellantwort_auf(self):
        roh = (
            "Sehr geehrte Damen und Herren,\n\n"
            "## Begründung\n\n"
            "Im **Modul 4** bin ich unselbständig ¹. Das belegt [2].\n\n"
            "Mit freundlichen Grüßen\nMax Muster"
        )
        sauber = pdf.prepare_begruendung(roh)
        assert "Sehr geehrte" not in sauber
        assert "Grüßen" not in sauber
        assert "**" not in sauber
        assert "¹" not in sauber and "[2]" not in sauber
        assert "Modul 4" in sauber

    def test_vertraegt_leere_eingabe(self):
        assert pdf.prepare_begruendung("") == ""
        assert pdf.prepare_begruendung(None) == ""


# ---------------------------------------------------------------------------
# PFLICHTANGABEN
# ---------------------------------------------------------------------------
@pytest.fixture
def vollstaendige_daten() -> pdf.LetterData:
    return pdf.LetterData(
        absender_name="Michaela Müller-Groß",
        absender_strasse="Musterweg 1",
        absender_plz_ort="99999 Musterstadt",
        ort="Musterstadt",
        kasse_name="Pflegekasse bei der Musterkrankenkasse",
        kasse_strasse="Kassenstraße 5",
        kasse_plz_ort="12345 Kassenstadt",
        versichert_name="Michaela Müller-Groß",
        versichert_nr="A123456789",
        aktenzeichen="AZ-2026/4711",
        bescheid_datum="14.03.2026",
        begruendung="Im Modul 4 wurde die tägliche Hilfe nicht berücksichtigt.",
    )


class TestValidate:
    def test_vollstaendige_angaben_werden_akzeptiert(self, vollstaendige_daten):
        assert pdf.validate(vollstaendige_daten) == []

    def test_leere_angaben_werden_bemaengelt(self):
        assert len(pdf.validate(pdf.LetterData())) >= 5

    def test_platzhalterdatum_gilt_als_fehlend(self, vollstaendige_daten):
        vollstaendige_daten.bescheid_datum = "TT.MM.JJJJ"
        assert "Das Datum des Bescheids" in pdf.validate(vollstaendige_daten)

    def test_fristwahrende_variante_braucht_keine_begruendung(self):
        daten = pdf.LetterData(
            absender_name="Max Muster", absender_strasse="Weg 2",
            absender_plz_ort="11111 Stadt", kasse_name="Kasse",
            bescheid_datum="01.02.2026", begruendung_folgt=True,
        )
        assert pdf.validate(daten) == []


# ---------------------------------------------------------------------------
# PDF-ERZEUGUNG
# ---------------------------------------------------------------------------
def _pdf_text(daten: pdf.LetterData, tmp_path) -> str:
    pfad = tmp_path / "brief.pdf"
    pfad.write_bytes(pdf.build_letter_pdf(daten))
    return "\n".join(seite.extract_text() for seite in PdfReader(str(pfad)).pages)


class TestBuildLetterPdf:
    def test_erzeugt_gueltiges_pdf(self, vollstaendige_daten):
        rohdaten = pdf.build_letter_pdf(vollstaendige_daten)
        assert rohdaten.startswith(b"%PDF-")
        assert len(rohdaten) > 1000

    @pytest.mark.parametrize(
        "erwartet",
        ["Michaela", "Pflegekasse bei der Musterkrankenkasse", "Widerspruch",
         "AZ-2026/4711", "14.03.2026", "A123456789",
         "Bestätigung des Eingangs", "Begründung"],
    )
    def test_enthaelt_alle_pflichtbestandteile(self, vollstaendige_daten, tmp_path, erwartet):
        assert erwartet in _pdf_text(vollstaendige_daten, tmp_path)

    def test_folgt_dem_musterbrief_aufbau(self, vollstaendige_daten, tmp_path):
        # Zeilenumbrüche zusammenfassen: der Satzbau darf über Zeilen laufen.
        text = " ".join(_pdf_text(vollstaendige_daten, tmp_path).split())
        # Reihenfolge: Anrede -> Einleitung -> "Widerspruch" -> Begründung -> Gruß
        assert text.index("Sehr geehrte") < text.index("form- und fristgerecht")
        assert text.index("form- und fristgerecht") < text.index("Begründung")
        assert text.index("Begründung") < text.index("freundlichen")

    def test_widerspruch_steht_im_satz_und_nicht_allein(self, vollstaendige_daten, tmp_path):
        """Das hervorgehobene Wort gehört in den Satz.

        Stand es zentriert auf einer eigenen Zeile, hing das abschließende Wort
        allein in der nächsten Zeile und wirkte wie ein Satzfehler.
        """
        text = " ".join(_pdf_text(vollstaendige_daten, tmp_path).split())
        assert "form- und fristgerecht Widerspruch eingelegt." in text

    def test_anlagen_erscheinen_unter_der_unterschrift(self, vollstaendige_daten, tmp_path):
        vollstaendige_daten.anlagen = "Kopie des Bescheids\n- Pflegetagebuch\n\n"
        text = " ".join(_pdf_text(vollstaendige_daten, tmp_path).split())
        assert text.index("freundlichen") < text.index("Anlagen")
        assert "Kopie des Bescheids" in text
        # Führende Aufzählungszeichen werden entfernt.
        assert "- Pflegetagebuch" not in text
        assert "Pflegetagebuch" in text

    def test_ohne_anlagen_kein_leerer_abschnitt(self, vollstaendige_daten, tmp_path):
        vollstaendige_daten.anlagen = "   \n \n"
        assert "Anlagen" not in _pdf_text(vollstaendige_daten, tmp_path)

    def test_fristwahrende_variante_kuendigt_begruendung_an(self, tmp_path):
        daten = pdf.LetterData(
            absender_name="Max Muster", absender_strasse="Weg 2",
            absender_plz_ort="11111 Stadt", kasse_name="Kasse",
            bescheid_datum="01.02.2026", begruendung_folgt=True,
        )
        assert "in Kürze" in _pdf_text(daten, tmp_path)

    def test_bricht_bei_langem_text_auf_mehrere_seiten_um(self, vollstaendige_daten, tmp_path):
        vollstaendige_daten.begruendung = "Ein sehr langer Absatz zur Prüfung. " * 200
        pfad = tmp_path / "lang.pdf"
        pfad.write_bytes(pdf.build_letter_pdf(vollstaendige_daten))
        leser = PdfReader(str(pfad))
        assert len(leser.pages) > 1
        assert "Seite 1 von" in leser.pages[0].extract_text()

    def test_enthaelt_keine_erzeugerangaben(self, vollstaendige_daten, tmp_path):
        pfad = tmp_path / "meta.pdf"
        pfad.write_bytes(pdf.build_letter_pdf(vollstaendige_daten))
        metadaten = PdfReader(str(pfad)).metadata or {}
        assert not metadaten.get("/Author")
        assert not metadaten.get("/Creator")

    def test_ohne_belegziffern_im_brief(self, vollstaendige_daten, tmp_path):
        vollstaendige_daten.begruendung = pdf.prepare_begruendung(
            "Ich bin unselbständig ¹. Beleg [2] zeigt das."
        )
        text = _pdf_text(vollstaendige_daten, tmp_path)
        assert not any(z in text for z in "⁰¹²³⁴⁵⁶⁷⁸⁹")


# ---------------------------------------------------------------------------
# UNPERSÖNLICHE FORMULIERUNG
# ---------------------------------------------------------------------------
class TestNeutraleFormulierung:
    """Der Brief legt nicht fest, wer ihn schreibt.

    Frühere Fassungen kannten eine wählbare Perspektive: Ich-Form für die
    betroffene Person, dritte Person für Angehörige. Das war eine Entscheidung,
    die niemand treffen wollte, und erzeugte zwei Textfassungen mit doppeltem
    Fehlerpotenzial. Jetzt gibt es nur eine, unpersönliche Fassung - sie stimmt
    für beide Fälle.
    """

    def _daten(self, **zusatz) -> pdf.LetterData:
        werte = dict(
            absender_name="Sabine Müller", absender_strasse="Musterweg 1",
            absender_plz_ort="99999 Musterstadt", kasse_name="Muster-Pflegekasse",
            bescheid_datum="12.03.2025",
            begruendung="Der Hilfebedarf wurde zu niedrig bewertet.",
            versichert_name="Elfriede Müller",
        )
        werte.update(zusatz)
        return pdf.LetterData(**werte)

    def test_einleitung_nennt_die_betroffene_person_ohne_verhaeltnis(self):
        satz = self._daten().einleitungssatz
        assert "für Elfriede Müller" in satz
        assert satz.startswith("hiermit wird")
        assert "eingelegt." in satz

    @pytest.mark.parametrize(
        "wort", ["ich ", "meine ", "mein ", " mir ", " uns ", "Mandantin", "Patientin"]
    )
    def test_kein_persoenliches_wort_im_brief(self, wort, tmp_path):
        text = " ".join(_pdf_text(self._daten(), tmp_path).split())
        assert wort.lower() not in text.lower(), f"„{wort.strip()}“ steht im Brief"

    def test_schlusssatz_redet_die_kasse_nicht_mit_sie_an(self, tmp_path):
        text = _pdf_text(self._daten(), tmp_path)
        assert "Um eine Bestätigung des Eingangs wird gebeten." in text
        assert "bestätigen Sie" not in text

    def test_fristwahrende_variante_ist_ebenfalls_unpersoenlich(self, tmp_path):
        daten = self._daten(begruendung="", begruendung_folgt=True)
        text = _pdf_text(daten, tmp_path)
        assert "wird in Kürze nachgereicht" in text
        assert "reiche ich" not in text

    def test_unterschriftshinweis_legt_die_rolle_nicht_fest(self):
        hinweis = pdf.LetterData().unterschrift_hinweis
        assert hinweis == "(eigenhändige Unterschrift)"

    def test_ohne_namen_der_betroffenen_gilt_der_absender(self):
        daten = self._daten(versichert_name="")
        assert daten.name_versicherte == "Sabine Müller"
        # Der Brief bleibt vollständig - der Name ist keine Pflichtangabe mehr.
        assert pdf.validate(daten) == []

    def test_pdf_nennt_die_betroffene_person(self, tmp_path):
        text = _pdf_text(self._daten(), tmp_path)
        assert "Elfriede Müller" in text
        assert "Versicherte Person" in text


# ---------------------------------------------------------------------------
# ORT DER DATUMSZEILE
# ---------------------------------------------------------------------------
class TestOrtszeile:
    """Der Ort muss nicht zweimal eingetippt werden."""

    def test_leeres_feld_nimmt_den_ort_aus_der_anschrift(self):
        daten = pdf.LetterData(absender_plz_ort="30159 Hannover")
        assert daten.ortszeile == "Hannover"

    def test_eigene_angabe_hat_vorrang(self):
        daten = pdf.LetterData(absender_plz_ort="30159 Hannover", ort="Berlin")
        assert daten.ortszeile == "Berlin"

    @pytest.mark.parametrize(
        "anschrift,erwartet",
        [
            ("30159 Hannover", "Hannover"),
            ("1010 Wien", "Wien"),                      # vierstellig (AT/CH)
            ("99999 Bad Neuenahr-Ahrweiler", "Bad Neuenahr-Ahrweiler"),
            ("60313 Frankfurt am Main", "Frankfurt am Main"),
            ("Hannover", "Hannover"),                   # ganz ohne Postleitzahl
            ("", ""),
        ],
    )
    def test_postleitzahl_wird_abgeschnitten(self, anschrift, erwartet):
        assert pdf.LetterData(absender_plz_ort=anschrift).ortszeile == erwartet

    def test_ort_erscheint_in_der_datumszeile_des_pdf(self, tmp_path):
        daten = pdf.LetterData(
            absender_name="Max Muster", absender_strasse="Weg 2",
            absender_plz_ort="20095 Hamburg", kasse_name="Kasse",
            bescheid_datum="01.02.2026", begruendung="Zur Probe.",
        )
        assert "Hamburg, den" in _pdf_text(daten, tmp_path)


class TestDin5008:
    """Aufbau als Geschäftsbrief."""

    def test_enthaelt_ruecksendeangabe_ueber_dem_anschriftfeld(self, vollstaendige_daten, tmp_path):
        text = _pdf_text(vollstaendige_daten, tmp_path)
        # Die Rücksendeangabe steht vor der Empfängeranschrift.
        assert text.index("Musterweg 1") < text.index("Pflegekasse bei der")

    def test_datum_steht_mit_ort_im_informationsblock(self, vollstaendige_daten, tmp_path):
        assert "Musterstadt, den" in _pdf_text(vollstaendige_daten, tmp_path)

    def test_verwendet_eine_echte_schrift_wenn_vorhanden(self, vollstaendige_daten, tmp_path):
        import os
        vorhanden = any(os.path.exists(k[1]) for k in pdf.SCHRIFT_KANDIDATEN)
        if not vorhanden:
            pytest.skip("Keine Systemschrift vorhanden")
        # Umlaute überleben nur mit eingebetteter Schrift unverändert.
        assert "Müller-Groß" in _pdf_text(vollstaendige_daten, tmp_path)


class TestStripContextHeaders:
    """Technische Trennzeilen dürfen niemals im Brief landen."""

    @pytest.mark.parametrize(
        "roh",
        ["----- [3] ----- Herkunft: Gutachten.pdf | Kapitel: Modul 4\nDer Inhalt.",
         "----- Herkunft: Bescheid der Pflegekasse.pdf\nDer Inhalt.",
         "-----  Herkunft: a.pdf\nDer Inhalt.",
         "----- 3 ----- Herkunft: a.pdf\nDer Inhalt."],
    )
    def test_entfernt_alle_schreibweisen(self, roh):
        assert "Herkunft" not in pdf.strip_context_headers(roh)

    def test_laesst_gedankenstriche_in_ruhe(self):
        text = "Im Modul 4 - Selbstversorgung - wurde gekürzt."
        assert pdf.strip_context_headers(text) == text

    def test_anrede_wird_nach_dem_entfernen_erkannt(self):
        # Stünde die Trennzeile davor, bliebe die Anrede stehen und
        # erschiene doppelt im Brief.
        roh = "----- Herkunft: a.pdf\nSehr geehrte Damen und Herren,\n\nDer Inhalt."
        assert pdf.prepare_begruendung(roh) == "Der Inhalt."

    def test_pdf_bleibt_frei_von_trennzeilen(self, vollstaendige_daten, tmp_path):
        vollstaendige_daten.begruendung = pdf.prepare_begruendung(
            "----- Herkunft: Gutachten.pdf\nDie Bewertung ist zu niedrig ausgefallen."
        )
        assert "Herkunft" not in _pdf_text(vollstaendige_daten, tmp_path)

    @pytest.mark.parametrize("marke", ["[Einleitung]", "[Hauptteil]", "[Schluss]", "[Begründung]:"])
    def test_entfernt_gliederungsmarken(self, marke):
        roh = f"{marke}\nDer eigentliche Inhalt des Absatzes."
        assert marke.strip(":") not in pdf.strip_context_headers(roh)

    def test_marke_vor_der_anrede_verhindert_keine_bereinigung(self):
        roh = "[Einleitung]\nSehr geehrte Damen und Herren,\n\nDer Inhalt."
        assert pdf.prepare_begruendung(roh) == "Der Inhalt."

    def test_laesst_klammern_im_satz_stehen(self):
        text = "Im Modul 4 (Selbstversorgung) wurden 15 Punkte vergeben."
        assert pdf.strip_context_headers(text) == text

    def test_beginnt_nach_entfernter_anrede_gross(self):
        roh = "Sehr geehrte Damen und Herren,\n\nmit Schreiben vom 12.03.2025 haben Sie entschieden."
        assert pdf.prepare_begruendung(roh).startswith("Mit Schreiben")


# ---------------------------------------------------------------------------
# PERSÖNLICHE FORMULIERUNGEN
# ---------------------------------------------------------------------------
class TestMeinungsfloskeln:
    """Floskeln, die nur den Schreiber markieren, werden gestrichen.

    Ihr Wegfall ändert die Aussage nicht - anders als eine Umformung von
    "haben Sie ... eingestuft" ins Passiv, die eine Rate über Ein- oder
    Mehrzahl verlangt und deshalb bewusst unterbleibt.
    """

    @pytest.mark.parametrize(
        "eingabe, erwartet",
        [
            ("Der Bedarf wurde unserer Meinung nach nicht erfasst.",
             "Der Bedarf wurde nicht erfasst."),
            ("Der Bedarf wurde meiner Ansicht nach nicht erfasst.",
             "Der Bedarf wurde nicht erfasst."),
            ("Aus unserer Sicht ist die Bewertung zu niedrig.",
             "Ist die Bewertung zu niedrig."),
            ("Die Einstufung ist, wie wir meinen, nicht haltbar.",
             "Die Einstufung ist nicht haltbar."),
        ],
    )
    def test_streicht_floskeln(self, eingabe, erwartet):
        ergebnis = pdf.strip_meinungsfloskeln(eingabe).strip()
        # Groß-/Kleinschreibung am Satzanfang regelt prepare_begruendung.
        assert ergebnis[:1].upper() + ergebnis[1:] == erwartet

    def test_laesst_sachtext_unangetastet(self):
        text = "Im Modul 4 wurde die tägliche Hilfe beim Duschen nicht berücksichtigt."
        assert pdf.strip_meinungsfloskeln(text) == text

    def test_prepare_begruendung_streicht_sie_mit(self):
        roh = "Der Hilfebedarf wurde unserer Meinung nach zu niedrig bewertet."
        assert "unserer Meinung" not in pdf.prepare_begruendung(roh)


class TestFindPersonalWording:
    """Was nicht gefahrlos umschreibbar ist, muss wenigstens auffallen."""

    @pytest.mark.parametrize(
        "text, erwartet",
        [
            ("Mit Schreiben vom 12.03.2025 haben Sie sie eingestuft.", "Sie"),
            ("Ich lege Widerspruch ein.", "Ich"),
            ("Meine Mutter benötigt Hilfe.", "Meine"),
            ("Wir bitten um erneute Begutachtung.", "Wir"),
            ("Bitte prüfen Sie Ihre Unterlagen.", "Sie"),
        ],
    )
    def test_erkennt_persoenliche_woerter(self, text, erwartet):
        assert erwartet in pdf.find_personal_wording(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Mit Bescheid vom 12.03.2025 wurde Elfriede Müller eingestuft.",
            "Es wird um eine erneute Begutachtung gebeten.",
            "Nach den Begutachtungs-Richtlinien ist eine höhere Bewertung angezeigt.",
            "Der Hilfebedarf beim Duschen wurde nicht berücksichtigt.",
        ],
    )
    def test_meldet_bei_neutralem_text_nichts(self, text):
        assert pdf.find_personal_wording(text) == []

    def test_meldet_jedes_wort_nur_einmal(self):
        text = "ich schreibe, weil ich betroffen bin, und ich bitte darum."
        assert pdf.find_personal_wording(text).count("ich") == 1

    @pytest.mark.parametrize(
        "text",
        [
            "Sie benötigt beim Duschen Hilfe.",          # Satzanfang: die Person
            "Ihr Gang ist unsicher.",                     # Satzanfang: ihr Gang
        ],
    )
    def test_kleingeschriebenes_sie_ist_kein_verstoss(self, text):
        """„sie benötigt Hilfe“ meint die betroffene Person und ist richtig.

        Am Satzanfang steht es großgeschrieben und lässt sich nicht von der
        Höflichkeitsanrede unterscheiden - dieser Fehlalarm wird bewusst in
        Kauf genommen, weil er selten ist. Mitten im Satz darf er nicht
        auftreten, sonst warnte fast jeder Brief grundlos.
        """
        mitten_im_satz = "Der Bericht zeigt: " + text[0].lower() + text[1:]
        assert pdf.find_personal_wording(mitten_im_satz) == []


class TestStripWarnbanner:
    """Der Warnbanner ist Beiwerk der Anzeige, kein Briefinhalt.

    Belegt am 2026-08-13: Eine Antwort ohne Belegstelle bekommt den Banner
    vorangestellt. Er landete dadurch im Briefentwurf - und weil die Anrede
    danach nicht mehr am Textanfang stand, blieb auch sie stehen.
    """

    BANNER = (
        "> ⚠️ **Diese Antwort stützt sich auf keine Textstelle** aus Ihren Unterlagen. "
        "Bitte prüfen Sie sie nach.\n\n"
    )

    def test_entfernt_den_banner(self):
        assert pdf.strip_warnbanner(self.BANNER + "Der Inhalt.") == "Der Inhalt."

    def test_anrede_wird_danach_wieder_erkannt(self):
        roh = self.BANNER + "Sehr geehrte Damen und Herren,\n\nDer Inhalt.\n\nMit freundlichen Grüßen"
        fertig = pdf.prepare_begruendung(roh)
        assert "Sehr geehrte" not in fertig
        assert "Grüßen" not in fertig
        assert "prüfen Sie" not in fertig
        assert fertig == "Der Inhalt."

    def test_laesst_fliesstext_unangetastet(self):
        text = "Im Modul 4 wurde der Hilfebedarf beim Duschen nicht berücksichtigt."
        assert pdf.strip_warnbanner(text) == text
