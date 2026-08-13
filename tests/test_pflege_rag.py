"""Tests für die Bausteine der Such- und Antwortkette.

Geprüft werden ausschließlich die reinen Funktionen: Qualitätsfilter,
Metadaten-Erkennung, Rangfusion, Belegstellen und Ausschnittwahl. Für Modelle
und Datenbank wären laufende Dienste nötig; die bleiben hier außen vor.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import pflege_rag as rag


# ---------------------------------------------------------------------------
# TEXTAUFBEREITUNG UND QUALITÄTSFILTER
# ---------------------------------------------------------------------------
class TestCleanText:
    def test_entfernt_steuerzeichen_und_leerzeilen(self):
        # Steuerzeichen werden zu einem Leerzeichen, damit Wörter nicht
        # zusammenkleben; Leerzeilen fallen weg.
        assert rag.clean_text("a\x00b\n\n\nc") == "a b\nc"

    def test_fasst_mehrfache_leerzeichen_zusammen(self):
        assert rag.clean_text("viel     Abstand") == "viel Abstand"


class TestIsInformative:
    """Der Filter hält inhaltsleere Abschnitte aus der Wissensbasis heraus."""

    @pytest.mark.parametrize(
        "text, grund",
        [
            ("| ---- | ---- | ---- |", "Tabellenrahmen"),
            ("Seite 47 von 332", "Seitenzahl"),
            ("1.2 Mobilität ......................... 14", "Inhaltsverzeichnis"),
            ("12 15 18 21 24 27 30 33 36 39 42 45", "reine Zahlenreihe"),
            ("", "leer"),
            ("Zu kurz.", "unter der Mindestlänge"),
        ],
    )
    def test_weist_inhaltsleere_abschnitte_ab(self, text, grund):
        assert not rag.is_informative(text), grund

    def test_erkennt_echten_fliesstext(self):
        text = (
            "Im Modul 4 Selbstversorgung werden dreizehn Einzelkriterien bewertet. "
            "Die Bewertung reicht von null bis drei Punkten je Kriterium."
        )
        assert rag.is_informative(text)


class TestNormalize:
    def test_gleicht_umlaute_an(self):
        assert rag.normalize("Mobilität") == rag.normalize("Mobilitaet")

    def test_ist_unabhaengig_von_grossschreibung(self):
        assert rag.normalize("SELBSTVERSORGUNG") == rag.normalize("selbstversorgung")


class TestDetectModules:
    @pytest.mark.parametrize(
        "text, erwartet",
        [
            ("Bei Modul 4 zeigt sich Bedarf.", {4}),
            ("Die Mobilität ist eingeschränkt.", {1}),
            ("Mobilitaet ohne Umlaut", {1}),
            ("Modul 2 und Modul 6", {2, 6}),
            ("Kein Bezug zu einem Bereich.", set()),
        ],
    )
    def test_erkennt_module(self, text, erwartet):
        assert erwartet <= set(rag.detect_modules(text))

    def test_kennt_alle_sechs_module(self):
        assert sorted(rag.MODULE_NAMES) == [1, 2, 3, 4, 5, 6]


class TestClassifyDocument:
    @pytest.mark.parametrize(
        "dateiname, erwartet",
        [
            ("Musterbrief_Widerspruch.pdf", "Musterbrief"),
            ("MD-Bund_Begutachtungs-Richtlinien.pdf", "Amtliche Richtlinie"),
            ("BMG_Ratgeber_Pflege.pdf", "Amtlicher Ratgeber"),
            ("Gutachten3.pdf", "Gutachten"),
            ("https://www.pflege.de/artikel", "Webseite"),
        ],
    )
    def test_ordnet_dokumentart_zu(self, dateiname, erwartet):
        assert rag.classify_document(dateiname) == erwartet


# ---------------------------------------------------------------------------
# ZERLEGUNG
# ---------------------------------------------------------------------------
class TestSplitDocuments:
    def test_behaelt_die_herkunftsangabe(self):
        doc = Document(
            page_content=("Modul 4 Selbstversorgung wird mit 40 Prozent gewichtet. " * 20),
            metadata={"source": "richtlinie.pdf"},
        )
        abschnitte = rag.split_documents([doc])
        assert abschnitte
        assert all(a.metadata["source"] == "richtlinie.pdf" for a in abschnitte)

    def test_ergaenzt_dokumentart_und_module(self):
        doc = Document(
            page_content=("Im Modul 4 Selbstversorgung gelten Besonderheiten. " * 20),
            metadata={"source": "MD-Bund_Begutachtungs-Richtlinien.pdf"},
        )
        abschnitt = rag.split_documents([doc])[0]
        assert abschnitt.metadata["doc_kind"] == "Amtliche Richtlinie"
        assert 4 in abschnitt.metadata["modules"]

    def test_wirft_inhaltsleere_abschnitte_weg(self):
        doc = Document(page_content="| a | b |\n| - | - |", metadata={"source": "x.pdf"})
        assert rag.split_documents([doc]) == []


# ---------------------------------------------------------------------------
# STICHWORTSUCHE
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_entfernt_haeufige_woerter(self):
        assert "und" not in rag.tokenize("Mobilität und Selbstversorgung")

    def test_gleicht_umlaute_an(self):
        assert rag.tokenize("Mobilität") == rag.tokenize("Mobilitaet")

    def test_behaelt_paragrafenzeichen(self):
        assert any("§" in wort for wort in rag.tokenize("Nach § 18 SGB XI"))


class TestReciprocalRankFusion:
    def _doc(self, nummer: int) -> Document:
        return Document(page_content=f"Inhalt {nummer}", metadata={"source": "a.pdf"})

    def test_belohnt_treffer_aus_beiden_listen(self):
        a, b, c = self._doc(1), self._doc(2), self._doc(3)
        # c steht in beiden Listen weit oben, a nur in einer.
        fusion = rag.reciprocal_rank_fusion([[a, b, c], [c, b]])
        assert fusion[0].page_content == "Inhalt 3"

    def test_entfernt_doppelte_treffer(self):
        a = self._doc(1)
        assert len(rag.reciprocal_rank_fusion([[a], [a], [a]])) == 1

    def test_vertraegt_leere_listen(self):
        assert rag.reciprocal_rank_fusion([]) == []


# ---------------------------------------------------------------------------
# BELEGSTELLEN
# ---------------------------------------------------------------------------
class TestCitations:
    def test_wandelt_in_hochziffern(self):
        assert rag.render_citations("Beleg [1].", [1]) == "Beleg ¹."

    def test_entfernt_ungueltige_nummern(self):
        ergebnis = rag.render_citations("Erfunden [9].", [1, 2])
        assert "[9]" not in ergebnis and "9" not in ergebnis

    def test_liest_zitierte_nummern_aus(self):
        assert rag.cited_numbers("Erst [1], dann [3], nochmal [1].") == [1, 3]

    def test_to_superscript(self):
        assert rag.to_superscript(12) == "¹²"


# ---------------------------------------------------------------------------
# AUSSCHNITTE FÜR DIE QUELLENANZEIGE
# ---------------------------------------------------------------------------
class TestStripJunkLines:
    def test_entfernt_tabellenzeilen(self):
        assert "|" not in rag.strip_junk_lines("| a | b |\nEchter Satz mit Inhalt.")

    def test_entfernt_trennlinien(self):
        assert "===" not in rag.strip_junk_lines("=====\nEchter Satz mit Inhalt.")


class TestBestExcerpt:
    def test_liefert_ganze_saetze_ohne_zeichensalat(self):
        abschnitt = (
            "| Modul | Punkte |\n| --- | --- |\n"
            "Die Selbstversorgung wird mit 40 Prozent gewichtet. "
            "Sie umfasst Körperpflege und Ernährung."
        )
        ausschnitt = rag.best_excerpt(abschnitt, "Wie wird Selbstversorgung gewichtet?")
        assert "40 Prozent" in ausschnitt
        assert "|" not in ausschnitt and "---" not in ausschnitt

    def test_beginnt_nicht_mit_einer_seitenzahl(self):
        abschnitt = "Seite 12\nDer Widerspruch ist innerhalb eines Monats einzulegen."
        assert not rag.best_excerpt(abschnitt, "Frist").startswith("Seite")

    def test_haelt_die_laengenvorgabe_ein(self):
        abschnitt = "Ein vollständiger Satz zur Prüfung der Länge. " * 40
        assert len(rag.best_excerpt(abschnitt, "Prüfung", max_chars=200)) < 320


# ---------------------------------------------------------------------------
# KONTEXT UND SCHNELLAKTIONEN
# ---------------------------------------------------------------------------
class TestBuildMessages:
    def test_begrenzt_den_gespraechsverlauf(self):
        verlauf = [{"role": "user", "content": f"Frage {i}"} for i in range(30)]
        nachrichten = rag.build_messages("System", verlauf)
        assert len(nachrichten) == rag.MAX_HISTORY_MESSAGES + 1
        assert nachrichten[0]["role"] == "system"

    def test_behaelt_die_juengsten_nachrichten(self):
        verlauf = [{"role": "user", "content": f"Frage {i}"} for i in range(30)]
        assert rag.build_messages("System", verlauf)[-1]["content"] == "Frage 29"


class TestQuickActions:
    def test_es_gibt_vier_aufgaben(self):
        assert len(rag.QUICK_ACTIONS) == 4

    def test_jede_aufgabe_ist_vollstaendig(self):
        for aktion in rag.QUICK_ACTIONS:
            assert aktion.titel and aktion.beschreibung
            assert aktion.nutzertext and aktion.prompt
            # Der Anzeigetext darf nicht die lange Anweisung sein.
            assert len(aktion.nutzertext) < len(aktion.prompt)

    def test_modulweise_aufgaben_haben_zusatzfragen(self):
        for schluessel in ("differenz", "argumente", "schreiben"):
            assert len(rag.QUICK_ACTION_BY_KEY[schluessel].zusatzfragen) == 6

    def test_briefaufgabe_verbietet_belegziffern(self):
        prompt = rag.QUICK_ACTION_BY_KEY["schreiben"].prompt
        assert "KEINE Belegnummern" in prompt

    def test_fuehrt_keine_beispieldokumente_als_vorhanden_auf(self):
        # Werden Dokumentarten als Beispiel aufgezaehlt, nimmt das Modell ihr
        # Vorhandensein an und erfindet Inhalte dazu. In einer Verbots-Anweisung
        # duerfen sie dagegen vorkommen.
        for aktion in rag.QUICK_ACTIONS:
            assert "belegen (Pflegetagebuch" not in aktion.prompt
            assert "Unterlagen (Pflegetagebuch" not in aktion.prompt


class TestFormatNumberedContext:
    def test_nummeriert_die_abschnitte(self):
        quellen = [
            rag.SourceRef(nummer=1, quelle="a.pdf", art="Gutachten",
                          ausschnitt="…", bewertung=0.9, herkunft="nutzer")
        ]
        dokumente = [Document(page_content="Inhalt", metadata={})]
        ergebnis = rag.format_numbered_context(quellen, dokumente, "Titel")
        assert "[1]" in ergebnis and "Inhalt" in ergebnis

    def test_meldet_leere_trefferliste(self):
        assert "keine passenden" in rag.format_numbered_context([], [], "Fachwissen")


# ---------------------------------------------------------------------------
# ANSCHLUSSFRAGEN
# ---------------------------------------------------------------------------
class TestNeedsCondensing:
    """Entscheidet, wann eine Frage ohne den Verlauf unverständlich ist."""

    def _verlauf(self):
        return [
            {"role": "user", "content": "Wie wird Modul 4 bewertet?"},
            {"role": "assistant", "content": "Modul 4 zählt 40 Prozent."},
        ]

    @pytest.mark.parametrize(
        "frage",
        ["Und was heißt das für mich?", "Wie komme ich dazu?", "Warum ist das so?"],
    )
    def test_erkennt_rueckbezuege(self, frage):
        assert rag.needs_condensing(frage, self._verlauf())

    def test_laesst_eigenstaendige_fragen_in_ruhe(self):
        frage = "Welche Frist gilt für den Widerspruch gegen einen Pflegegradbescheid?"
        assert not rag.needs_condensing(frage, self._verlauf())

    def test_erste_frage_braucht_keine_aufloesung(self):
        assert not rag.needs_condensing("Und was heißt das?", [])

    def test_lange_fragen_stehen_fuer_sich(self):
        lang = "Das Gutachten sagt " + "x" * 200
        assert not rag.needs_condensing(lang, self._verlauf())


class TestCondenseQuestion:
    class _LLM:
        def __init__(self, antwort): self.antwort = antwort
        def invoke(self, _):
            return type("Antwort", (), {"content": self.antwort})()

    def _verlauf(self):
        return [
            {"role": "user", "content": "Wie wird Modul 4 bewertet?"},
            {"role": "assistant", "content": "Modul 4 zählt 40 Prozent."},
        ]

    def test_uebernimmt_die_umformulierung(self):
        llm = self._LLM("Was bedeutet die Gewichtung von Modul 4 für meinen Pflegegrad?")
        ergebnis = rag.condense_question(llm, "Und was heißt das für mich?", self._verlauf())
        assert "Modul 4" in ergebnis

    def test_faellt_bei_fehler_auf_die_frage_zurueck(self):
        class Kaputt:
            def invoke(self, _): raise RuntimeError("nicht erreichbar")

        frage = "Und was heißt das für mich?"
        assert rag.condense_question(Kaputt(), frage, self._verlauf()) == frage

    def test_verwirft_unbrauchbare_antworten(self):
        frage = "Und was heißt das für mich?"
        assert rag.condense_question(self._LLM("ok"), frage, self._verlauf()) == frage

    def test_ruft_das_modell_bei_eigenstaendiger_frage_nicht_auf(self):
        class Verboten:
            def invoke(self, _): raise AssertionError("darf nicht aufgerufen werden")

        frage = "Welche Frist gilt für den Widerspruch gegen einen Pflegegradbescheid?"
        assert rag.condense_question(Verboten(), frage, self._verlauf()) == frage


class TestStripContextHeaders:
    """Das Modell übernimmt gelegentlich die Trennzeilen des Kontexts."""

    def test_entfernt_uebernommene_trennzeile(self):
        roh = ("----- [1] ----- Herkunft: § 15 SGB XI | Kapitel: Ermittlung\n"
               "Der Pflegegrad ergibt sich aus den Punkten.")
        assert rag.strip_context_headers(roh) == "Der Pflegegrad ergibt sich aus den Punkten."

    def test_entfernt_trennzeile_ohne_klammern(self):
        roh = "----- 3 ----- Herkunft: a.pdf\nInhalt."
        assert "Herkunft" not in rag.strip_context_headers(roh)

    def test_entfernt_uebrig_gebliebene_striche(self):
        assert rag.strip_context_headers("----- [2] -----\nInhalt.").strip() == "Inhalt."

    def test_laesst_normale_gedankenstriche_stehen(self):
        text = "Im Modul 4 - Selbstversorgung - wurde gekürzt."
        assert rag.strip_context_headers(text) == text

    def test_wird_beim_rendern_angewandt(self):
        roh = "----- [1] ----- Herkunft: a.pdf\nAussage [1]."
        assert rag.render_citations(roh, [1]) == "Aussage ¹."


class TestKontextbudget:
    """Das örtliche Modell hat 8192 Token; zu viel Kontext zerstört die Antwort."""

    def _belege(self, anzahl: int, laenge: int = 900):
        return [
            (Document(page_content="x" * laenge, metadata={"source": f"{i}.pdf"}), 1.0 - i / 100)
            for i in range(anzahl)
        ]

    def test_schneidet_auf_das_budget_zu(self):
        behalten, rest = rag._passe_in_kontext(self._belege(20), 4500)
        assert len(behalten) == 5
        assert rest == 0

    def test_behaelt_die_bestbewerteten(self):
        behalten, _ = rag._passe_in_kontext(self._belege(10), 2700)
        assert [d.metadata["source"] for d, _ in behalten] == ["0.pdf", "1.pdf", "2.pdf"]

    def test_behaelt_mindestens_einen_beleg(self):
        behalten, _ = rag._passe_in_kontext(self._belege(3, laenge=99999), 100)
        assert len(behalten) == 1

    def test_vertraegt_leere_liste(self):
        assert rag._passe_in_kontext([], 1000) == ([], 1000)
