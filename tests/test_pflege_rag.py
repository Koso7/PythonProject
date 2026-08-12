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
