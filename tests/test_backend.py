"""Tests für den Sitzungsdienst.

Jeder Test arbeitet auf einer eigenen Datenbankdatei, damit die Tests weder
untereinander noch mit der laufenden Anwendung in Konflikt geraten.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import timedelta

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient


@pytest.fixture
def dienst(tmp_path, monkeypatch):
    """Lädt den Dienst mit einer frischen Datenbank und festem Schlüssel neu."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("SESSION_LIFETIME_DAYS", "28")
    monkeypatch.setenv("SESSION_EXTEND_DAYS", "3")
    sys.modules.pop("backend", None)
    modul = importlib.import_module("backend")
    yield modul
    sys.modules.pop("backend", None)


@pytest.fixture
def client(dienst):
    return TestClient(dienst.app)


def neue_sitzung(client) -> str:
    antwort = client.post("/session")
    assert antwort.status_code == 200
    return antwort.json()["token"]


# ---------------------------------------------------------------------------
# ZUGANGSCODES
# ---------------------------------------------------------------------------
class TestSessionErstellen:
    def test_liefert_code_und_ablaufdatum(self, client):
        daten = client.post("/session").json()
        assert daten["token"] and daten["expires_at"]

    def test_code_ist_ausreichend_lang(self, client):
        # 32 Zufallsbytes ergeben 43 Zeichen in der URL-sicheren Schreibweise.
        assert len(neue_sitzung(client)) >= 40

    def test_codes_werden_nie_doppelt_vergeben(self, client):
        codes = {neue_sitzung(client) for _ in range(50)}
        assert len(codes) == 50

    def test_datenbank_weist_doppelte_codes_ab(self, dienst, client):
        from sqlalchemy.exc import IntegrityError

        vorhanden = neue_sitzung(client)
        db = dienst.SessionLocal()
        try:
            db.add(dienst.SessionRecord(
                token=vorhanden, expires_at=dienst.utcnow(), data_encrypted=b""))
            with pytest.raises(IntegrityError):
                db.commit()
        finally:
            db.rollback()
            db.close()


# ---------------------------------------------------------------------------
# SPEICHERN UND LADEN
# ---------------------------------------------------------------------------
class TestSessionInhalt:
    def test_neue_sitzung_ist_leer(self, client):
        assert client.get(f"/session/{neue_sitzung(client)}").json()["data"] == {}

    def test_speichert_und_liest_zurueck(self, client):
        token = neue_sitzung(client)
        nutzdaten = {
            "messages": [{"role": "user", "content": "Hallo"}],
            "user_documents": [{"page_content": "Inhalt", "metadata": {"source": "a.pdf"}}],
            "absender_name": "Michaela Müller-Groß",
        }
        assert client.put(f"/session/{token}", json={"data": nutzdaten}).status_code == 200
        assert client.get(f"/session/{token}").json()["data"] == nutzdaten

    def test_ueberschreibt_den_alten_stand(self, client):
        token = neue_sitzung(client)
        client.put(f"/session/{token}", json={"data": {"a": 1}})
        client.put(f"/session/{token}", json={"data": {"b": 2}})
        assert client.get(f"/session/{token}").json()["data"] == {"b": 2}

    def test_daten_liegen_verschluesselt_in_der_datenbank(self, dienst, client):
        token = neue_sitzung(client)
        client.put(f"/session/{token}", json={"data": {"geheim": "Pflegetagebuch-Eintrag"}})
        db = dienst.SessionLocal()
        try:
            eintrag = db.query(dienst.SessionRecord).filter_by(token=token).first()
            assert b"Pflegetagebuch" not in eintrag.data_encrypted
        finally:
            db.close()

    def test_unbekannter_code_wird_abgewiesen(self, client):
        assert client.get("/session/gibtesnicht").status_code == 404


# ---------------------------------------------------------------------------
# FRISTEN
# ---------------------------------------------------------------------------
class TestFristen:
    def test_verlaengert_um_drei_tage(self, client):
        token = neue_sitzung(client)
        vorher = client.get(f"/session/{token}").json()["expires_at"]
        nachher = client.post(f"/session/{token}/extend").json()["expires_at"]
        assert nachher > vorher

    def test_abgelaufene_sitzung_wird_geloescht(self, dienst, client):
        token = neue_sitzung(client)
        db = dienst.SessionLocal()
        try:
            eintrag = db.query(dienst.SessionRecord).filter_by(token=token).first()
            eintrag.expires_at = dienst.utcnow() - timedelta(seconds=1)
            db.commit()
        finally:
            db.close()

        # Erster Zugriff meldet den Ablauf, danach ist der Code unbekannt.
        assert client.get(f"/session/{token}").status_code == 410
        assert client.get(f"/session/{token}").status_code == 404

    def test_speichern_verlaengert_die_frist_nicht(self, client):
        token = neue_sitzung(client)
        vorher = client.get(f"/session/{token}").json()["expires_at"]
        nachher = client.put(f"/session/{token}", json={"data": {"x": 1}}).json()["expires_at"]
        assert nachher == vorher


# ---------------------------------------------------------------------------
# LÖSCHUNG
# ---------------------------------------------------------------------------
class TestLoeschung:
    def test_entfernt_die_sitzung(self, client):
        token = neue_sitzung(client)
        assert client.delete(f"/session/{token}").status_code == 200
        assert client.get(f"/session/{token}").status_code == 404

    def test_unbekannter_code_meldet_keinen_fehler(self, client):
        assert client.delete("/session/gibtesnicht").status_code == 200

    def test_entfernt_den_code_aus_der_datenbankdatei(self, dienst, client, tmp_path):
        token = neue_sitzung(client)
        client.put(f"/session/{token}", json={"data": {"x": "y"}})
        datei = tmp_path / "test.db"
        assert token.encode() in datei.read_bytes()

        client.delete(f"/session/{token}")
        # secure_delete und das Neuschreiben der Datei entfernen den Code wirklich.
        assert token.encode() not in datei.read_bytes()


# ---------------------------------------------------------------------------
# VERSCHLÜSSELUNG
# ---------------------------------------------------------------------------
class TestVerschluesselung:
    def test_umlaute_bleiben_erhalten(self, dienst):
        daten = {"text": "Größe, Höhe, Übung, weiß"}
        assert dienst._decrypt(dienst._encrypt(daten)) == daten

    def test_fremder_schluessel_liefert_leeren_stand(self, dienst):
        fremd = Fernet(Fernet.generate_key()).encrypt(b'{"a": 1}')
        assert dienst._decrypt(fremd) == {}

    def test_leerer_inhalt_ist_zulaessig(self, dienst):
        assert dienst._decrypt(None) == {}


# ---------------------------------------------------------------------------
# UNTERLAGEN UND ASSISTENT
# ---------------------------------------------------------------------------
class TestActions:
    def test_liefert_die_vier_aufgaben(self, client):
        aufgaben = client.get("/actions").json()
        assert len(aufgaben) == 4
        assert {a["schluessel"] for a in aufgaben} == {
            "einlesen", "differenz", "argumente", "schreiben"
        }

    def test_jede_aufgabe_ist_beschriftet(self, client):
        for aufgabe in client.get("/actions").json():
            assert aufgabe["titel"] and aufgabe["beschreibung"] and aufgabe["nutzertext"]


class TestStatus:
    def test_meldet_die_betriebsart(self, client):
        daten = client.get("/status").json()
        assert "vektordatenbank" in daten and daten["sprachmodell"]


class TestDokumente:
    def _pdf_bytes(self) -> bytes:
        # Ein minimales, aber gültiges PDF mit Text.
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 6, text=("Im Modul 4 Selbstversorgung besteht taeglich Hilfebedarf "
                                   "beim Waschen und Ankleiden. " * 6))
        return bytes(pdf.output())

    def test_weist_zu_grosse_dateien_ab(self, dienst, client, monkeypatch):
        monkeypatch.setattr(dienst, "MAX_UPLOAD_BYTES", 10)
        token = neue_sitzung(client)
        antwort = client.post(
            f"/session/{token}/documents",
            files={"files": ("gross.pdf", self._pdf_bytes(), "application/pdf")},
        )
        ergebnis = antwort.json()["ergebnisse"][0]
        assert not ergebnis["erfolgreich"] and "größer" in ergebnis["hinweis"]

    def test_unbekannter_code_wird_abgewiesen(self, client):
        antwort = client.post(
            "/session/gibtesnicht/documents",
            files={"files": ("a.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert antwort.status_code == 404

    def test_entfernt_eine_unterlage(self, dienst, client):
        token = neue_sitzung(client)
        # Unterlagen direkt in die Sitzung legen, ohne den PDF-Umweg.
        client.put(f"/session/{token}", json={"data": {
            "user_documents": [
                {"page_content": "Inhalt A", "metadata": {"source": "a.pdf"}},
                {"page_content": "Inhalt B", "metadata": {"source": "b.pdf"}},
            ]
        }})
        antwort = client.delete(f"/session/{token}/documents/a.pdf")
        assert antwort.json()["dokumente"] == ["b.pdf"]
        verbleibend = client.get(f"/session/{token}").json()["data"]["user_documents"]
        assert [d["metadata"]["source"] for d in verbleibend] == ["b.pdf"]


class TestChatEingaben:
    def test_leere_anfrage_wird_abgewiesen(self, client):
        token = neue_sitzung(client)
        antwort = client.post(f"/session/{token}/chat", json={})
        assert antwort.status_code == 400

    def test_unbekannter_code_wird_abgewiesen(self, client):
        antwort = client.post("/session/gibtesnicht/chat", json={"frage": "Test"})
        assert antwort.status_code == 404


class TestChatAbbruch:
    """Ein Fehler mitten im Antwortstrom muss die Oberfläche erreichen.

    Bricht die Beantwortung ab, ohne dass etwas gesendet wird, endet der Strom
    stillschweigend. Die Oberfläche zeigt dann bis in alle Ewigkeit "wird
    durchsucht ..." an, ohne dass jemand erfährt, was los ist.
    """

    def test_meldet_einen_abbruch_als_ereignis(self, dienst, client, monkeypatch):
        import pflege_service

        def bricht_ab(*args, **kwargs):
            yield {"art": "status", "text": "Wird gesucht …"}
            raise RuntimeError("Das Sprachmodell antwortet nicht.")

        monkeypatch.setattr(pflege_service, "beantworte", bricht_ab)
        token = neue_sitzung(client)
        antwort = client.post(f"/session/{token}/chat", json={"frage": "Test"})

        arten = [
            json.loads(zeile[6:])["art"]
            for zeile in antwort.text.splitlines()
            if zeile.startswith("data: ")
        ]
        assert arten == ["status", "fehler"]

    def test_der_wortlaut_des_fehlers_bleibt_im_dienst(self, dienst, client, monkeypatch):
        """Fehlertexte können Auszüge aus den Unterlagen enthalten."""
        import pflege_service

        def bricht_ab(*args, **kwargs):
            raise RuntimeError("Diagnose F00.1 aus dem Gutachten von Frau Müller")
            yield  # pragma: no cover - macht die Funktion zum Erzeuger

        monkeypatch.setattr(pflege_service, "beantworte", bricht_ab)
        token = neue_sitzung(client)
        antwort = client.post(f"/session/{token}/chat", json={"frage": "Test"})

        assert "Müller" not in antwort.text
        assert "F00.1" not in antwort.text
        assert "fehler" in antwort.text


class TestIndexFreigabe:
    """Beim Löschen einer Sitzung muss auch der Suchindex verschwinden."""

    def test_loeschen_gibt_den_index_frei(self, dienst, client):
        import pflege_service

        token = neue_sitzung(client)
        pflege_service.user_indices._eintraege[token] = pflege_service.UserIndexEintrag(
            index=None, fingerabdruck="test"
        )
        client.delete(f"/session/{token}")
        assert token not in pflege_service.user_indices._eintraege
