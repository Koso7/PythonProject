"""Hintergrunddienst für die Sitzungsverwaltung.

Verwaltet anonyme Sitzungen ohne Registrierung: Eine Sitzung wird ausschließlich
über einen zufälligen Zugangscode angesprochen. Der gespeicherte Arbeitsstand
(Gesprächsverlauf, Textabschnitte der Unterlagen, Widerspruchsentwurf,
Formularangaben) liegt verschlüsselt in der Datenbank.

Datenschutz:
* Der Zugangscode ist Primärschlüssel und wird nie doppelt vergeben.
* Gelöschte Sitzungen werden durch ``secure_delete`` und ein Neuschreiben der
  Datei tatsächlich aus der Datenbank entfernt, nicht nur als frei markiert.
* Ein Hintergrundlauf entfernt abgelaufene Sitzungen selbsttätig.
* Der Dienst lauscht nur auf der örtlichen Netzwerkschnittstelle.

Start:  uvicorn backend:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, LargeBinary, String, create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession, declarative_base, sessionmaker

import pflege_pdf
import pflege_rag
import pflege_service

load_dotenv()

SESSION_LIFETIME_DAYS = int(os.getenv("SESSION_LIFETIME_DAYS", "28"))
SESSION_EXTEND_DAYS = int(os.getenv("SESSION_EXTEND_DAYS", "3"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pflege_sicher.db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
# Eingescannte Gutachten sind erfahrungsgemäß groß.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "30")) * 1024 * 1024

if not ENCRYPTION_KEY:
    # Nur für lokale Entwicklung: Ohne festen Key in .env wird bei jedem Neustart
    # ein neuer Schlüssel erzeugt -> bereits gespeicherte Sitzungen werden dann
    # unlesbar (bleiben aber automatisch nach Ablauf der Frist gelöscht).
    # Für echten Betrieb MUSS ein fester ENCRYPTION_KEY in .env gesetzt werden.
    ENCRYPTION_KEY = Fernet.generate_key().decode()
    print("⚠️  Kein ENCRYPTION_KEY in .env gefunden – generiere temporären Key für diesen Lauf.")
    print("    Für dauerhaft lesbare Sitzungen bitte in .env fest eintragen:")
    print(f"    ENCRYPTION_KEY={ENCRYPTION_KEY}")

fernet = Fernet(ENCRYPTION_KEY.encode() if isinstance(ENCRYPTION_KEY, str) else ENCRYPTION_KEY)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    """Sorgt dafür, dass gelöschte Daten tatsächlich überschrieben werden.

    Ohne `secure_delete` markiert SQLite gelöschte Zeilen nur als frei; der
    Klartext bliebe in der Datenbankdatei stehen und wäre mit einfachen
    Werkzeugen wiederherstellbar. Bei Gesundheits- und Pflegedaten ist das
    nicht hinnehmbar.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA secure_delete=ON")
    cursor.close()


def _rewrite_database_file() -> None:
    """Schreibt die Datenbankdatei neu und gibt gelöschte Seiten endgültig frei."""
    if not DATABASE_URL.startswith("sqlite"):
        return
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.exec_driver_sql("VACUUM")
    except Exception as error:  # pragma: no cover - rein vorsorglich
        print(f"Hinweis: Die Datenbankdatei konnte nicht neu geschrieben werden: {error}")


def utcnow() -> datetime:
    # Naiv (ohne Zeitzonenangabe) gehalten, weil SQLite Zeitzoneninformationen
    # beim Speichern verwirft - ein Mix aus naiven und zeitzonenbehafteten
    # Werten würde beim Vergleich einen TypeError auslösen.
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ------------------------------------------------------------
# DATENMODELL
# ------------------------------------------------------------
class SessionRecord(Base):
    """Eine anonyme, token-basierte Sitzung. Kein Nutzerkonto, keine E-Mail,
    kein Passwort - der Token selbst ist der einzige Zugriffsschlüssel."""

    __tablename__ = "sessions"
    token = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_accessed_at = Column(DateTime, default=utcnow)
    # Verschlüsselter JSON-Blob mit Chatverlauf, hochgeladenen Dokumentabschnitten,
    # zuletzt generiertem Widerspruchstext und PDF-Formularfeldern.
    data_encrypted = Column(LargeBinary, nullable=True)


Base.metadata.create_all(bind=engine)


# ------------------------------------------------------------
# SCHEMAS
# ------------------------------------------------------------
class SessionCreateResponse(BaseModel):
    token: str
    expires_at: datetime


class SessionSyncRequest(BaseModel):
    data: dict = Field(default_factory=dict)


class SessionLoadResponse(BaseModel):
    token: str
    expires_at: datetime
    data: dict


class ExtendResponse(BaseModel):
    expires_at: datetime


class UploadErgebnisModel(BaseModel):
    """Rückmeldung zu einer hochgeladenen Datei."""

    dateiname: str
    erfolgreich: bool
    abschnitte: int = 0
    hinweis: str = ""


class UploadResponse(BaseModel):
    ergebnisse: List[UploadErgebnisModel]
    dokumente: List[str]


class ChatRequest(BaseModel):
    """Eine Frage an den Assistenten.

    Entweder ``aktion`` (eine der vorbereiteten Aufgaben) oder ``frage``
    (freie Eingabe). Die Perspektive bestimmt, ob der Widerspruch in der
    Ich-Form oder für eine andere Person geschrieben wird.
    """

    aktion: Optional[str] = None
    frage: Optional[str] = None
    perspektive: str = "selbst"
    versicherte_name: str = ""
    verhaeltnis: str = ""


class ActionModel(BaseModel):
    schluessel: str
    titel: str
    beschreibung: str
    nutzertext: str
    braucht_perspektive: bool


class StatusResponse(BaseModel):
    wissensbasis_abschnitte: int
    vektordatenbank: str
    neubewertung: str
    sprachmodell: str


class LetterRequest(BaseModel):
    """Alle Angaben für das Widerspruchsschreiben."""

    absender_name: str = ""
    absender_strasse: str = ""
    absender_plz_ort: str = ""
    ort: str = ""
    kasse_name: str = ""
    kasse_strasse: str = ""
    kasse_plz_ort: str = ""
    versichert_name: str = ""
    versichert_nr: str = ""
    aktenzeichen: str = ""
    bescheid_datum: str = ""
    begruendung: str = ""
    begruendung_folgt: bool = False
    perspektive: str = "selbst"
    verhaeltnis: str = ""
    # Beigefügte Unterlagen, eine je Zeile. Optional.
    anlagen: str = ""


class LetterCheckResponse(BaseModel):
    fehlende_angaben: List[str]
    offene_platzhalter: List[str]


def _brief_daten(angaben: LetterRequest) -> "pflege_pdf.LetterData":
    """Übersetzt die Anfrage in die Briefvorlage."""
    return pflege_pdf.LetterData(**angaben.model_dump())


# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
async def _cleanup_loop():
    """Löscht abgelaufene Sitzungen periodisch, auch wenn niemand mehr mit
    dem zugehörigen Token zurückkehrt (Anforderung: automatische Löschung
    nach 4 Wochen inkl. aller Daten)."""
    while True:
        try:
            db = SessionLocal()
            try:
                expired = db.query(SessionRecord).filter(SessionRecord.expires_at <= utcnow()).all()
                for record in expired:
                    db.delete(record)
                if expired:
                    db.commit()
                    db.close()
                    _rewrite_database_file()
                    print(f"🧹 {len(expired)} abgelaufene Sitzung(en) vollständig gelöscht.")
            finally:
                db.close()
        except Exception as e:
            print(f"Fehler bei der Bereinigung abgelaufener Sitzungen: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    cleanup_task.cancel()


app = FastAPI(title="Pflege-Assistent Session-API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Nur die Oberfläche auf diesem Rechner darf den Dienst ansprechen.
    # Die Streamlit-Oberfläche spricht den Dienst serverseitig an, für sie
    # spielt CORS keine Rolle; die Einstellung bleibt für den Fall, dass
    # wieder eine Oberfläche im Browser dazukommt.
    allow_origins=sorted({FRONTEND_ORIGIN, "http://127.0.0.1:8501"}),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
    expose_headers=["Content-Disposition"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _encrypt(data: dict) -> bytes:
    return fernet.encrypt(json.dumps(data).encode("utf-8"))


def _decrypt(blob: Optional[bytes]) -> dict:
    if not blob:
        return {}
    try:
        return json.loads(fernet.decrypt(blob).decode("utf-8"))
    except InvalidToken:
        # Blob wurde mit einem anderen ENCRYPTION_KEY verschlüsselt (z. B. nach
        # Neustart ohne festen Key) -> Daten gelten als nicht wiederherstellbar.
        return {}


def _get_valid_record(db: DbSession, token: str) -> SessionRecord:
    record = db.query(SessionRecord).filter(SessionRecord.token == token).first()
    if not record:
        raise HTTPException(status_code=404, detail="Token unbekannt.")
    if record.expires_at <= utcnow():
        db.delete(record)
        db.commit()
        _rewrite_database_file()
        raise HTTPException(
            status_code=410,
            detail="Zugangscode abgelaufen. Die zugehörigen Daten wurden vollständig gelöscht.",
        )
    return record


# ------------------------------------------------------------
# ENDPUNKTE
# ------------------------------------------------------------
MAX_TOKEN_VERSUCHE = 8


@app.post("/session", response_model=SessionCreateResponse)
def create_session(db: DbSession = Depends(get_db)):
    """Legt eine neue, leere Sitzung ohne jede Registrierung an.

    Der Zugangscode wird garantiert nicht doppelt vergeben: Die Spalte ist
    Primärschlüssel, zusätzlich wird vor dem Einfügen geprüft und ein
    gleichzeitig entstandener Doppelgänger über die Ausnahme abgefangen.
    Erst wenn eine Sitzung gelöscht ist, könnte ihr Code theoretisch wieder
    vergeben werden.
    """
    now = utcnow()
    for _ in range(MAX_TOKEN_VERSUCHE):
        token = secrets.token_urlsafe(32)
        if db.query(SessionRecord).filter(SessionRecord.token == token).first() is not None:
            continue
        record = SessionRecord(
            token=token,
            created_at=now,
            expires_at=now + timedelta(days=SESSION_LIFETIME_DAYS),
            last_accessed_at=now,
            data_encrypted=_encrypt({}),
        )
        db.add(record)
        try:
            db.commit()
        except IntegrityError:
            # Zwei Anfragen haben zeitgleich denselben Code erzeugt.
            db.rollback()
            continue
        return SessionCreateResponse(token=token, expires_at=record.expires_at)

    raise HTTPException(
        status_code=503,
        detail="Es konnte kein eindeutiger Zugangscode erzeugt werden. Bitte erneut versuchen.",
    )


@app.get("/session/{token}", response_model=SessionLoadResponse)
def load_session(token: str, db: DbSession = Depends(get_db)):
    record = _get_valid_record(db, token)
    record.last_accessed_at = utcnow()
    db.commit()
    return SessionLoadResponse(token=record.token, expires_at=record.expires_at, data=_decrypt(record.data_encrypted))


@app.put("/session/{token}", response_model=SessionCreateResponse)
def sync_session(token: str, payload: SessionSyncRequest, db: DbSession = Depends(get_db)):
    """Überschreibt den gespeicherten Stand vollständig mit dem aktuellen
    Frontend-Zustand (Chat, Dokumentabschnitte, Widerspruchstext, Formulardaten).
    Verlängert die Ablauffrist NICHT - das passiert ausschließlich über /extend."""
    record = _get_valid_record(db, token)
    record.data_encrypted = _encrypt(payload.data)
    record.last_accessed_at = utcnow()
    db.commit()
    return SessionCreateResponse(token=record.token, expires_at=record.expires_at)


@app.post("/session/{token}/extend", response_model=ExtendResponse)
def extend_session(token: str, db: DbSession = Depends(get_db)):
    """Verlängert eine bestehende, noch gültige Sitzung um SESSION_EXTEND_DAYS
    (Standard: 3 Tage), ausgehend vom bisherigen Ablaufdatum."""
    record = _get_valid_record(db, token)
    record.expires_at = record.expires_at + timedelta(days=SESSION_EXTEND_DAYS)
    db.commit()
    return ExtendResponse(expires_at=record.expires_at)


@app.delete("/session/{token}")
def delete_session(token: str, db: DbSession = Depends(get_db)):
    """Löscht Zugangscode und alle zugehörigen Daten sofort und unwiderruflich."""
    record = db.query(SessionRecord).filter(SessionRecord.token == token).first()
    if record:
        db.delete(record)
        db.commit()
        db.close()
        # Erst das Neuschreiben der Datei entfernt die Inhalte endgültig.
        _rewrite_database_file()
    # Auch den Suchindex im Arbeitsspeicher freigeben.
    pflege_service.user_indices.entferne(token)
    return {"message": "Sitzung und alle zugehörigen Daten wurden vollständig gelöscht."}


# ---------------------------------------------------------------------------
# UNTERLAGEN
# ---------------------------------------------------------------------------
def _dokumente_aus_sitzung(daten: dict) -> List[Document]:
    return [
        Document(page_content=e.get("page_content", ""), metadata=e.get("metadata", {}))
        for e in daten.get("user_documents", []) or []
    ]


def _dokumente_speichern(daten: dict, dokumente: Sequence[Document]) -> None:
    daten["user_documents"] = [
        {"page_content": d.page_content, "metadata": d.metadata} for d in dokumente
    ]
    daten["document_names"] = sorted(
        {d.metadata.get("source", "") for d in dokumente if d.metadata.get("source")}
    )


@app.post("/session/{token}/documents", response_model=UploadResponse)
async def upload_documents(
    token: str, files: List[UploadFile] = File(...), db: DbSession = Depends(get_db)
):
    """Nimmt PDF-Dateien entgegen, liest sie ein und legt sie in der Sitzung ab.

    Die Verarbeitung geschieht hier und nicht in der Oberfläche: Damit kommt
    jede Oberfläche mit demselben Verhalten aus, und die Dateien müssen nicht
    für jede Anfrage erneut übertragen werden.
    """
    record = _get_valid_record(db, token)
    daten = _decrypt(record.data_encrypted)
    vorhandene = _dokumente_aus_sitzung(daten)
    bekannte_namen = {d.metadata.get("source") for d in vorhandene}

    ergebnisse: List[UploadErgebnisModel] = []
    for datei in files:
        if datei.filename in bekannte_namen:
            ergebnisse.append(UploadErgebnisModel(
                dateiname=datei.filename, erfolgreich=False,
                hinweis="Diese Datei ist bereits eingelesen."))
            continue

        inhalt = await datei.read()
        if len(inhalt) > MAX_UPLOAD_BYTES:
            ergebnisse.append(UploadErgebnisModel(
                dateiname=datei.filename, erfolgreich=False,
                hinweis=f"Die Datei ist größer als {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."))
            continue

        abschnitte, ergebnis = pflege_service.verarbeite_pdf(inhalt, datei.filename)
        vorhandene.extend(abschnitte)
        bekannte_namen.add(datei.filename)
        ergebnisse.append(UploadErgebnisModel(**ergebnis.__dict__))

    _dokumente_speichern(daten, vorhandene)
    record.data_encrypted = _encrypt(daten)
    record.last_accessed_at = utcnow()
    db.commit()
    pflege_service.user_indices.entferne(token)  # Index wird bei der nächsten Frage neu gebaut
    return UploadResponse(ergebnisse=ergebnisse, dokumente=daten["document_names"])


@app.delete("/session/{token}/documents/{dateiname}")
def delete_document(token: str, dateiname: str, db: DbSession = Depends(get_db)):
    """Entfernt eine einzelne Unterlage aus der Sitzung."""
    record = _get_valid_record(db, token)
    daten = _decrypt(record.data_encrypted)
    verbleibend = [
        d for d in _dokumente_aus_sitzung(daten) if d.metadata.get("source") != dateiname
    ]
    _dokumente_speichern(daten, verbleibend)
    record.data_encrypted = _encrypt(daten)
    db.commit()
    pflege_service.user_indices.entferne(token)
    return {"dokumente": daten["document_names"]}


# ---------------------------------------------------------------------------
# ASSISTENT
# ---------------------------------------------------------------------------
@app.get("/actions", response_model=List[ActionModel])
def list_actions():
    """Die vorbereiteten Aufgaben, damit jede Oberfläche dieselben anbietet."""
    return [
        ActionModel(
            schluessel=a.schluessel, titel=a.titel, beschreibung=a.beschreibung,
            nutzertext=a.nutzertext, braucht_perspektive=a.braucht_perspektive,
        )
        for a in pflege_rag.QUICK_ACTIONS
    ]


@app.get("/status", response_model=StatusResponse)
def status():
    """Betriebszustand - hilfreich bei der Einrichtung."""
    return StatusResponse(
        wissensbasis_abschnitte=pflege_service.ressourcen.wissensbasis_umfang(),
        vektordatenbank=pflege_rag.qdrant_betriebsart(),
        neubewertung=pflege_rag.reranker_backend(),
        sprachmodell=pflege_rag.LLM_MODEL,
    )


@app.post("/session/{token}/letter/validate", response_model=LetterCheckResponse)
def validate_letter(token: str, angaben: LetterRequest, db: DbSession = Depends(get_db)):
    """Prüft die Briefangaben, ohne das PDF zu erzeugen.

    So kann die Oberfläche schon beim Ausfüllen zeigen, was noch fehlt,
    statt erst beim Erzeugen abzubrechen.
    """
    _get_valid_record(db, token)
    daten = _brief_daten(angaben)
    return LetterCheckResponse(
        fehlende_angaben=pflege_pdf.validate(daten),
        offene_platzhalter=pflege_pdf.find_placeholders(daten.begruendung),
    )


@app.post("/session/{token}/letter")
def build_letter(token: str, angaben: LetterRequest, db: DbSession = Depends(get_db)):
    """Erzeugt das Widerspruchsschreiben als PDF."""
    record = _get_valid_record(db, token)
    daten = _brief_daten(angaben)
    fehlend = pflege_pdf.validate(daten)
    if fehlend:
        raise HTTPException(status_code=400, detail="; ".join(fehlend))

    # Die Angaben mitspeichern, damit sie beim Wiedereinstieg wieder da sind.
    stand = _decrypt(record.data_encrypted)
    stand["brief"] = angaben.model_dump()
    record.data_encrypted = _encrypt(stand)
    record.last_accessed_at = utcnow()
    db.commit()

    return Response(
        content=pflege_pdf.build_letter_pdf(daten),
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="Widerspruch_Pflegegrad.pdf"'},
    )


@app.post("/session/{token}/chat")
def chat(token: str, anfrage: ChatRequest, db: DbSession = Depends(get_db)):
    """Beantwortet eine Frage und sendet den Fortschritt fortlaufend.

    Die Antwort kommt als Ereignisstrom (Server-Sent Events), damit die
    Oberfläche den Text beim Entstehen anzeigen kann statt am Ende auf einen
    Block zu warten.
    """
    record = _get_valid_record(db, token)
    daten = _decrypt(record.data_encrypted)
    dokumente = _dokumente_aus_sitzung(daten)
    verlauf = list(daten.get("messages", []) or [])

    aktion = pflege_rag.QUICK_ACTION_BY_KEY.get(anfrage.aktion or "")
    person = pflege_rag.Antragsteller(
        perspektive=anfrage.perspektive,
        versicherte_name=anfrage.versicherte_name,
        verhaeltnis=anfrage.verhaeltnis,
    )
    if aktion is not None:
        anzeige, anweisung = aktion.nutzertext, aktion.render(person)
        zusatzfragen = aktion.zusatzfragen
    else:
        anzeige = anweisung = (anfrage.frage or "").strip()
        zusatzfragen = ()
    if not anzeige:
        raise HTTPException(status_code=400, detail="Es wurde keine Frage übergeben.")

    def ereignisse():
        antwort, quellen = "", []
        try:
            for meldung in pflege_service.beantworte(
                token, anzeige, anweisung, verlauf, dokumente, zusatzfragen
            ):
                if meldung["art"] == "ergebnis":
                    antwort, quellen = meldung["antwort"], meldung["quellen"]
                yield f"data: {json.dumps(meldung, ensure_ascii=False)}\n\n"
        except Exception as fehler:
            # Ohne diese Meldung endet der Strom stillschweigend und die
            # Oberfläche wartet endlos auf eine Antwort, die nie kommt.
            # Der Wortlaut des Fehlers bleibt im Dienst: er könnte Auszüge aus
            # den Unterlagen enthalten.
            print(f"❌ Fehler bei der Beantwortung: {type(fehler).__name__}: {fehler}")
            traceback.print_exc()
            hinweis = {
                "art": "fehler",
                "text": "Die Antwort konnte nicht erzeugt werden. Bitte versuchen Sie es "
                        "noch einmal. Bleibt es dabei, hilft ein Blick in das Fenster des "
                        "Hintergrunddienstes.",
            }
            yield f"data: {json.dumps(hinweis, ensure_ascii=False)}\n\n"
            return

        # Erst nach dem vollständigen Durchlauf speichern, mit eigener
        # Datenbanksitzung: die des Aufrufs ist zu diesem Zeitpunkt geschlossen.
        if antwort:
            eigene = SessionLocal()
            try:
                eintrag = eigene.query(SessionRecord).filter(SessionRecord.token == token).first()
                if eintrag is not None:
                    stand = _decrypt(eintrag.data_encrypted)
                    nachrichten = list(stand.get("messages", []) or [])
                    nachrichten.append({"role": "user", "content": anzeige})
                    nachrichten.append({"role": "assistant", "content": antwort})
                    stand["messages"] = nachrichten
                    stand["last_sources"] = quellen
                    if aktion is not None and aktion.schluessel == "schreiben":
                        stand["last_generated_letter"] = antwort
                        # Zusätzlich die briefreife Fassung ablegen: ohne Anrede,
                        # Grußformel und Belegziffern. Die Oberfläche übernimmt
                        # sie unverändert, damit dort nichts Ungeputztes steht.
                        stand["letter_draft"] = pflege_pdf.prepare_begruendung(antwort)
                    eintrag.data_encrypted = _encrypt(stand)
                    eintrag.last_accessed_at = utcnow()
                    eigene.commit()
            finally:
                eigene.close()

    return StreamingResponse(
        ereignisse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    # Bewusst nur auf der örtlichen Schnittstelle: Die Sitzungsdaten dürfen
    # nicht über das Netzwerk erreichbar sein.
    uvicorn.run(app, host="127.0.0.1", port=8000)
