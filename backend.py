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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, LargeBinary, String, create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DbSession, declarative_base, sessionmaker

load_dotenv()

SESSION_LIFETIME_DAYS = int(os.getenv("SESSION_LIFETIME_DAYS", "28"))
SESSION_EXTEND_DAYS = int(os.getenv("SESSION_EXTEND_DAYS", "3"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "3600"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pflege_sicher.db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

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
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type"],
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
    return {"message": "Sitzung und alle zugehörigen Daten wurden vollständig gelöscht."}


if __name__ == "__main__":
    import uvicorn

    # Bewusst nur auf der örtlichen Schnittstelle: Die Sitzungsdaten dürfen
    # nicht über das Netzwerk erreichbar sein.
    uvicorn.run(app, host="127.0.0.1", port=8000)
