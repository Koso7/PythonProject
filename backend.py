from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
import random
import os

# --- 1. KONFIGURATION ---
app = FastAPI(title="Pflege-KI Backend")

# CORS aktivieren, damit Streamlit (Port 8501) zugreifen darf
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "ein_sehr_geheimer_schluessel_fuer_die_tokens"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# bcrypt für Passwort-Hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- 2. DATENBANK-SETUP ---
# Die Datenbank-Datei wird im aktuellen Ordner erstellt
DB_PATH = "./pflege_sicher.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Tabellen-Modell
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String)


# Erstellt die Tabellen, falls sie noch nicht existieren
Base.metadata.create_all(bind=engine)


# --- 3. DATEN-MODELLE (Pydantic) ---
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


class VerifyCode(BaseModel):
    username: str
    code: str


class Token(BaseModel):
    access_token: str
    token_type: str


# --- 4. HILFSFUNKTIONEN ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Ungültiger Token")
        user = db.query(UserDB).filter(UserDB.username == username).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User nicht gefunden")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Token abgelaufen oder ungültig")


# --- 5. API-ENDPUNKTE ---

@app.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    # Prüfen, ob User schon existiert
    if db.query(UserDB).filter(UserDB.username == user.username).first():
        raise HTTPException(status_code=400, detail="Benutzername bereits vergeben")
    if db.query(UserDB).filter(UserDB.email == user.email).first():
        raise HTTPException(status_code=400, detail="E-Mail bereits registriert")

    v_code = str(random.randint(100000, 999999))
    new_user = UserDB(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        verification_code=v_code
    )

    try:
        db.add(new_user)
        db.commit()
        # Den Code im Terminal ausgeben (statt E-Mail-Versand)
        print(f"\n{'=' * 30}")
        print(f"VERIFIZIERUNGS-CODE FÜR {user.username}: {v_code}")
        print(f"{'=' * 30}\n")
        return {"message": "Registrierung erfolgreich, Code wurde generiert."}
    except Exception as e:
        db.rollback()
        print(f"Datenbankfehler: {e}")
        raise HTTPException(status_code=500, detail="Fehler beim Speichern in der Datenbank.")


@app.post("/verify")
def verify_user(data: VerifyCode, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden")
    if user.verification_code != data.code:
        raise HTTPException(status_code=400, detail="Der eingegebene Code ist falsch.")

    user.is_verified = True
    db.commit()
    return {"message": "E-Mail erfolgreich verifiziert!"}


@app.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Benutzername oder Passwort falsch")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Bitte verifizieren Sie zuerst Ihre E-Mail.")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/extend")
def extend_account(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    # Logik für Fristverlängerung (z.B. created_at anpassen)
    current_user.created_at = datetime.now(timezone.utc)
    db.commit()
    return {"message": "Account-Frist erfolgreich verlängert."}


@app.delete("/users/me")
def delete_user_account(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(current_user)
    db.commit()
    return {"message": "Account und alle Daten wurden erfolgreich gelöscht."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)