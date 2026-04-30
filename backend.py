from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
import random

# --- 1. SICHERHEITS-KONFIGURATION ---
SECRET_KEY = "euer_super_geheimer_schluessel_fuer_das_projekt"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- 2. DATENBANK-SETUP ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./pflege_sicher.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)  # NEU: E-Mail Feld
    hashed_password = Column(String)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)  # NEU: E-Mail Verifizierung
    verification_code = Column(String)  # NEU: Der 6-stellige Code


Base.metadata.create_all(bind=engine)


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class VerifyCode(BaseModel):
    username: str
    code: str


class Token(BaseModel):
    access_token: str
    token_type: str


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
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ungültiges Token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None: raise credentials_exception
    return user


# --- 3. DIE API-ENDPUNKTE ---
app = FastAPI(title="Pflege-KI Backend UX-Update")


@app.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == user.username).first():
        raise HTTPException(status_code=400, detail="Benutzername bereits vergeben.")
    if db.query(UserDB).filter(UserDB.email == user.email).first():
        raise HTTPException(status_code=400, detail="Diese E-Mail ist bereits registriert.")

    # 6-stelligen Code generieren
    v_code = str(random.randint(100000, 999999))

    hashed_pw = get_password_hash(user.password)
    new_user = UserDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_pw,
        verification_code=v_code
    )
    db.add(new_user)
    db.commit()

    # E-MAIL SIMULATION (Wird im PyCharm Terminal angezeigt)
    print("\n" + "=" * 50)
    print(f"📧 SIMULIERTE E-MAIL AN: {user.email}")
    print(f"Betreff: Ihr Bestätigungscode für den Pflege-Assistenten")
    print(f"Hallo {user.username},\nIhr 6-stelliger Code lautet: {v_code}")
    print("=" * 50 + "\n")

    return {"message": "Code generiert. Bitte überprüfen Sie Ihr Postfach."}


@app.post("/verify")
def verify_user(data: VerifyCode, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data.username).first()
    if not user:
        raise HTTPException(status_code=404, detail="Benutzer nicht gefunden.")
    if user.verification_code != data.code:
        raise HTTPException(status_code=400, detail="Der eingegebene Code ist falsch.")

    user.is_verified = True
    db.commit()
    return {"message": "E-Mail erfolgreich bestätigt. Sie können sich nun einloggen."}


@app.post("/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # UX Feature: Login mit E-Mail ODER Username
    if "@" in form_data.username:
        user = db.query(UserDB).filter(UserDB.email == form_data.username).first()
    else:
        user = db.query(UserDB).filter(UserDB.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="E-Mail/Name oder Passwort falsch.")

    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Bitte bestätigen Sie zuerst Ihre E-Mail-Adresse mit dem Code.")

    if datetime.utcnow() > user.created_at + timedelta(days=28):
        db.delete(user)
        db.commit()
        raise HTTPException(status_code=403, detail="Account wurde gemäß DSGVO gelöscht.")

    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}


@app.delete("/users/me")
def delete_user_account(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(current_user)
    db.commit()
    return {"message": "Account gelöscht."}


@app.post("/users/extend")
def extend_account_lifetime(current_user: UserDB = Depends(get_current_user), db: Session = Depends(get_db)):
    current_user.created_at = current_user.created_at + timedelta(days=7)
    db.commit()
    return {"message": "Frist verlängert."}