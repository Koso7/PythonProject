from datetime import datetime, timedelta
import os
import random

import bcrypt
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ------------------------------------------------------------
# KONFIGURATION
# ------------------------------------------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_please_change")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ACCOUNT_LIFETIME_DAYS = int(os.getenv("ACCOUNT_LIFETIME_DAYS", "28"))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pflege_sicher.db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")

# ------------------------------------------------------------
# APP
# ------------------------------------------------------------
app = FastAPI(
    title="Pflege-Assistent API",
    description="Backend für Registrierung, Login und Authentifizierung.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ------------------------------------------------------------
# DATENBANK
# ------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    is_verified = Column(Boolean, default=False)
    verification_code = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.utcnow())
    expires_at = Column(
        DateTime,
        default=lambda: datetime.utcnow() + timedelta(days=ACCOUNT_LIFETIME_DAYS),
    )


Base.metadata.create_all(bind=engine)


# ------------------------------------------------------------
# SCHEMAS
# ------------------------------------------------------------
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class VerifyRequest(BaseModel):
    username: str
    code: str = Field(min_length=6, max_length=6)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class MeResponse(BaseModel):
    username: str
    email: EmailStr
    is_verified: bool
    expires_at: datetime


# ------------------------------------------------------------
# HILFSFUNKTIONEN
# ------------------------------------------------------------
def now_utc() -> datetime:
    return datetime.utcnow()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def purge_expired_users(db: Session):
    expired_users = db.query(UserDB).filter(UserDB.expires_at < now_utc()).all()

    for user in expired_users:
        db.delete(user)

    if expired_users:
        db.commit()


def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = now_utc() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentifizierung fehlgeschlagen.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    purge_expired_users(db)

    user = db.query(UserDB).filter(UserDB.username == username).first()

    if user is None:
        raise credentials_exception

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Nutzerkonto ist noch nicht verifiziert.",
        )

    if user.expires_at < now_utc():
        db.delete(user)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nutzerkonto ist abgelaufen.",
        )

    return user


# ------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Pflege-Assistent API",
    }


@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    purge_expired_users(db)

    existing_username = db.query(UserDB).filter(UserDB.username == user.username).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Der Nutzername ist bereits vergeben.",
        )

    existing_email = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Diese E-Mail-Adresse ist bereits registriert.",
        )

    verification_code = str(random.randint(100000, 999999))
    hashed_password = get_password_hash(user.password)

    new_user = UserDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        verification_code=verification_code,
        is_verified=False,
        created_at=now_utc(),
        expires_at=now_utc() + timedelta(days=ACCOUNT_LIFETIME_DAYS),
    )

    db.add(new_user)
    db.commit()

    print("\n" + "=" * 60)
    print(f"VERIFIZIERUNGSCODE FÜR {user.username}: {verification_code}")
    print("=" * 60 + "\n")

    return {
        "message": "Registrierung erfolgreich. Bitte Verifizierungscode eingeben.",
        "demo_hint": "Im Prototyp steht der Code im Backend-Terminal.",
    }


@app.post("/verify")
def verify(data: VerifyRequest, db: Session = Depends(get_db)):
    purge_expired_users(db)

    user = db.query(UserDB).filter(UserDB.username == data.username).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nutzer nicht gefunden.",
        )

    if user.expires_at < now_utc():
        db.delete(user)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dieses Konto ist abgelaufen.",
        )

    if user.verification_code != data.code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Der Verifizierungscode ist falsch.",
        )

    user.is_verified = True
    user.verification_code = None
    db.commit()

    return {
        "message": "Konto erfolgreich verifiziert. Sie können sich jetzt einloggen.",
    }


@app.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    purge_expired_users(db)

    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nutzername oder Passwort ist falsch.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user.expires_at < now_utc():
        db.delete(user)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Dieses Konto ist abgelaufen.",
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Das Konto ist noch nicht verifiziert.",
        )

    token = create_access_token({"sub": user.username})

    return {
        "access_token": token,
        "token_type": "bearer",
    }


@app.get("/me", response_model=MeResponse)
def read_me(current_user: UserDB = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "email": current_user.email,
        "is_verified": current_user.is_verified,
        "expires_at": current_user.expires_at,
    }