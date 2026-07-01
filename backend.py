from datetime import datetime, timedelta
import os
import random

import bcrypt
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_please_change")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
ACCOUNT_LIFETIME_DAYS = int(os.getenv("ACCOUNT_LIFETIME_DAYS", "28"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pflege_sicher.db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")

app = FastAPI(title="Pflege-Assistent Auth-API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
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
    expires_at = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(days=ACCOUNT_LIFETIME_DAYS))

Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class VerifyRequest(BaseModel):
    username: str
    code: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class MeResponse(BaseModel):
    username: str
    email: str
    is_verified: bool
    expires_at: datetime

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None: raise Exception
    except:
        raise HTTPException(status_code=401, detail="Authentifizierung fehlgeschlagen.")
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user: raise HTTPException(status_code=401, detail="User nicht gefunden.")
    if not user.is_verified: raise HTTPException(status_code=403, detail="Konto nicht verifiziert.")
    return user

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == user.username).first(): raise HTTPException(status_code=400, detail="Name vergeben.")
    code = str(random.randint(100000, 999999))
    new_user = UserDB(username=user.username, email=user.email, hashed_password=bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"), verification_code=code)
    db.add(new_user); db.commit()
    print(f"\n[DEMO] Code für {user.username}: {code}\n")
    return {"message": "Registrierung erfolgreich."}

@app.post("/verify")
def verify(data: VerifyRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data.username).first()
    if not user or user.verification_code != data.code: raise HTTPException(status_code=400, detail="Code falsch.")
    user.is_verified = True; user.verification_code = None; db.commit()
    return {"message": "Erfolgreich verifiziert."}

@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    if not user or not bcrypt.checkpw(form_data.password.encode("utf-8"), user.hashed_password.encode("utf-8")) or not user.is_verified:
        raise HTTPException(status_code=401, detail="Login fehlgeschlagen.")
    return {"access_token": jwt.encode({"sub": user.username, "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}, SECRET_KEY, algorithm=ALGORITHM), "token_type": "bearer"}

@app.get("/me", response_model=MeResponse)
def read_me(current_user: UserDB = Depends(get_current_user)):
    return current_user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)