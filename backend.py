from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import bcrypt
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
import random

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SECRET_KEY = "dein_geheimnis_fuer_den_pflegeraht"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

SQLALCHEMY_DATABASE_URL = "sqlite:///./pflege_sicher.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True)
    hashed_password = Column(String)
    is_verified = Column(Boolean, default=False)
    verification_code = Column(String)


Base.metadata.create_all(bind=engine)


def get_password_hash(password: str):
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=60)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str


@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == user.username).first():
        raise HTTPException(400, "Name vergeben")

    code = str(random.randint(100000, 999999))
    hashed = get_password_hash(user.password)

    new_user = UserDB(username=user.username, email=user.email, hashed_password=hashed, verification_code=code)
    db.add(new_user)
    db.commit()

    print(f"\n--- CODE FÜR {user.username}: {code} ---\n")
    return {"msg": "Registriert"}


@app.post("/verify")
def verify(data: dict, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == data.get("username")).first()
    if user and user.verification_code == data.get("code"):
        user.is_verified = True
        db.commit()
        return {"msg": "OK"}
    raise HTTPException(400, "Falscher Code")


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(401, "Falsche Logindaten")
    if not user.is_verified:
        raise HTTPException(403, "Nicht verifiziert")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}