import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_PATH = "backend/clinical_audit.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CaseAudit(Base):
    __tablename__ = "case_audit"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    patient_id = Column(String, index=True)
    filename = Column(String)
    age = Column(Float)
    gender = Column(String)
    temperature = Column(Float)
    spo2 = Column(Integer)
    heart_rate = Column(Integer)
    cough_severity = Column(String)
    
    # Model Outputs
    prediction_class = Column(Integer)  # 0 = Normal, 1 = Pneumonia
    raw_probability = Column(Float)
    calibrated_probability = Column(Float)
    uncertainty = Column(Float)
    confidence_category = Column(String)
    
    # Reliability Outputs
    is_ood = Column(Boolean)
    ood_distance = Column(Float)
    ood_risk = Column(String)
    
    # Clinician Interaction / Audit Log
    clinician_label = Column(Integer, nullable=True) # Clinician override: 0 or 1
    override_reason = Column(String, nullable=True)
    override_timestamp = Column(DateTime, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
