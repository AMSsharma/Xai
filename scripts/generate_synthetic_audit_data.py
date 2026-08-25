import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import Base, CaseAudit, DB_PATH

def main():
    print("=== Seeding Database with Simulated human-in-the-loop Audit Trail ===")
    
    # Connect to database and drop/recreate table
    engine = create_engine(f"sqlite:///{DB_PATH}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    # Recreate tables to start clean
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Load splits to get realistic patient metadata
    train_df = pd.read_csv("data/processed/train_split.csv")
    test_df = pd.read_csv("data/processed/test_split.csv")
    
    combined_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    n_samples = len(combined_df)
    
    rng = np.random.default_rng(42)
    
    # Generate 100 simulated audits
    cases = []
    base_time = datetime.utcnow() - timedelta(days=10)
    
    override_reasons = {
        "FN": [
            "SIMULATED DEMO: Clinical SpO2 and HR shift indicated severe distress despite low radiographic probability.",
            "SIMULATED DEMO: Early consolidation suspected in visual assessment, overridden for patient safety.",
            "SIMULATED DEMO: Patient has underlying asthma comorbidity masking symptoms on radiography."
        ],
        "FP": [
            "SIMULATED DEMO: Radiograph artifact was mistaken for lung consolidation by visual backbone.",
            "SIMULATED DEMO: Patient is asymptomatic with normal clinical vitals, chest consolidations are chronic scarring.",
            "SIMULATED DEMO: Resolution decay caused model to overestimate consolidations."
        ],
        "agree": [
            "SIMULATED DEMO: Clinician confirmed clear lung consolidations.",
            "SIMULATED DEMO: Vitals and radiograph are fully consistent with clinical guidelines.",
            "SIMULATED DEMO: Clinician consensus matches negative model prediction."
        ]
    }
    
    for i in range(100):
        # Pick a random template row from the dataset
        row_idx = rng.integers(0, n_samples)
        row = combined_df.iloc[row_idx]
        
        # Simulate predictions
        # Make them mostly correct (75% accuracy) but inject uncertainty and anomalies
        actual_label = int(row['label'])
        
        # Generate raw probability with some noise
        if actual_label == 1:
            raw_prob = float(rng.uniform(0.3, 0.99))
        else:
            raw_prob = float(rng.uniform(0.01, 0.6))
            
        # Simulate MC Dropout uncertainty: higher for borderline probability
        uncertainty = float(0.15 * (1.0 - abs(raw_prob - 0.5) / 0.5) + rng.uniform(0.01, 0.05))
        if uncertainty > 0.15:
            conf_cat = "HIGH UNCERTAINTY"
        elif uncertainty > 0.08:
            conf_cat = "MODERATE UNCERTAINTY"
        else:
            conf_cat = "LOW UNCERTAINTY"
            
        # Calibrate (similar to temperature scaling)
        calibrated_prob = float(1.0 / (1.0 + np.exp(-(raw_prob - 0.5) / 0.15)))
        # Map to decision class based on optimal threshold (t = 0.17)
        pred_class = 1 if calibrated_prob >= 0.17 else 0
        
        # Simulate OOD status: 10% rate
        is_ood = bool(rng.uniform() < 0.1)
        if is_ood:
            ood_distance = float(rng.uniform(13.0, 35.0))
            ood_risk = "CRITICAL" if ood_distance > 19.0 else "HIGH"
        else:
            ood_distance = float(rng.uniform(1.0, 12.0))
            ood_risk = "LOW"
            
        # Clinician decision loop (Human-in-the-Loop)
        # Agreement rate: ~70%
        # Overrides: ~30% (highly concentrated in high uncertainty or OOD cases)
        override_rate_bias = 0.7 if (conf_cat == "HIGH UNCERTAINTY" or is_ood) else 0.15
        
        override = bool(rng.uniform() < override_rate_bias)
        
        if override:
            clinician_label = 1 - pred_class # flip decision
            override_time = base_time + timedelta(hours=int(rng.integers(1, 4)))
            
            # Select reason
            if pred_class == 1 and clinician_label == 0:
                reason = rng.choice(override_reasons["FP"])
            else:
                reason = rng.choice(override_reasons["FN"])
        else:
            clinician_label = pred_class
            reason = rng.choice(override_reasons["agree"])
            override_time = None
            
        timestamp = base_time + timedelta(hours=i * 2)
        
        case = CaseAudit(
            timestamp=timestamp,
            patient_id=row['patient_id'],
            filename=row['filename'],
            age=float(row['age']),
            gender=row['gender'],
            temperature=float(row['temperature']),
            spo2=int(row['spo2']),
            heart_rate=int(row['heart_rate']),
            cough_severity=row['cough_severity'],
            prediction_class=pred_class,
            raw_probability=raw_prob,
            calibrated_probability=calibrated_prob,
            uncertainty=uncertainty,
            confidence_category=conf_cat,
            is_ood=is_ood,
            ood_distance=ood_distance,
            ood_risk=ood_risk,
            clinician_label=clinician_label if override else None,
            override_reason=reason if override else None,
            override_timestamp=override_time
        )
        cases.append(case)
        
    db.add_all(cases)
    db.commit()
    db.close()
    
    print("Database successfully seeded with 100 CaseAudit records!")
    print("Simulated Oversight Log is ready for clinician evaluation.")

if __name__ == "__main__":
    main()
