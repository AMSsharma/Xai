import os
import sys
import json
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.database import CaseAudit, DB_PATH

def main():
    print("=== Upgraded Human-in-the-Loop Disagreement Analysis ===")
    
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file {DB_PATH} not found. Run generate_synthetic_audit_data first.")
        return
        
    engine = create_engine(f"sqlite:///{DB_PATH}")
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    total_cases = db.query(CaseAudit).count()
    if total_cases == 0:
        print("Error: No cases in database.")
        db.close()
        return
        
    # Overrides are cases where clinician_label is set (not null)
    overrides = db.query(CaseAudit).filter(CaseAudit.clinician_label != None).count()
    agreements = total_cases - overrides
    
    agreement_rate = float(agreements / total_cases)
    override_rate = float(overrides / total_cases)
    
    # False Positive overrides: model predicted 1, clinician corrected to 0
    fp_overrides = db.query(CaseAudit).filter(CaseAudit.prediction_class == 1, CaseAudit.clinician_label == 0).count()
    # False Negative overrides: model predicted 0, clinician corrected to 1
    fn_overrides = db.query(CaseAudit).filter(CaseAudit.prediction_class == 0, CaseAudit.clinician_label == 1).count()
    
    # OOD overrides
    ood_overrides = db.query(CaseAudit).filter(CaseAudit.is_ood == True, CaseAudit.clinician_label != None).count()
    total_ood = db.query(CaseAudit).filter(CaseAudit.is_ood == True).count()
    ood_override_rate = float(ood_overrides / total_ood) if total_ood > 0 else 0.0
    
    # High-uncertainty overrides
    high_unc_overrides = db.query(CaseAudit).filter(CaseAudit.uncertainty > 0.15, CaseAudit.clinician_label != None).count()
    total_high_unc = db.query(CaseAudit).filter(CaseAudit.uncertainty > 0.15).count()
    high_unc_override_rate = float(high_unc_overrides / total_high_unc) if total_high_unc > 0 else 0.0
    
    # Borderline cases overrides (probability between 0.10 and 0.40)
    borderline_overrides = db.query(CaseAudit).filter(
        CaseAudit.calibrated_probability >= 0.10,
        CaseAudit.calibrated_probability <= 0.40,
        CaseAudit.clinician_label != None
    ).count()
    total_borderline = db.query(CaseAudit).filter(
        CaseAudit.calibrated_probability >= 0.10,
        CaseAudit.calibrated_probability <= 0.40
    ).count()
    borderline_override_rate = float(borderline_overrides / total_borderline) if total_borderline > 0 else 0.0
    
    records = [
        {"Metric Name": "Total Audited Cases", "Count": total_cases, "Percentage / Rate": 1.0, "Details": "All cases logged in clinical_audit.db"},
        {"Metric Name": "Implicit Agreements", "Count": agreements, "Percentage / Rate": agreement_rate, "Details": "Model predictions accepted by clinician"},
        {"Metric Name": "Clinician Overrides (Disagreements)", "Count": overrides, "Percentage / Rate": override_rate, "Details": "Model predictions changed by clinician"},
        {"Metric Name": "False Positive Overrides", "Count": fp_overrides, "Percentage / Rate": float(fp_overrides / overrides) if overrides > 0 else 0.0, "Details": "Model predicted positive, clinician corrected to negative"},
        {"Metric Name": "False Negative Overrides", "Count": fn_overrides, "Percentage / Rate": float(fn_overrides / overrides) if overrides > 0 else 0.0, "Details": "Model predicted negative, clinician corrected to positive"},
        {"Metric Name": "OOD-related Overrides", "Count": ood_overrides, "Percentage / Rate": ood_override_rate, "Details": "Overrides out of all OOD cases"},
        {"Metric Name": "High-Uncertainty Overrides", "Count": high_unc_overrides, "Percentage / Rate": high_unc_override_rate, "Details": "Overrides out of all high uncertainty cases"},
        {"Metric Name": "Borderline Cases Overrides (0.10-0.40)", "Count": borderline_overrides, "Percentage / Rate": borderline_override_rate, "Details": "Overrides out of all borderline cases"}
    ]
    
    df_out = pd.DataFrame(records)
    out_path = "reports/human_model_agreement.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved human-model disagreement analysis to {out_path}")
    print("\nSimulated Oversight Metrics (SIMULATION ONLY):")
    print(df_out.to_string(index=False))
    
    db.close()

if __name__ == "__main__":
    main()
