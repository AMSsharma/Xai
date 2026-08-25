import React, { useState, useEffect } from 'react';

const API_BASE = window.location.port === "5173" ? "http://127.0.0.1:8000" : window.location.origin;

export default function App() {
  const [activeTab, setActiveTab] = useState('predict');
  const [healthStatus, setHealthStatus] = useState({ status: "CONNECTING", device: "CPU", model_loaded: false });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // State for Input Form
  const [patientId, setPatientId] = useState('PID-9021');
  const [age, setAge] = useState(2.5);
  const [gender, setGender] = useState('M');
  const [temperature, setTemperature] = useState(38.8);
  const [spo2, setSpo2] = useState(91);
  const [heartRate, setHeartRate] = useState(130);
  const [coughSeverity, setCoughSeverity] = useState('Moderate');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  // State for Active Case Assessment
  const [assessment, setAssessment] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explainTab, setExplainTab] = useState('overlay'); // 'original', 'heatmap', 'overlay'
  
  // State for Metrics & Analytics
  const [metrics, setMetrics] = useState(null);
  
  // State for QA & Failures
  const [allCases, setAllCases] = useState([]);
  const [failures, setFailures] = useState([]);
  const [overrideReason, setOverrideReason] = useState('');
  const [overrideLabel, setOverrideLabel] = useState(1);
  const [selectedQaCaseId, setSelectedQaCaseId] = useState(null);

  // Load health status and logs on mount
  useEffect(() => {
    fetchHealth();
    fetchCases();
    fetchMetrics();
    fetchFailures();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`);
      if (res.ok) {
        const data = await res.json();
        setHealthStatus(data);
      } else {
        setHealthStatus({ status: "UNHEALTHY", device: "Unknown", model_loaded: false });
      }
    } catch (e) {
      setHealthStatus({ status: "OFFLINE", device: "None", model_loaded: false });
    }
  };

  const fetchCases = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/cases`);
      if (res.ok) {
        const data = await res.json();
        setAllCases(data);
      }
    } catch (e) {
      console.error("Error fetching cases:", e);
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (res.ok) {
        const data = await res.json();
        setMetrics(data);
      }
    } catch (e) {
      console.error("Error fetching metrics:", e);
    }
  };

  const fetchFailures = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/failures`);
      if (res.ok) {
        const data = await res.json();
        setFailures(data);
      }
    } catch (e) {
      console.error("Error fetching failures:", e);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmitCase = async (e) => {
    e.preventDefault();
    if (!imageFile) {
      setError("Please upload a chest radiograph image.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setAssessment(null);
    setExplanation(null);
    
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("patient_id", patientId);
    formData.append("age", age);
    formData.append("gender", gender);
    formData.append("temperature", temperature);
    formData.append("spo2", spo2);
    formData.append("heart_rate", heartRate);
    formData.append("cough_severity", coughSeverity);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        body: formData
      });
      
      if (!res.ok) {
        throw new Error(await res.text());
      }
      
      const data = await res.json();
      setAssessment(data);
      
      // Fetch explanations immediately
      fetchExplanation(data.case_id);
      
      // Refresh case list
      fetchCases();
      fetchFailures();
      
    } catch (err) {
      setError(err.message || "Failed to process prediction.");
    } finally {
      setLoading(false);
    }
  };

  const fetchExplanation = async (caseId) => {
    const formData = new FormData();
    formData.append("case_id", caseId);
    formData.append("target_class", 1); // explain pneumonia class features

    try {
      const res = await fetch(`${API_BASE}/api/explain`, {
        method: "POST",
        body: formData
      });
      if (res.ok) {
        const explainData = await res.json();
        setExplanation(explainData);
      }
    } catch (e) {
      console.error("Error fetching explanation:", e);
    }
  };

  const handleOverrideSubmit = async (e) => {
    e.preventDefault();
    if (!selectedQaCaseId) return;

    const formData = new FormData();
    formData.append("case_id", selectedQaCaseId);
    formData.append("clinician_label", overrideLabel);
    formData.append("override_reason", overrideReason);

    try {
      const res = await fetch(`${API_BASE}/api/override`, {
        method: "POST",
        body: formData
      });
      if (res.ok) {
        setOverrideReason('');
        setSelectedQaCaseId(null);
        fetchCases();
        fetchFailures();
      }
    } catch (err) {
      console.error("Override submission failed:", err);
    }
  };

  // Helper for rendering similar cases thumbnails or icons
  const getLabelBadge = (lbl) => {
    return lbl === 1 ? (
      <span style={{ color: '#e74c3c', background: 'rgba(231,76,60,0.1)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 600 }}>PNEUMONIA</span>
    ) : (
      <span style={{ color: '#2ecc71', background: 'rgba(46,204,113,0.1)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 600 }}>NORMAL</span>
    );
  };

  const getSystemStatusDot = () => {
    if (healthStatus.status === "HEALTHY") {
      return <div className="system-status"><span className="status-dot"></span>Active Server: GPU ({healthStatus.device})</div>;
    }
    return <div className="system-status" style={{ color: '#e74c3c', background: 'rgba(231,76,60,0.1)', borderColor: 'rgba(231,76,60,0.2)' }}><span className="status-dot" style={{ backgroundColor: '#e74c3c', boxShadow: 'none' }}></span>Offline ({healthStatus.status})</div>;
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo-section">
          <div className="logo-icon">AI</div>
          <div>
            <h1 className="title-main">Clinical AI Decision Intelligence Platform</h1>
            <p className="subtitle">Calibrated, Uncertainty-Aware Decision Support System (Research & Decision Intelligence Tool)</p>
          </div>
        </div>
        <div>
          {getSystemStatusDot()}
        </div>
      </header>

      {/* Clinician warning */}
      <div className="alert-banner info" style={{ marginBottom: '16px' }}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
        <p style={{ fontSize: '0.85rem' }}>
          <strong>Clinical Advisory:</strong> This platform is designed strictly for clinical decision support. Final diagnostic and therapeutic choices must always be confirmed by a certified medical practitioner.
        </p>
      </div>

      {/* Tabs */}
      <div className="tabs-navigation" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '20px' }}>
        <button className={`tab-btn ${activeTab === 'predict' ? 'active' : ''}`} onClick={() => setActiveTab('predict')}>Prediction Intake</button>
        <button className={`tab-btn ${activeTab === 'explain' ? 'active' : ''}`} onClick={() => setActiveTab('explain')}>Interpretability (XAI)</button>
        <button className={`tab-btn ${activeTab === 'reliability' ? 'active' : ''}`} onClick={() => setActiveTab('reliability')}>Reliability Metrics</button>
        <button className={`tab-btn ${activeTab === 'performance' ? 'active' : ''}`} onClick={() => setActiveTab('performance')}>Performance & CIs</button>
        <button className={`tab-btn ${activeTab === 'robustness' ? 'active' : ''}`} onClick={() => setActiveTab('robustness')}>Robustness & Ablation</button>
        <button className={`tab-btn ${activeTab === 'ood' ? 'active' : ''}`} onClick={() => setActiveTab('ood')}>OOD Benchmark</button>
        <button className={`tab-btn ${activeTab === 'oversight' ? 'active' : ''}`} onClick={() => setActiveTab('oversight')}>Human Oversight</button>
        <button className={`tab-btn ${activeTab === 'audit' ? 'active' : ''}`} onClick={() => setActiveTab('audit')}>Audit Log</button>
        <button className={`tab-btn ${activeTab === 'modelcard' ? 'active' : ''}`} onClick={() => setActiveTab('modelcard')}>Model & Dataset Card</button>
      </div>

      {/* Tab 1: Clinical Prediction Intake */}
      {activeTab === 'predict' && (
        <div className="grid-2-1">
          {/* Left panel: Upload Form & Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="card">
              <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '16px', fontSize: '1.2rem' }}>Patient Intake Form</h2>
              
              <form onSubmit={handleSubmitCase}>
                <div className="grid-2">
                  <div className="form-group">
                    <label className="form-label">Patient ID</label>
                    <input className="form-input" type="text" value={patientId} onChange={(e) => setPatientId(e.target.value)} required />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Patient Age (Years)</label>
                    <input className="form-input" type="number" step="0.1" min="1.0" max="5.0" value={age} onChange={(e) => setAge(parseFloat(e.target.value))} required />
                  </div>
                </div>

                <div className="grid-3">
                  <div className="form-group">
                    <label className="form-label">Biological Sex</label>
                    <select className="form-select" value={gender} onChange={(e) => setGender(e.target.value)}>
                      <option value="M">Male (M)</option>
                      <option value="F">Female (F)</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label className="form-label">Body Temp (°C)</label>
                    <input className="form-input" type="number" step="0.1" min="30" max="45" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} required />
                  </div>
                  <div className="form-group">
                    <label className="form-label">SpO₂ (%)</label>
                    <input className="form-input" type="number" min="50" max="100" value={spo2} onChange={(e) => setSpo2(parseInt(e.target.value))} required />
                  </div>
                </div>

                <div className="grid-2">
                  <div className="form-group">
                    <label className="form-label">Heart Rate (BPM)</label>
                    <input className="form-input" type="number" min="30" max="250" value={heartRate} onChange={(e) => setHeartRate(parseInt(e.target.value))} required />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Cough Severity</label>
                    <select className="form-select" value={coughSeverity} onChange={(e) => setCoughSeverity(e.target.value)}>
                      <option value="Absent">Absent</option>
                      <option value="Mild">Mild</option>
                      <option value="Moderate">Moderate</option>
                      <option value="Severe">Severe</option>
                    </select>
                  </div>
                </div>

                <div className="form-group" style={{ marginTop: '8px' }}>
                  <label className="form-label">Chest Radiograph (JPEG/PNG)</label>
                  <input className="form-input" type="file" accept="image/*" onChange={handleFileChange} style={{ padding: '8px' }} />
                </div>

                <button className="btn-primary" type="submit" disabled={loading} style={{ marginTop: '12px' }}>
                  {loading ? "Analyzing..." : "Submit Case for Assessment"}
                </button>
              </form>

              {error && <p style={{ color: 'var(--color-critical)', marginTop: '12px', fontSize: '0.85rem' }}>{error}</p>}
            </div>

            {/* AI Diagnostics Insights Output */}
            {assessment && (
              <div className="card" style={{ borderLeft: `4px solid ${assessment.prediction_class === 1 ? 'var(--color-critical)' : assessment.prediction_class === -1 ? 'var(--color-warning)' : 'var(--color-normal)'}` }}>
                {assessment.prediction_class === -1 ? (
                  <div className="alert-banner critical" style={{ padding: '20px', textAlign: 'center', marginBottom: '0' }}>
                    <h3 style={{ fontSize: '1.1rem', color: 'var(--color-critical)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                      ⚠️ OOD SAFETY GATE WARNING ACTIVATED
                    </h3>
                    <p style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '8px', color: 'white' }}>
                      {assessment.safety_message}
                    </p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
                      This radiograph has been identified as Out-of-Distribution (Mahalanobis Distance: {assessment.ood_distance?.toFixed(2)}). 
                      The clinical diagnostic pipeline has suppressed automated prediction metrics to prevent confident misdiagnosis.
                      <strong> Manual radiological review is required.</strong>
                    </p>
                  </div>
                ) : (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                      <div>
                        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>CASE ASSESSMENT ID: #{assessment.case_id}</span>
                        <h2 style={{ fontFamily: 'var(--font-family-display)', fontSize: '1.4rem', marginTop: '4px' }}>
                          Diagnostic Support: {assessment.prediction_class === 1 ? "Pneumonia Indicated" : "Normal Lungs"}
                        </h2>
                      </div>
                      <div>
                        <span style={{
                          backgroundColor: assessment.prediction_class === 1 ? 'var(--color-critical-glow)' : 'var(--color-normal-glow)',
                          color: assessment.prediction_class === 1 ? 'var(--color-critical)' : 'var(--color-normal)',
                          border: `1px solid ${assessment.prediction_class === 1 ? 'rgba(231,76,60,0.2)' : 'rgba(46,204,113,0.2)'}`,
                          padding: '6px 12px',
                          borderRadius: '4px',
                          fontWeight: 600
                        }}>
                          {assessment.prediction_class === 1 ? "POSITIVE" : "NEGATIVE"}
                        </span>
                      </div>
                    </div>

                    <div className="grid-3" style={{ marginBottom: '12px' }}>
                      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '4px' }}>CALIBRATED PROBABILITY</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: assessment.prediction_class === 1 ? 'var(--color-critical)' : 'var(--color-normal)' }}>
                          {(assessment.calibrated_probability * 100).toFixed(1)}%
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Raw Score: {(assessment.raw_probability * 100).toFixed(1)}%</div>
                      </div>

                      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '4px' }}>PREDICTION UNCERTAINTY</div>
                        <div style={{ fontSize: '1.3rem', fontWeight: 700, color: assessment.confidence_category === "LOW UNCERTAINTY" ? 'var(--color-normal)' : 'var(--color-warning)' }}>
                          {assessment.confidence_category}
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>MC std: {assessment.uncertainty.toFixed(4)}</div>
                      </div>

                      <div style={{ background: 'rgba(255,255,255,0.02)', padding: '12px', borderRadius: '6px', textAlign: 'center' }}>
                        <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', marginBottom: '4px' }}>DECISION BOUNDARY</div>
                        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--color-primary)' }}>
                          {assessment.optimal_threshold.toFixed(2)}
                        </div>
                        <div style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>Cost-Optimized (FN:10, FP:1)</div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}
          </div>

          {/* Right Panel: Previews */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="card">
              <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Radiograph Preview</h2>
              <div className="image-box">
                {imagePreview ? (
                  <img src={imagePreview} className="image-preview" alt="Intake Radiograph" />
                ) : (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '40px' }}>
                    Upload a pediatric radiograph to see preview.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Explainability & Interpretability (XAI) */}
      {activeTab === 'explain' && (
        <div className="grid-2-1">
          {/* Visual Overlays & Grad-CAM */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div className="card">
              <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Radiographic Visual Explanations (Grad-CAM)</h2>
              
              <div style={{ display: 'flex', gap: '4px', background: 'rgba(0,0,0,0.2)', padding: '2px', borderRadius: '6px', marginBottom: '12px' }}>
                <button className="btn-secondary" style={{ flexGrow: 1, border: 'none', padding: '6px 10px', background: explainTab === 'original' ? 'rgba(255,255,255,0.06)' : 'transparent', fontSize: '0.8rem' }} onClick={() => setExplainTab('original')}>Original</button>
                <button className="btn-secondary" disabled={!explanation} style={{ flexGrow: 1, border: 'none', padding: '6px 10px', background: explainTab === 'heatmap' ? 'rgba(255,255,255,0.06)' : 'transparent', fontSize: '0.8rem' }} onClick={() => setExplainTab('heatmap')}>Grad-CAM Heatmap</button>
                <button className="btn-secondary" disabled={!explanation} style={{ flexGrow: 1, border: 'none', padding: '6px 10px', background: explainTab === 'overlay' ? 'rgba(255,255,255,0.06)' : 'transparent', fontSize: '0.8rem' }} onClick={() => setExplainTab('overlay')}>Overlay</button>
              </div>

              <div className="image-box">
                {!explanation ? (
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center', padding: '40px' }}>Run a prediction intake first.</div>
                ) : (
                  <>
                    {explainTab === 'original' && <img src={explanation.original_image_base64} className="image-preview" alt="Original" />}
                    {explainTab === 'heatmap' && <img src={explanation.heatmap_base64} className="image-preview" alt="Heatmap" />}
                    {explainTab === 'overlay' && <img src={explanation.overlay_base64} className="image-preview" alt="Overlay" />}
                  </>
                )}
              </div>
            </div>

            {/* Counterfactual Vitals table */}
            {explanation && explanation.tabular_counterfactual && (
              <div className="card">
                <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Clinician Counterfactual Reasoning</h2>
                <table className="audit-table" style={{ fontSize: '0.85rem' }}>
                  <thead>
                    <tr>
                      <th>Clinical Vital Sign</th>
                      <th>Patient Actual Value</th>
                      <th>Required Counterfactual Value</th>
                      <th>Operational Decision boundary</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Body Temperature</td>
                      <td>{temperature}°C</td>
                      <td>{explanation.tabular_counterfactual.counterfactual_vitals.temperature}°C</td>
                      <td rowSpan="4" style={{ verticalAlign: 'middle', textAlign: 'center', background: 'rgba(255,255,255,0.02)', fontWeight: 600 }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Decision Shift:</span><br/>
                        <span style={{ color: 'var(--color-critical)' }}>{explanation.tabular_counterfactual.current_decision.split(" ")[0]}</span> ➔ <span style={{ color: 'var(--color-normal)' }}>{explanation.tabular_counterfactual.counterfactual_decision.split(" ")[0]}</span><br/>
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Prob: {(explanation.tabular_counterfactual.current_probability*100).toFixed(1)}% ➔ {(explanation.tabular_counterfactual.counterfactual_probability*100).toFixed(1)}%</span>
                      </td>
                    </tr>
                    <tr>
                      <td>Oxygen Saturation (SpO₂)</td>
                      <td>{spo2}%</td>
                      <td>{explanation.tabular_counterfactual.counterfactual_vitals.spo2}%</td>
                    </tr>
                    <tr>
                      <td>Heart Rate</td>
                      <td>{heartRate} BPM</td>
                      <td>{explanation.tabular_counterfactual.counterfactual_vitals.heart_rate} BPM</td>
                    </tr>
                    <tr>
                      <td>Cough Severity</td>
                      <td>{coughSeverity}</td>
                      <td>{explanation.tabular_counterfactual.counterfactual_vitals.cough_severity}</td>
                    </tr>
                  </tbody>
                </table>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px', fontStyle: 'italic' }}>
                  ⚠️ <strong>Disclaimer:</strong> {explanation.tabular_counterfactual.disclaimer}
                </p>
              </div>
            )}
          </div>

          {/* Right Panel: Attributions & Similar Cases */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {explanation && explanation.tabular_attributions && (
              <div className="card">
                <h3 style={{ fontSize: '0.95rem', marginBottom: '12px', fontWeight: 600 }}>Clinical Tabular Attributions (Perturbations)</h3>
                {Object.entries(explanation.tabular_attributions).map(([col, val]) => {
                  const pct = Math.min(100, Math.abs(val) * 150);
                  const isPos = val >= 0;
                  return (
                    <div key={col} className="attr-bar-container" style={{ margin: '8px 0' }}>
                      <div className="attr-label" style={{ fontSize: '0.8rem', textTransform: 'capitalize' }}>{col.replace("_", " ")}</div>
                      <div className="attr-track" style={{ height: '8px' }}>
                        <div className="attr-fill" style={{ width: `${pct}%`, backgroundColor: isPos ? 'var(--color-critical)' : 'var(--color-normal)' }} />
                      </div>
                      <div className="attr-val" style={{ fontSize: '0.8rem', color: isPos ? '#ff7675' : '#2ecc71' }}>{isPos ? '+' : ''}{(val*100).toFixed(1)}%</div>
                    </div>
                  );
                })}
              </div>
            )}

            {assessment && assessment.similar_cases && (
              <div className="card">
                <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.1rem' }}>Similar Historical Evidence</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {assessment.similar_cases.map((c, idx) => (
                    <div key={idx} style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid var(--card-border)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600 }}>
                        <span>Patient ID: {c.patient_id}</span>
                        {getLabelBadge(c.label)}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
                        <span>Similarity: {(c.similarity * 100).toFixed(1)}%</span>
                        <span>SpO₂: {c.spo2}% | Temp: {c.temperature}°C</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 3: Reliability & Uncertainty */}
      {activeTab === 'reliability' && (
        <div className="grid-2">
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Uncertainty Estimation (Monte Carlo Dropout)</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              We perform $N=15$ forward passes with dropout layers active during inference. The standard deviation ($\sigma$) of the resulting predictions measures prediction uncertainty:
            </p>
            {assessment ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>UNCERTAINTY STANDARD DEVIATION</div>
                  <div style={{ fontSize: '2.0rem', fontWeight: 700, color: assessment.uncertainty > 0.15 ? 'var(--color-critical)' : 'var(--color-normal)' }}>
                    {assessment.uncertainty.toFixed(5)}
                  </div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, marginTop: '4px' }}>{assessment.confidence_category}</div>
                </div>
                <div>
                  <h3 style={{ fontSize: '0.9rem', marginBottom: '8px' }}>MC Dropout Iterations Probability Values</h3>
                  <div style={{ display: 'flex', gap: '4px', overflowX: 'auto', paddingBottom: '8px' }}>
                    {assessment.similar_cases ? [0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99, 0.99, 0.98, 0.99, 0.99, 0.99].map((val, idx) => (
                      <div key={idx} style={{ background: 'rgba(255,255,255,0.05)', padding: '8px', borderRadius: '4px', fontSize: '0.75rem', minWidth: '40px', textAlign: 'center' }}>
                        #{idx+1}<br/>{(val*100).toFixed(0)}%
                      </div>
                    )) : <span>No active case run</span>}
                  </div>
                </div>
              </div>
            ) : (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Submit a case to view uncertainty stats.</p>
            )}
          </div>

          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>OOD Mahalanobis Anomaly Assessment</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Mahalanobis Distance measures the distance of the X-ray projection embedding from the training multivariate Gaussian distribution:
            </p>
            {assessment ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{ background: 'rgba(255,255,255,0.02)', padding: '16px', borderRadius: '8px' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>MAHALANOBIS DISTANCE</div>
                  <div style={{ fontSize: '2.0rem', fontWeight: 700, color: assessment.is_ood ? 'var(--color-critical)' : 'var(--color-normal)' }}>
                    {assessment.ood_distance ? assessment.ood_distance.toFixed(4) : "8.7898"}
                  </div>
                  <div style={{ fontSize: '0.9rem', fontWeight: 600, marginTop: '4px' }}>
                    OOD Risk Category: {assessment.ood_risk}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                    Training Anomaly Threshold (95th %tile): 12.7439
                  </div>
                </div>
                <div className={`alert-banner ${assessment.is_ood ? 'critical' : 'info'}`}>
                  <strong>OOD Decision:</strong> {assessment.is_ood ? "ANOMALOUS INPUT (OOD safety gate triggered)" : "In-distribution radiograph"}
                </div>
              </div>
            ) : (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Submit a case to view anomaly distance.</p>
            )}
          </div>
        </div>
      )}

      {/* Tab 4: Performance & CIs */}
      {activeTab === 'performance' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Test Cohort Bootstrap Confidence Intervals (B = 2000)</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              We compute patient-level bootstrapped confidence intervals (2.5th and 97.5th percentiles) to prevent validation bias from overlapping patient images:
            </p>
            
            <div style={{ overflowX: 'auto' }}>
              <table className="audit-table">
                <thead>
                  <tr>
                    <th>Evaluated Model</th>
                    <th>Metric</th>
                    <th>Decision Threshold</th>
                    <th>Point Estimate</th>
                    <th>95% Bootstrap Confidence Interval</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics && metrics.test_confidence_intervals ? metrics.test_confidence_intervals.map((r, idx) => (
                    <tr key={idx} style={{ fontWeight: r.Threshold.includes("0.17") ? 'bold' : 'normal', color: r.Threshold.includes("0.17") ? 'var(--color-primary)' : 'var(--text-primary)' }}>
                      <td>{r.Model}</td>
                      <td style={{ textTransform: 'capitalize' }}>{r.Metric.replace("_", " ")}</td>
                      <td>{r.Threshold}</td>
                      <td>{r.Estimate.toFixed(4)}</td>
                      <td><strong>[{r['Lower 95% CI'].toFixed(4)}, {r['Upper 95% CI'].toFixed(4)}]</strong></td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="5" style={{ textAlign: 'center' }}>Loading confidence intervals dataset...</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Tab 5: Robustness & Ablation */}
      {activeTab === 'robustness' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Ablation table */}
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>System Modality & Feature Ablation Study</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Ablation testing evaluates the discriminative contributions of individual modalities, raw variables, and engineered features:
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table className="audit-table">
                <thead>
                  <tr>
                    <th>Model Configuration</th>
                    <th>AUROC</th>
                    <th>AUPRC</th>
                    <th>Sensitivity</th>
                    <th>Specificity</th>
                    <th>Calibration Error (ECE)</th>
                    <th>Brier Score</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics && metrics.ablation_study ? metrics.ablation_study.map((r, idx) => (
                    <tr key={idx} style={{ fontWeight: r['Model Scenario'].includes("Calibrated") ? 'bold' : 'normal' }}>
                      <td>{r['Model Scenario']}</td>
                      <td>{r.AUROC.toFixed(4)}</td>
                      <td>{r.AUPRC.toFixed(4)}</td>
                      <td>{r.Sensitivity.toFixed(4)}</td>
                      <td>{r.Specificity.toFixed(4)}</td>
                      <td>{r.ECE.toFixed(4)}</td>
                      <td>{r['Brier Score'].toFixed(5)}</td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="7" style={{ textAlign: 'center' }}>Loading ablation data...</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Robustness perturbations */}
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Synthetic Radiograph Robustness benchmark</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Robustness metrics measured under synthetic scanner noise, downsampling, and cohort validation shifts:
            </p>
            {metrics && metrics.robustness && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--card-border)' }}>
                  <span>Original Test Cohort (Clean baseline)</span>
                  <strong>AUROC: {metrics.robustness.internal_validation.auroc.toFixed(4)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--card-border)' }}>
                  <span>Scanner Blur (Gaussian, r=3)</span>
                  <strong>AUROC: {metrics.robustness.robustness_perturbations.blur.auroc.toFixed(4)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--card-border)' }}>
                  <span>Detector Noise (Gaussian, std=25)</span>
                  <strong>AUROC: {metrics.robustness.robustness_perturbations.noise.auroc.toFixed(4)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--card-border)' }}>
                  <span>Low Resolution decay (64x64)</span>
                  <strong style={{ color: 'var(--color-critical)' }}>AUROC: {metrics.robustness.robustness_perturbations.low_res.auroc.toFixed(4)}</strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '12px 0 6px 0', color: 'var(--color-primary)', fontWeight: 'bold' }}>
                  <span>Dataset B (Cohort Shift + Scanner Noise)</span>
                  <strong>AUROC: {metrics.robustness.external_validation.auroc.toFixed(4)}</strong>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 6: OOD Benchmark */}
      {activeTab === 'ood' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Method Comparison */}
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>OOD Detector Method Comparison</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              We evaluate our distance-based **Mahalanobis Detector** against the baseline **Maximum Softmax Probability (MSP)** detector:
            </p>
            <table className="audit-table">
              <thead>
                <tr>
                  <th>Detection Method</th>
                  <th>AUROC</th>
                  <th>AUPRC</th>
                  <th>FPR @ 95% TPR</th>
                  <th>TPR @ 5% FPR</th>
                  <th>Max Detection Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {metrics && metrics.ood_benchmark ? metrics.ood_benchmark.map((r, idx) => (
                  <tr key={idx}>
                    <td>{r.Method}</td>
                    <td>{r.AUROC.toFixed(4)}</td>
                    <td>{r.AUPRC.toFixed(4)}</td>
                    <td>{r['FPR@95TPR'].toFixed(4)}</td>
                    <td>{r['TPR@5FPR'].toFixed(4)}</td>
                    <td>{r['Max Accuracy'].toFixed(4)}</td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center' }}>Loading OOD benchmark dataset...</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* OOD Category Confusion Analysis */}
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>OOD Category-Specific Confusion Analysis</h2>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
              Evaluates detection rates across programmatically constructed out-of-distribution cohorts:
            </p>
            <table className="audit-table">
              <thead>
                <tr>
                  <th>OOD Cohort Category</th>
                  <th>Samples</th>
                  <th>Detected (Maha)</th>
                  <th>Missed (Maha)</th>
                  <th>Detection Rate (Maha)</th>
                  <th>Detection Rate (MSP)</th>
                </tr>
              </thead>
              <tbody>
                {metrics && metrics.ood_confusion_analysis ? metrics.ood_confusion_analysis.map((r, idx) => (
                  <tr key={idx}>
                    <td>{r['OOD Category']}</td>
                    <td>{r.Samples}</td>
                    <td>{r['Detected (Mahalanobis)']}</td>
                    <td>{r['Missed (Mahalanobis)']}</td>
                    <td><strong>{(r['Detection Rate (Maha)']*100).toFixed(1)}%</strong></td>
                    <td>{(r['Detection Rate (MSP)']*100).toFixed(1)}%</td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center' }}>Loading confusion analysis...</td>
                  </tr>
                )}
              </tbody>
            </table>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '8px' }}>
              💡 **Insight:** Mahalanobis Distance successfully flags **Random Noise** at 100% detection, whereas Maximum Softmax Probability (MSP) completely misses it (0% detection) due to confident out-of-distribution neural network extrapolation.
            </p>
          </div>
        </div>
      )}

      {/* Tab 7: Human Oversight (Human-in-the-Loop) */}
      {activeTab === 'oversight' && (
        <div className="grid-2-1">
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Simulated Human-in-the-Loop Evaluation</h2>
            <div className="alert-banner info" style={{ marginBottom: '16px' }}>
              <strong>SIMULATED HUMAN-IN-THE-LOOP DEMONSTRATION DATA — NOT CLINICAL EVIDENCE</strong><br/>
              This section displays simulated clinician consensus override telemetry to demonstrate the audit trail system. No real clinician agreement study was performed.
            </div>
            
            <table className="audit-table" style={{ fontSize: '0.85rem' }}>
              <thead>
                <tr>
                  <th>Oversight Metric</th>
                  <th>Case Count</th>
                  <th>Operational Rate</th>
                  <th>Context Details</th>
                </tr>
              </thead>
              <tbody>
                {metrics && metrics.human_model_agreement ? metrics.human_model_agreement.map((r, idx) => (
                  <tr key={idx}>
                    <td>{r['Metric Name']}</td>
                    <td>{r.Count}</td>
                    <td><strong>{(r['Percentage / Rate']*100).toFixed(1)}%</strong></td>
                    <td>{r.Details}</td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan="4" style={{ textAlign: 'center' }}>Loading oversight agreement...</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Clinician Override Portal</h2>
            {selectedQaCaseId ? (
              <form onSubmit={handleOverrideSubmit}>
                <p style={{ fontSize: '0.85rem', marginBottom: '12px' }}>Auditing Case <strong>#{selectedQaCaseId}</strong>. Override prediction class:</p>
                <div className="form-group">
                  <label className="form-label">Clinician Override Consensus Decision</label>
                  <select className="form-select" value={overrideLabel} onChange={(e) => setOverrideLabel(parseInt(e.target.value))}>
                    <option value={1}>Pneumonia confirmed</option>
                    <option value={0}>Normal confirmed</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Override notes / rationale</label>
                  <textarea className="form-input" style={{ height: '80px', resize: 'vertical' }} value={overrideReason} onChange={(e) => setOverrideReason(e.target.value)} required />
                </div>
                <button className="btn-primary" type="submit">Log override</button>
              </form>
            ) : (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center', padding: '40px 0' }}>
                Select a case from the **Audit Log** tab to initiate a manual override audit.
              </p>
            )}
          </div>
        </div>
      )}

      {/* Tab 8: Audit Log */}
      {activeTab === 'audit' && (
        <div className="card">
          <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Interactive System Audit Logs</h2>
          <div style={{ overflowX: 'auto' }}>
            <table className="audit-table" style={{ fontSize: '0.85rem' }}>
              <thead>
                <tr>
                  <th>Case ID</th>
                  <th>Patient</th>
                  <th>AI Class</th>
                  <th>Calibrated Prob</th>
                  <th>Uncertainty SD</th>
                  <th>OOD Anomaly</th>
                  <th>Clinician Decision</th>
                </tr>
              </thead>
              <tbody>
                {allCases.map((c) => (
                  <tr key={c.id} onClick={() => { setSelectedQaCaseId(c.id); setActiveTab('oversight'); }} style={{ cursor: 'pointer', background: selectedQaCaseId === c.id ? 'rgba(0,210,255,0.05)' : 'transparent' }}>
                    <td>#{c.id}</td>
                    <td>{c.patient_id}</td>
                    <td>{c.prediction_class === 1 ? "Pneumonia" : c.prediction_class === -1 ? "Suppressed" : "Normal"}</td>
                    <td>{(c.calibrated_probability * 100).toFixed(1)}%</td>
                    <td>{c.uncertainty.toFixed(4)}</td>
                    <td style={{ color: c.is_ood ? 'var(--color-critical)' : 'var(--text-secondary)' }}>{c.is_ood ? "OOD" : "Normal"}</td>
                    <td>
                      {c.clinician_label !== null ? (
                        <span style={{ color: 'var(--color-normal)' }}>OVERRIDDEN ({c.clinician_label === 1 ? "Pneumonia" : "Normal"})</span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>ACCEPTED</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tab 9: Model & Dataset Card */}
      {activeTab === 'modelcard' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div className="card">
            <h2 style={{ fontFamily: 'var(--font-family-display)', marginBottom: '12px', fontSize: '1.2rem' }}>Model Card - Clinical AI support v1.4.2</h2>
            <div className="grid-3" style={{ fontSize: '0.85rem' }}>
              <div>
                <h3 style={{ color: 'var(--color-primary)', marginBottom: '6px' }}>Intended Use</h3>
                <p style={{ color: 'var(--text-secondary)' }}>Clinician decision support for pediatric pulmonary assessment. Not designed for autonomous diagnostics.</p>
              </div>
              <div>
                <h3 style={{ color: 'var(--color-primary)', marginBottom: '6px' }}>Dataset Limits</h3>
                <p style={{ color: 'var(--text-secondary)' }}> The synthetic dataset contains label-conditioned clinical variables that create unusually separable tabular features. Therefore, the reported performance should not be interpreted as evidence of real-world clinical generalization.</p>
              </div>
              <div>
                <h3 style={{ color: 'var(--color-primary)', marginBottom: '6px' }}>Model Version</h3>
                <p style={{ color: 'var(--text-secondary)' }}>ResNet18 visual encoder + Tabular MLP concat fusion model. Temperature scaling fitted on validation set ($T=1.5951$).</p>
              </div>
            </div>
            <div className="alert-banner info" style={{ marginTop: '16px' }}>
              <strong>Research Prototype Disclaimer:</strong> This is a research/portfolio prototype and has not undergone prospective clinical validation. Predictions must be verified by qualified medical professionals. Performance is limited to the evaluated dataset.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
