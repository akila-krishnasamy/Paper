import React, { useState } from 'react';
import { Activity, ShieldAlert, AlertTriangle, CheckCircle, Info, Stethoscope, RefreshCw, Droplets, Sun, Sparkles } from 'lucide-react';

export default function DeficiencyScreening() {
  const [formData, setFormData] = useState({
    child_sex: 'Female',
    child_age_months: '24',
    hemoglobin: '10.2',
    mch: '21.5',
    mchc: '28.9',
    mcv: '74.5',
    dietary_diversity: '2',
    measles_vaccine: 'No',
    diarrhea_recent: 'Yes'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleScreen = async (e) => {
    if (e) e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('http://localhost:3000/api/predict/deficiencies', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      if (!res.ok) throw new Error('Deficiency screening service returned an error.');
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Unable to connect to deficiency screening service.');
    } finally {
      setLoading(false);
    }
  };

  const setSample = (type) => {
    if (type === 'anemia_flagged') {
      setFormData({
        child_sex: 'Female',
        child_age_months: '18',
        hemoglobin: '8.7',
        mch: '18.2',
        mchc: '27.4',
        mcv: '68.0',
        dietary_diversity: '1',
        measles_vaccine: 'No',
        diarrhea_recent: 'Yes'
      });
    } else {
      setFormData({
        child_sex: 'Male',
        child_age_months: '30',
        hemoglobin: '12.4',
        mch: '26.8',
        mchc: '32.1',
        mcv: '86.5',
        dietary_diversity: '6',
        measles_vaccine: 'Yes',
        diarrhea_recent: 'No'
      });
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', animation: 'fadeIn 0.3s ease' }}>
      
      {/* Header Panel */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Droplets size={24} color="#f43f5e" />
              Early Vitamin & Micronutrient Deficiency Screening
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
              Laboratory-backed screening for pediatric anemia and evidence-based screening protocols for micronutrient risks.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              className="btn btn-outline"
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem', borderColor: '#ef4444', color: '#ef4444' }}
              onClick={() => setSample('anemia_flagged')}
            >
              Load Sample: Anemic / Low Diversity
            </button>
            <button
              type="button"
              className="btn btn-outline"
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.75rem', borderColor: '#10b981', color: '#10b981' }}
              onClick={() => setSample('healthy')}
            >
              Load Sample: Normal Lab Values
            </button>
          </div>
        </div>
      </div>

      {/* Clinical Input Form */}
      <form onSubmit={handleScreen} className="glass-panel" style={{ padding: '1.75rem' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#38bdf8' }}>
          <Stethoscope size={18} /> Laboratory & Clinical Indicator Workup
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>
          
          <div className="form-group">
            <label className="form-label">Hemoglobin (Hb, g/dL)</label>
            <input
              type="number"
              step="0.1"
              className="form-control"
              value={formData.hemoglobin}
              onChange={(e) => setFormData(prev => ({ ...prev, hemoglobin: e.target.value }))}
              placeholder="e.g. 10.2 (WHO cutoff: 11.0)"
            />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>WHO Normal &ge; 11.0 g/dL</span>
          </div>

          <div className="form-group">
            <label className="form-label">Mean Corpuscular Hb (MCH, pg)</label>
            <input
              type="number"
              step="0.1"
              className="form-control"
              value={formData.mch}
              onChange={(e) => setFormData(prev => ({ ...prev, mch: e.target.value }))}
              placeholder="e.g. 21.5"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Mean Corpuscular Volume (MCV, fL)</label>
            <input
              type="number"
              step="0.1"
              className="form-control"
              value={formData.mcv}
              onChange={(e) => setFormData(prev => ({ ...prev, mcv: e.target.value }))}
              placeholder="e.g. 74.5"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Dietary Diversity Score (0–8 food groups)</label>
            <select
              className="form-control"
              value={formData.dietary_diversity}
              onChange={(e) => setFormData(prev => ({ ...prev, dietary_diversity: e.target.value }))}
            >
              <option value="1">1 Group — Severely Inadequate</option>
              <option value="2">2 Groups — Low Diversity</option>
              <option value="3">3 Groups — Borderline Diversity</option>
              <option value="4">4 Groups — Adequate Diversity</option>
              <option value="5">5 Groups — High Diversity</option>
              <option value="6">6+ Groups — Optimum Diversity</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Measles Vaccine Received</label>
            <select
              className="form-control"
              value={formData.measles_vaccine}
              onChange={(e) => setFormData(prev => ({ ...prev, measles_vaccine: e.target.value }))}
            >
              <option value="Yes">Yes (Vaccinated)</option>
              <option value="No">No (Unvaccinated)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Recent Diarrhea Episode</label>
            <select
              className="form-control"
              value={formData.diarrhea_recent}
              onChange={(e) => setFormData(prev => ({ ...prev, diarrhea_recent: e.target.value }))}
            >
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>

        </div>

        <div style={{ textAlign: 'right' }}>
          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary"
            style={{ padding: '0.65rem 1.75rem', fontSize: '0.95rem' }}
          >
            {loading ? <RefreshCw className="animate-spin" size={16} /> : <Sparkles size={16} />}
            Run Deficiency Risk Screening
          </button>
        </div>
      </form>

      {/* Error state */}
      {error && (
        <div className="glass-panel" style={{ padding: '1rem', borderColor: '#ef4444', color: '#ef4444' }}>
          {error}
        </div>
      )}

      {/* Screening Cards Breakdown */}
      {result && result.deficiencies && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
          
          {/* 1. Iron Deficiency & Anemia */}
          <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: `4px solid ${result.deficiencies.iron_anemia.is_flagged ? '#ef4444' : '#10b981'}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#fff' }}>
                1. Iron Deficiency & Anemia Screening
              </h4>
              <span className={`badge ${result.deficiencies.iron_anemia.severity.toLowerCase()}`}>
                {result.deficiencies.iron_anemia.severity}
              </span>
            </div>

            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Status: <strong style={{ color: '#10b981' }}>{result.deficiencies.iron_anemia.status}</strong>
              {result.deficiencies.iron_anemia.risk_score !== null && (
                <span style={{ marginLeft: '0.75rem' }}>
                  Risk Probability: <strong style={{ color: result.deficiencies.iron_anemia.is_flagged ? '#ef4444' : '#10b981' }}>{result.deficiencies.iron_anemia.risk_score}%</strong>
                </span>
              )}
            </div>

            {result.deficiencies.iron_anemia.lab_markers_used && (
              <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.6rem', borderRadius: '6px', fontSize: '0.8rem', marginBottom: '0.75rem' }}>
                <div><strong>Hemoglobin:</strong> {result.deficiencies.iron_anemia.lab_markers_used.hemoglobin}</div>
                <div><strong>MCH:</strong> {result.deficiencies.iron_anemia.lab_markers_used.mch}</div>
                <div><strong>MCV:</strong> {result.deficiencies.iron_anemia.lab_markers_used.mcv}</div>
              </div>
            )}

            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              {result.deficiencies.iron_anemia.clinical_notes}
            </p>
          </div>

          {/* 2. Vitamin A */}
          <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: `4px solid ${result.deficiencies.vitamin_a.is_flagged ? '#f59e0b' : '#10b981'}` }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#fff' }}>
                2. Vitamin A Deficiency Risk
              </h4>
              <span className={`badge ${result.deficiencies.vitamin_a.severity.toLowerCase()}`}>
                {result.deficiencies.vitamin_a.severity}
              </span>
            </div>

            <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Status: <strong style={{ color: '#38bdf8' }}>{result.deficiencies.vitamin_a.status}</strong>
              {result.deficiencies.vitamin_a.risk_score !== null && (
                <span style={{ marginLeft: '0.75rem' }}>
                  Screening Index: <strong style={{ color: result.deficiencies.vitamin_a.is_flagged ? '#f59e0b' : '#10b981' }}>{result.deficiencies.vitamin_a.risk_score}%</strong>
                </span>
              )}
            </div>

            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              {result.deficiencies.vitamin_a.clinical_notes}
            </p>
          </div>

          {/* 3. Vitamin D (Explicitly Marked Unavailable to prevent fabrication) */}
          <div className="glass-panel" style={{ padding: '1.5rem', opacity: 0.85, borderLeft: '4px solid #64748b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                3. Vitamin D Deficiency Risk
              </h4>
              <span className="badge" style={{ background: 'rgba(100,116,139,0.2)', color: '#94a3b8' }}>
                UNAVAILABLE
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              {result.deficiencies.vitamin_d.clinical_notes}
            </p>
            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#f59e0b', fontStyle: 'italic' }}>
              Safety standard: No probability fabricated in the absence of genuine 25(OH)D laboratory assays.
            </div>
          </div>

          {/* 4. Vitamin B12 */}
          <div className="glass-panel" style={{ padding: '1.5rem', opacity: 0.85, borderLeft: '4px solid #64748b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                4. Vitamin B12 Deficiency Risk
              </h4>
              <span className="badge" style={{ background: 'rgba(100,116,139,0.2)', color: '#94a3b8' }}>
                UNAVAILABLE
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              {result.deficiencies.vitamin_b12.clinical_notes}
            </p>
            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#f59e0b', fontStyle: 'italic' }}>
              Safety standard: Requires serum cobalamin / methylmalonic acid testing.
            </div>
          </div>

          {/* 5. Folate */}
          <div className="glass-panel" style={{ padding: '1.5rem', opacity: 0.85, borderLeft: '4px solid #64748b' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-secondary)' }}>
                5. Folate Deficiency Risk
              </h4>
              <span className="badge" style={{ background: 'rgba(100,116,139,0.2)', color: '#94a3b8' }}>
                UNAVAILABLE
              </span>
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>
              {result.deficiencies.folate.clinical_notes}
            </p>
            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#f59e0b', fontStyle: 'italic' }}>
              Safety standard: Requires red blood cell folate laboratory panels.
            </div>
          </div>

        </div>
      )}

      {/* Clinical Disclaimer */}
      <div className="glass-panel" style={{ padding: '1rem 1.5rem', borderColor: 'rgba(99,102,241,0.3)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <Info size={20} color="#818cf8" style={{ flexShrink: 0 }} />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <strong>Clinical Research Note:</strong> Deficiency risks are screening estimates from validated laboratory models and dietary protocols. Anemia is not exclusively caused by iron deficiency and requires formal differential medical diagnosis.
        </span>
      </div>

    </div>
  );
}
