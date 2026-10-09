import React, { useState, useEffect } from 'react';
import { Activity, Cpu, Sparkles, AlertCircle, ShieldAlert, User, Heart, Home, Stethoscope } from 'lucide-react';

export default function ChildForm({ onSubmit, loading, selectedModel, setSelectedModel }) {
  const [formData, setFormData] = useState({
    childName: 'Child #1042',
    child_age_months: '24',
    child_sex: 'Female',
    height_cm: '85',
    weight_kg: '11',
    birth_weight: '2.9',
    birth_size: 'Average',
    breastfeeding_duration: '18',
    birth_order: '2',
    mother_bmi: '21.5',
    education: 'Secondary',
    anc_visits: '5',
    wealth_quintile: 'Middle',
    residence: 'Rural',
    hhsize: '5',
    sanitation_risk_index: '1',
    diarrhea_recent: 'No',
    fever_recent: 'No',
    cough_recent: 'No',
    measles_vaccine: 'Yes'
  });

  const [errors, setErrors] = useState({});

  const applyPreset = (type) => {
    if (type === 'high_risk') {
      setFormData({
        childName: 'Priya (High-Risk Screening)',
        child_age_months: '18',
        child_sex: 'Female',
        height_cm: '76',
        weight_kg: '8.5',
        birth_weight: '2.1',
        birth_size: 'Smaller than Average',
        breastfeeding_duration: '8',
        birth_order: '4',
        mother_bmi: '17.2',
        education: 'No Education',
        anc_visits: '1',
        wealth_quintile: 'Poorest',
        residence: 'Rural',
        hhsize: '7',
        sanitation_risk_index: '3',
        diarrhea_recent: 'Yes',
        fever_recent: 'Yes',
        cough_recent: 'Yes',
        measles_vaccine: 'No'
      });
    } else if (type === 'low_risk') {
      setFormData({
        childName: 'Aarav (Healthy Baseline)',
        child_age_months: '24',
        child_sex: 'Male',
        height_cm: '88',
        weight_kg: '13',
        birth_weight: '3.3',
        birth_size: 'Average',
        breastfeeding_duration: '20',
        birth_order: '1',
        mother_bmi: '23.8',
        education: 'Higher',
        anc_visits: '8',
        wealth_quintile: 'Richest',
        residence: 'Urban',
        hhsize: '4',
        sanitation_risk_index: '0',
        diarrhea_recent: 'No',
        fever_recent: 'No',
        cough_recent: 'No',
        measles_vaccine: 'Yes'
      });
    } else if (type === 'moderate_risk') {
      setFormData({
        childName: 'Kavya (Borderline Profile)',
        child_age_months: '14',
        child_sex: 'Female',
        height_cm: '78',
        weight_kg: '9.5',
        birth_weight: '2.6',
        birth_size: 'Average',
        breastfeeding_duration: '12',
        birth_order: '2',
        mother_bmi: '19.4',
        education: 'Primary',
        anc_visits: '3',
        wealth_quintile: 'Poorer',
        residence: 'Rural',
        hhsize: '6',
        sanitation_risk_index: '2',
        diarrhea_recent: 'No',
        fever_recent: 'Yes',
        cough_recent: 'No',
        measles_vaccine: 'Yes'
      });
    }
  };

  const handleChange = (field, val) => {
    setFormData(prev => ({ ...prev, [field]: val }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  const validate = () => {
    const errs = {};
    const age = parseFloat(formData.child_age_months);
    const bw = parseFloat(formData.birth_weight);
    const mbmi = parseFloat(formData.mother_bmi);

    if (isNaN(age) || age < 0 || age > 60) errs.child_age_months = 'Age must be 0–60 months.';
    if (isNaN(bw) || bw < 0.5 || bw > 6.0) errs.birth_weight = 'Birth weight must be between 0.5 and 6.0 kg.';
    if (isNaN(mbmi) || mbmi < 10 || mbmi > 55) errs.mother_bmi = 'Mother BMI must be between 10 and 55.';

    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validate()) return;

    onSubmit({
      ...formData,
      child_age_months: parseFloat(formData.child_age_months),
      height_cm: parseFloat(formData.height_cm),
      weight_kg: parseFloat(formData.weight_kg),
      birth_weight: parseFloat(formData.birth_weight),
      breastfeeding_duration: parseFloat(formData.breastfeeding_duration),
      birth_order: parseFloat(formData.birth_order),
      mother_bmi: parseFloat(formData.mother_bmi),
      anc_visits: parseFloat(formData.anc_visits),
      hhsize: parseFloat(formData.hhsize),
      sanitation_risk_index: parseFloat(formData.sanitation_risk_index)
    });
  };

  const modelsList = ['Compare All', 'Anthropometric', 'XGBoost', 'Transformer', 'DNN', 'TabNet'];

  return (
    <form onSubmit={handleSubmit} className="glass-panel" style={{ padding: '2rem' }}>
      {/* Header & Model Selector */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Activity color="#6366f1" />
            Pediatric Intake & Survey Indicators
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            Enter DHS/ICDS indicators to screen for Stunting, Wasting, and Malnutrition risk.
          </p>
        </div>

        {/* Preset Case Buttons */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn btn-outline"
            style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', borderColor: '#ef4444', color: '#ef4444' }}
            onClick={() => applyPreset('high_risk')}
          >
            🔴 Preset: High-Risk
          </button>
          <button
            type="button"
            className="btn btn-outline"
            style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', borderColor: '#eab308', color: '#eab308' }}
            onClick={() => applyPreset('moderate_risk')}
          >
            🟡 Preset: Borderline
          </button>
          <button
            type="button"
            className="btn btn-outline"
            style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', borderColor: '#22c55e', color: '#22c55e' }}
            onClick={() => applyPreset('low_risk')}
          >
            🟢 Preset: Healthy
          </button>
        </div>
      </div>

      {/* Model Selection Tabs */}
      <div style={{ marginBottom: '2rem', padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', display: 'block', marginBottom: '0.6rem' }}>
          Selected DL Screening Model:
        </span>
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          {modelsList.map(m => (
            <button
              key={m}
              type="button"
              className={`btn ${selectedModel === m ? 'btn-primary' : 'btn-outline'}`}
              style={{ padding: '0.4rem 1rem', fontSize: '0.85rem' }}
              onClick={() => setSelectedModel(m)}
            >
              {m === 'Compare All' && <Sparkles size={14} style={{ marginRight: '0.3rem', display: 'inline' }} />}
              {m}
            </button>
          ))}
        </div>
      </div>

      {/* 4 Thematic Form Grid Sections */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
        
        {/* Section 1: Child Demographic & Birth */}
        <div className="glass-panel" style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#818cf8', marginBottom: '1rem' }}>
            <User size={16} /> Child Profile & Birth Metrics
          </h3>

          <div className="form-group" style={{ marginBottom: '0.85rem' }}>
            <label className="form-label">Child Name / ID</label>
            <input
              type="text"
              className="form-control"
              value={formData.childName}
              onChange={(e) => handleChange('childName', e.target.value)}
            />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
            <div className="form-group">
              <label className="form-label">Current Height (cm)</label>
              <input type="number" step="0.1" className="form-control" value={formData.height_cm} onChange={(e) => handleChange('height_cm', e.target.value)} />
            </div>
            <div className="form-group">
              <label className="form-label">Current Weight (kg)</label>
              <input type="number" step="0.1" className="form-control" value={formData.weight_kg} onChange={(e) => handleChange('weight_kg', e.target.value)} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
            <div className="form-group">
              <label className="form-label">Age (months)</label>
              <input
                type="number"
                className="form-control"
                value={formData.child_age_months}
                onChange={(e) => handleChange('child_age_months', e.target.value)}
              />
              {errors.child_age_months && <span style={{ color: '#ef4444', fontSize: '0.75rem' }}>{errors.child_age_months}</span>}
            </div>

            <div className="form-group">
              <label className="form-label">Sex</label>
              <select
                className="form-control"
                value={formData.child_sex}
                onChange={(e) => handleChange('child_sex', e.target.value)}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
            <div className="form-group">
              <label className="form-label">Birth Weight (kg)</label>
              <input
                type="number"
                step="0.1"
                className="form-control"
                value={formData.birth_weight}
                onChange={(e) => handleChange('birth_weight', e.target.value)}
              />
              {errors.birth_weight && <span style={{ color: '#ef4444', fontSize: '0.75rem' }}>{errors.birth_weight}</span>}
            </div>

            <div className="form-group">
              <label className="form-label">Birth Size</label>
              <select
                className="form-control"
                value={formData.birth_size}
                onChange={(e) => handleChange('birth_size', e.target.value)}
              >
                <option value="Very Large">Very Large</option>
                <option value="Larger than Average">Larger than Avg</option>
                <option value="Average">Average</option>
                <option value="Smaller than Average">Smaller than Avg</option>
                <option value="Very Small">Very Small</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group">
              <label className="form-label">Breastfeeding (mo)</label>
              <input
                type="number"
                className="form-control"
                value={formData.breastfeeding_duration}
                onChange={(e) => handleChange('breastfeeding_duration', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Birth Order</label>
              <input
                type="number"
                className="form-control"
                value={formData.birth_order}
                onChange={(e) => handleChange('birth_order', e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Section 2: Maternal Health & Nutrition */}
        <div className="glass-panel" style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#ec4899', marginBottom: '1rem' }}>
            <Heart size={16} /> Maternal Health & Care
          </h3>

          <div className="form-group" style={{ marginBottom: '0.85rem' }}>
            <label className="form-label">Mother's BMI (kg/m²)</label>
            <input
              type="number"
              step="0.1"
              className="form-control"
              value={formData.mother_bmi}
              onChange={(e) => handleChange('mother_bmi', e.target.value)}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>Normal: 18.5–24.9 | Underweight: &lt; 18.5</span>
            {errors.mother_bmi && <span style={{ color: '#ef4444', fontSize: '0.75rem' }}>{errors.mother_bmi}</span>}
          </div>

          <div className="form-group" style={{ marginBottom: '0.85rem' }}>
            <label className="form-label">Mother's Education</label>
            <select
              className="form-control"
              value={formData.education}
              onChange={(e) => handleChange('education', e.target.value)}
            >
              <option value="No Education">No Education</option>
              <option value="Primary">Primary Education</option>
              <option value="Secondary">Secondary Education</option>
              <option value="Higher">Higher Education</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Antenatal Care (ANC) Visits</label>
            <input
              type="number"
              className="form-control"
              value={formData.anc_visits}
              onChange={(e) => handleChange('anc_visits', e.target.value)}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-secondary)' }}>WHO Recommendation: ≥ 4 visits</span>
          </div>
        </div>

        {/* Section 3: Household & Environment */}
        <div className="glass-panel" style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#06b6d4', marginBottom: '1rem' }}>
            <Home size={16} /> Household & Living Setting
          </h3>

          <div className="form-group" style={{ marginBottom: '0.85rem' }}>
            <label className="form-label">Wealth Quintile</label>
            <select
              className="form-control"
              value={formData.wealth_quintile}
              onChange={(e) => handleChange('wealth_quintile', e.target.value)}
            >
              <option value="Poorest">Poorest (Lowest 20%)</option>
              <option value="Poorer">Poorer</option>
              <option value="Middle">Middle (Median)</option>
              <option value="Richer">Richer</option>
              <option value="Richest">Richest (Top 20%)</option>
            </select>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
            <div className="form-group">
              <label className="form-label">Residence</label>
              <select
                className="form-control"
                value={formData.residence}
                onChange={(e) => handleChange('residence', e.target.value)}
              >
                <option value="Rural">Rural</option>
                <option value="Urban">Urban</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Household Size</label>
              <input
                type="number"
                className="form-control"
                value={formData.hhsize}
                onChange={(e) => handleChange('hhsize', e.target.value)}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Sanitation Risk (0–3)</label>
            <select
              className="form-control"
              value={formData.sanitation_risk_index}
              onChange={(e) => handleChange('sanitation_risk_index', e.target.value)}
            >
              <option value="0">0 — Safe Water, Improved Toilet & Clean Fuel</option>
              <option value="1">1 — Mild Deficit (e.g. Solid Cooking Fuel)</option>
              <option value="2">2 — Moderate Deficit (Unsafe Water / Unimproved Toilet)</option>
              <option value="3">3 — Severe Deficit (Unsafe Water, Open Defecation, Biomass)</option>
            </select>
          </div>
        </div>

        {/* Section 4: Morbidity & Symptoms */}
        <div className="glass-panel" style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#10b981', marginBottom: '1rem' }}>
            <Stethoscope size={16} /> Child Morbidity Symptoms
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
            <div className="form-group">
              <label className="form-label">Recent Diarrhea</label>
              <select
                className="form-control"
                value={formData.diarrhea_recent}
                onChange={(e) => handleChange('diarrhea_recent', e.target.value)}
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Recent Fever</label>
              <select
                className="form-control"
                value={formData.fever_recent}
                onChange={(e) => handleChange('fever_recent', e.target.value)}
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
            <div className="form-group">
              <label className="form-label">Recent Cough</label>
              <select
                className="form-control"
                value={formData.cough_recent}
                onChange={(e) => handleChange('cough_recent', e.target.value)}
              >
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Measles Vaccine</label>
              <select
                className="form-control"
                value={formData.measles_vaccine}
                onChange={(e) => handleChange('measles_vaccine', e.target.value)}
              >
                <option value="Yes">Yes (Vaccinated)</option>
                <option value="No">No (Unvaccinated)</option>
              </select>
            </div>
          </div>
        </div>

      </div>

      {/* Submit Button */}
      <div style={{ textAlign: 'right' }}>
        <button
          type="submit"
          disabled={loading}
          className="btn btn-primary"
          style={{ padding: '0.75rem 2rem', fontSize: '1rem', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}
        >
          {loading ? (
            <>
              <Cpu className="animate-spin" size={18} />
              Evaluating 4-Model Suite...
            </>
          ) : (
            <>
              <Sparkles size={18} />
              Screen Malnutrition Risk ({selectedModel})
            </>
          )}
        </button>
      </div>
    </form>
  );
}
