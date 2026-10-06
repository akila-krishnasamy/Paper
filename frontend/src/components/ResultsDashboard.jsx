import React, { useState } from 'react';
import { 
  AlertTriangle, CheckCircle, ShieldAlert, Cpu, ChevronDown, ChevronUp, 
  Sparkles, Activity, HeartPulse, ArrowRight, Stethoscope, BarChart3, 
  Printer, CheckSquare, Square, Filter, TrendingUp, Info
} from 'lucide-react';

export default function ResultsDashboard({ result, childInfo, onReset }) {
  const [selectedModelTab, setSelectedModelTab] = useState('best'); // 'best' | 'xgboost' | 'transformer' | 'dnn' | 'tabnet'
  const [riskFactorFilter, setRiskFactorFilter] = useState('all'); // 'all' | 'high' | 'moderate'
  const [checkedActions, setCheckedActions] = useState({});

  if (!result) return null;

  const isFourModel = Boolean(result.xgboost && result.transformer && result.dnn && result.tabnet);

  // Active displayed model
  let displayedResult = result;
  if (isFourModel) {
    if (selectedModelTab === 'best') {
      const bestName = (result.best_model || 'XGBoost').toLowerCase();
      displayedResult = result[bestName] || result.xgboost || result.dnn || result.tabnet;
    } else {
      displayedResult = result[selectedModelTab] || result.xgboost;
    }
  }

  const {
    model,
    architecture,
    overall_risk,
    prediction,
    probability,
    percentage,
    conditions = [],
    top_factors = [],
    explainability_method,
    medical_disclaimer
  } = displayedResult;

  const formattedDate = new Date().toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });

  const getRiskColor = (risk) => {
    if (!risk) return '#22c55e';
    const r = String(risk).toUpperCase();
    if (r.includes('HIGH')) return '#ef4444';
    if (r.includes('MODERATE')) return '#eab308';
    return '#22c55e';
  };

  const getRiskBadgeClass = (risk) => {
    if (!risk) return 'low';
    const r = String(risk).toUpperCase();
    if (r.includes('HIGH')) return 'high';
    if (r.includes('MODERATE')) return 'moderate';
    return 'low';
  };

  const stuntingCondition = conditions.find(c => c.condition === 'Stunting');
  const wastingCondition = conditions.find(c => c.condition === 'Wasting');
  const malnutritionCondition = conditions.find(c => c.condition === 'Malnutrition') || {
    percentage: percentage || (probability ? (probability * 100).toFixed(1) : 50.0),
    severity: overall_risk || 'LOW'
  };

  const probVal = parseFloat(malnutritionCondition.percentage) || 0;
  // Arc calculation for semi-circle radial gauge (circumference of r=65 is ~408; half arc is ~204)
  const arcDashOffset = 204 - (204 * (Math.min(100, Math.max(0, probVal)) / 100));

  const toggleAction = (key) => {
    setCheckedActions(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', animation: 'fadeIn 0.4s ease' }}>
      
      {/* Top Banner & Assessment Summary Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.35rem' }}>
            <HeartPulse size={26} color="#6366f1" className="animate-spin-slow" />
            <h2 className="shimmer-text" style={{ fontSize: '1.45rem', fontWeight: 800 }}>
              AI Pediatric Malnutrition Assessment
            </h2>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Child: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.childName || 'Child #1042'}</strong> | 
            Age: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.child_age_months || childInfo?.age_months} mo</strong> | 
            Sex: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.child_sex || childInfo?.gender || 'Female'}</strong> | 
            Residence: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.residence || 'Rural'}</strong> | 
            Assessment Date: {formattedDate}
          </p>
        </div>

        {/* Action Buttons & Overall Risk Score Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn btn-outline"
            style={{ fontSize: '0.8rem', padding: '0.45rem 0.9rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
            onClick={handlePrint}
            title="Print or Export Clinical PDF"
          >
            <Printer size={15} />
            Print Report
          </button>

          <div 
            className={`glass-panel ${(overall_risk || '').toUpperCase() === 'HIGH' ? 'animate-pulse-high' : ((overall_risk || '').toUpperCase() === 'MODERATE' ? 'animate-pulse-amber' : 'animate-pulse-green')}`}
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '1rem', 
              padding: '0.75rem 1.25rem', 
              borderRadius: '12px', 
              border: `1px solid ${getRiskColor(overall_risk)}55` 
            }}
          >
            <div>
              <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 700 }}>
                Clinical Risk Status
              </div>
              <div style={{ fontSize: '1.35rem', fontWeight: 900, color: getRiskColor(overall_risk) }}>
                {overall_risk || 'LOW'} RISK
              </div>
            </div>
            {(overall_risk || '').toUpperCase() === 'HIGH' ? (
              <ShieldAlert size={36} color="#ef4444" />
            ) : (overall_risk || '').toUpperCase() === 'MODERATE' ? (
              <AlertTriangle size={36} color="#eab308" />
            ) : (
              <CheckCircle size={36} color="#22c55e" />
            )}
          </div>
        </div>
      </div>

      {/* Model Selection Switcher */}
      {isFourModel && (
        <div className="glass-panel" style={{ padding: '1rem 1.5rem', background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.25)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Sparkles size={20} color="#818cf8" />
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', color: '#818cf8', letterSpacing: '0.05em' }}>
                Active Consensus Model Engine
              </span>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Showing attributions for: <strong style={{ color: '#fff' }}>{model}</strong>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'best' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('best')}
            >
              ★ Leading Model ({result.best_model || 'XGBoost'})
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'xgboost' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('xgboost')}
            >
              XGBoost
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'transformer' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('transformer')}
            >
              FT-Transformer
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'dnn' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('dnn')}
            >
              DNN
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'tabnet' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('tabnet')}
            >
              TabNet
            </button>
          </div>
        </div>
      )}

      {/* Primary Malnutrition Prediction Card & Details */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
        
        {/* Main Result Card with Animated Radial Arc Gauge */}
        <div className="glass-panel" style={{ padding: '1.75rem', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: '-10px', right: '-10px', width: '90px', height: '90px', background: `radial-gradient(circle, ${getRiskColor(overall_risk)}22 0%, transparent 70%)` }} />
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Primary Malnutrition Screening
              </span>
              <h3 style={{ fontSize: '1.35rem', fontWeight: 800, marginTop: '0.2rem', color: 'var(--text-primary)' }}>
                {prediction || 'Normal (Well-Nourished)'}
              </h3>
            </div>
            <span className={`badge ${getRiskBadgeClass(overall_risk)}`}>
              {overall_risk || 'LOW'}
            </span>
          </div>

          {/* Semi-Circle SVG Radial Gauge */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '0.5rem 0 1rem 0' }}>
            <div style={{ position: 'relative', width: '180px', height: '105px', display: 'flex', justifyContent: 'center' }}>
              <svg width="180" height="105" viewBox="0 0 180 105">
                <defs>
                  <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#22c55e" />
                    <stop offset="50%" stopColor="#eab308" />
                    <stop offset="100%" stopColor="#ef4444" />
                  </linearGradient>
                </defs>
                {/* Background Arc */}
                <path
                  d="M 25 95 A 65 65 0 0 1 155 95"
                  fill="none"
                  stroke="rgba(255,255,255,0.08)"
                  strokeWidth="14"
                  strokeLinecap="round"
                />
                {/* Value Arc */}
                <path
                  d="M 25 95 A 65 65 0 0 1 155 95"
                  fill="none"
                  stroke="url(#gaugeGrad)"
                  strokeWidth="14"
                  strokeLinecap="round"
                  strokeDasharray="204"
                  strokeDashoffset={arcDashOffset}
                  style={{ transition: 'stroke-dashoffset 1s ease-out' }}
                />
              </svg>

              <div style={{ position: 'absolute', bottom: '5px', textAlign: 'center' }}>
                <span style={{ fontSize: '1.75rem', fontWeight: 900, color: getRiskColor(overall_risk), lineHeight: 1 }}>
                  {malnutritionCondition.percentage}%
                </span>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Risk Index
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: '0.75rem', color: 'var(--text-muted)', padding: '0 0.5rem' }}>
              <span>0% Low</span>
              <span>50% Threshold</span>
              <span>100% Critical</span>
            </div>
          </div>

          <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <strong>Engine:</strong> {model} &bull; <span style={{ color: '#a5b4fc' }}>{architecture || 'Multi-Output Neural Tabular'}</span>
          </div>
        </div>

        {/* Stunting & Wasting Breakdown Card */}
        <div className="glass-panel" style={{ padding: '1.75rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} color="#06b6d4" />
            WHO Anthropometric Condition Breakdown
          </h3>

          {/* Stunting Card */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <div>
                <strong style={{ fontSize: '0.95rem' }}>Stunting (Height-for-Age Deficit)</strong>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Chronic nutritional deprivation (HAZ &le; -2.0 SD)</div>
              </div>
              <span className={`badge ${getRiskBadgeClass(stuntingCondition?.severity || 'low')}`}>
                {stuntingCondition?.severity || 'LOW'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Risk Probability:</span>
              <strong style={{ color: getRiskColor(stuntingCondition?.severity) }}>{stuntingCondition?.percentage || 0}%</strong>
            </div>
          </div>

          {/* Wasting Card */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <div>
                <strong style={{ fontSize: '0.95rem' }}>Wasting (Weight-for-Height Deficit)</strong>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Acute nutritional deficiency (WHZ &le; -2.0 SD)</div>
              </div>
              <span className={`badge ${getRiskBadgeClass(wastingCondition?.severity || 'low')}`}>
                {wastingCondition?.severity || 'LOW'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Risk Probability:</span>
              <strong style={{ color: getRiskColor(wastingCondition?.severity) }}>{wastingCondition?.percentage || 0}%</strong>
            </div>
          </div>
        </div>

      </div>

      {/* Model Explainability & Feature Contribution */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <BarChart3 size={20} color="#6366f1" />
              Model Explainability & Key Driving Factors
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Attribution method: <strong style={{ color: '#a5b4fc' }}>{explainability_method || 'Feature Importance Analysis'}</strong>
            </p>
          </div>
        </div>

        {top_factors && top_factors.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            {top_factors.map((factor, idx) => (
              <div
                key={idx}
                style={{
                  padding: '1rem',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '10px',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderLeft: factor.impact_level === 'High Impact' ? '4px solid #ef4444' : (factor.impact_level === 'Moderate Impact' ? '4px solid #eab308' : '4px solid #6366f1')
                }}
              >
                <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.35rem' }}>
                  {factor.feature}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Relative Weight:</span>
                  <strong style={{ color: '#fff' }}>{factor.contribution_pct}%</strong>
                </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: factor.impact_level === 'High Impact' ? '#ef4444' : (factor.impact_level === 'Moderate Impact' ? '#eab308' : '#818cf8'), fontWeight: 600 }}>
                  ● {factor.impact_level}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>No feature importance data returned for this case.</p>
        )}
      </div>

      {/* Identified Patient Risk Factors Panel */}
      {(() => {
        const factors = [];
        const motherBmi = parseFloat(childInfo?.mother_bmi || 21);
        const birthWt = parseFloat(childInfo?.birth_weight || 3.0);
        const anc = parseFloat(childInfo?.anc_visits || 4);
        const age = parseFloat(childInfo?.child_age_months || childInfo?.age_months || 24);
        const bf = parseFloat(childInfo?.breastfeeding_duration || 12);
        const sanitation = parseFloat(childInfo?.sanitation_risk_index || 0);
        const hhsize = parseFloat(childInfo?.hhsize || 5);

        if (motherBmi < 18.5) {
          factors.push({
            title: "Maternal Underweight / Energy Deficiency",
            desc: `Mother's BMI is ${motherBmi.toFixed(1)} kg/m² (< 18.5). Intergenerational transmission triples stunting risk.`,
            severity: "High",
            badgeColor: "#ef4444"
          });
        }
        if (birthWt < 2.5) {
          factors.push({
            title: "Low Birth Weight (LBW / Preterm)",
            desc: `Birth weight of ${birthWt} kg (< 2.5 kg). Child started life with impaired organ growth and low fat stores.`,
            severity: "High",
            badgeColor: "#ef4444"
          });
        }
        if (anc < 4) {
          factors.push({
            title: "Inadequate Antenatal Care (ANC)",
            desc: `Only ${anc} antenatal visit(s) attended (WHO recommends ≥ 4 visits for optimal maternal-fetal nutrition).`,
            severity: "Moderate",
            badgeColor: "#eab308"
          });
        }
        if (age <= 24 && bf < 6) {
          factors.push({
            title: "Suboptimal / Premature Weaning",
            desc: `Breastfeeding duration of ${bf} month(s). Missing protective maternal antibodies and exclusive feeding window.`,
            severity: "Moderate",
            badgeColor: "#eab308"
          });
        }
        if (childInfo?.diarrhea_recent === 'Yes') {
          factors.push({
            title: "Active Enteric Infection / Diarrhea",
            desc: "Recent diarrheal episode within 14 days causing acute nutrient loss, mucosal blunting, and electrolyte depletion.",
            severity: "High",
            badgeColor: "#ef4444"
          });
        }
        if (childInfo?.fever_recent === 'Yes' || childInfo?.cough_recent === 'Yes') {
          factors.push({
            title: "Recent Febrile / Respiratory Morbidity",
            desc: "Acute infectious illness elevates hypermetabolic catabolism, rapidly triggering nutritional wasting.",
            severity: "Moderate",
            badgeColor: "#eab308"
          });
        }
        if (sanitation >= 2) {
          factors.push({
            title: "High Environmental & Water Sanitation Risk",
            desc: `Sanitation index ${sanitation}/3. Unimproved water or toilet exposure leads to Environmental Enteric Dysfunction (EED).`,
            severity: "High",
            badgeColor: "#ef4444"
          });
        }
        if (childInfo?.measles_vaccine === 'No') {
          factors.push({
            title: "Unvaccinated Against Measles",
            desc: "Unimmunized children face up to 3x higher post-measles severe wasting and immunosuppression.",
            severity: "Moderate",
            badgeColor: "#eab308"
          });
        }
        if (childInfo?.wealth_quintile === 'Poorest' || childInfo?.wealth_quintile === 'Poorer') {
          factors.push({
            title: "Severe Socioeconomic Deprivation",
            desc: `Household in the ${childInfo?.wealth_quintile} wealth tier with limited dietary diversity access.`,
            severity: "Moderate",
            badgeColor: "#eab308"
          });
        }

        const filteredFactors = factors.filter(f => {
          if (riskFactorFilter === 'high') return f.severity === 'High';
          if (riskFactorFilter === 'moderate') return f.severity === 'Moderate';
          return true;
        });

        return (
          <div className="glass-panel" style={{ padding: '1.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div>
                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <AlertTriangle size={20} color="#f59e0b" />
                  Patient-Specific Risk Factors Identified ({factors.length} Active)
                </h3>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  Clinical and socio-environmental risk drivers detected from the child's screening profile:
                </p>
              </div>

              {/* Filter Buttons */}
              <div style={{ display: 'flex', gap: '0.4rem', background: 'rgba(255,255,255,0.03)', padding: '0.2rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                <button
                  type="button"
                  onClick={() => setRiskFactorFilter('all')}
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.25rem 0.65rem',
                    borderRadius: '6px',
                    border: 'none',
                    background: riskFactorFilter === 'all' ? '#6366f1' : 'transparent',
                    color: riskFactorFilter === 'all' ? '#fff' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  All ({factors.length})
                </button>
                <button
                  type="button"
                  onClick={() => setRiskFactorFilter('high')}
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.25rem 0.65rem',
                    borderRadius: '6px',
                    border: 'none',
                    background: riskFactorFilter === 'high' ? '#ef4444' : 'transparent',
                    color: riskFactorFilter === 'high' ? '#fff' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  High ({factors.filter(f => f.severity === 'High').length})
                </button>
                <button
                  type="button"
                  onClick={() => setRiskFactorFilter('moderate')}
                  style={{
                    fontSize: '0.75rem',
                    padding: '0.25rem 0.65rem',
                    borderRadius: '6px',
                    border: 'none',
                    background: riskFactorFilter === 'moderate' ? '#f59e0b' : 'transparent',
                    color: riskFactorFilter === 'moderate' ? '#fff' : 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontWeight: 600
                  }}
                >
                  Moderate ({factors.filter(f => f.severity === 'Moderate').length})
                </button>
              </div>
            </div>

            {filteredFactors.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
                {filteredFactors.map((f, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '1rem',
                      background: 'rgba(255,255,255,0.02)',
                      borderRadius: '10px',
                      border: '1px solid rgba(255,255,255,0.06)',
                      borderLeft: `4px solid ${f.badgeColor}`,
                      transition: 'transform 0.2s ease, border-color 0.2s ease'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                      <strong style={{ fontSize: '0.9rem', color: '#fff' }}>{f.title}</strong>
                      <span style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: `${f.badgeColor}22`, color: f.badgeColor, fontWeight: 700 }}>
                        {f.severity} Risk
                      </span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.4 }}>
                      {f.desc}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: '1rem', background: 'rgba(34, 197, 94, 0.05)', border: '1px solid rgba(34, 197, 94, 0.2)', borderRadius: '8px', color: '#22c55e', fontSize: '0.85rem', marginTop: '1rem' }}>
                ✓ No active risk factors in this filter category.
              </div>
            )}
          </div>
        );
      })()}

      {/* Severe Secondary Disease & Complication Risk Alerts */}
      <div className="glass-panel" style={{ padding: '1.75rem', background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.05), rgba(99, 102, 241, 0.03))', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#f87171' }}>
          <ShieldAlert size={20} color="#ef4444" />
          Severe Secondary Disease Prognosis & Complication Alerts
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
          Predicted clinical vulnerability to severe comorbidities based on NFHS-5 multi-condition epidemiology:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
          
          {/* Anaemia requires a labelled haemoglobin measurement model. */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>Pediatric Anaemia</span>
              <span style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: '#64748b22', color: '#94a3b8', fontWeight: 700 }}>
                Not Evaluated
              </span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4, margin: '0 0 0.5rem 0' }}>
              No haemoglobin measurement or labelled anaemia outcome is available in this screening dataset.
            </p>
            <div style={{ fontSize: '0.72rem', color: '#a5b4fc' }}>
              Action: Obtain a validated haemoglobin test and clinical assessment.
            </div>
          </div>

          {/* Alert 2: Acute Pneumonia / ALRI */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>Acute Lower Respiratory (ALRI)</span>
              <span style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: (childInfo?.cough_recent === 'Yes' || childInfo?.fever_recent === 'Yes' ? '#ef444422' : '#22c55e22'), color: (childInfo?.cough_recent === 'Yes' || childInfo?.fever_recent === 'Yes' ? '#ef4444' : '#22c55e'), fontWeight: 700 }}>
                {childInfo?.cough_recent === 'Yes' ? 'Active Warning' : 'Low Active Risk'}
              </span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4, margin: '0 0 0.5rem 0' }}>
              Wasting weakens diaphragm and intercostal muscles, increasing pediatric pneumonia fatality up to 9-fold.
            </p>
            <div style={{ fontSize: '0.72rem', color: '#a5b4fc' }}>
              Action: Count respiratory rate (&gt;50/min is tachypnea); screen for chest indrawing.
            </div>
          </div>

          {/* Alert 3: Invasive Dysentery & Dehydration */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#fff' }}>Severe Enteric Colitis & Dysentery</span>
              <span style={{ fontSize: '0.7rem', padding: '0.2rem 0.5rem', borderRadius: '4px', background: (childInfo?.diarrhea_recent === 'Yes' ? '#ef444422' : '#22c55e22'), color: (childInfo?.diarrhea_recent === 'Yes' ? '#ef4444' : '#22c55e'), fontWeight: 700 }}>
                {childInfo?.diarrhea_recent === 'Yes' ? 'Active Infection' : 'Low Active Risk'}
              </span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.4, margin: '0 0 0.5rem 0' }}>
              Blunted gut mucosal lining risks bacterial invasion, hypovolemic dehydration, and septic shock.
            </p>
            <div style={{ fontSize: '0.72rem', color: '#a5b4fc' }}>
              Action: Immediate low-osmolarity ORS + Zinc supplementation (20 mg/day for 14 days).
            </div>
          </div>

        </div>
      </div>

      {/* Clinical & Nutritional Support Recommendations (WHO / ICDS Guidelines) */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, marginBottom: '0.4rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <Stethoscope size={20} color="#10b981" />
          Clinical & Nutritional Support Recommendations (WHO &amp; ICDS Protocol)
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          Tailored intervention guidelines matched to the child's <strong>{overall_risk || 'LOW'} RISK</strong> tier:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
          
          {/* Recommendation 1: Nutrition */}
          <div style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#34d399', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🥗 Targeted Nutritional Rehabilitation
            </h4>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6, paddingLeft: '1.2rem', margin: 0 }}>
              {overall_risk === 'HIGH' ? (
                <>
                  <li>Enroll in Anganwadi supplementary therapeutic ration (THR / fortified bal ahar).</li>
                  <li>Provide Ready-to-Use Therapeutic Food (RUTF) or calorie-dense mashed foods (khichdi with ghee, mashed egg).</li>
                  <li>Increase meal frequency to <strong>5–6 small nutrient-rich feeds daily</strong>.</li>
                  <li>Continue intensive breastfeeding alongside energy-dense complementary feeding.</li>
                </>
              ) : overall_risk === 'MODERATE' ? (
                <>
                  <li>Increase daily dietary diversity to at least 4 food groups (dairy, eggs/pulses, yellow fruits, greens).</li>
                  <li>Provide 3–4 meals plus 2 nutritious snacks per day (banana, roasted chana, milk).</li>
                  <li>Maternal dietary counseling on clean food preparation and child active feeding.</li>
                </>
              ) : (
                <>
                  <li>Maintain age-appropriate balanced family diet rich in proteins, vegetables, and fruit.</li>
                  <li>Ensure regular intake of milk, pulses, and locally available seasonal produce.</li>
                </>
              )}
            </ul>
          </div>

          {/* Recommendation 2: Medical & Micronutrients */}
          <div style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#60a5fa', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              💊 Micronutrient & Medical Prophylaxis
            </h4>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6, paddingLeft: '1.2rem', margin: 0 }}>
              <li><strong>Iron-Folic Acid (IFA)</strong>: 1 ml pediatric syrup (20 mg elemental iron) daily for 100 days.</li>
              <li><strong>Vitamin A Dose</strong>: Biannual high-dose oral vitamin A (100,000 IU for &lt;1 yr, 200,000 IU for 1–5 yrs).</li>
              <li><strong>Deworming</strong>: Age-appropriate Albendazole (200 mg if &lt;2 yrs, 400 mg if &gt;2 yrs) every 6 months.</li>
              {childInfo?.diarrhea_recent === 'Yes' && (
                <li style={{ color: '#f87171' }}><strong>Zinc &amp; ORS</strong>: 20 mg Zinc daily for 14 days + low-osmolarity ORS solution.</li>
              )}
              {childInfo?.measles_vaccine === 'No' && (
                <li style={{ color: '#f87171' }}><strong>Catch-Up Immunization</strong>: Schedule immediate Measles-Rubella (MR) vaccine dose.</li>
              )}
            </ul>
          </div>

          {/* Recommendation 3: Clinical Care Protocol */}
          <div style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f472b6', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              🏥 Clinical Care & Referral Protocol
            </h4>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6, paddingLeft: '1.2rem', margin: 0 }}>
              {overall_risk === 'HIGH' ? (
                <>
                  <li style={{ color: '#f87171', fontWeight: 700 }}>Immediate referral to Nutrition Rehabilitation Centre (NRC) or Taluk/District Hospital.</li>
                  <li>Complete physical anthropometric assessment: Height/Length (infantometer) and Weight (Salter scale).</li>
                  <li>Clinical evaluation for bilateral pedal edema (Kwashiorkor check).</li>
                  <li>Weekly growth monitoring follow-up with ASHA/ANM health worker.</li>
                </>
              ) : overall_risk === 'MODERATE' ? (
                <>
                  <li>Refer to Primary Health Centre (PHC) / Anganwadi worker for monthly growth chart plotting.</li>
                  <li>Monitor weight gain trajectory over the next 30 days.</li>
                  <li>Re-screen if any fever, cough, or diarrheal episode recurs.</li>
                </>
              ) : (
                <>
                  <li>Continue routine bi-annual growth monitoring at local health sub-centre.</li>
                  <li>Ensure adherence to universal immunization schedule.</li>
                </>
              )}
            </ul>
          </div>

          {/* Recommendation 4: WASH & Sanitation */}
          <div style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fbbf24', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              💧 WASH & Infection Prevention
            </h4>
            <ul style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6, paddingLeft: '1.2rem', margin: 0 }}>
              <li>Boil drinking water for at least 1 minute or use reliable chlorine water purification.</li>
              <li>Strict handwashing with soap before preparing infant food and after toilet use.</li>
              <li>Safe disposal of infant feces in an improved latrine to prevent enteric environmental enteropathy.</li>
              <li>Store cooked complementary foods in clean, covered containers consumed within 2 hours.</li>
            </ul>
          </div>

        </div>

        {/* Interactive Clinical Directive Action Tracker */}
        {(() => {
          const actionList = [
            { id: 'act1', label: 'Dispense 100-day pediatric Iron-Folic Acid (IFA) syrup prescription' },
            { id: 'act2', label: 'Verify & administer bi-annual oral high-dose Vitamin A capsule' },
            { id: 'act3', label: overall_risk === 'HIGH' ? 'Generate emergency referral slip for Nutrition Rehabilitation Centre (NRC)' : 'Schedule 30-day follow-up growth chart plotting at Anganwadi' },
            { id: 'act4', label: 'Provide maternal feeding & dietary diversity counseling (≥4 food groups)' },
            { id: 'act5', label: childInfo?.diarrhea_recent === 'Yes' ? 'Dispense 14-day Zinc tablet blister pack + 2 low-osmolarity ORS packets' : 'Schedule biannual pediatric Albendazole deworming dose' }
          ];

          const completedCount = actionList.filter(a => checkedActions[a.id]).length;
          const pctDone = Math.round((completedCount / actionList.length) * 100);

          return (
            <div style={{ marginTop: '1.75rem', padding: '1.25rem', background: 'rgba(99, 102, 241, 0.05)', borderRadius: '12px', border: '1px solid rgba(99, 102, 241, 0.2)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <CheckSquare size={18} color="#818cf8" />
                  <strong style={{ fontSize: '0.95rem', color: '#fff' }}>Clinician / Health Worker Directive Tracker</strong>
                </div>
                <span style={{ fontSize: '0.8rem', color: pctDone === 100 ? '#22c55e' : '#a5b4fc', fontWeight: 700 }}>
                  {completedCount} of {actionList.length} Directives Enacted ({pctDone}%)
                </span>
              </div>

              {/* Progress bar */}
              <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.08)', borderRadius: '999px', overflow: 'hidden', marginBottom: '1rem' }}>
                <div
                  style={{
                    width: `${pctDone}%`,
                    height: '100%',
                    background: pctDone === 100 ? '#22c55e' : 'linear-gradient(90deg, #6366f1, #38bdf8)',
                    transition: 'width 0.4s ease'
                  }}
                />
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {actionList.map(item => {
                  const isDone = Boolean(checkedActions[item.id]);
                  return (
                    <div
                      key={item.id}
                      className={`checklist-item ${isDone ? 'completed' : ''}`}
                      onClick={() => toggleAction(item.id)}
                    >
                      {isDone ? (
                        <CheckSquare size={18} color="#22c55e" style={{ flexShrink: 0 }} />
                      ) : (
                        <Square size={18} color="var(--text-muted)" style={{ flexShrink: 0 }} />
                      )}
                      <span style={{ fontSize: '0.825rem', color: isDone ? '#22c55e' : 'var(--text-primary)' }}>
                        {item.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })()}

      </div>

      {/* Mandatory Medical Safety Disclaimer */}
      <div className="glass-panel" style={{ padding: '1.25rem 1.75rem', background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.25)', display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
        <Stethoscope size={24} color="#ef4444" style={{ flexShrink: 0, marginTop: '0.2rem' }} />
        <div>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ef4444', marginBottom: '0.25rem' }}>
            Clinical Safety & Ethical Disclaimer
          </h4>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            This system provides <strong>AI-based malnutrition risk screening for academic and research purposes</strong> and is <strong>not a medical diagnosis</strong>. 
            Any child flagged with Moderate or High risk should immediately be referred to a qualified pediatrician or public health worker (ANM/ICDS) for clinical measurement and intervention.
          </p>
        </div>
      </div>

      {/* Action Footer */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
        <button
          type="button"
          className="btn btn-outline"
          onClick={onReset}
        >
          Screen Another Child
        </button>
      </div>

    </div>
  );
}
