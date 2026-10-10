import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, Cell
} from 'recharts';
import { BarChart3, PieChart, Activity, AlertCircle, RefreshCw, Layers, ShieldCheck, Flame } from 'lucide-react';

export default function AnalyticsCharts() {
  const [distributionData, setDistributionData] = useState([]);
  const [importanceData, setImportanceData] = useState([]);
  const [cmData, setCmData] = useState(null);
  const [selectedCmModel, setSelectedCmModel] = useState('DNN');
  const [loading, setLoading] = useState(true);

  const verifiedAccuracyData = [
    { model: 'Early-risk DNN', accuracy: 52.62, baseline: 50.08 },
    { model: 'Current-status DNN', accuracy: 98.03, baseline: null },
    { model: 'NFHS-5 anemia DNN', accuracy: 67.20, baseline: 66.78 },
    { model: 'NFHS-5 ARI DNN', accuracy: 97.70, baseline: 97.70 },
  ];

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        // 1. Distribution
        const resDist = await fetch('http://localhost:3000/api/analytics/malnutrition-distribution');
        if (resDist.ok) {
          const d = await resDist.json();
          setDistributionData(d.categories || []);
        }

        // 2. Feature Importance
        const resImp = await fetch('http://localhost:3000/api/analytics/feature-importance');
        if (resImp.ok) {
          const imp = await resImp.json();
          setImportanceData(imp.features || []);
        }

        // 3. Confusion Matrix
        const resCm = await fetch('http://localhost:3000/api/models/confusion-matrix');
        if (resCm.ok) {
          const cm = await resCm.json();
          setCmData(cm);
        }

      } catch (err) {
        console.error('Failed to fetch analytics data:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const deficiencyData = [
    { name: 'Anemia / Iron Deficiency', screenedRisk: 38.3, color: '#ef4444', status: 'Active (Laboratory Model)' },
    { name: 'Vitamin A Deficiency Risk', screenedRisk: 22.5, color: '#f59e0b', status: 'Dietary Protocol' },
    { name: 'Vitamin D Deficiency', screenedRisk: 0, color: '#64748b', status: 'Unavailable (No Biomarker)' },
    { name: 'Vitamin B12 Deficiency', screenedRisk: 0, color: '#64748b', status: 'Unavailable (No Biomarker)' },
    { name: 'Folate Deficiency', screenedRisk: 0, color: '#64748b', status: 'Unavailable (No Biomarker)' }
  ];

  const earlyWarningData = [
    { level: 'High Risk (Flagged for Immediate Care)', count: 71888, pct: '36.2%', color: '#ef4444' },
    { level: 'Moderate Risk (Targeted Supplementary Nutrition)', count: 27385, pct: '13.8%', color: '#f59e0b' },
    { level: 'Low / Normal Risk (Routine Growth Monitoring)', count: 99576, pct: '50.1%', color: '#10b981' }
  ];

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <RefreshCw size={28} className="animate-spin" style={{ margin: '0 auto 1rem', display: 'block' }} />
        <p>Loading Deep Learning Analytics and Visualization Dashboards...</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeIn 0.3s ease' }}>
      
      {/* Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <BarChart3 size={24} color="#6366f1" />
          Pediatric Malnutrition & Deep Learning Analytics Suite
        </h2>
        <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
          Real-time visualizations connecting NFHS-5 survey populations, DL architecture benchmark comparisons, early deficiency screening, and SHAP feature importance.
        </p>
      </div>

      {/* Grid: Chart A & Chart B */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '1.5rem' }}>
        
        {/* Chart A: Malnutrition Distribution */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <PieChart size={18} color="#10b981" />
            Chart A: NFHS-5 Malnutrition Category Distribution (198,849 Records)
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Prevalence across WHO standard nutritional classifications in survey records.
          </p>

          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="category" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-15} textAnchor="end" height={60} />
                <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  formatter={(val, name, item) => [`${val.toLocaleString()} children (${item.payload.percentage}%)`, 'Count']}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {distributionData.map((entry, idx) => (
                    <Cell key={`cell-${idx}`} fill={entry.color || '#6366f1'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart B: DL Performance Grouped Bar Chart */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={18} color="#6366f1" />
            Chart B: Verified DNN Accuracy by Prediction Task (%)
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Prediction-derived held-out results. Current-status and early-risk malnutrition are different tasks.
          </p>

          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={verifiedAccuracyData} margin={{ top: 10, right: 20, left: 0, bottom: 55 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="model" interval={0} angle={-18} textAnchor="end" height={65} tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                <Legend />
                <Bar dataKey="accuracy" name="Test accuracy" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="baseline" name="Majority baseline" fill="#64748b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Grid: Chart C & Chart D */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '1.5rem' }}>
        
        {/* Chart C: Vitamin & Micronutrient Deficiency Screening */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} color="#f43f5e" />
            Chart C: Early Micronutrient Deficiency Screening Risk (%)
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Screening rates across supported biomarkers. Unsupported vitamins explicitly marked as 0% / Unavailable.
          </p>

          <div style={{ height: '280px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={deficiencyData} layout="vertical" margin={{ top: 10, right: 30, left: 80, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={120} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  formatter={(val, name, item) => [`${val}% (${item.payload.status})`, 'Risk']}
                />
                <Bar dataKey="screenedRisk" radius={[0, 4, 4, 0]}>
                  {deficiencyData.map((entry, idx) => (
                    <Cell key={`cell-def-${idx}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart D: Early-Warning Dashboard */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Flame size={18} color="#f59e0b" />
            Chart D: Clinical Early-Warning Triage Breakdown
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Children flagged for immediate nutritional intervention under calibrated 99% screening thresholds.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
            {earlyWarningData.map((item, idx) => (
              <div key={idx} style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', borderLeft: `4px solid ${item.color}` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                  <strong style={{ fontSize: '0.9rem', color: '#fff' }}>{item.level}</strong>
                  <span style={{ fontSize: '0.9rem', fontWeight: 800, color: item.color }}>{item.count.toLocaleString()} ({item.pct})</span>
                </div>
                <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div style={{ width: item.pct, height: '100%', background: item.color, borderRadius: '4px' }} />
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Grid: Chart E & Chart F */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '1.5rem' }}>
        
        {/* Chart E: Feature Importance */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <ShieldCheck size={18} color="#38bdf8" />
            Chart E: Model Explainability — Driving Predictors (SHAP / Gradient)
          </h3>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
            Relative global influence of features on pediatric growth outcome predictions.
          </p>

          <div style={{ height: '280px', width: '100%' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importanceData} layout="vertical" margin={{ top: 10, right: 20, left: 60, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} width={110} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                  formatter={(val) => [(val * 100).toFixed(1) + '%', 'Attribution']}
                />
                <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart F: Confusion Matrix */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={18} color="#ec4899" />
              Chart F: Confusion Matrix (Held-Out Test Set)
            </h3>
            <select
              className="form-control"
              style={{ width: 'auto', padding: '0.2rem 0.5rem', fontSize: '0.8rem' }}
              value={selectedCmModel}
              onChange={(e) => setSelectedCmModel(e.target.value)}
            >
              <option value="DNN">DNN (Deep Neural Network)</option>
              <option value="FT-Transformer">FT-Transformer</option>
              <option value="XGBoost">XGBoost</option>
            </select>
          </div>

          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
            True vs Predicted classifications on 29,828 held-out records for Malnutrition screening.
          </p>

          {cmData && cmData[selectedCmModel] && cmData[selectedCmModel].Malnutrition ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', textAlign: 'center' }}>
              <div style={{ padding: '1.25rem', background: 'rgba(16,185,129,0.15)', border: '1px solid #10b981', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#34d399' }}>
                  {cmData[selectedCmModel].Malnutrition.tp.toLocaleString()}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  True Positive (Correct Malnourished)
                </div>
              </div>

              <div style={{ padding: '1.25rem', background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f87171' }}>
                  {cmData[selectedCmModel].Malnutrition.fp.toLocaleString()}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  False Positive (False Alarm)
                </div>
              </div>

              <div style={{ padding: '1.25rem', background: 'rgba(239,68,68,0.15)', border: '1px solid #ef4444', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f87171' }}>
                  {cmData[selectedCmModel].Malnutrition.fn.toLocaleString()}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  False Negative (Missed Detection)
                </div>
              </div>

              <div style={{ padding: '1.25rem', background: 'rgba(16,185,129,0.15)', border: '1px solid #10b981', borderRadius: '8px' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#34d399' }}>
                  {cmData[selectedCmModel].Malnutrition.tn.toLocaleString()}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  True Negative (Correct Normal)
                </div>
              </div>
            </div>
          ) : (
            <p style={{ color: 'var(--text-muted)' }}>No confusion matrix data available.</p>
          )}

        </div>

      </div>

    </div>
  );
}
