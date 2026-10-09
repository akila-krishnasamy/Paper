import React, { useState, useEffect } from 'react';
import { BarChart2, CheckCircle, Cpu, Activity, RefreshCw, Sparkles, Award, ShieldCheck, Layers, Eye } from 'lucide-react';

export default function ModelComparison() {
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchBenchmark = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:3000/api/prediction/comparison');
      if (!res.ok) throw new Error('Failed to fetch 4-model comparison');
      const data = await res.json();
      setBenchmarkData(data);
    } catch (err) {
      setError(err.message || 'Unable to retrieve comparison data from server.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBenchmark();
  }, []);

  if (loading) {
    return (
      <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        <RefreshCw size={28} className="animate-spin" style={{ margin: '0 auto 1rem', display: 'block' }} />
        <p>Loading 4-model evaluation benchmark metrics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center', color: '#ef4444' }}>
        <p>Error: {error}</p>
        <button type="button" onClick={fetchBenchmark} className="btn btn-primary" style={{ marginTop: '1rem', padding: '0.5rem 1rem', fontSize: '0.85rem' }}>
          Retry Loading
        </button>
      </div>
    );
  }

  const {
    dataset = 'NFHS-5 dhs_clean.parquet (198,849 records)',
    best_overall_model = 'XGBoost',
    best_model_reason = 'Evaluated on 29,828 unseen test records.',
    overall_summary = {},
    per_condition_comparison = {}
  } = benchmarkData || {};

  const modelsList = ['XGBoost', 'Transformer', 'DNN', 'TabNet'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', animation: 'fadeIn 0.4s ease' }}>
      
      {/* Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <BarChart2 size={24} color="#6366f1" />
              4-Model DL Benchmark & Comparative Evaluation
            </h2>
            <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.35rem' }}>
              All 4 architectures trained on socio-demographic features and evaluated on <strong>29,828 held-out test records</strong> from {dataset} with calibrated high-confidence screening.
            </p>
          </div>

          <button
            type="button"
            className="btn btn-outline"
            style={{ fontSize: '0.8rem', padding: '0.4rem 0.8rem' }}
            onClick={fetchBenchmark}
          >
            <RefreshCw size={14} style={{ marginRight: '0.35rem', display: 'inline' }} />
            Refresh Metrics
          </button>
        </div>
      </div>

      {/* Leading Model Announcement Card */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(16, 185, 129, 0.1))', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Award size={36} color="#fbbf24" style={{ flexShrink: 0 }} />
          <div>
            <span style={{ fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', color: '#fbbf24', letterSpacing: '0.05em' }}>
              Primary Benchmark Winner
            </span>
            <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: '#fff', margin: '0.2rem 0' }}>
              {best_overall_model}
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              {best_model_reason}
            </p>
          </div>
        </div>
      </div>

      {/* 4-Model Overall Performance Table */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', overflowX: 'auto' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Layers size={18} color="#06b6d4" />
          Overall Test Benchmark Comparison (29,828 Held-Out Records)
        </h3>

        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.875rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-secondary)' }}>
              <th style={{ padding: '0.75rem 0.5rem' }}>Model Architecture</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Screening Accuracy (≥99% Calibrated)</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Macro F1</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Weighted F1</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Macro ROC-AUC</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Hamming Loss</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Inference Latency</th>
              <th style={{ padding: '0.75rem 0.5rem' }}>Train Time</th>
            </tr>
          </thead>
          <tbody>
            {modelsList.map(mKey => {
              const row = overall_summary[mKey] || {};
              const isLeader = mKey === best_overall_model;
              return (
                <tr
                  key={mKey}
                  style={{
                    borderBottom: '1px solid rgba(255,255,255,0.05)',
                    background: isLeader ? 'rgba(99, 102, 241, 0.08)' : 'transparent'
                  }}
                >
                  <td style={{ padding: '0.75rem 0.5rem', fontWeight: 700, color: isLeader ? '#818cf8' : 'var(--text-primary)' }}>
                    {mKey} {isLeader && '★'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem', fontWeight: 700, color: '#34d399' }}>
                    {row.accuracy !== undefined ? `${(row.accuracy * 100).toFixed(2)}%` : '—'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem', fontWeight: 700, color: '#a5b4fc' }}>
                    {row.macro_f1 !== undefined ? row.macro_f1.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>
                    {row.weighted_f1 !== undefined ? row.weighted_f1.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>
                    {row.macro_roc_auc !== undefined ? row.macro_roc_auc.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-secondary)' }}>
                    {row.hamming_loss !== undefined ? row.hamming_loss.toFixed(4) : '—'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem' }}>
                    {row.per_sample_ms !== undefined ? `${row.per_sample_ms.toFixed(4)} ms` : '—'}
                  </td>
                  <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-secondary)' }}>
                    {row.train_time_sec !== undefined ? `${row.train_time_sec.toFixed(1)}s` : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Per-Condition Breakdown Across Models */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', overflowX: 'auto' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Activity size={18} color="#10b981" />
          Per-Condition Test Accuracy & F1 Scores
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          {['Stunting', 'Wasting', 'Malnutrition'].map(cond => {
            const condData = per_condition_comparison[cond] || {};
            return (
              <div key={cond} className="glass-panel" style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)' }}>
                <h4 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem', color: '#fff' }}>
                  {cond}
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.825rem' }}>
                  {modelsList.map(mKey => {
                    const mCond = condData[mKey] || {};
                    return (
                      <div key={mKey} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0.6rem', background: 'rgba(255,255,255,0.03)', borderRadius: '6px' }}>
                        <span style={{ fontWeight: 600 }}>{mKey}:</span>
                        <span>
                          Acc: <strong style={{ color: '#34d399' }}>{mCond.accuracy !== undefined ? `${(mCond.accuracy * 100).toFixed(1)}%` : '—'}</strong> | 
                          F1: <strong style={{ color: '#818cf8' }}>{mCond.f1_score !== undefined ? mCond.f1_score.toFixed(4) : '—'}</strong> | 
                          AUC: <strong style={{ color: '#06b6d4' }}>{mCond.roc_auc !== undefined ? mCond.roc_auc.toFixed(4) : '—'}</strong>
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Architecture Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1.25rem' }}>
        
        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h4 style={{ fontWeight: 700, color: '#818cf8', marginBottom: '0.4rem' }}>1. XGBoost Baseline</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            Multi-output gradient tree ensemble. Fast, highly robust to tabular skew and non-linear interactions with SHAP explainability.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h4 style={{ fontWeight: 700, color: '#06b6d4', marginBottom: '0.4rem' }}>2. FT-Transformer</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            Feature Tokenizer Transformer with multi-head self-attention, continuous linear embeddings, and categorical entity projections.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h4 style={{ fontWeight: 700, color: '#ec4899', marginBottom: '0.4rem' }}>3. Deep Neural Network (DNN)</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            Multi-layer feedforward network with Batch Normalization, ReLU non-linearities, Dropout regularization, and Input Gradient attributions.
          </p>
        </div>

        <div className="glass-panel" style={{ padding: '1.25rem' }}>
          <h4 style={{ fontWeight: 700, color: '#10b981', marginBottom: '0.4rem' }}>4. TabNet</h4>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
            Attentive tabular transformer utilizing sequential attention, sparse regularization, and interpretable feature selection masks per decision step.
          </p>
        </div>

      </div>

    </div>
  );
}
