import React, { useState } from 'react';
import ChildForm from './components/ChildForm';
import ResultsDashboard from './components/ResultsDashboard';
import ModelComparison from './components/ModelComparison';
import { Activity, Shield, Cpu, RefreshCw, AlertCircle, Sparkles } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('screening'); // 'screening' | 'comparison'
  const [selectedModel, setSelectedModel] = useState('Compare All');
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [lastSubmittedChild, setLastSubmittedChild] = useState(null);
  const [apiError, setApiError] = useState(null);

  const handleFormSubmit = async (formData) => {
    setLoading(true);
    setApiError(null);
    setPredictionResult(null);
    setLastSubmittedChild(formData);

    let endpoint = 'http://localhost:3000/api/prediction/all';
    if (selectedModel === 'Anthropometric') {
      endpoint = 'http://localhost:3000/api/prediction/anthropometric';
    } else if (selectedModel === 'XGBoost') {
      endpoint = 'http://localhost:3000/api/prediction/xgboost';
    } else if (selectedModel === 'Transformer') {
      endpoint = 'http://localhost:3000/api/prediction/transformer';
    } else if (selectedModel === 'DNN') {
      endpoint = 'http://localhost:3000/api/prediction/dnn';
    } else if (selectedModel === 'TabNet') {
      endpoint = 'http://localhost:3000/api/prediction/tabnet';
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || errData.message || 'The AI prediction service encountered an issue.');
      }

      const result = await response.json();
      setPredictionResult(result);
    } catch (err) {
      setApiError(err.message || 'Unable to connect to prediction backend server on port 3000.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setPredictionResult(null);
    setApiError(null);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Header Navbar */}
      <header className="navbar">
        <div className="brand-logo">
          <Shield size={26} color="#6366f1" />
          <span>NutriPredict AI — 4-Model Pediatric Malnutrition Suite</span>
        </div>

        <nav className="nav-tabs">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'screening' ? 'active' : ''}`}
            onClick={() => setActiveTab('screening')}
          >
            <Activity size={16} style={{ display: 'inline', marginRight: '0.4rem' }} />
            Malnutrition Risk Screening
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveTab('comparison')}
          >
            <Cpu size={16} style={{ display: 'inline', marginRight: '0.4rem' }} />
            4-Model Benchmark Comparison
          </button>
        </nav>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '2rem 1.5rem', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
        {activeTab === 'screening' && (
          <div>
            {!predictionResult && (
              <ChildForm
                onSubmit={handleFormSubmit}
                loading={loading}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
              />
            )}

            {apiError && (
              <div className="glass-panel" style={{ padding: '1.5rem', marginTop: '1.5rem', borderColor: '#ef4444', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <AlertCircle size={28} color="#ef4444" />
                <div>
                  <h4 style={{ color: '#ef4444', fontWeight: 700 }}>Assessment Service Error</h4>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{apiError}</p>
                </div>
              </div>
            )}

            {predictionResult && (
              <ResultsDashboard
                result={predictionResult}
                childInfo={lastSubmittedChild}
                onReset={handleReset}
              />
            )}
          </div>
        )}

        {activeTab === 'comparison' && (
          <ModelComparison />
        )}
      </main>

      {/* Footer with Medical Disclaimer */}
      <footer style={{ borderTop: '1px solid rgba(255,255,255,0.06)', padding: '1.25rem 2rem', textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
        <p>
          <strong>Clinical Research Disclaimer:</strong> This system provides AI-based malnutrition risk screening for academic and research purposes and is not a medical diagnosis.
        </p>
        <p style={{ marginTop: '0.35rem', color: 'var(--text-muted)' }}>
          Powered by XGBoost, FT-Transformer, Deep Neural Network (DNN), and TabNet • Trained on NFHS-5 Survey Records
        </p>
      </footer>

    </div>
  );
}
