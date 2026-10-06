const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 3000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:5005";

app.use(cors());
app.use(express.json());

// Health Check
app.get("/api/health", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 3000 });
    return res.status(200).json({
      status: "healthy",
      backend: "Node.js Express",
      ml_service: response.data
    });
  } catch (error) {
    return res.status(200).json({
      status: "degraded",
      backend: "Node.js Express",
      ml_service: "ML Service not reachable. Please start Python Flask service on port 5005."
    });
  }
});

// All 4 Models Consensus Prediction Endpoint
app.post(["/api/prediction/all", "/api/prediction/dual", "/api/prediction"], async (req, res) => {

  // Anthropometric high-accuracy prediction endpoint
  app.post("/api/prediction/anthropometric", async (req, res) => {
    try {
      const response = await axios.post(`${ML_SERVICE_URL}/predict/anthropometric`, req.body, {
        timeout: 10000,
        headers: { "Content-Type": "application/json" }
      });
      return res.status(200).json(response.data);
    } catch (error) {
      if (error.response) return res.status(error.response.status).json(error.response.data);
      return res.status(503).json({ error: "ML Service Unavailable", message: "Unable to connect to the Python ML prediction service on port 5005." });
    }
  });
  try {
    const payload = req.body;
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, payload, {
      timeout: 20000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(503).json({
      error: "ML Service Unavailable",
      message: "Unable to connect to the Python ML prediction service on port 5005."
    });
  }
});

// XGBoost Prediction Endpoint
app.post("/api/prediction/xgboost", async (req, res) => {
  try {
    const payload = req.body;
    const response = await axios.post(`${ML_SERVICE_URL}/predict/xgboost`, payload, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(503).json({
      error: "ML Service Unavailable",
      message: "Unable to connect to the Python ML prediction service on port 5005."
    });
  }
});

// FT-Transformer Prediction Endpoint
app.post("/api/prediction/transformer", async (req, res) => {
  try {
    const payload = req.body;
    const response = await axios.post(`${ML_SERVICE_URL}/predict/transformer`, payload, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(503).json({
      error: "ML Service Unavailable",
      message: "Unable to connect to the Python ML prediction service on port 5005."
    });
  }
});

// DNN Prediction Endpoint
app.post("/api/prediction/dnn", async (req, res) => {
  try {
    const payload = req.body;
    const response = await axios.post(`${ML_SERVICE_URL}/predict/dnn`, payload, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(503).json({
      error: "ML Service Unavailable",
      message: "Unable to connect to the Python ML prediction service on port 5005."
    });
  }
});

// TabNet Prediction Endpoint
app.post("/api/prediction/tabnet", async (req, res) => {
  try {
    const payload = req.body;
    const response = await axios.post(`${ML_SERVICE_URL}/predict/tabnet`, payload, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(503).json({
      error: "ML Service Unavailable",
      message: "Unable to connect to the Python ML prediction service on port 5005."
    });
  }
});

// 4-Model Benchmark Comparison Endpoint
app.get(["/api/prediction/comparison", "/api/prediction/benchmark"], async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/compare`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) {
      return res.status(error.response.status).json(error.response.data);
    }
    return res.status(503).json({
      error: "ML Service Unavailable",
      message: "Unable to retrieve model comparison metrics."
    });
  }
});

app.listen(PORT, () => {
  console.log(`Node.js Express Backend running on http://localhost:${PORT}`);
  console.log(`Proxying 4-Model ML requests to ${ML_SERVICE_URL}`);
});
