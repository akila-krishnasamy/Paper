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
      dl_service: response.data
    });
  } catch (error) {
    return res.status(200).json({
      status: "degraded",
      backend: "Node.js Express",
      dl_service: "DL Service not reachable. Please start Python Flask service on port 5005."
    });
  }
});

// Anthropometric high-accuracy prediction endpoint (99.8% test accuracy)
app.post("/api/prediction/anthropometric", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/anthropometric`, req.body, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to connect to the Python DL prediction service on port 5005." });
  }
});

// All Models Consensus Prediction Endpoint
app.post(["/api/prediction/all", "/api/prediction/dual", "/api/prediction", "/api/predict/malnutrition"], async (req, res) => {
  try {
    const payload = req.body;
    const response = await axios.post(`${ML_SERVICE_URL}/predict`, payload, {
      timeout: 20000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({
      error: "DL Service Unavailable",
      message: "Unable to connect to the Python DL prediction service on port 5005."
    });
  }
});

// Individual Model Endpoints
app.post("/api/prediction/xgboost", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/xgboost`, req.body, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to connect to the Python DL prediction service on port 5005." });
  }
});

app.post("/api/prediction/transformer", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/transformer`, req.body, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to connect to the Python DL prediction service on port 5005." });
  }
});

app.post("/api/prediction/dnn", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/dnn`, req.body, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to connect to the Python DL prediction service on port 5005." });
  }
});

app.post("/api/prediction/tabnet", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/tabnet`, req.body, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to connect to the Python DL prediction service on port 5005." });
  }
});

// Early Vitamin & Micronutrient Deficiency Screening
app.post("/api/predict/deficiencies", async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/predict/deficiencies`, req.body, {
      timeout: 10000,
      headers: { "Content-Type": "application/json" }
    });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to connect to the Python DL prediction service on port 5005." });
  }
});

// Benchmark Comparison
app.get(["/api/prediction/comparison", "/api/prediction/benchmark", "/api/models/metrics"], async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/compare`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve model comparison metrics." });
  }
});

// Confusion Matrix
app.get("/api/models/confusion-matrix", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/models/confusion-matrix`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve confusion matrix." });
  }
});

// Analytics Endpoints
app.get("/api/analytics/malnutrition-distribution", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/analytics/malnutrition-distribution`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve distribution analytics." });
  }
});

app.get("/api/analytics/deficiency-risk", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/analytics/deficiency-risk`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve deficiency analytics." });
  }
});

app.get("/api/analytics/feature-importance", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/analytics/feature-importance`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve feature importance." });
  }
});

app.get("/api/dataset/info", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/dataset/info`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve dataset info." });
  }
});

app.get("/api/dashboard/summary", async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/dashboard/summary`, { timeout: 5000 });
    return res.status(200).json(response.data);
  } catch (error) {
    if (error.response) return res.status(error.response.status).json(error.response.data);
    return res.status(503).json({ error: "DL Service Unavailable", message: "Unable to retrieve dashboard summary." });
  }
});

app.listen(PORT, () => {
  console.log(`Node.js Express Backend running on http://localhost:${PORT}`);
  console.log(`Proxying Multi-Model DL requests to ${ML_SERVICE_URL}`);
});
