import React from "react";
import { createRoot } from "react-dom/client";
import RetailIntelligenceDashboard from "../retail_intelligence_dashboard.jsx";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RetailIntelligenceDashboard />
  </React.StrictMode>
);
