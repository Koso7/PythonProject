import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

createRoot(document.getElementById("wurzel")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
