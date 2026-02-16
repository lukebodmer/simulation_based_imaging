import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./styles/global.css";
import App from "./App";
import { useThemeStore } from "./stores/themeStore";

// Initialize theme from persisted state
const theme = useThemeStore.getState().theme;
document.documentElement.setAttribute("data-theme", theme);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
