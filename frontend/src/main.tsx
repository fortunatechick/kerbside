import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { EngineBoot } from "./components/EngineBoot";
import { bootEngine } from "./lib/engine";
import "./styles/index.css";

// Start downloading the Python runtime before React has even rendered.
bootEngine();

const client = new QueryClient({
  defaultOptions: {
    queries: { refetchOnWindowFocus: false, retry: 1, staleTime: 2000 },
  },
});

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <EngineBoot>
        <App />
      </EngineBoot>
    </QueryClientProvider>
  </React.StrictMode>,
);
