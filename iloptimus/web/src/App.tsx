import { Routes, Route, useLocation } from "react-router-dom";
import { ThemeProvider } from "./components/ThemeProvider";
import Navbar from "./components/Navbar";
import ChatPage from "./pages/ChatPage";
import ModelLibraryPage from "./pages/ModelLibraryPage";
import OptimusLabPage from "./pages/OptimusLabPage";
import EnvironmentBuilderPage from "./pages/EnvironmentBuilderPage";
import MyEnvironmentsPage from "./pages/MyEnvironmentsPage";
import EnvironmentPlayPage from "./pages/EnvironmentPlayPage";
import AppErrorBoundary from "./components/AppErrorBoundary";

export default function App() {
  const location = useLocation();
  return (
    <ThemeProvider>
      <div className="app-shell">
        <Navbar />
        <main className="app-content">
          <AppErrorBoundary key={location.pathname}><Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/models" element={<ModelLibraryPage />} />
            <Route path="/studio" element={<OptimusLabPage />} />
            <Route path="/pipelines" element={<EnvironmentBuilderPage />} />
            <Route path="/environments" element={<MyEnvironmentsPage />} />
            <Route path="/environments/:environmentId/play" element={<EnvironmentPlayPage />} />
          </Routes></AppErrorBoundary>
        </main>
      </div>
    </ThemeProvider>
  );
}
