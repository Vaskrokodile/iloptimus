import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import DashboardPage from "./pages/DashboardPage";
import ModelsPage from "./pages/ModelsPage";
import ILStudioPage from "./pages/ILStudioPage";
import TasksetsPage from "./pages/TasksetsPage";

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/studio" element={<ILStudioPage />} />
          <Route path="/tasksets" element={<TasksetsPage />} />
        </Routes>
      </main>
    </div>
  );
}
