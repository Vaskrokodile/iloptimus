import { NavLink } from "react-router-dom";
import {
  Brain,
  Cpu,
  FlaskConical,
  LayoutDashboard,
  Boxes,
} from "lucide-react";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/models", label: "Models", icon: Cpu },
  { to: "/studio", label: "IL-Studio", icon: FlaskConical },
  { to: "/tasksets", label: "Tasksets", icon: Boxes },
];

export default function Navbar() {
  return (
    <nav className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 bg-brand-600 rounded-lg flex items-center justify-center group-hover:bg-brand-500 transition-colors">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg text-white hidden sm:block">
              IL Optimus
            </span>
          </NavLink>

          {/* Nav links */}
          <div className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-brand-600/20 text-brand-400"
                        : "text-gray-400 hover:text-gray-200 hover:bg-gray-800"
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{item.label}</span>
                </NavLink>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
