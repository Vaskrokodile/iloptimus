import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Brain,
  Cpu,
  FlaskConical,
  LayoutDashboard,
  Boxes,
} from "lucide-react";
import ThemeToggle from "./ThemeToggle";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/models", label: "Models", icon: Cpu },
  { to: "/studio", label: "IL-Studio", icon: FlaskConical },
  { to: "/tasksets", label: "Tasksets", icon: Boxes },
];

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 glass border-b border-white/5">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2.5 group">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent to-accent-hover flex items-center justify-center shadow-lg shadow-accent/20"
            >
              <Brain className="w-4.5 h-4.5 text-white" strokeWidth={2.5} />
            </motion.div>
            <span className="font-bold text-lg tracking-tight text-fg-primary hidden sm:block">
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
                    `relative flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? "text-accent"
                        : "text-fg-secondary hover:text-fg-primary"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      {isActive && (
                        <motion.div
                          layoutId="nav-active"
                          className="absolute inset-0 rounded-xl bg-accent/10 border border-accent/20"
                          transition={{ type: "spring", stiffness: 400, damping: 30 }}
                        />
                      )}
                      <Icon className="w-4 h-4 relative z-10" strokeWidth={2} />
                      <span className="hidden sm:inline relative z-10">{item.label}</span>
                    </>
                  )}
                </NavLink>
              );
            })}
            <div className="w-px h-6 bg-fg-muted/15 mx-1.5" />
            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
}
