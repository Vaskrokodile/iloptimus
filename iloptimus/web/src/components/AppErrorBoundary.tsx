import { Component, type ErrorInfo, type ReactNode } from "react";

export default class AppErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("IL Optimus page error", error, info);
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="route-error">
          <span>Page interrupted</span>
          <h1>This workspace hit a temporary error.</h1>
          <p>Your saved models, runs, and environments are safe.</p>
          <button onClick={() => window.location.reload()}>Reload workspace</button>
        </div>
      );
    }
    return this.props.children;
  }
}
