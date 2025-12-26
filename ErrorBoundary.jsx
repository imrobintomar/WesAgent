import React from "react";

export default class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <pre style={{ color: "red", padding: "16px" }}>
          {this.state.error?.toString()}
        </pre>
      );
    }
    return this.props.children;
  }
}
