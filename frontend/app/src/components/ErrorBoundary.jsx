import { Component } from 'react';

/** Catches render errors so a bad payload can't blank the whole app. */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{
        height: '100vh', display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: 12,
        background: '#0f1117', color: '#8892a4',
        fontFamily: "'JetBrains Mono', monospace",
      }}>
        <div style={{ fontSize: 13, color: '#ffb4b4' }}>Something broke while rendering.</div>
        <div style={{ fontSize: 11, maxWidth: 520, textAlign: 'center', color: '#5a6375' }}>
          {String(this.state.error?.message || this.state.error)}
        </div>
        <button
          onClick={() => { this.setState({ error: null }); window.location.reload(); }}
          style={{
            marginTop: 8, padding: '7px 16px', borderRadius: 4, cursor: 'pointer',
            background: 'rgba(29,233,182,0.12)', border: '1px solid rgba(29,233,182,0.4)',
            color: '#1de9b6', fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
          }}>
          Reload app
        </button>
      </div>
    );
  }
}
