const COLORS = {
  info:    { fg: '#7c83e8', border: 'rgba(124,131,232,0.35)' },
  success: { fg: '#1de9b6', border: 'rgba(29,233,182,0.35)' },
  error:   { fg: '#ffb4b4', border: 'rgba(239,68,68,0.4)' },
};

/** Bottom-right toast stack. Toasts: [{id, text, kind}]. */
export default function Toasts({ toasts, onDismiss }) {
  if (!toasts.length) return null;
  return (
    <div style={{
      position: 'fixed', bottom: 18, right: 18, zIndex: 600,
      display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 340,
    }}>
      {toasts.map(t => {
        const c = COLORS[t.kind] || COLORS.info;
        return (
          <div key={t.id}
            onClick={() => onDismiss(t.id)}
            style={{
              padding: '9px 14px', borderRadius: 4, cursor: 'pointer',
              background: 'rgba(24,28,38,0.96)', border: `1px solid ${c.border}`,
              color: c.fg, fontSize: 11, fontFamily: "'JetBrains Mono', monospace",
              lineHeight: 1.5, boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
            }}>
            {t.text}
          </div>
        );
      })}
    </div>
  );
}
