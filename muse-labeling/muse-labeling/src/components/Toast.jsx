export default function Toast({ msg, isError }) {
  return (
    <div style={{
      position: 'fixed',
      bottom: 24,
      left: '50%',
      transform: 'translateX(-50%)',
      background: isError ? '#e74c3c' : '#27ae60',
      color: '#fff',
      padding: '8px 20px',
      borderRadius: 6,
      fontSize: 13,
      zIndex: 999,
      pointerEvents: 'none',
      boxShadow: '0 2px 12px rgba(0,0,0,0.4)',
    }}>
      {msg}
    </div>
  )
}
