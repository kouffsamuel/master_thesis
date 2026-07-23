import { useState, useEffect } from 'react'
import './TopBar.css'

export default function TopBar({
  onOpen, frameIdx, frameTotal, frameData, onPrev, onNext, onJump, labeled
}) {
  const [jumpVal, setJumpVal] = useState('')

  useEffect(() => { setJumpVal(String(frameIdx + 1)) }, [frameIdx])

  const frameLabel = frameData
    ? `Frame ${String(frameData.frame_index).padStart(5,'0')}`
    : 'No folder opened'

  const handleJumpKey = (e) => {
    if (e.key !== 'Enter') return
    const n = parseInt(jumpVal, 10)
    if (!isNaN(n)) onJump(n - 1)
  }

  return (
    <div className="topbar">
      <span className="topbar-title">MUSE Labeling Tool</span>
      <button className="tb-btn" onClick={onOpen}>📂 Open Folder</button>

      <div className="frame-nav">
        <button className="tb-btn" onClick={onPrev} disabled={!frameTotal || frameIdx === 0}>◀</button>
        <span className="frame-label">{frameLabel}</span>
        <button className="tb-btn" onClick={onNext} disabled={!frameTotal || frameIdx === frameTotal - 1}>▶</button>

        {frameTotal > 0 && (
          <span className="frame-jump">
            <input
              className="frame-jump-input"
              type="number"
              value={jumpVal}
              onChange={e => setJumpVal(e.target.value)}
              onKeyDown={handleJumpKey}
            />
            <span className="frame-jump-total">/ {frameTotal}</span>
          </span>
        )}
      </div>

      {frameTotal > 0 && (
        <span className="progress-label">Labeled: {labeled} / {frameTotal}</span>
      )}
    </div>
  )
}
