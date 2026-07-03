import { useState, useEffect } from 'react'
import './TopBar.css'

export default function TopBar({
  onOpen, jsonFiles = [], activeJsonFile = '', onJsonFileChange,
  isDirty = false,
  frameIdx, frameTotal, frameData, onPrev, onNext, onJump, labeled
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

      {jsonFiles.length > 0 && (
        <select
          className="json-select"
          value={activeJsonFile}
          onChange={e => onJsonFileChange(e.target.value)}
          title="Open a JSON file from the selected data folder"
        >
          {jsonFiles.map(file => (
            <option key={file.name} value={file.name}>
              {file.name}
            </option>
          ))}
        </select>
      )}

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
      {isDirty && <span className="dirty-label">Unsaved</span>}
    </div>
  )
}
