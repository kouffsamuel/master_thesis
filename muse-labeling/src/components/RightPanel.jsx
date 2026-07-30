import { useEffect, useRef } from 'react'
import './RightPanel.css'

export default function RightPanel({
  boxes, setBoxes, selectedBox, setSelectedBox,
  radarDetections, selectedRadar, labeling,
  pairPendingFor, boxLinkPendingFor,
  onObject, onObjectNoBox, onNoise, onPair, onClear,
  onAutoLabel, canAutoLabel,
  autoLabelAlways, onToggleAutoLabelAlways,
  labelHelpers,
  onSave, onBeforeEdit,
  onUndo, onRedo, canUndo, canRedo,
  logs
}) {
  const logRef = useRef(null)
  const { isObject, isNoise, isPaired, getPartner, getObjectBox } = labelHelpers

  const b = selectedBox >= 0 ? boxes[selectedBox] : null
  const d = selectedRadar >= 0 ? radarDetections[selectedRadar] : null
  const radarId = d?.track_id

  const objOn   = d != null && isObject(labeling, radarId)
  const noiseOn = d != null && isNoise(labeling, radarId)
  const paired  = d != null && isPaired(labeling, radarId)
  const partner = d != null ? getPartner(labeling, radarId) : null
  const objBox  = objOn ? getObjectBox(labeling, radarId) : undefined

  const isPairPending = pairPendingFor === selectedRadar && pairPendingFor >= 0
  const isBoxPending  = boxLinkPendingFor === selectedRadar && boxLinkPendingFor >= 0

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  const setField = (field, val) => {
    if (selectedBox < 0) return
    onBeforeEdit?.()
    setBoxes(prev => {
      const next = prev.map(b => ({...b, pos1:[...b.pos1], pos2:[...b.pos2]}))
      if (field === 'x1') next[selectedBox].pos1[0] = val
      if (field === 'y1') next[selectedBox].pos1[1] = val
      if (field === 'x2') next[selectedBox].pos2[0] = val
      if (field === 'y2') next[selectedBox].pos2[1] = val
      return next
    })
  }
  const nudge = (field, delta) => {
    if (!b) return
    const vals = { x1:b.pos1[0], y1:b.pos1[1], x2:b.pos2[0], y2:b.pos2[1] }
    setField(field, vals[field] + delta)
  }
  const deleteBox = () => {
    if (selectedBox < 0) return
    onBeforeEdit?.()
    setBoxes(prev => prev.filter((_,i) => i !== selectedBox))
    setSelectedBox(-1)
  }

  return (
    <div className="panel">

      {/* Camera box editor */}
      <div className="panel-section">
        <div className="panel-title">CAMERA BOX</div>
        {b ? (
          <>
            <div className="box-info">box #{b.track_id}</div>
            <div className="coord-row">
              <label>type</label>
              <input
                type="text"
                value={b.label ?? ''}
                placeholder="e.g. car, truck…"
                onChange={e => {
                  onBeforeEdit?.()
                  setBoxes(prev => {
                    const next = prev.map(x => ({...x}))
                    next[selectedBox].label = e.target.value
                    return next
                  })
                }}
              />
            </div>
            {[['x1', b.pos1[0]], ['y1', b.pos1[1]], ['x2', b.pos2[0]], ['y2', b.pos2[1]]].map(([field, val]) => (
              <div className="coord-row" key={field}>
                <label>{field}</label>
                <button className="spin" onClick={() => nudge(field, -1)}>−</button>
                <input type="number" value={Math.round(val)} onChange={e => setField(field, parseFloat(e.target.value))} />
                <button className="spin" onClick={() => nudge(field, 1)}>+</button>
              </div>
            ))}
            <button className="delete-btn" onClick={deleteBox}>🗑 Delete box</button>
              {b.original && (
                <button
                  className="restore-btn"
                  onClick={() => {
                    onBeforeEdit?.()
                    setBoxes(prev => {
                      const next = prev.map(x => ({...x, pos1:[...x.pos1], pos2:[...x.pos2]}))
                      const o = next[selectedBox].original
                      next[selectedBox].pos1 = [...o.pos1]
                      next[selectedBox].pos2 = [...o.pos2]
                      next[selectedBox].label = o.label
                      if (o.confidence != null) next[selectedBox].confidence = o.confidence
                      return next
                    })
                  }}
                >↺ Restore original YOLO box</button>
              )}
          </>
        ) : (
          <div className="empty-hint">Click a box to select<br/>or drag to draw a new one</div>
        )}
      </div>

      {/* Radar labeler */}
      <div className="panel-section">
        <div className="panel-title">RADAR POINT</div>
        {d ? (
          <div className="box-info">
            ID {radarId} — {d.is_confirmed ? 'confirmed' : 'unconfirmed'}
            {d.energy != null && d.energy > 0 && (
              <><br/><span className="tag tag-energy">⚡ {(10 * Math.log10(d.energy)).toFixed(1)} dB</span></>
            )}
            {objOn   && <><br/><span className="tag tag-object">○ object {objBox != null ? `→ box ${objBox}` : '(no box)'}</span></>}
            {noiseOn && <><br/><span className="tag tag-noise">△ noise</span></>}
            {paired  && <><br/><span className="tag tag-pair">□ paired with {partner}</span></>}
          </div>
        ) : (
          <div className="empty-hint">Click a centroid (✕) to select a point</div>
        )}

        {isPairPending && <div className="pending-hint">⏳ Click the partner point (Esc to cancel)</div>}
        {isBoxPending  && <div className="pending-hint">⏳ Click a box on the camera (Esc to cancel)</div>}

        <div className="toggle-row">
          <button className={`toggle-btn ${objOn ? 'lit-object' : 'dim'}`} onClick={onObject} disabled={!d}>
            ○ Object {objOn && objBox != null ? `(box ${objBox})` : ''}
          </button>
          {d && !objOn && boxes.length > 0 && (
            <button className="sub-btn" onClick={onObjectNoBox}>↳ object with no box</button>
          )}
          <button className={`toggle-btn ${paired ? 'lit-pair' : (isPairPending ? 'pending' : 'dim')}`} onClick={onPair} disabled={!d}>
            {paired ? `□ Paired (#${partner})` : '□ Pair'}
          </button>
          <button className={`toggle-btn ${noiseOn ? 'lit-noise' : 'dim'}`} onClick={onNoise} disabled={!d}>
            △ Noise
          </button>
        </div>

        <button className="clear-all-btn" onClick={onClear} disabled={!d}>✕ Clear all</button>

        <div className="hint" style={{marginTop:8}}>
          Select a point, then toggle.<br/>
          Object ↔ Noise mutually exclusive.<br/>
          Noise cascades across a pair.<br/>
          Object asks you to click a box.
        </div>
      </div>

      {/* Actions */}
      <div className="panel-section">
        <div className="panel-title">ACTIONS</div>
        <button className="auto-label-btn" onClick={onAutoLabel} disabled={!canAutoLabel}>
          ⚙ Auto-label this frame
        </button>

        <label
          className={`switch-bar ${autoLabelAlways ? 'on' : ''} ${!canAutoLabel ? 'disabled' : ''}`}
          title="Run Auto-label on every frame you move to. Frames that already carry labels are skipped, so your manual work is never overwritten."
        >
          <input
            type="checkbox"
            checked={autoLabelAlways}
            onChange={onToggleAutoLabelAlways}
            disabled={!canAutoLabel}
          />
          <span className="switch-track"><span className="switch-thumb" /></span>
          <span className="switch-text">
            Auto on frame change
            <span className="switch-sub">{autoLabelAlways ? 'ON — applies on every frame' : 'OFF — manual only'}</span>
          </span>
        </label>
        <button className="save-btn" onClick={onSave}>💾 Save</button>
        <div className="undo-redo-row">
          <button className="undo-btn" onClick={onUndo} disabled={!canUndo}>↩ Undo</button>
          <button className="undo-btn" onClick={onRedo} disabled={!canRedo}>↪ Redo</button>
        </div>
        <div className="hint">
          <kbd>Ctrl+S</kbd> save · <kbd>Ctrl+Z</kbd> undo<br/>
          <kbd>←→</kbd> navigate · <kbd>Del</kbd> delete box · <kbd>Esc</kbd> cancel
        </div>
      </div>

      {/* Log */}
      <div className="panel-section panel-log-section">
        <div className="panel-title">LOG</div>
        <div className="log-box" ref={logRef}>
          {logs.length === 0
            ? <span className="log-empty">No activity yet</span>
            : logs.map((l, i) => <div key={i} className="log-line">{l}</div>)}
        </div>
      </div>

    </div>
  )
}