import { useEffect, useRef, useState } from 'react'
import './RightPanel.css'

function validateId (item, draft, items, selectedIndex, setDraft) {
  if (!item) return null
  const newId = parseInt(draft, 10)

  if (Number.isNaN(newId) || newId === item.id) {
    setDraft(String(item.id))
    return null
  }

  if (items.some((x, i) => i !== selectedIndex && x.id === newId)) {
    return null
  }

  return newId
}

export default function RightPanel({
  boxes, setBoxes, selectedBox, setSelectedBox,
  radarDetections, selectedRadar, onRadarEdit,
  lidarClusters, selectedLidar, onLidarEdit,
  onSave, onBeforeEdit,
  onUndo, onRedo, canUndo, canRedo,
  logs
}) {
  const logRef = useRef(null)

  const box = selectedBox >= 0 ? boxes[selectedBox] : null
  const radarCluster = selectedRadar >= 0 ? radarDetections[selectedRadar] : null
  const lidarPoint = selectedLidar >= 0 ? lidarClusters[selectedLidar]: null

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  // Camera Id

  const [idDraft, setIdDraft] = useState('')
  useEffect(() => { setIdDraft(box ? String(box.id) : '') }, [selectedBox, box?.id])
  
  const idConflict = box != null && idDraft !== '' && Number(idDraft) !== box.id && boxes.some((x, i) => i !== selectedBox && x.id === Number(idDraft))
  const commitCameraId = () => {
    const newId = validateId(box,idDraft, boxes,selectedBox, setIdDraft)
    onBeforeEdit?.()
    setBoxes(prev => {
      const next = prev.map(x => ({ ...x }))
      next[selectedBox].id = newId
      return next
    })
  }

  // Camera box field

  const setField = (field, val) => {
    if (selectedBox < 0) return
    onBeforeEdit?.()
    setBoxes(prev => {
      const next = prev.map(b => ({ ...b }))
      const box = next[selectedBox]
      if (field === 'x1') { const x2 = box.cx + box.width / 2; box.cx = (val + x2) / 2; box.width = x2 - val }
      if (field === 'y1') { const y2 = box.cy + box.height / 2; box.cy = (val + y2) / 2; box.height = y2 - val }
      if (field === 'x2') { const x1 = box.cx - box.width / 2; box.cx = (x1 + val) / 2; box.width = val - x1 }
      if (field === 'y2') { const y1 = box.cy - box.height / 2; box.cy = (y1 + val) / 2; box.height = val - y1 }
      return next
    })
  }
  const nudge = (field, delta) => {
    if (!box) return
    const x1 = box.cx - box.width / 2, y1 = box.cy - box.height / 2
    const x2 = box.cx + box.width / 2, y2 = box.cy + box.height / 2
    const vals = { x1, y1, x2, y2 }
    setField(field, vals[field] + delta)
  }
  const deleteBox = () => {
    if (selectedBox < 0) return
    onBeforeEdit?.()
    setBoxes(prev => prev.filter((_, i) => i !== selectedBox))
    setSelectedBox(-1)
  }

  // Radar field

  const [radarIdDraft, setRadarIdDraft] = useState('')
  const [velocityDraft, setVelocityDraft] = useState('')
  const [rangeDraft, setRangeDraft] = useState('')

  const radarIdConflict = radarCluster != null && radarIdDraft !== '' && Number(radarIdDraft) !== radarCluster.id && radarDetections.some((x, i) =>i !== selectedRadar && x.id === Number(radarIdDraft))


  useEffect(() => {
    if (radarCluster) {
      setRadarIdDraft(String(radarCluster.id ?? ''))
      setVelocityDraft(String(radarCluster.radar_kmh ?? ''))
      setRangeDraft(String(radarCluster.radar_m ?? ''))
    } else {
      setRadarIdDraft('')
      setVelocityDraft('')
      setRangeDraft('')
    }
  }, [selectedRadar, radarCluster?.id, radarCluster?.radar_kmh, radarCluster?.radar_m])

  // Radar ID
  const commitRadarId = () => {
    const newId = validateId(radarCluster, radarIdDraft, radarDetections ,selectedRadar, setRadarIdDraft)
    onBeforeEdit?.()
    onRadarEdit(selectedRadar, 'id', newId)
  }

  //LiDAR ID

  const [lidarIdDraft, setLidarIdDraft] = useState('')

  const lidarIdConflict = lidarPoint != null && lidarIdDraft !== '' && Number(lidarIdDraft) !== lidarPoint.id && lidarClusters.some((x, i) =>i !== selectedLidar && x.id === Number(lidarIdDraft))


  useEffect(() => {
    if(lidarPoint){
      setLidarIdDraft(String(lidarPoint.id ?? ''))
    }else{
      setLidarIdDraft('')
    }
  }, [selectedLidar, lidarPoint?.id])

  const commitLidarId = () => {
    const newId = validateId(lidarPoint, lidarIdDraft, lidarClusters ,selectedLidar, setLidarIdDraft)
    onBeforeEdit?.()
    onLidarEdit(selectedLidar, 'id', newId)
  }

  return (
    <div className="panel">

      {/* Camera box editor */}
      <div className="panel-section">
        <div className="panel-title">CAMERA BOX</div>
        {box ? (
          <>
            <div className="coord-row">
              <label>id</label>
              <input
                type="number"
                value={idDraft}
                onChange={e => setIdDraft(e.target.value)}
                onBlur={commitCameraId}
                onKeyDown={e => { if (e.key === 'Enter') e.target.blur() }}
                style={idConflict ? { borderColor: '#e94560' } : undefined}
              />
            </div>
            {idConflict && <div className="pending-hint" style={{ color: '#e94560' }}>⚠ ID déjà utilisé dans cette frame</div>}

            <div className="coord-row">
              <label>type</label>
              <input
                type="text"
                value={box.class ?? ''}
                placeholder="e.g. car, truck…"
                onChange={e => {
                  onBeforeEdit?.()
                  setBoxes(prev => {
                    const next = prev.map(x => ({ ...x }))
                    next[selectedBox].class = e.target.value
                    return next
                  })
                }}
              />
            </div>
            {[
              ['x1', box.cx - box.width / 2],
              ['y1', box.cy - box.height / 2],
              ['x2', box.cx + box.width / 2],
              ['y2', box.cy + box.height / 2],
            ].map(([field, val]) => (
              <div className="coord-row" key={field}>
                <label>{field}</label>
                <button className="spin" onClick={() => nudge(field, -1)}>−</button>
                <input type="number" value={Math.round(val)} onChange={e => setField(field, parseFloat(e.target.value))} />
                <button className="spin" onClick={() => nudge(field, 1)}>+</button>
              </div>
            ))}
            <button className="delete-btn" onClick={deleteBox}>🗑 Delete box</button>
          </>
        ) : (
          <div className="empty-hint">Click a box to select<br/>or drag to draw a new one</div>
        )}
      </div>

      {/* Radar labeler */}
      <div className="panel-section">
        <div className="panel-title">RADAR POINT</div>
          {radarCluster ? (
            <>
        <div className="coord-row">
          <label>id</label>
          <input type="number" value={radarIdDraft} onChange={e => setRadarIdDraft(e.target.value)}
                onBlur={commitRadarId}
                onKeyDown={e => {
                  if (e.key === 'Enter') e.target.blur()
                }}
                style={
                  radarIdConflict
                    ? { borderColor: '#e94560' }
                    : undefined
                }
              />
        </div>
        {radarIdConflict && <div className="pending-hint" style={{ color: '#e94560' }}>⚠ ID déjà utilisé dans cette frame</div>}

        <div className="coord-row">
          <label>velocity</label>
          <input type="text" value={radarCluster.radar_kmh != null ? `${radarCluster.radar_kmh.toFixed(2)} km/h` : ''} readOnly />
        </div>

        <div className="coord-row">
          <label>range</label>
          <input type="text" value={radarCluster.radar_m != null ? `${radarCluster.radar_m.toFixed(2)} m` : ''} readOnly/>
        </div>
      </>
    ) : (
      <div className="empty-hint">
        Click a centroid (✕) to select a point
      </div>
    )}
      </div>
      <div className="panel-section">
        <div className="panel-title">LIDAR POINT</div>
        {lidarPoint ? (<>
        <div className="coord-row">
          <label>id</label>
          <input type="number" value={lidarIdDraft} onChange={e => setLidarIdDraft(e.target.value)}
                onBlur={commitLidarId}
                onKeyDown={e => {
                  if (e.key === 'Enter') e.target.blur()
                }}
                style={
                  radarIdConflict
                    ? { borderColor: '#e94560' }
                    : undefined
                }
              />
        </div>
        {lidarIdConflict && <div className="pending-hint" style={{ color: '#e94560' }}>⚠ ID déjà utilisé dans cette frame</div>}
        </>):
        <div className="empty-hint">
          Click on a LiDAR cluster to select a point
        </div>
        }
      </div>
      

      {/* Actions */}
      <div className="panel-section">
        <div className="panel-title">ACTIONS</div>
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