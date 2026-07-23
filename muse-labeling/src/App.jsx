import { useState, useEffect, useCallback, useRef } from 'react'
import TopBar from './components/TopBar'
import CameraCanvas from './components/CameraCanvas'
import ClusterCanvas from './components/ClusterCanvas'
import TrackCanvas from './components/TrackCanvas'
import StaticImage from './components/StaticImage'
import RDMapCanvas from './components/RDMapCanvas'
import RightPanel from './components/RightPanel'
import Toast from './components/Toast'
import { getFileURL, getFileFloat32, padIndex } from './utils/fs'
import {
  normalizeLabeling, emptyLabeling,
  isObject, isNoise, getPartner, isPaired, getObjectBox,
  toggleObject, toggleNoise, linkPair, unlinkPair, clearRadar
} from './utils/labeling'
import { autoLabelFrame } from './utils/autoLabel'
import './App.css'

function frameIndexFromKey(key) {
  const m = key?.match(/frame_(\d+)\.jpeg/)
  return m ? parseInt(m[1], 10) : 0
}

export default function App() {
  // jsonData is the ONLY source of truth. Everything shown is derived from it.
  const [dirHandle, setDirHandle] = useState(null)
  const [jsonData,  setJsonData]  = useState({})
  const [frameKeys, setFrameKeys] = useState([])
  const [frameIdx,  setFrameIdx]  = useState(0)

  const [selectedBox,     setSelectedBox]     = useState(-1)
  const [selectedRadar,   setSelectedRadar]   = useState(-1)   // index into radar_detections
  const [pairPendingFor,  setPairPendingFor]  = useState(-1)   // radar index awaiting partner
  const [boxLinkPendingFor, setBoxLinkPendingFor] = useState(-1) // radar index awaiting box click

  // const [rdURL,     setRdURL]     = useState('')
  const [rdData,    setRdData]    = useState(null)   // Float32Array of rd_power
  const [cameraURL, setCameraURL] = useState('')
  // const [cameraURL, setCameraURL] = useState('')
  const [toast,     setToast]     = useState(null)
  const [logs,       setLogs]     = useState([])
  // Display-only camera brightness (1 = original). Lives outside jsonData
  // and the undo system on purpose — it never modifies any data.
  const [brightness, setBrightness] = useState(1)
  // Algorithm suggestions (Stage 3): display-only, never persisted to JSON.
  // { candidates: [trackId], needsReview: bool } | null
  const [suggestions, setSuggestions] = useState(null)

  const [history,   setHistory]   = useState([])
  const [redoStack, setRedoStack] = useState([])

  const addLog = (msg) => setLogs(prev => [...prev.slice(-99), msg])

  // ── Derived view of current frame ──────────────────────────────────────
  const currentFrameKey = frameKeys[frameIdx]
  const frameData       = jsonData[currentFrameKey] || null
  const boxesData       = frameData?.box_detections   || []
  const radarDetections = frameData?.radar_detections || []
  const trackHistory    = frameData?.track_history    || []
  const labeling        = normalizeLabeling(frameData?.labeling)

  // ── Undo / Redo: snapshot whole jsonData ───────────────────────────────
  const pushHistory = useCallback(() => {
    setHistory(h => [...h.slice(-49), jsonData])
    setRedoStack([])
  }, [jsonData])

  const undo = useCallback(() => {
    if (!history.length) return
    setRedoStack(r => [jsonData, ...r])
    setJsonData(history[history.length - 1])
    setHistory(h => h.slice(0, -1))
  }, [history, jsonData])

  const redo = useCallback(() => {
    if (!redoStack.length) return
    setHistory(h => [...h, jsonData])
    setJsonData(redoStack[0])
    setRedoStack(r => r.slice(1))
  }, [redoStack, jsonData])

  // ── Mutators that write into the current frame ─────────────────────────
  const patchFrame = (patch) => {
    setJsonData(prev => {
      const key = frameKeys[frameIdx]
      if (!key || !prev[key]) return prev
      return { ...prev, [key]: { ...prev[key], ...patch } }
    })
  }

  const setBoxes = (updater) => {
    setJsonData(prev => {
      const key = frameKeys[frameIdx]
      if (!key || !prev[key]) return prev
      const frame = prev[key]
      const prevBoxes = frame.box_detections || []
      const newBoxes = typeof updater === 'function' ? updater(prevBoxes) : updater
      return { ...prev, [key]: { ...frame, box_detections: newBoxes } }
    })
  }

  const setLabeling = (newLabeling) => patchFrame({ labeling: newLabeling })

  // ── Open folder ─────────────────────────────────────────────────────────
  const openFolder = async () => {
    try {
      const dh = await window.showDirectoryPicker({ mode: 'readwrite' })
      const jh = await dh.getFileHandle('yolo_tracking_data.json')
      const jf = await jh.getFile()
      const data = JSON.parse(await jf.text())
      const keys = Object.keys(data).sort()
      setDirHandle(dh)
      setJsonData(data)
      setFrameKeys(keys)
      setFrameIdx(0)
      setHistory([]); setRedoStack([])
      addLog(`[System] Opened folder — ${keys.length} frames found`)
    } catch (e) {
      if (e.name !== 'AbortError') showToast('Error: ' + e.message, true)
    }
  }

  // ── Reset transient selection when navigating ──────────────────────────
  useEffect(() => {
    setSelectedBox(-1)
    setSelectedRadar(-1)
    setPairPendingFor(-1)
    setBoxLinkPendingFor(-1)
    setSuggestions(null)   // algorithm suggestions live per-frame only
  }, [frameIdx])

  // // ── Fetch images (independent of label edits) ──────────────────────────
  // useEffect(() => {
  //   if (!dirHandle || !frameKeys.length) return
  //   const key = frameKeys[frameIdx]
  //   const fi = padIndex(frameIndexFromKey(key))
  //   ;(async () => {
  //     setRdURL(    await getFileURL(dirHandle, `rd/frame_${fi}_rd.jpeg`))
  //     setCameraURL(await getFileURL(dirHandle, `camera/frame_${fi}_camera.jpeg`))
  //     addLog(`[Frame ${fi}] Loaded`)
  //   })()
  // }, [frameIdx, dirHandle, frameKeys])
  
  // ── Fetch images + RD raw (independent of label edits) ─────────────────
  useEffect(() => {
    if (!dirHandle || !frameKeys.length) return
    const key = frameKeys[frameIdx]
    const fi = padIndex(frameIndexFromKey(key))
    const rdRawPath = jsonData[key]?.rd_raw_file
    ;(async () => {
      setCameraURL(await getFileURL(dirHandle, `camera/frame_${fi}_camera.jpeg`))
      if (rdRawPath) {
        setRdData(await getFileFloat32(dirHandle, rdRawPath))
      } else {
        setRdData(null)
      }
      addLog(`[Frame ${fi}] Loaded`)
    })()
  }, [frameIdx, dirHandle, frameKeys])

  // ── Save ────────────────────────────────────────────────────────────────
  const save = useCallback(async () => {
    if (!dirHandle) return
    try {
      const fh = await dirHandle.getFileHandle('yolo_tracking_data.json', { create: true })
      const w  = await fh.createWritable()
      await w.write(JSON.stringify(jsonData, null, 2))
      await w.close()
      showToast('Saved')
      addLog(`[Save] All frames written to disk`)
    } catch (e) {
      showToast('Save failed: ' + e.message, true)
      addLog(`[Error] Save failed: ${e.message}`)
    }
  }, [dirHandle, jsonData])

  // ── Keyboard ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.ctrlKey && e.key === 's') { e.preventDefault(); save(); return }
      if (e.target.tagName === 'INPUT') return
      if (e.ctrlKey && e.shiftKey && e.key.toLowerCase() === 'z') { e.preventDefault(); redo(); return }
      if (e.ctrlKey && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return }
      if (e.key === 'ArrowLeft')  gotoFrame(frameIdx - 1)
      if (e.key === 'ArrowRight') gotoFrame(frameIdx + 1)
      if (e.key === 'Escape') { setPairPendingFor(-1); setBoxLinkPendingFor(-1) }
      if (e.key === 'Delete' && selectedBox >= 0) {
        pushHistory()
        setBoxes(prev => prev.filter((_, i) => i !== selectedBox))
        setSelectedBox(-1)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [frameIdx, save, undo, redo, selectedBox, pushHistory])

  const gotoFrame = (idx) => {
    if (!frameKeys.length) return
    setFrameIdx(Math.max(0, Math.min(frameKeys.length - 1, idx)))
  }

  const showToast = (msg, isError = false) => {
    setToast({ msg, isError })
    setTimeout(() => setToast(null), 1800)
  }

  // ── Radar point click ────────────────────────────────────────────────────
  const handleRadarClick = (idx) => {
    const radarId = radarDetections[idx]?.track_id

    // completing a pair link
    if (pairPendingFor >= 0) {
      if (pairPendingFor === idx) {
        setPairPendingFor(-1)
        addLog(`[Radar ${radarId}] Pair cancelled`)
        return
      }
      const idA = radarDetections[pairPendingFor]?.track_id
      pushHistory()
      setLabeling(linkPair(labeling, idA, radarId))
      addLog(`[Pair] Radar ${idA} ↔ Radar ${radarId} linked`)
      setPairPendingFor(-1)
      setSelectedRadar(idx)
      return
    }

    setSelectedRadar(idx)
  }

  // ── Box click (used both for normal select AND for object→box linking) ──
  const handleBoxClick = (idx) => {
    if (boxLinkPendingFor >= 0) {
      const radarId = radarDetections[boxLinkPendingFor]?.track_id
      const boxId   = boxesData[idx]?.track_id
      pushHistory()
      setLabeling(toggleObject(
        isObject(labeling, radarId) ? unlinkObjectFirst(labeling, radarId) : labeling,
        radarId, boxId
      ))
      addLog(`[Object] Radar ${radarId} → Box ${boxId}`)
      setBoxLinkPendingFor(-1)
      setSelectedBox(idx)
      return
    }
    setSelectedBox(idx)
  }

  // helper: if already object, ensure toggleObject re-adds with new box
  const unlinkObjectFirst = (l, radarId) => toggleObject(l, radarId)

  // ── Right-panel actions on the selected radar point ─────────────────────
  const selRadarId = selectedRadar >= 0 ? radarDetections[selectedRadar]?.track_id : null

  const actObject = () => {
    if (selRadarId == null) return
    if (isObject(labeling, selRadarId)) {
      // turning OFF
      pushHistory()
      setLabeling(toggleObject(labeling, selRadarId))
      addLog(`[Radar ${selRadarId}] object removed`)
    } else {
      // turning ON — need to pick a box (or set null if no boxes)
      if (boxesData.length === 0) {
        pushHistory()
        setLabeling(toggleObject(labeling, selRadarId, null))
        addLog(`[Radar ${selRadarId}] object (no box this frame)`)
      } else {
        setBoxLinkPendingFor(selectedRadar)
        addLog(`[Radar ${selRadarId}] object armed — click a box (or Esc for null)`)
      }
    }
  }

  // set object with null box explicitly (button for "object, no matching box")
  const actObjectNoBox = () => {
    if (selRadarId == null) return
    pushHistory()
    const l = isObject(labeling, selRadarId)
      ? toggleObject(toggleObject(labeling, selRadarId), selRadarId, null) // off then on(null)
      : toggleObject(labeling, selRadarId, null)
    setLabeling(l)
    setBoxLinkPendingFor(-1)
    addLog(`[Radar ${selRadarId}] object (no box)`)
  }

  const actNoise = () => {
    if (selRadarId == null) return
    pushHistory()
    setLabeling(toggleNoise(labeling, selRadarId))
    const partner = getPartner(labeling, selRadarId)
    addLog(partner != null
      ? `[Radar ${selRadarId}] noise toggled (cascaded to paired ${partner})`
      : `[Radar ${selRadarId}] noise toggled`)
  }

  const actPair = () => {
    if (selRadarId == null) return
    if (isPaired(labeling, selRadarId)) {
      pushHistory()
      setLabeling(unlinkPair(labeling, selRadarId))
      addLog(`[Radar ${selRadarId}] pair unlinked`)
    } else {
      setPairPendingFor(selectedRadar)
      addLog(`[Radar ${selRadarId}] pair armed — click partner point`)
    }
  }

  const actClear = () => {
    if (selRadarId == null) return
    pushHistory()
    setLabeling(clearRadar(labeling, selRadarId))
    addLog(`[Radar ${selRadarId}] all labels cleared`)
  }

  // ── Auto-label: run the algorithm on the current frame ──────────────────
  // ONE pushHistory + ONE setLabeling = one undo step. Ctrl+Z restores the
  // entire pre-button state. The algorithm output replaces the frame's
  // labeling wholesale (deterministic and idempotent).
  const runAutoLabel = () => {
    if (!frameData) return
    pushHistory()
    // labeling = decisions (noise + pairs) -> jsonData, one undo step.
    // suggestions = Stage 3 hints -> display state only, never saved.
    const { labeling: auto, suggestions: sug } = autoLabelFrame(frameData)
    setLabeling(auto)
    setSuggestions(sug)
    addLog(
      `[Auto] noise: ${auto.noise.length}, pairs: ${auto.pairs.length}, ` +
      `candidates: ${sug.candidates.length}` +
      (sug.needsReview ? ' — ⚠ sensors disagree, please review' : '')
    )
  }

  // ── Progress ─────────────────────────────────────────────────────────────
  const labeledCount = frameKeys.filter(k => {
    const l = jsonData[k]?.labeling
    return l && (Object.keys(l.object || {}).length || (l.noise || []).length || (l.pairs || []).length)
  }).length

  return (
    <div className="app">
      <TopBar
        onOpen={openFolder}
        frameIdx={frameIdx}
        frameTotal={frameKeys.length}
        frameData={frameData}
        onPrev={() => gotoFrame(frameIdx - 1)}
        onNext={() => gotoFrame(frameIdx + 1)}
        onJump={gotoFrame}
        labeled={labeledCount}
      />

      <div className="main-grid">
        <div className="cell cell-rd">
          <div className="cell-title">Range-Doppler Map</div>
          {/* <div className="cell-body"><StaticImage src={rdURL} alt="RD map" /></div> */}
          <div className="cell-body"><RDMapCanvas data={rdData} /></div>
        </div>

        <div className="cell cell-camera">
          <div className="cell-title" style={{ display:'flex', justifyContent:'space-between', alignItems:'center', gap:8 }}>
            <span>
              Camera{boxLinkPendingFor >= 0 ? ' — click a box to link as object (Esc to cancel)' : ' — draw or adjust boxes'}
            </span>
            <span style={{ display:'flex', alignItems:'center', gap:6, flexShrink:0 }}
                  title="Display-only brightness — does not modify the image or any data">
              <span style={{ opacity:0.7 }}>☀</span>
              <input
                type="range" min="1" max="3" step="0.1"
                value={brightness}
                onChange={e => setBrightness(parseFloat(e.target.value))}
                onDoubleClick={() => setBrightness(1)}
                style={{ width: 90 }}
              />
              <span style={{ width: 34, textAlign:'right', opacity:0.7 }}>{brightness.toFixed(1)}×</span>
            </span>
          </div>
          <div className="cell-body">
            <CameraCanvas
              imageURL={cameraURL}
              boxes={boxesData}
              setBoxes={setBoxes}
              selectedBox={selectedBox}
              setSelectedBox={handleBoxClick}
              onBeforeEdit={pushHistory}
              linkMode={boxLinkPendingFor >= 0}
              brightness={brightness}
            />
          </div>
        </div>

        <div className="cell cell-panel">
          <RightPanel
            boxes={boxesData}
            setBoxes={setBoxes}
            selectedBox={selectedBox}
            setSelectedBox={setSelectedBox}
            radarDetections={radarDetections}
            selectedRadar={selectedRadar}
            labeling={labeling}
            pairPendingFor={pairPendingFor}
            boxLinkPendingFor={boxLinkPendingFor}
            onObject={actObject}
            onObjectNoBox={actObjectNoBox}
            onNoise={actNoise}
            onPair={actPair}
            onClear={actClear}
            onAutoLabel={runAutoLabel}
            canAutoLabel={!!frameData}
            labelHelpers={{ isObject, isNoise, isPaired, getPartner, getObjectBox }}
            onSave={save}
            onBeforeEdit={pushHistory}
            onUndo={undo}
            onRedo={redo}
            canUndo={history.length > 0}
            canRedo={redoStack.length > 0}
            logs={logs}
          />
        </div>

        <div className="cell cell-track">
          <div className="cell-title">Track History</div>
          <div className="cell-body"><TrackCanvas trackHistory={trackHistory} /></div>
        </div>

        <div className="cell cell-cluster">
          <div className="cell-title">Radar Detections — click ✕ to select</div>
          <div className="cell-body">
            <ClusterCanvas
              radarDetections={radarDetections}
              labeling={labeling}
              selectedRadar={selectedRadar}
              pairPendingFor={pairPendingFor}
              candidates={suggestions?.candidates || []}
              onRadarClick={handleRadarClick}
              labelHelpers={{ isObject, isNoise, getPartner }}
            />
          </div>
        </div>
      </div>

      {toast && <Toast msg={toast.msg} isError={toast.isError} />}
    </div>
  )
}