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
import './App.css'

function frameIndexFromKey(key) {
  const m = key?.match(/frame_(\d+)\.jpeg/)
  return m ? parseInt(m[1], 10) : 0
}

function fileAccessErrorMessage(error) {
  if (error?.name === 'NotReadableError') {
    return [
      'Chrome could not read the selected data folder.',
      'Re-open the folder, make sure it was not moved or synced while loading,',
      'and grant Chrome access to Desktop/Documents in macOS Privacy settings if needed.'
    ].join(' ')
  }
  if (error?.name === 'NotAllowedError') {
    return 'Folder access was not allowed. Re-open the folder and approve read/write access.'
  }
  return error?.message || 'Unknown file access error'
}

const MAX_AUTO_LOAD_BYTES = 400 * 1024 * 1024

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return ''
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

function isLabelJsonName(name) {
  return /(^|_)yolo_tracking_data.*\.json(\.bak)?$/i.test(name)
}

export default function App() {
  // jsonData is the ONLY source of truth. Everything shown is derived from it.
  const [dirHandle, setDirHandle] = useState(null)
  const [jsonFiles, setJsonFiles] = useState([])
  const [activeJsonFile, setActiveJsonFile] = useState('')
  const [jsonData,  setJsonData]  = useState({})
  const [frameKeys, setFrameKeys] = useState([])
  const [frameIdx,  setFrameIdx]  = useState(0)
  const [isDirty, setIsDirty] = useState(false)

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
    const key = frameKeys[frameIdx]
    if (!key) return
    setIsDirty(true)
    setJsonData(prev => {
      if (!key || !prev[key]) return prev
      return { ...prev, [key]: { ...prev[key], ...patch } }
    })
  }

  const setBoxes = (updater) => {
    const key = frameKeys[frameIdx]
    if (!key) return
    setIsDirty(true)
    setJsonData(prev => {
      if (!key || !prev[key]) return prev
      const frame = prev[key]
      const prevBoxes = frame.box_detections || []
      const newBoxes = typeof updater === 'function' ? updater(prevBoxes) : updater
      return { ...prev, [key]: { ...frame, box_detections: newBoxes } }
    })
  }

  const setLabeling = (newLabeling) => patchFrame({ labeling: newLabeling })

  const clearDataset = () => {
    setJsonData({})
    setFrameKeys([])
    setFrameIdx(0)
    setHistory([])
    setRedoStack([])
    setSelectedBox(-1)
    setSelectedRadar(-1)
    setPairPendingFor(-1)
    setBoxLinkPendingFor(-1)
    setRdData(null)
    setCameraURL('')
    setIsDirty(false)
  }

  const listJsonFiles = async (dh) => {
    const files = []
    for await (const [name, handle] of dh.entries()) {
      if (handle.kind !== 'file' || !isLabelJsonName(name)) continue
      const file = await handle.getFile()
      files.push({ name, size: file.size })
    }
    return files.sort((a, b) => a.name.localeCompare(b.name))
  }

  const loadJsonFile = useCallback(async (fileName, handle = dirHandle) => {
    if (!handle || !fileName) return
    if (isDirty && fileName !== activeJsonFile && !window.confirm('Discard unsaved changes and open another JSON file?')) {
      return
    }
    try {
      const jh = await handle.getFileHandle(fileName)
      const jf = await jh.getFile()
      if (jf.size > MAX_AUTO_LOAD_BYTES) {
        clearDataset()
        setActiveJsonFile(fileName)
        const msg = `${fileName} is ${formatBytes(jf.size)}, too large to load safely in Chrome. Split it into smaller JSON files first.`
        showToast(msg, true)
        addLog(`[Error] ${msg}`)
        return
      }
      const data = JSON.parse(await jf.text())
      const keys = Object.keys(data).sort()
      setJsonData(data)
      setFrameKeys(keys)
      setFrameIdx(0)
      setHistory([])
      setRedoStack([])
      setActiveJsonFile(fileName)
      setIsDirty(false)
      addLog(`[System] Opened ${fileName} — ${keys.length} frames found`)
    } catch (e) {
      if (e.name !== 'AbortError') {
        const msg = fileAccessErrorMessage(e)
        showToast('Error: ' + msg, true)
        addLog(`[Error] ${msg}`)
      }
    }
  }, [dirHandle, isDirty, activeJsonFile])

  // ── Open folder ─────────────────────────────────────────────────────────
  const openFolder = async () => {
    if (isDirty && !window.confirm('Discard unsaved changes and open another folder?')) {
      return
    }
    try {
      const dh = await window.showDirectoryPicker({ mode: 'readwrite' })
      if (dh.queryPermission && dh.requestPermission) {
        const opts = { mode: 'readwrite' }
        const permission = await dh.queryPermission(opts)
        if (permission !== 'granted') {
          const requested = await dh.requestPermission(opts)
          if (requested !== 'granted') {
            throw new DOMException('Folder read/write access was denied', 'NotAllowedError')
          }
        }
      }
      const files = await listJsonFiles(dh)
      setDirHandle(dh)
      setJsonFiles(files)
      clearDataset()
      addLog(`[System] Opened folder — ${files.length} JSON files found`)

      const preferred = files.find(f => f.name === 'yolo_tracking_data.json') || files[0]
      if (!preferred) {
        showToast('No yolo_tracking_data JSON file found in this folder.', true)
        return
      }
      if (preferred.size > MAX_AUTO_LOAD_BYTES) {
        setActiveJsonFile(preferred.name)
        const msg = `${preferred.name} is ${formatBytes(preferred.size)}. Choose a smaller split JSON file, or run split_label_json.py first.`
        showToast(msg, true)
        addLog(`[System] ${msg}`)
        return
      }
      await loadJsonFile(preferred.name, dh)
    } catch (e) {
      if (e.name !== 'AbortError') {
        const msg = fileAccessErrorMessage(e)
        showToast('Error: ' + msg, true)
        addLog(`[Error] ${msg}`)
      }
    }
  }

  // ── Reset transient selection when navigating ──────────────────────────
  useEffect(() => {
    setSelectedBox(-1)
    setSelectedRadar(-1)
    setPairPendingFor(-1)
    setBoxLinkPendingFor(-1)
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
  }, [frameIdx, dirHandle, frameKeys, jsonData])

  // ── Save ────────────────────────────────────────────────────────────────
  const save = useCallback(async () => {
    if (!dirHandle || !activeJsonFile) return
    try {
      const fh = await dirHandle.getFileHandle(activeJsonFile, { create: true })
      const w  = await fh.createWritable()
      await w.write(JSON.stringify(jsonData, null, 2))
      await w.close()
      setIsDirty(false)
      showToast(`Saved ${activeJsonFile}`)
      addLog(`[Save] ${activeJsonFile} written to disk`)
    } catch (e) {
      showToast('Save failed: ' + e.message, true)
      addLog(`[Error] Save failed: ${e.message}`)
    }
  }, [dirHandle, activeJsonFile, jsonData])

  // ── Keyboard ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e) => {
      if (e.ctrlKey && e.key === 's') { e.preventDefault(); save(); return }
      if (['INPUT', 'SELECT'].includes(e.target.tagName)) return
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

  // ── Progress ─────────────────────────────────────────────────────────────
  const labeledCount = frameKeys.filter(k => {
    const l = jsonData[k]?.labeling
    return l && (Object.keys(l.object || {}).length || (l.noise || []).length || (l.pairs || []).length)
  }).length

  return (
    <div className="app">
      <TopBar
        onOpen={openFolder}
        jsonFiles={jsonFiles}
        activeJsonFile={activeJsonFile}
        onJsonFileChange={loadJsonFile}
        isDirty={isDirty}
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
          <div className="cell-title">
            Camera{boxLinkPendingFor >= 0 ? ' — click a box to link as object (Esc to cancel)' : ' — draw or adjust boxes'}
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
