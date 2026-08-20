import { useState, useEffect, useCallback, useRef } from 'react'
import TopBar from './components/TopBar'
import CameraCanvas from './components/CameraCanvas'
import ClusterCanvas from './components/ClusterCanvas'
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
import { loadTheme, saveTheme } from './utils/theme'
import './App.css'
import LidarCanvas from './components/LidarCanvas'

function frameIndexFromKey(key) {
  const m = key?.match(/frame_(\d+)\.jpeg/)
  return m ? parseInt(m[1], 10) : 0
}

// A frame counts as labeled once it carries any decision at all.
function hasLabels(l) {
  return !!l && !!(
    Object.keys(l.object || {}).length || (l.noise || []).length || (l.pairs || []).length
  )
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

  const [rdData,    setRdData]    = useState(null)   // Float32Array of rd_power
  const [cameraURL, setCameraURL] = useState('')

  const [toast,     setToast]     = useState(null)
  const [logs,       setLogs]     = useState([])
  // Display-only camera brightness (1 = original). Lives outside jsonData
  // and the undo system on purpose — it never modifies any data.
  const [brightness, setBrightness] = useState(1)
  // Algorithm suggestions (Stage 3): display-only, never persisted to JSON.
  // { candidates: [trackId], needsReview: bool } | null
  const [suggestions, setSuggestions] = useState(null)
  // Switch: re-run Auto-label on every frame change. Session-only — it drives
  // writes into jsonData, so it deliberately doesn't survive a reload.
  const [autoLabelAlways, setAutoLabelAlways] = useState(false)
  const autoAlwaysRef = useRef(autoLabelAlways)
  useEffect(() => { autoAlwaysRef.current = autoLabelAlways }, [autoLabelAlways])

  const [history,   setHistory]   = useState([])
  const [redoStack, setRedoStack] = useState([])

  // Display-only theme: drives the CSS variables via <html data-theme> and is
  // passed to the canvases (they can't read CSS variables). Never touches data.
  const [theme, setTheme] = useState(loadTheme)
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    saveTheme(theme)
  }, [theme])
  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'))

  const addLog = (msg) => setLogs(prev => [...prev.slice(-99), msg])

  // ── Derived view of current frame ──────────────────────────────────────
  const currentFrame = frameKeys[frameIdx]
  const frameData       = currentFrame ? (jsonData[currentFrame.session]?.[currentFrame.ts] ?? null) : null
  const boxesData       = frameData?.camera_detections || []
  const radarClusters = frameData?.radar_clusters  || []
  const lidarDetections = frameData?.lidar_clusters || []
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
    const cur = frameKeys[frameIdx]
    if (!cur) return prev
    const sessionObj = prev[cur.session]
    const frame = sessionObj?.[cur.ts]
    if (!frame) return prev
    const prevBoxes = frame.camera_detections || []
    const newBoxes = typeof updater === 'function' ? updater(prevBoxes) : updater
    return {
      ...prev,
      [cur.session]: {
        ...sessionObj,
        [cur.ts]: { ...frame, camera_detections: newBoxes }
      }
    }
  })
}

  const setLabeling = (newLabeling) => patchFrame({ labeling: newLabeling })

  // ── Open folder ─────────────────────────────────────────────────────────
  const openFolder = async () => {
    try {
      const dh = await window.showDirectoryPicker({ mode: 'readwrite' })
      const jh = await dh.getFileHandle('labels.json')
      const jf = await jh.getFile()

      const data = JSON.parse(await jf.text())
      
      const keys = Object.keys(data).sort().flatMap(session =>
      Object.keys(data[session])
        .sort((a, b) => parseFloat(a) - parseFloat(b))
        .map(ts => ({ session, ts }))
      )
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
  const cur = frameKeys[frameIdx]
  const frame = cur ? jsonData[cur.session]?.[cur.ts] : null

  const tCamera   = frame?.t_camera
  const tRadar = frame?.t_radar
  ;(async () => {
    if (tCamera != null) {
      const path = `camera/${tCamera}.jpeg`
      
      try {
        setCameraURL(await getFileURL(dirHandle, path))
      } catch (e) {
        setCameraURL('')
        addLog(`[Error] Camera image not found: ${path} (${e.message})`)
      }
    } else {
      setCameraURL('')
      addLog(`[Warn] Frame ${cur?.session}/${cur?.ts} has no t_camera`)
    }

    if (tRadar != null) {
      const path = `radar/${tRadar}.raw`
      try{
        setRdData(await getFileFloat32(dirHandle, path))
      }catch(e){
        setRdData('')
        addLog(`[Error] Radar data not found: ${path} (${e.message})`)
      }
    } else {
      setRdData('')
      addLog(`[Warn] Frame ${cur?.session}/${cur?.ts} has no t_radar`)
    }
    addLog(`[Frame ${cur ? `${cur.session}/${cur.ts}` : '?'}] Loaded`)
  })()
}, [frameIdx, dirHandle, frameKeys])

  // ── Save ────────────────────────────────────────────────────────────────
  const save = useCallback(async () => {
    if (!dirHandle) return
    try {
      const fh = await dirHandle.getFileHandle('labels.json', { create: true })
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
      if (e.key === 'Delete') {
        if (selectedBox >= 0) {
          pushHistory()
          setBoxes(prev => prev.filter((_, i) => i !== selectedBox))
          setSelectedBox(-1)
          return
        }

        if (selectedRadar >= 0) {
          handleRadarDelete(selectedRadar)
          return
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)

  }, [ frameIdx, save, undo, redo, selectedBox,selectedRadar, pushHistory])

  const gotoFrame = (idx) => {
    if (!frameKeys.length) return
    setFrameIdx(Math.max(0, Math.min(frameKeys.length - 1, idx)))
  }

  const showToast = (msg, isError = false) => {
    setToast({ msg, isError })
    setTimeout(() => setToast(null), 1800)
  }

  // ── Radar point click ────────────────────────────────────────────────────
  const handleRadarCreate = ({radar_kmh, radar_m}) => {
    const clusters = frameData?.radar_clusters || []
    const usedIds = new Set(clusters.map(c => c.id))

    let newId = 0
    while(usedIds.has(newId)){
      newId++
    }

    const newCluster = {
      id:newId,
      radar_kmh,
      radar_m
    }

    pushHistory()

    setJsonData(prev => {
      const cur = frameKeys[frameIdx]
      if (!cur) return prev

      const sessionObj = prev[cur.session]
      const frame = sessionObj?.[cur.ts]

      if(!frame) return prev

      return {
        ...prev,
        [cur.session]: {
          ...sessionObj,
          [cur.ts]: {
            ...frame,
            radar_clusters: [
              ...(frame.radar_clusters || []),
              newCluster
            ]
          }
        }
      }
    })

    setSelectedRadar(clusters.length)

    addLog(
      `[Radar] Cluster ${newId} created at ` +
      `${radar_kmh.toFixed(2)} km/h, ${radar_m.toFixed(2)} m`
    )
  }

  const handleRadarMove = (index, radar_kmh, radar_m) => {
    setJsonData(prev => {
      const cur = frameKeys[frameIdx]
      if (!cur) return prev

      const sessionObj = prev[cur.session]
      const frame = sessionObj?.[cur.ts]
      if (!frame) return prev

      const clusters = frame.radar_clusters || []

      if (!clusters[index]) return prev

      const newClusters = clusters.map((cluster, i) =>
        i === index
          ? {
              ...cluster,
              radar_kmh,
              radar_m
            }
          : cluster
      )

      return {
        ...prev,
        [cur.session]: {
          ...sessionObj,
          [cur.ts]: {
            ...frame,
            radar_clusters: newClusters
          }
        }
      }
    })
  }

  const handleRadarDelete = (index) => {
    if (index == null || index < 0) return

    const cluster = radarClusters[index]
    if (!cluster) return

    pushHistory()

    setJsonData(prev => {
      const cur = frameKeys[frameIdx]
      if (!cur) return prev

      const sessionObj = prev[cur.session]
      const frame = sessionObj?.[cur.ts]
      if (!frame) return prev

      const clusters = frame.radar_clusters || []

      return {
        ...prev,
        [cur.session]: {
          ...sessionObj,
          [cur.ts]: {
            ...frame,
            radar_clusters: clusters.filter((_, i) => i !== index)
          }
        }
      }
    })

    setSelectedRadar(-1)

    addLog(`[Radar] Cluster ${cluster.id} deleted`)
  }

  const handleRadarMoveStart = () => {
    pushHistory()
  }

  // ── Box click (used both for normal select AND for object→box linking) ──
  const handleBoxClick = (idx) => {
    setSelectedBox(idx)
  }

  const handleRadarEdit = (index, field, value) => {
    setJsonData(prev => {
      const cur = frameKeys[frameIdx]
      if (!cur) return prev

      const sessionObj = prev[cur.session]
      const frame = sessionObj?.[cur.ts]

      if (!frame) return prev

      const clusters = frame.radar_clusters || []

      if (!clusters[index]) return prev

      const newClusters = clusters.map((cluster, i) =>
        i === index
          ? {
              ...cluster,
              [field]: value
            }
          : cluster
      )

      return {
        ...prev,
        [cur.session]: {
          ...sessionObj,
          [cur.ts]: {
            ...frame,
            radar_clusters: newClusters
          }
        }
      }
    })
  }


  // Frame changed (or a folder was just opened) with the switch on.
  // Declared after the selection-reset effect on purpose: effects fire in
  // declaration order, so `suggestions` is cleared before this refills it.
  useEffect(() => { 
  }, [frameIdx, frameKeys])

  // ── Progress ─────────────────────────────────────────────────────────────
  const labeledCount = frameKeys.filter(k => hasLabels(jsonData[k]?.labeling)).length

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
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <div className="main-grid">
        <div className="cell cell-rd">
          <div className="cell-title">Range-Doppler Map</div>
          {/* <div className="cell-body"><StaticImage src={rdURL} alt="RD map" /></div> */}
          <div className="cell-body">
            <RDMapCanvas data={rdData} 
            radarClusters={radarClusters} 
            selectedRadar={selectedRadar} 
            onRadarSelect={setSelectedRadar}
            onRadarCreate={handleRadarCreate}
            onRadarMove={handleRadarMove}
            onRadarMoveStart={handleRadarMoveStart}
            theme={theme} /></div>
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

            radarDetections={radarClusters}
            selectedRadar={selectedRadar}
            onRadarEdit={handleRadarEdit}

            onSave={save}
            onBeforeEdit={pushHistory}
            onUndo={undo}
            onRedo={redo}
            canUndo={history.length > 0}
            canRedo={redoStack.length > 0}
            logs={logs}
          />
        </div>

        <div className="cell cell-lidar">
          <div className="cell-title">Lidar data</div>
          {/* <div className="cell-body"><LidarCanvas  theme={theme} /></div> */}
        </div>

        {/* <div className="cell cell-cluster">
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
              theme={theme}
            />
          </div>
        </div> */}
      </div>

      {toast && <Toast msg={toast.msg} isError={toast.isError} />}
    </div>
  )
}