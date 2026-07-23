import { useRef, useEffect, useState, useCallback } from 'react'

const HANDLE_R = 6

export default function CameraCanvas({
  imageURL, boxes, setBoxes, selectedBox, setSelectedBox, onBeforeEdit, linkMode,
  brightness = 1
}) {
  const canvasRef = useRef(null)
  const imgRef    = useRef(new Image())
  const stateRef  = useRef({ boxes, selectedBox })
  const brightRef = useRef(1)   // display-only brightness; never touches image data

  const [drag, setDrag] = useState(null)

  useEffect(() => { stateRef.current = { boxes, selectedBox } }, [boxes, selectedBox])
  useEffect(() => { brightRef.current = brightness }, [brightness])

  useEffect(() => {
    if (!imageURL) return
    const img = new Image()
    img.onload = () => { imgRef.current = img; render() }
    img.src = imageURL
  }, [imageURL])

  useEffect(() => { render() }, [boxes, selectedBox, drag, brightness])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => {
      canvas.width  = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
      render()
    })
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [])

  const getScale = useCallback(() => {
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img.naturalWidth) return { scale:1, ox:0, oy:0 }
    const scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight)
    const ox = (canvas.width  - img.naturalWidth  * scale) / 2
    const oy = (canvas.height - img.naturalHeight * scale) / 2
    return { scale, ox, oy }
  }, [])

  const toImg = (cx, cy) => {
    const { scale, ox, oy } = getScale()
    return { x: (cx - ox) / scale, y: (cy - oy) / scale }
  }
  const toCanvas = (ix, iy) => {
    const { scale, ox, oy } = getScale()
    return { x: ix * scale + ox, y: iy * scale + oy }
  }
  const getPos = (e) => {
    const r = canvasRef.current.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  const hitHandle = (cx, cy) => {
    const { selectedBox, boxes } = stateRef.current
    if (selectedBox < 0) return null
    const b = boxes[selectedBox]
    const corners = [
      { corner:'tl', ix:b.pos1[0], iy:b.pos1[1] },
      { corner:'tr', ix:b.pos2[0], iy:b.pos1[1] },
      { corner:'bl', ix:b.pos1[0], iy:b.pos2[1] },
      { corner:'br', ix:b.pos2[0], iy:b.pos2[1] },
    ]
    for (const c of corners) {
      const { x, y } = toCanvas(c.ix, c.iy)
      if (Math.hypot(cx-x, cy-y) < HANDLE_R + 3) return { boxIdx: selectedBox, corner: c.corner }
    }
    return null
  }

  const hitBox = (cx, cy) => {
    const { boxes } = stateRef.current
    const { x:ix, y:iy } = toImg(cx, cy)
    for (let i = boxes.length-1; i >= 0; i--) {
      const b = boxes[i]
      if (ix >= b.pos1[0] && ix <= b.pos2[0] && iy >= b.pos1[1] && iy <= b.pos2[1]) return i
    }
    return -1
  }

  const onMouseDown = (e) => {
    if (linkMode) return   // in link mode, clicks select a box (handled by onClick), never draw
    const pos = getPos(e)
    const handle = hitHandle(pos.x, pos.y)
    if (handle) {
      onBeforeEdit?.()              // record history once at drag-start, not per-move
      setDrag({ type:'resize', handle, start: pos })
      return
    }
    const hit = hitBox(pos.x, pos.y)
    if (hit >= 0) { setSelectedBox(hit); return }
    setSelectedBox(-1)
    onBeforeEdit?.()                // record history once before starting a new box
    setDrag({ type:'new', start: pos, current: pos })
  }

  const onMouseMove = (e) => {
    if (!drag) return
    const pos = getPos(e)
    if (drag.type === 'new') {
      setDrag(d => ({ ...d, current: pos }))
    } else if (drag.type === 'resize') {
      const { x:ix, y:iy } = toImg(pos.x, pos.y)
      setBoxes(prev => {
        const next = prev.map(b => ({...b, pos1:[...b.pos1], pos2:[...b.pos2]}))
        const b = next[drag.handle.boxIdx]
        if (drag.handle.corner === 'tl') { b.pos1[0]=ix; b.pos1[1]=iy }
        if (drag.handle.corner === 'tr') { b.pos2[0]=ix; b.pos1[1]=iy }
        if (drag.handle.corner === 'bl') { b.pos1[0]=ix; b.pos2[1]=iy }
        if (drag.handle.corner === 'br') { b.pos2[0]=ix; b.pos2[1]=iy }
        return next
      })
    }
  }

  const onMouseUp = (e) => {
    if (!drag) return
    if (drag.type === 'new' && drag.current) {
      const p1 = toImg(drag.start.x,   drag.start.y)
      const p2 = toImg(drag.current.x, drag.current.y)
      if (Math.abs(p2.x-p1.x) > 5 && Math.abs(p2.y-p1.y) > 5) {
        setBoxes(prev => {
          // assign next id = max existing id + 1 (ids start at 1 if none yet)
          const maxId = prev.reduce((m, b) => Math.max(m, b.track_id ?? 0), 0)
          const newBox = {
            pos1: [Math.min(p1.x,p2.x), Math.min(p1.y,p2.y)],
            pos2: [Math.max(p1.x,p2.x), Math.max(p1.y,p2.y)],
            label: 'unknown', confidence: 1.0, track_id: maxId + 1, thickness: 2
          }
          setSelectedBox(prev.length)
          return [...prev, newBox]
        })
      }
    }
    setDrag(null)
  }

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    ctx.clearRect(0,0,W,H)

    const img = imgRef.current
    if (img.naturalWidth) {
      const { scale, ox, oy } = getScale()
      // Display-only brightness: filter wraps ONLY the photo draw call,
      // so boxes/labels keep their exact colors. Image data is untouched.
      const b = brightRef.current
      if (b !== 1) ctx.filter = `brightness(${b})`
      ctx.drawImage(img, ox, oy, img.naturalWidth*scale, img.naturalHeight*scale)
      ctx.filter = 'none'
    }

    const { boxes, selectedBox } = stateRef.current
    boxes.forEach((b, i) => {
      const { scale, ox, oy } = getScale()
      const x1 = b.pos1[0]*scale+ox, y1 = b.pos1[1]*scale+oy
      const x2 = b.pos2[0]*scale+ox, y2 = b.pos2[1]*scale+oy
      const isSel = i === selectedBox
      ctx.strokeStyle = isSel ? '#e94560' : '#2ecc71'
      ctx.lineWidth   = isSel ? 2.5 : 1.5
      ctx.strokeRect(x1,y1,x2-x1,y2-y1)
      ctx.fillStyle = isSel ? '#e94560' : '#2ecc71'
      ctx.font = '11px Segoe UI'
      const confTxt = b.confidence != null && b.confidence < 1 ? ` (${(b.confidence*100).toFixed(0)}%)` : ''
      ctx.fillText(`${b.label} #${b.track_id}${confTxt}`, x1+2, y1-4)
      if (isSel) {
        [[x1,y1],[x2,y1],[x1,y2],[x2,y2]].forEach(([hx,hy]) => {
          ctx.fillStyle = '#e94560'
          ctx.beginPath(); ctx.arc(hx,hy,HANDLE_R,0,Math.PI*2); ctx.fill()
        })
      }
    })

    if (drag?.type === 'new' && drag.current) {
      const { x:sx,y:sy } = drag.start
      const { x:cx,y:cy } = drag.current
      ctx.strokeStyle = '#f1c40f'
      ctx.lineWidth = 1.5
      ctx.setLineDash([4,3])
      ctx.strokeRect(sx,sy,cx-sx,cy-sy)
      ctx.setLineDash([])
    }
  }, [getScale, drag])

  const onClick = (e) => {
    if (!linkMode) return  // normal mode handles selection in onMouseDown
    const pos = getPos(e)
    const hit = hitBox(pos.x, pos.y)
    if (hit >= 0) setSelectedBox(hit)
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', cursor: linkMode ? 'pointer' : 'crosshair' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onClick={onClick}
    />
  )
}