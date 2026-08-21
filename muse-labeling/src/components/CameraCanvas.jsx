import { useRef, useEffect, useState, useCallback } from 'react'
import { canvasTheme, trackColor } from '../utils/theme'

const HANDLE_R = 6

const toCorners = (b) => ({
  x1: b.cx - b.width / 2,
  y1: b.cy - b.height / 2,
  x2: b.cx + b.width / 2,
  y2: b.cy + b.height / 2,
})
const fromCorners = (x1, y1, x2, y2) => ({
  cx: (x1 + x2) / 2,
  cy: (y1 + y2) / 2,
  width: Math.abs(x2 - x1),
  height: Math.abs(y2 - y1),
})

export default function CameraCanvas({
  imageURL, boxes, setBoxes, selectedBox, setSelectedBox, onBeforeEdit, linkMode,
  brightness = 1, theme="dark"
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
    const { x1, y1, x2, y2 } = toCorners(boxes[selectedBox])
    const corners = [
      { corner:'tl', ix:x1, iy:y1 },
      { corner:'tr', ix:x2, iy:y1 },
      { corner:'bl', ix:x1, iy:y2 },
      { corner:'br', ix:x2, iy:y2 },
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
      const { x1, y1, x2, y2 } = toCorners(boxes[i])
      if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) return i
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
      const { x:ixRaw, y:iyRaw } = toImg(pos.x, pos.y)
      const maxX = imgRef.current.naturalWidth
      const maxY = imgRef.current.naturalHeight
      const ix = Math.max(0, Math.min(maxX, ixRaw))
      const iy = Math.max(0, Math.min(maxY, iyRaw))
      setBoxes(prev => {
        const next = prev.map(b => ({ ...b }))
        const b = next[drag.handle.boxIdx]
        const { x1, y1, x2, y2 } = toCorners(b)
        // fixed corner = the one opposite the dragged corner
        let fx = x1, fy = y1, nx = ix, ny = iy
        if (drag.handle.corner === 'tl') { fx = x2; fy = y2 }
        if (drag.handle.corner === 'tr') { fx = x1; fy = y2 }
        if (drag.handle.corner === 'bl') { fx = x2; fy = y1 }
        if (drag.handle.corner === 'br') { fx = x1; fy = y1 }
        const merged = fromCorners(fx, fy, nx, ny)
        next[drag.handle.boxIdx] = { ...b, ...merged }
        return next
      })
    }
  }

  const onMouseUp = (e) => {
    if (!drag) return
    if (drag.type === 'new' && drag.current) {
      const p1Raw = toImg(drag.start.x, drag.start.y)
      const p2Raw = toImg(drag.current.x, drag.current.y)

      const maxX = imgRef.current.naturalWidth
      const maxY = imgRef.current.naturalHeight

      const p1 = {
        x: Math.max(0, Math.min(maxX, p1Raw.x)),
        y: Math.max(0, Math.min(maxY, p1Raw.y)),
      }

      const p2 = {
        x: Math.max(0, Math.min(maxX, p2Raw.x)),
        y: Math.max(0, Math.min(maxY, p2Raw.y)),
      }
      if (Math.abs(p2.x-p1.x) > 5 && Math.abs(p2.y-p1.y) > 5) {
        setBoxes(prev => {
          // assign next id = max existing id + 1 (ids start at 1 if none yet)
          const maxId = prev.reduce((m, b) => Math.max(m, b.id ?? 0), 0)
          const { cx, cy, width, height } = fromCorners(
            Math.min(p1.x,p2.x), Math.min(p1.y,p2.y),
            Math.max(p1.x,p2.x), Math.max(p1.y,p2.y)
          )
          const newBox = { id: maxId + 1, cx, cy, width, height, class: 'unknown' }
          setSelectedBox(prev.length)
          return [...prev, newBox]
        })
      }
    }
    setDrag(null)
  }

  const render = useCallback(() => {
    const C = canvasTheme(theme)
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
      const { x1, y1, x2, y2 } = toCorners(b)
      const cx1 = x1*scale+ox, cy1 = y1*scale+oy
      const cx2 = x2*scale+ox, cy2 = y2*scale+oy
      const isSel = i === selectedBox
      ctx.strokeStyle = isSel ? '#e94560' : trackColor(C, b.id)
      ctx.lineWidth   = isSel ? 2.5 : 1.5
      ctx.strokeRect(cx1,cy1,cx2-cx1,cy2-cy1)
      ctx.fillStyle = isSel ? '#e94560' : trackColor(C, b.id)
      ctx.font = '11px Segoe UI'
      ctx.fillText(`${b.class} #${b.id}`, cx1+2, cy1-4)
      if (isSel) {
        [[cx1,cy1],[cx2,cy1],[cx1,cy2],[cx2,cy2]].forEach(([hx,hy]) => {
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