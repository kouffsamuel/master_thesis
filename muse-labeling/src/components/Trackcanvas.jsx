import { useRef, useEffect, useCallback } from 'react'
import { getSquareFrame } from '../utils/squareFrame'
import { canvasTheme, trackColor } from '../utils/theme'

const VEL_MIN = -80, VEL_MAX = 80, RNG_MIN = 0, RNG_MAX = 70
const PAD = 32

function toCanvas(vel, range, frame) {
  const { size, offsetX, offsetY } = frame
  const x = offsetX + PAD + (vel - VEL_MIN) / (VEL_MAX - VEL_MIN) * (size - 2*PAD)
  const y = offsetY + size - PAD - (range - RNG_MIN) / (RNG_MAX - RNG_MIN) * (size - 2*PAD)
  return { x, y }
}

export default function TrackCanvas({ trackHistory, theme = 'dark' }) {
  const canvasRef = useRef(null)

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const frame = getSquareFrame(canvas)
    const { size, offsetX, offsetY } = frame
    const C = canvasTheme(theme)

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = C.bg
    ctx.fillRect(0, 0, W, H)

    ctx.fillStyle = C.plot                       // plot area, tinted off the bg
    ctx.fillRect(offsetX, offsetY, size, size)

    ctx.strokeStyle = C.frame
    ctx.lineWidth = 1
    ctx.strokeRect(offsetX, offsetY, size, size)

    // grid
    ctx.strokeStyle = C.grid
    ctx.lineWidth = 0.5
    ctx.font = '9px Segoe UI'
    for (let v = -80; v <= 80; v += 20) {
      const { x } = toCanvas(v, 0, frame)
      ctx.beginPath(); ctx.moveTo(x, offsetY+PAD); ctx.lineTo(x, offsetY+size-PAD); ctx.stroke()
      ctx.fillStyle = C.tick; ctx.textAlign = 'center'
      ctx.fillText(v, x, offsetY+size-PAD+12)
    }
    for (let r = 0; r <= 70; r += 10) {
      const { y } = toCanvas(0, r, frame)
      ctx.beginPath(); ctx.moveTo(offsetX+PAD, y); ctx.lineTo(offsetX+size-PAD, y); ctx.stroke()
      ctx.fillStyle = C.tick; ctx.textAlign = 'right'
      ctx.fillText(r+'m', offsetX+PAD-4, y+3)
    }

    // v=0 axis
    const { x: zx } = toCanvas(0, 0, frame)
    ctx.strokeStyle = C.axis; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(zx, offsetY+PAD); ctx.lineTo(zx, offsetY+size-PAD); ctx.stroke()

    ctx.fillStyle = C.tick; ctx.font = '10px Segoe UI'; ctx.textAlign = 'center'
    ctx.fillText('Velocity (km/h)', offsetX+size/2, Math.min(offsetY+size+12, H-2))
    ctx.save(); ctx.translate(Math.max(offsetX-18, 8), offsetY+size/2); ctx.rotate(-Math.PI/2)
    ctx.fillText('Range (m)', 0, 0); ctx.restore()

    if (!trackHistory.length) return

    trackHistory.forEach(tr => {
      if (!tr.history || tr.history.length < 2) return
      const color = trackColor(C, tr.track_id)

      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()
      tr.history.forEach((pt, i) => {
        const { x, y } = toCanvas(pt.velocity_kmh, pt.range_m, frame)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.stroke()

      const last = tr.history[tr.history.length - 1]
      const { x: lx, y: ly } = toCanvas(last.velocity_kmh, last.range_m, frame)
      ctx.fillStyle = color
      ctx.beginPath(); ctx.arc(lx, ly, 4, 0, Math.PI*2); ctx.fill()

      ctx.fillStyle = color; ctx.font = '9px Segoe UI'; ctx.textAlign = 'left'
      ctx.fillText(`ID ${tr.track_id}`, lx+6, ly+3)
    })
  }, [trackHistory, theme])

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
  }, [render])

  useEffect(() => { render() }, [render])

  return (
    <canvas
      ref={canvasRef}
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%' }}
    />
  )
}
