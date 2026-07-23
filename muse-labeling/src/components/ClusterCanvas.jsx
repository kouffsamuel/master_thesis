import { useRef, useEffect, useCallback } from 'react'
import { getSquareFrame } from '../utils/squareFrame'

const TAB20 = [
  '#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#2ca02c',
  '#98df8a','#d62728','#ff9896','#9467bd','#c5b0d5',
  '#8c564b','#c49c94','#e377c2','#f7b6d2','#7f7f7f',
  '#c7c7c7','#bcbd22','#dbdb8d','#17becf','#9edae5'
]
const OBJ_COLOR = '#2ecc71'
const NOISE_COLOR = '#7f8c8d'
const PAIR_PALETTE = ['#f39c12','#e74c3c','#9b59b6','#1abc9c','#3498db','#e67e22','#16a085','#e91e63']

const VEL_MIN = -80, VEL_MAX = 80, RNG_MIN = 0, RNG_MAX = 70
const PAD = 32
const RING_BASE = 9
const RING_GAP = 6

function toCanvas(vel, range, frame) {
  const { size, offsetX, offsetY } = frame
  const x = offsetX + PAD + (vel - VEL_MIN) / (VEL_MAX - VEL_MIN) * (size - 2*PAD)
  const y = offsetY + size - PAD - (range - RNG_MIN) / (RNG_MAX - RNG_MIN) * (size - 2*PAD)
  return { x, y }
}

function pairColorFor(radarId, partnerId) {
  const key = Math.min(radarId, partnerId)
  const i = ((key % PAIR_PALETTE.length) + PAIR_PALETTE.length) % PAIR_PALETTE.length
  return PAIR_PALETTE[i]
}

export default function ClusterCanvas({
  radarDetections, labeling, selectedRadar, pairPendingFor,
  candidates = [],   // Stage 3 nominations (display-only)
  onRadarClick, labelHelpers
}) {
  const canvasRef = useRef(null)
  const { isObject, isNoise, getPartner } = labelHelpers

  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width, H = canvas.height
    const frame = getSquareFrame(canvas)
    const { size, offsetX, offsetY } = frame

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = '#0d1117'
    ctx.fillRect(0, 0, W, H)
    ctx.strokeStyle = '#22324a'; ctx.lineWidth = 1
    ctx.strokeRect(offsetX, offsetY, size, size)

    // grid
    ctx.strokeStyle = '#1e2d3d'; ctx.lineWidth = 0.5; ctx.font = '9px Segoe UI'
    for (let v = -80; v <= 80; v += 20) {
      const { x } = toCanvas(v, 0, frame)
      ctx.beginPath(); ctx.moveTo(x, offsetY+PAD); ctx.lineTo(x, offsetY+size-PAD); ctx.stroke()
      ctx.fillStyle = '#555'; ctx.textAlign = 'center'; ctx.fillText(v, x, offsetY+size-PAD+12)
    }
    for (let r = 0; r <= 70; r += 10) {
      const { y } = toCanvas(0, r, frame)
      ctx.beginPath(); ctx.moveTo(offsetX+PAD, y); ctx.lineTo(offsetX+size-PAD, y); ctx.stroke()
      ctx.fillStyle = '#555'; ctx.textAlign = 'right'; ctx.fillText(r+'m', offsetX+PAD-4, y+3)
    }
    const { x: zx } = toCanvas(0, 0, frame)
    ctx.strokeStyle = '#2c3e50'; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(zx, offsetY+PAD); ctx.lineTo(zx, offsetY+size-PAD); ctx.stroke()
    ctx.fillStyle = '#555'; ctx.font = '10px Segoe UI'; ctx.textAlign = 'center'
    ctx.fillText('Velocity (km/h)', offsetX+size/2, Math.min(offsetY+size+12, H-2))
    ctx.save(); ctx.translate(Math.max(offsetX-18, 8), offsetY+size/2); ctx.rotate(-Math.PI/2)
    ctx.fillText('Range (m)', 0, 0); ctx.restore()

    if (!radarDetections.length) return

    radarDetections.forEach((d, i) => {
      const radarId = d.track_id
      const baseColor = TAB20[(radarId >= 0 ? radarId : 19) % 20]

      // raw points
      if (d.points) {
        d.points.forEach(p => {
          const { x, y } = toCanvas(p.velocity_kmh, p.range_m, frame)
          ctx.fillStyle = baseColor + '55'; ctx.fillRect(x-2, y-2, 4, 4)
        })
      }

      const { x: cx, y: cy } = toCanvas(d.centroid.velocity_kmh, d.centroid.range_m, frame)
      const isSel = i === selectedRadar
      const isPend = i === pairPendingFor

      // base X marker
      const xsz = isSel ? 6 : 4
      ctx.strokeStyle = isSel ? '#fff' : baseColor
      ctx.lineWidth = isSel ? 2.5 : 1.5
      ctx.beginPath()
      ctx.moveTo(cx-xsz, cy-xsz); ctx.lineTo(cx+xsz, cy+xsz)
      ctx.moveTo(cx+xsz, cy-xsz); ctx.lineTo(cx-xsz, cy+xsz)
      ctx.stroke()

      // labeling outlines — read from frame-level labeling object
      const obj = isObject(labeling, radarId)
      const noi = isNoise(labeling, radarId)
      const partner = getPartner(labeling, radarId)
      ctx.lineWidth = 2
      const hasInner = obj || noi

      if (obj) {
        ctx.strokeStyle = OBJ_COLOR
        ctx.beginPath(); ctx.arc(cx, cy, RING_BASE, 0, Math.PI*2); ctx.stroke()
      }
      if (noi) {
        ctx.strokeStyle = NOISE_COLOR
        ctx.beginPath()
        ctx.moveTo(cx, cy-RING_BASE); ctx.lineTo(cx+RING_BASE, cy+RING_BASE); ctx.lineTo(cx-RING_BASE, cy+RING_BASE)
        ctx.closePath(); ctx.stroke()
      }
      if (partner != null) {
        const pr = hasInner ? RING_BASE + RING_GAP : RING_BASE
        ctx.strokeStyle = pairColorFor(radarId, partner)
        ctx.strokeRect(cx-pr, cy-pr, pr*2, pr*2)
      }

      const outerMax = RING_BASE + RING_GAP
      // Stage 3 candidate: dashed gold ring — a suggestion awaiting the
      // human's object↔box link, visually distinct from decided labels.
      if (candidates.includes(radarId)) {
        ctx.strokeStyle = '#f0c419'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3])
        ctx.beginPath(); ctx.arc(cx, cy, outerMax + 3, 0, Math.PI*2); ctx.stroke()
        ctx.setLineDash([])
      }
      if (isSel) {
        ctx.strokeStyle = '#e94560'; ctx.lineWidth = 2
        ctx.beginPath(); ctx.arc(cx, cy, outerMax+6, 0, Math.PI*2); ctx.stroke()
      }
      if (isPend) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.setLineDash([3,2])
        ctx.beginPath(); ctx.arc(cx, cy, outerMax+10, 0, Math.PI*2); ctx.stroke()
        ctx.setLineDash([])
      }

      ctx.fillStyle = baseColor; ctx.font = '9px Segoe UI'; ctx.textAlign = 'left'
      ctx.fillText(`ID ${radarId}`, cx+outerMax+8, cy+3)
      if (!d.is_confirmed) {
        ctx.fillStyle = '#555'; ctx.fillText('unconfirmed', cx+outerMax+8, cy+13)
      }
    })
  }, [radarDetections, labeling, selectedRadar, pairPendingFor, candidates, isObject, isNoise, getPartner])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth; canvas.height = canvas.offsetHeight; render()
    })
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [render])

  useEffect(() => { render() }, [render])

  const onClick = (e) => {
    if (!radarDetections.length) return
    const canvas = canvasRef.current
    const r = canvas.getBoundingClientRect()
    const mx = e.clientX - r.left, my = e.clientY - r.top
    const frame = getSquareFrame(canvas)
    let best = -1, bestD = 18
    radarDetections.forEach((d, i) => {
      const { x, y } = toCanvas(d.centroid.velocity_kmh, d.centroid.range_m, frame)
      const dist = Math.hypot(mx-x, my-y)
      if (dist < bestD) { bestD = dist; best = i }
    })
    if (best >= 0) onRadarClick(best)
  }

  return (
    <canvas
      ref={canvasRef}
      style={{ position:'absolute', top:0, left:0, width:'100%', height:'100%', cursor:'pointer' }}
      onClick={onClick}
    />
  )
}