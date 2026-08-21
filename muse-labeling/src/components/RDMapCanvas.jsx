  import { useRef, useEffect, useCallback, useState } from 'react'
  import { getSquareFrame } from '../utils/squareFrame'
  import { canvasTheme, trackColor } from '../utils/theme'

  const N = 256

  // Radar parameters
  const c = 3e8
  const fc = 24.125e9
  const lam = c / fc
  const BW = 554e6
  const clk = 38461538
  const delay = 2214

  const delta_v =
    (lam * clk * 3.6) /
    (2 * N * (12 * (N + 4) + delay))

  const Vmax = delta_v * Math.floor(N / 2)

  const range_bins = Array.from(
    { length: N },
    (_, i) => i * (c / (2 * BW))
  )

  const velocity_bins = Array.from(
    { length: N },
    (_, i) => i * delta_v - Vmax
  )

  // Display limits
  const VEL_MIN = velocity_bins[0]
  const VEL_MAX = velocity_bins[N - 1]

  const RNG_MIN = range_bins[0]
  const RNG_MAX = range_bins[N - 1]

  const PAD = 32
  const RING_BASE = 9
  const RING_GAP = 6


  function toCanvas(vel, range, frame) {
    const { size, offsetX, offsetY } = frame
    const x = offsetX + PAD + (vel - VEL_MIN) / (VEL_MAX - VEL_MIN) * (size - 2 * PAD)
    const y = offsetY + size - PAD - (range - RNG_MIN) / (RNG_MAX - RNG_MIN) * (size - 2 * PAD)
    return { x, y }
  }

  function findClusterAtMouse(radarClusters, mx, my, frame){
    let best = -1
    let bestDist = 15

    radarClusters.forEach((cluster, i) => {
      const {x,y} = toCanvas(cluster.radar_kmh, cluster.radar_m, frame)
      const dist = Math.hypot(mx - x, my - y)
      if (dist < bestDist){
        bestDist = dist
        best = i
      }
    })

    return best
  }

  function getMousePosition(canvas, e) {
    const rect = canvas.getBoundingClientRect()

    return {
      mx: e.clientX - rect.left,
      my: e.clientY - rect.top
    }
  }

  function canvasToRadar(mx, my, frame) {
    const { size, offsetX, offsetY } = frame
    const plotSize = size - 2 * PAD

    const velocityRaw = VEL_MIN +((mx - offsetX - PAD) / plotSize) * (VEL_MAX - VEL_MIN)

    const rangeRaw = RNG_MAX -((my - offsetY - PAD) / plotSize) * (RNG_MAX - RNG_MIN)

    const velocity = Math.max(VEL_MIN, Math.min(VEL_MAX, velocityRaw))

    const range = Math.max(RNG_MIN,Math.min(RNG_MAX, rangeRaw))

    return { velocity, range }
  }

  function isInsideRDMap(mx, my, frame){
    const { size, offsetX, offsetY } = frame

    return (
      mx >= offsetX + PAD &&
      mx <= offsetX + size - PAD &&
      my >= offsetY + PAD &&
      my <= offsetY + size - PAD
    )
  }

  export default function RDMapCanvas({data, radarClusters, selectedRadar, onRadarSelect, onRadarCreate, onRadarMove, onRadarMoveStart, theme = 'dark'}) {
    const canvasRef = useRef(null)

    const [draggingIndex, setDraggingIndex] = useState(null)
    const [didDrag, setDidDrag] = useState(false)
    const dragStartRef = useRef(null)

    const render = useCallback(() => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext('2d')

      const W = canvas.width
      const H = canvas.height

      const frame = getSquareFrame(canvas)

      const { size, offsetX, offsetY } = frame
      const C = canvasTheme(theme)
      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = C.bg
      ctx.fillRect(0, 0, W, H)

      // --------------------------------------------------
      // Plot area
      // --------------------------------------------------
      ctx.fillStyle = C.plot
      ctx.fillRect(offsetX, offsetY, size, size)
      ctx.strokeStyle = C.frame
      ctx.lineWidth = 1
      ctx.strokeRect(offsetX, offsetY, size, size)

      // --------------------------------------------------
      // RD image
      // --------------------------------------------------

      if (data && data.length >= N * N) {

        const plotSize = size - 2 * PAD
        const img = ctx.createImageData(N, N)

        for (let r = 0; r < N; r++) {
          for (let c = 0; c < N; c++) {
            const v = data[r * N + c]
            // log power
            const db =10 * Math.log10(v > 0 ? v : 1e-12)
            const norm = Math.max(0,Math.min(1,(db - 0) / (30 - 0)))
            // gray_r
            const g = Math.round(C.rdWhite * (1 - norm))
            const px = r
            const py = (N - 1) - c
            const idx = (py * N + px) * 4
            img.data[idx] = g
            img.data[idx + 1] = g
            img.data[idx + 2] = g
            img.data[idx + 3] = 255
          }
        }

        // Offscreen image
        const off = document.createElement('canvas')

        off.width = N
        off.height = N

        const offCtx = off.getContext('2d')

        offCtx.putImageData(img, 0, 0)

        // Draw image exactly inside plot area
        ctx.imageSmoothingEnabled = false

        ctx.drawImage(off, offsetX + PAD, offsetY + PAD, plotSize, plotSize)
      }


      // --------------------------------------------------
      // Grid
      // --------------------------------------------------

      ctx.strokeStyle = C.grid
      ctx.lineWidth = 0.5
      ctx.font = '9px Segoe UI'

      // Velocity grid
      const velocityStep = 20

      for (let v = Math.ceil(VEL_MIN / velocityStep) * velocityStep; v <= VEL_MAX; v += velocityStep) {
        const { x } =toCanvas(v, 0, frame)

        ctx.beginPath()
        ctx.moveTo(x, offsetY + PAD)
        ctx.lineTo(x, offsetY + size - PAD)
        ctx.stroke()

        ctx.fillStyle = C.tick
        ctx.textAlign = 'center'

        ctx.fillText( `${v}`, x, offsetY + size - PAD + 12)
      }


      // Range grid
      const rangeStep = 10

      for (let r = 0; r <= RNG_MAX; r += rangeStep) {

        const { y } = toCanvas(0, r, frame)

        ctx.beginPath()
        ctx.moveTo(offsetX + PAD, y)
        ctx.lineTo(offsetX + size - PAD, y)
        ctx.stroke()

        ctx.fillStyle = C.tick
        ctx.textAlign = 'right'

        ctx.fillText(
          `${r}m`,
          offsetX + PAD - 4,
          y + 3
        )
      }


      // --------------------------------------------------
      // Zero velocity axis
      // --------------------------------------------------

      const { x: zeroX } = toCanvas(0, 0, frame)

      ctx.strokeStyle = C.axis
      ctx.lineWidth = 1

      ctx.beginPath()
      ctx.moveTo( zeroX, offsetY + PAD)

      ctx.lineTo(zeroX,offsetY + size - PAD)

      ctx.stroke()


      // --------------------------------------------------
      // Axis labels
      // --------------------------------------------------

      ctx.fillStyle = C.tick
      ctx.font = '10px Segoe UI'
      ctx.textAlign = 'center'
      ctx.fillText('Velocity (km/h)', offsetX + size / 2, Math.min(offsetY + size + 12, H - 2))
      ctx.save()
      ctx.translate(Math.max(offsetX - 18, 8), offsetY + size / 2)
      ctx.rotate(-Math.PI / 2)
      ctx.fillText('Range (m)', 0,0)
      ctx.restore()

      if (radarClusters && radarClusters.length > 0){
          radarClusters.forEach((c,i) => {
            const id = c.id
            const baseColor = trackColor(C, id)

            const {x: cx, y:cy} = toCanvas(c.radar_kmh, c.radar_m, frame)
            const isSel = i === selectedRadar

            const xsz = isSel ? 6 : 4
            ctx.strokeStyle = isSel ? C.marker : baseColor
            ctx.lineWidth = isSel ? 2.5 : 1.5
            ctx.beginPath()
            ctx.moveTo(cx-xsz, cy-xsz); ctx.lineTo(cx+xsz, cy+xsz)
            ctx.moveTo(cx+xsz, cy-xsz); ctx.lineTo(cx-xsz, cy+xsz)
            ctx.stroke()
            ctx.fillStyle = baseColor; ctx.font = '9px Segoe UI'; ctx.textAlign = 'left'
            
            const outerMax = RING_BASE + RING_GAP
            ctx.fillText(`ID ${id}`, cx+outerMax+8, cy+3)
          })
      }

    }, [data, radarClusters, selectedRadar, theme])


    // --------------------------------------------------
    // Resize
    // --------------------------------------------------

    useEffect(() => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ro = new ResizeObserver(() => {
        canvas.width = canvas.offsetWidth
        canvas.height = canvas.offsetHeight
        render()
      })
      ro.observe(canvas)
      return () => ro.disconnect()
    }, [render])


    useEffect(() => {render()}, [render])


    const onMouseDown = (e) => {
      const { mx, my } = getMousePosition(canvasRef.current, e)
      const frame = getSquareFrame(canvasRef.current)
      const index = findClusterAtMouse(radarClusters, mx, my, frame)

      if (!isInsideRDMap(mx, my, frame)) {
        return
      }

      if (index >= 0) {
        onRadarSelect(index)
        onRadarMoveStart()
        setDraggingIndex(index)
        setDidDrag(false)
        dragStartRef.current = {x: mx, y: my}
      } else {
        setDraggingIndex(null)
        setDidDrag(false)
      }
    }

    const onMouseMove = (e) => {
      if (draggingIndex == null) return

      const { mx, my } = getMousePosition(canvasRef.current, e)

      const dx = mx - dragStartRef.current.x
      const dy = my - dragStartRef.current.y

      if (!didDrag) {
        if (Math.hypot(dx, dy) <= 3) return
        setDidDrag(true)
      }

      const frame = getSquareFrame(canvasRef.current)
      const { velocity, range } = canvasToRadar(mx, my, frame)

      onRadarMove(draggingIndex, velocity, range)
    }

    const onMouseUp = (e) => {
      const { mx, my } = getMousePosition(canvasRef.current, e)

      if (draggingIndex != null) {
        setDraggingIndex(null)
        return
      }

      if (!didDrag) {
        const frame = getSquareFrame(canvasRef.current)
        
        if (!isInsideRDMap(mx, my, frame)) {
          return
        }

        const { velocity, range } = canvasToRadar(mx, my, frame)

        onRadarCreate({radar_kmh: velocity,radar_m: range})
      }

      setDraggingIndex(null)
  }

    return (
      <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        cursor: draggingIndex != null
          ? 'grabbing'
          : 'crosshair'
      }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      />
    )
  }